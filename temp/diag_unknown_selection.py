"""
[PHASE-A] Frozen-checkpoint diagnostic of pseudo-unknown selection (no training, no changes to core/).

What it does
  1. Registers the TRAIN split with TEST.MASK=0 (a cloned cfg used only for data) so unknown GT are loaded too.
  2. Runs the training forward path of RandBox (same proposal sampler as the checkpoint's cfg) on N images,
     calling the real matcher with KNOWN GT only (as in training), and caches per-proposal quantities.
  3. Labels every proposal by its final-stage box: known-dup / true-unknown(<class>) / background / ambiguous.
  4. Computes signals o (objectness), k (max known prob), g (max IoU to known GT), c (stage convergence),
     n (feature novelty vs batch known prototype) and reports their distributions + AUROC per label group.
  5. Runs several selectors on the SAME candidate set (~excl, as NC-SELECT builds it from the checkpoint cfg)
     and reports the composition of what each selects (bus/truck/motor share, known-dup, background, dup rate).

Usage (server):
  python tools/diag_unknown_selection.py --config-file configs/<BENCH>/t1.yaml --task <BENCH>/t1 \
      --ckpt R3p=output/mix/R3p_t1/model_final.pth --ckpt R0=output/orthogonaldet/baseline/t1/model_final.pth \
      --num-images 400 --out docs/results/phaseA \
      PROPOSAL.SAMPLER mixture PROPOSAL.SCALE_POOL_PATH output/scale_pool_<BENCH>_t1.npy \
      MODEL.NC_EXCLUDE all MODEL.NC_EXCLUDE_IOU 0.5 MODEL.NC_NMS_IOU 0.5 MODEL.NC_BACKFILL True
  (pass the SAME cfg overrides the checkpoint was trained with; the R0 checkpoint can be run in a second call
   with its own overrides, or in the same call if the only differences are NC_*/PROPOSAL.* — they are read per run
   via --ckpt NAME=PATH[;KEY=VAL;KEY=VAL] optional per-checkpoint overrides.)
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torchvision.ops as ops

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.structures import Instances

from core import DatasetMapper, add_config
from core.pascal_voc import register_pascal_voc
from core.util.model_ema import add_model_ema_configs, may_get_ema_checkpointer, EMADetectionCheckpointer, \
    apply_model_ema_and_restore

IOU_POS = 0.5      # proposal counts as "on" a GT
IOU_BG = 0.3       # below this to every GT -> background
NONLOOKALIKE = {"bus", "truck", "motor", "bicycle", "tricycle", "awning-tricycle"}   # VisDrone t1 unknowns that are not car-like


# ----------------------------------------------------------------------------------------------- cfg / data
def build_cfg(config_file, opts):
    cfg = get_cfg(); add_config(cfg); add_model_ema_configs(cfg)
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    return cfg


def register_train_with_unknowns(cfg, task, dataset_root, name):
    """Register the train split with TEST.MASK=0 on a cloned cfg -> all GT (known + unknown) are loaded."""
    data_cfg = cfg.clone(); data_cfg.defrost(); data_cfg.TEST.MASK = 0; data_cfg.freeze()
    super_split = task.split("/")[0]
    register_pascal_voc(name, dataset_root, super_split, task, data_cfg)
    return MetadataCatalog.get(name).thing_classes


def load_model(cfg, weights):
    model = build_model(cfg)
    kwargs = may_get_ema_checkpointer(cfg, model)
    if cfg.MODEL_EMA.ENABLED:
        EMADetectionCheckpointer(model, save_dir="/tmp", **kwargs).resume_or_load(weights, resume=False)
    else:
        DetectionCheckpointer(model, save_dir="/tmp", **kwargs).resume_or_load(weights, resume=False)
    model.eval()
    return model


# ----------------------------------------------------------------------------------------------- helpers
def split_instances(inst, n_known):
    known = inst[inst.gt_classes < n_known]
    unknown = inst[inst.gt_classes >= n_known]
    return known, unknown


def auroc(pos, neg):
    """Mann-Whitney AUROC, numpy, NaN-safe."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort().astype(float)
    # average ranks for ties
    order = np.argsort(allv); sorted_v = allv[order]; r = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg)))


def connected_components_iou(boxes, thr):
    """Union-find over pairs with IoU > thr. boxes (n,4). Returns comp id per box (n,)."""
    n = boxes.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    if n > 1:
        iou = ops.box_iou(boxes, boxes)
        ii, jj = torch.nonzero(torch.triu(iou > thr, diagonal=1), as_tuple=True)
        for a, b in zip(ii.tolist(), jj.tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    return torch.tensor([find(i) for i in range(n)])


def sinkhorn_log(C, a, b, eps, n_iter=200, tau_row=None, tau_col=None):
    """Entropic (unbalanced) OT in log domain. C (n,m), a (n,), b (m,) positive masses.
    tau_row/tau_col: KL relaxation strengths (None = exact marginal). Returns plan P (n,m)."""
    la, lb = torch.log(a.clamp_min(1e-12)), torch.log(b.clamp_min(1e-12))
    f = torch.zeros_like(a); g = torch.zeros_like(b)
    fr = 1.0 if tau_row is None else tau_row / (tau_row + eps)
    fc = 1.0 if tau_col is None else tau_col / (tau_col + eps)
    for _ in range(n_iter):
        f = fr * (eps * la - eps * torch.logsumexp((-C + g[None, :]) / eps, dim=1))
        g = fc * (eps * lb - eps * torch.logsumexp((-C + f[:, None]) / eps, dim=0))
    return torch.exp((f[:, None] + g[None, :] - C) / eps)


# ----------------------------------------------------------------------------------------------- per-image record
@torch.no_grad()
def run_batch(model, batched_inputs, n_known, class_names, nc_cfg):
    """Replicates RandBox.forward training path and returns one record per image."""
    device = model.device
    images, images_whwh = model.preprocess_image(batched_inputs)
    src = model.backbone(images.tensor)
    features = [src[f] for f in model.in_features]
    gt_all = [x["instances"].to(device) for x in batched_inputs]
    known_list, unknown_list = zip(*[split_instances(g, n_known) for g in gt_all])

    targets, x_boxes, _, t = model.prepare_targets(list(known_list))
    t = t.squeeze(-1)
    x_boxes = x_boxes * images_whwh[:, None, :]
    x_boxes = model._enforce_min_size(x_boxes)
    cls, obj, coord, feat = model.head(features, x_boxes, t, None)         # (6,B,N,*)
    output = {'pred_logits': cls[-1], 'pred_objectness': obj[-1], 'pred_boxes': coord[-1],
              'pred_proposal_features': feat[-1], 'all_stage_boxes': coord, 'init_boxes': x_boxes}
    matcher = model.criterion.matcher
    indices, matched_ids, ow_indices, _ = matcher(output, targets)      # final stage, known GT only

    recs = []
    for b in range(len(batched_inputs)):
        pb = coord[-1, b]                                  # (N,4) final boxes
        o = obj[-1, b].squeeze(-1).float()
        prob = torch.softmax(cls[-1, b].float(), dim=-1)
        k = prob[:, :n_known].amax(dim=-1)
        kb = known_list[b].gt_boxes.tensor; ub = unknown_list[b].gt_boxes.tensor
        iou_k = ops.box_iou(pb, kb) if len(kb) else pb.new_zeros((pb.shape[0], 0))
        iou_u = ops.box_iou(pb, ub) if len(ub) else pb.new_zeros((pb.shape[0], 0))
        g = iou_k.max(1).values if iou_k.shape[1] else torch.zeros_like(o)
        gu = iou_u.max(1).values if iou_u.shape[1] else torch.zeros_like(o)
        # labels
        lab = np.full(pb.shape[0], "ambiguous", dtype=object)
        best_all = torch.maximum(g, gu)
        lab[(best_all < IOU_BG).cpu().numpy()] = "background"
        on_k = (g >= IOU_POS) & (g >= gu); on_u = (gu >= IOU_POS) & (gu > g)
        lab[on_k.cpu().numpy()] = "known-dup"
        if iou_u.shape[1]:
            ucls = unknown_list[b].gt_classes[iou_u.argmax(1)].cpu().numpy()
            for i in np.nonzero(on_u.cpu().numpy())[0]:
                lab[i] = "unk:" + class_names[int(ucls[i])]
        # signals
        stages = coord[:, b]                                               # (6,N,4)
        c = torch.stack([ops.box_iou(stages[s], stages[-1]).diagonal() for s in range(2, 5)]).mean(0)  # stages 3..5 vs 6
        travel = ops.box_iou(x_boxes[b], stages[-1]).diagonal()            # init vs final
        pos_mask = indices[b][0].bool()
        fb = torch.nn.functional.normalize(feat[-1, b].float(), dim=-1)
        if pos_mask.sum() > 0:
            proto = torch.nn.functional.normalize(fb[pos_mask].mean(0, keepdim=True), dim=-1)
            n = 1.0 - (fb @ proto.T).squeeze(-1)
        else:
            n = torch.full_like(o, float("nan"))
        # dynamic_k per GT (replica of dynamic_k_matching step 1)
        if iou_k.shape[1]:
            topk = torch.topk(iou_k, min(matcher.ota_k, iou_k.shape[0]), dim=0).values
            dyn_k = torch.clamp(topk.sum(0).int(), min=1).cpu().numpy()
        else:
            dyn_k = np.zeros(0, int)
        # exclusion mask exactly as NC-SELECT builds it from cfg
        excl = torch.zeros(pb.shape[0], dtype=torch.bool, device=device)
        if nc_cfg.NC_EXCLUDE == "all":
            excl |= pos_mask
        else:
            excl[matched_ids[b].long()] = True
        if nc_cfg.NC_EXCLUDE_IOU > 0 and iou_k.shape[1]:
            excl |= g > nc_cfg.NC_EXCLUDE_IOU
        if nc_cfg.NC_NMS_IOU > 0:
            cand = torch.nonzero(~excl, as_tuple=True)[0]
            if cand.numel() > 1:
                keep = ops.nms(pb[cand], o[cand], nc_cfg.NC_NMS_IOU)
                sup = torch.ones_like(cand, dtype=torch.bool); sup[keep] = False
                excl[cand[sup]] = True
        actual_sel = torch.nonzero(ow_indices[b][0].bool(), as_tuple=True)[0].cpu()  # what the checkpoint's rule picked
        recs.append(dict(
            boxes=pb.cpu(), feat=fb.cpu(), o=o.cpu(), k=k.cpu(), g=g.cpu(), gu=gu.cpu(), c=c.cpu(), travel=travel.cpu(),
            n=n.cpu(), label=lab, excl=excl.cpu(), pos=pos_mask.cpu(), actual_sel=actual_sel, dyn_k=dyn_k,
            n_known_gt=int(len(kb)), n_unknown_gt=int(len(ub)),
            unk_gt_classes=[class_names[int(ci)] for ci in unknown_list[b].gt_classes.tolist()],
        ))
    return recs


# ----------------------------------------------------------------------------------------------- selectors
def _rank_select(score, cand, K):
    sc = score.clone(); sc[~cand] = float("-inf")
    k = min(K, int(cand.sum()))
    return torch.topk(sc, k).indices if k > 0 else torch.zeros(0, dtype=torch.long)


def selector_topk(rec, K, key="o"):
    return _rank_select(torch.nan_to_num(rec[key], nan=-1.0), ~rec["excl"], K)


def selector_product(rec, K, keys):
    s = torch.ones_like(rec["o"])
    for kk in keys:
        s = s * torch.nan_to_num(rec[kk], nan=0.0).clamp(0, 1)
    return _rank_select(s, ~rec["excl"], K)


def kub_scores(rec, w_k=1.0, eps=1e-6):
    o = rec["o"].clamp(0, 1); k = (rec["k"] * w_k).clamp(0, 1); g = rec["g"].clamp(0, 1)
    r = 1 - (1 - k) * (1 - g)
    s = torch.stack([o * r, o * (1 - r), 1 - o], 1).clamp(eps, 1.0)
    return s, -torch.log(s)


def selector_margin(rec, K, w_k=1.0):
    s, _ = kub_scores(rec, w_k)
    m = torch.log(s[:, 1]) - torch.log(torch.maximum(s[:, 0], s[:, 2]))
    return _rank_select(m, ~rec["excl"], K)


def selector_ot(rec, K, eps=0.1, tau_u=None, cluster_iou=0.0, w_k=1.0):
    """Flat 3-column OT on candidates (or on IoU clusters). tau_u=None -> U mass fixed at K; else KL-relaxed.
    K/B columns are always relaxed (tau=1e3 ~ free). Returns selected proposal indices."""
    cand = torch.nonzero(~rec["excl"], as_tuple=True)[0]
    if cand.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    _, C = kub_scores(rec, w_k)
    C = C[cand]
    if cluster_iou > 0:
        comp = connected_components_iou(rec["boxes"][cand], cluster_iou)
        uniq, inv = torch.unique(comp, return_inverse=True)
        # cluster cost = min over members (best proposal represents the object); representative = max objectness
        Cc = torch.full((len(uniq), 3), float("inf"))
        rep = torch.zeros(len(uniq), dtype=torch.long)
        best_o = torch.full((len(uniq),), -1.0)
        for i in range(len(cand)):
            ci = inv[i]; Cc[ci] = torch.minimum(Cc[ci], C[i])
            if rec["o"][cand[i]] > best_o[ci]:
                best_o[ci] = rec["o"][cand[i]]; rep[ci] = cand[i]
        C = Cc; units = rep
    else:
        units = cand
    n = C.shape[0]
    a = torch.full((n,), 1.0 / n)
    bU = min(K, n) / n
    bK = float((kub_scores(rec, w_k)[0][units][:, 0] > 0.5).float().mean().clamp(min=0.01))
    bB = max(1e-3, 1 - bU - bK)
    b = torch.tensor([bK, bU, bB]); b = b / b.sum()
    tau_col = None
    if tau_u is not None:
        tau_col = tau_u   # relax all columns with the same tau (semi-relaxed: rows exact)
    P = sinkhorn_log(C, a, b, eps, tau_col=tau_col)
    massU = P[:, 1]
    if tau_u is None:
        kk = min(K, n); sel = torch.topk(massU, kk).indices
    else:
        sel = torch.nonzero(massU > 0.5 * a, as_tuple=True)[0]          # row sends majority of its mass to U
        if sel.numel() > 2 * K:
            sel = sel[torch.topk(massU[sel], 2 * K).indices]
    return units[sel]


def selector_random(rec, K, gen):
    cand = torch.nonzero(~rec["excl"], as_tuple=True)[0]
    k = min(K, cand.numel())
    return cand[torch.randperm(cand.numel(), generator=gen)[:k]] if k > 0 else torch.zeros(0, dtype=torch.long)


# ----------------------------------------------------------------------------------------------- reporting
def composition(recs, sel_fn, K, class_names_unknown):
    counts = defaultdict(int); n_sel = 0; dup_pairs = 0; dup_den = 0; ks = []; corr_x = []; corr_y = []
    for rec in recs:
        sel = sel_fn(rec)
        ks.append(len(sel)); corr_x.append(len(sel)); corr_y.append(rec["n_unknown_gt"])
        if len(sel) == 0:
            continue
        for i in sel.tolist():
            counts[rec["label"][i]] += 1
        n_sel += len(sel)
        if len(sel) > 1:
            iou = ops.box_iou(rec["boxes"][sel], rec["boxes"][sel])
            dup_pairs += int(torch.triu(iou > 0.5, 1).sum()); dup_den += len(sel)
    out = {kname: counts[kname] / max(n_sel, 1) for kname in counts}
    unk_keys = [kname for kname in counts if kname.startswith("unk:")]
    out_summary = dict(
        n_selected_total=n_sel, k_mean=float(np.mean(ks)), k_std=float(np.std(ks)),
        known_dup=counts["known-dup"] / max(n_sel, 1), background=counts["background"] / max(n_sel, 1),
        ambiguous=counts["ambiguous"] / max(n_sel, 1),
        unknown_total=sum(counts[kname] for kname in unk_keys) / max(n_sel, 1),
        unknown_nonlookalike=sum(counts[kname] for kname in unk_keys if kname[4:] in NONLOOKALIKE) / max(n_sel, 1),
        dup_rate=dup_pairs / max(dup_den, 1),
        corr_k_vs_nunk=float(np.corrcoef(corr_x, corr_y)[0, 1]) if np.std(corr_x) > 0 and np.std(corr_y) > 0 else float("nan"),
        per_class={kname[4:]: counts[kname] / max(n_sel, 1) for kname in unk_keys},
    )
    return out_summary


def signal_table(recs):
    groups = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        cand = ~rec["excl"]
        for i in torch.nonzero(cand, as_tuple=True)[0].tolist():
            lab = rec["label"][i]
            grp = lab if not lab.startswith("unk:") else ("unk:nonlookalike" if lab[4:] in NONLOOKALIKE else "unk:lookalike")
            for key in ("o", "k", "g", "c", "travel", "n"):
                groups[grp][key].append(float(rec[key][i]))
    # per-class detail
    per_class = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        cand = ~rec["excl"]
        for i in torch.nonzero(cand, as_tuple=True)[0].tolist():
            lab = rec["label"][i]
            for key in ("o", "c", "n"):
                per_class[lab][key].append(float(rec[key][i]))
    return groups, per_class


def fmt_q(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return "-"
    q = np.percentile(v, [25, 50, 75])
    return f"{q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--dataset-root", default="./datasets/")
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH[;KEY=VAL;...] per-checkpoint cfg overrides")
    ap.add_argument("--num-images", type=int, default=400)
    ap.add_argument("--ims-per-batch", type=int, default=4)
    ap.add_argument("--k", type=int, default=0, help="selection budget; 0 = cfg.MODEL.FORWARD_K")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/results/phaseA")
    ap.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    base_cfg = build_cfg(args.config_file, args.opts)
    n_known = base_cfg.TEST.PREV_INTRODUCED_CLS + base_cfg.TEST.CUR_INTRODUCED_CLS
    class_names = register_train_with_unknowns(base_cfg, args.task, args.dataset_root, "phaseA_train")
    print(f"known classes: {class_names[:n_known]}  unknown in split: {class_names[n_known:-1]}")

    os.makedirs(args.out, exist_ok=True)
    md_all = [f"# Phase A — pseudo-unknown selection diagnostic\n", f"task={args.task} images={args.num_images} seed={args.seed}\n"]

    for spec in args.ckpt:
        name, rest = spec.split("=", 1)
        parts = rest.split(";"); path = parts[0]
        cfg = base_cfg.clone(); cfg.defrost()
        if len(parts) > 1:
            cfg.merge_from_list([tok for kv in parts[1:] for tok in kv.split("=", 1)])
        cfg.DATASETS.TRAIN = ("phaseA_train",); cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
        cfg.DATALOADER.NUM_WORKERS = 2; cfg.MODEL.WEIGHTS = path; cfg.freeze()
        K = args.k or cfg.MODEL.FORWARD_K
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model = load_model(cfg, path)
        loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True))
        recs = []; t0 = time.time()
        ctx = apply_model_ema_and_restore(model) if cfg.MODEL_EMA.ENABLED else torch.no_grad()
        with ctx:
            for batch in loader:
                recs += run_batch(model, batch, n_known, class_names, cfg.MODEL)
                if len(recs) >= args.num_images:
                    break
        recs = recs[:args.num_images]
        print(f"[{name}] cached {len(recs)} images in {time.time()-t0:.0f}s")
        torch.save(recs, os.path.join(args.out, f"cache_{name}.pt"))

        # self-check: our re-implemented NC-SELECT rule must reproduce the matcher's own selection
        jac = []
        for rec in recs:
            mine = set(selector_topk(rec, K).tolist()) if cfg.MODEL.NC_BACKFILL else None
            act = set(rec["actual_sel"].tolist())
            if mine is not None:
                jac.append(len(mine & act) / max(len(mine | act), 1))
        dynk = np.concatenate([r["dyn_k"] for r in recs]) if recs else np.zeros(0)
        md = [f"\n## checkpoint {name}  (`{path}`)\n",
              f"- NC cfg: EXCLUDE={cfg.MODEL.NC_EXCLUDE} IOU={cfg.MODEL.NC_EXCLUDE_IOU} NMS={cfg.MODEL.NC_NMS_IOU} BACKFILL={cfg.MODEL.NC_BACKFILL} K={K}; sampler={cfg.PROPOSAL.SAMPLER}",
              f"- self-check Jaccard(our topk rule, matcher's actual selection) = {np.mean(jac):.3f}" if jac else "- self-check skipped (BACKFILL=False)",
              f"- dynamic_k over known GT: mean {dynk.mean():.2f}, frac>1 = {(dynk>1).mean():.3f}, max {dynk.max() if len(dynk) else 0}",
              f"- candidates per image (~excl): mean {np.mean([int((~r['excl']).sum()) for r in recs]):.1f}; known GT/img {np.mean([r['n_known_gt'] for r in recs]):.1f}; unknown GT/img {np.mean([r['n_unknown_gt'] for r in recs]):.1f}"]

        # signals
        groups, per_class = signal_table(recs)
        md += ["\n### Signal distributions on candidates (p25/p50/p75)\n",
               "| group | n | o | k | g | c (stage conv.) | travel (init→final IoU) | n (novelty) |", "|---|---|---|---|---|---|---|---|"]
        for grp in ["unk:nonlookalike", "unk:lookalike", "known-dup", "background", "ambiguous"]:
            if grp in groups:
                gd = groups[grp]
                md.append(f"| {grp} | {len(gd['o'])} | {fmt_q(gd['o'])} | {fmt_q(gd['k'])} | {fmt_q(gd['g'])} | {fmt_q(gd['c'])} | {fmt_q(gd['travel'])} | {fmt_q(gd['n'])} |")
        md += ["\n| per-class (unknown) | n | o p50 | c p50 | n p50 |", "|---|---|---|---|---|"]
        for lab in sorted(k_ for k_ in per_class if k_.startswith("unk:")):
            d = per_class[lab]
            md.append(f"| {lab[4:]} | {len(d['o'])} | {np.median(d['o']):.3f} | {np.median(d['c']):.3f} | {np.nanmedian(d['n']):.3f} |")
        md += ["\n### AUROC (higher = signal separates the pair; 0.5 = useless)\n",
               "| pair | o | c | travel | n |", "|---|---|---|---|---|"]
        def pair(a, b):
            if a in groups and b in groups:
                return " | ".join(f"{auroc(groups[a][key], groups[b][key]):.3f}" for key in ("o", "c", "travel", "n"))
            return "- | - | - | -"
        md.append(f"| unk:nonlookalike vs background | {pair('unk:nonlookalike','background')} |")
        md.append(f"| unk:lookalike vs background | {pair('unk:lookalike','background')} |")
        md.append(f"| unk:nonlookalike vs known-dup | {pair('unk:nonlookalike','known-dup')} |")

        # selectors
        gen = torch.Generator().manual_seed(args.seed)
        selectors = {
            "random": lambda r: selector_random(r, K, gen),
            "topk_o (=current rule)": lambda r: selector_topk(r, K, "o"),
            "topk_c": lambda r: selector_topk(r, K, "c"),
            "topk_n": lambda r: selector_topk(r, K, "n"),
            "topk_o*c": lambda r: selector_product(r, K, ["o", "c"]),
            "topk_o*n": lambda r: selector_product(r, K, ["o", "n"]),
            "margin (w_k=1)": lambda r: selector_margin(r, K, 1.0),
            "margin (w_k=0)": lambda r: selector_margin(r, K, 0.0),
            "ot_flat eps0.1": lambda r: selector_ot(r, K, 0.1, None, 0.0),
            "uot tau0.3": lambda r: selector_ot(r, K, 0.1, 0.3, 0.0),
            "uot tau1.0": lambda r: selector_ot(r, K, 0.1, 1.0, 0.0),
            "uot_cluster tau0.3": lambda r: selector_ot(r, K, 0.1, 0.3, 0.5),
        }
        md += ["\n### Selection composition (fraction of selected proposals)\n",
               "| selector | k mean±std | known-dup | background | ambiguous | unknown total | **unknown non-lookalike** | dup rate | corr(k, #unk GT) | per-class |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        rows = {}
        for sname, fn in selectors.items():
            t1 = time.time()
            try:
                s = composition(recs, fn, K, class_names[n_known:-1])
            except Exception as e:   # keep going if one selector fails
                md.append(f"| {sname} | ERROR {e} |"); continue
            rows[sname] = s
            pc = ", ".join(f"{c_}:{v:.2f}" for c_, v in sorted(s["per_class"].items(), key=lambda kv: -kv[1])[:4])
            md.append(f"| {sname} | {s['k_mean']:.1f}±{s['k_std']:.1f} | {s['known_dup']:.3f} | {s['background']:.3f} | {s['ambiguous']:.3f} "
                      f"| {s['unknown_total']:.3f} | **{s['unknown_nonlookalike']:.3f}** | {s['dup_rate']:.3f} | {s['corr_k_vs_nunk']:.2f} | {pc} |")
        # ot_flat vs margin sanity
        if "ot_flat eps0.1" in rows and "margin (w_k=1)" in rows:
            j = []
            for rec in recs:
                a = set(selector_ot(rec, K, 0.1, None, 0.0).tolist()); b = set(selector_margin(rec, K, 1.0).tolist())
                j.append(len(a & b) / max(len(a | b), 1))
            md.append(f"\nsanity: Jaccard(ot_flat, margin) = {np.mean(j):.3f} (expected ≈1: flat OT ≡ margin ranking)")
        with open(os.path.join(args.out, f"phaseA_{name}.json"), "w") as f:
            json.dump(rows, f, indent=1)
        md_all += md
        del model; torch.cuda.empty_cache()

    text = "\n".join(md_all)
    open(os.path.join(args.out, "phaseA_summary.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
