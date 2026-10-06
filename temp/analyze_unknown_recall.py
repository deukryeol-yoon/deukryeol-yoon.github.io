"""
[EVAL-DUMP] Offline decomposition of unknown / known recall from dumped predictions.

Input : one or more prediction JSONs written by PascalVOCDetectionEvaluator when cfg.EVAL_DUMP_PATH is set
        ({"class_names": [...], "num_seen_classes": K, "predictions": {class_idx: ["img score xmin ymin xmax ymax", ...]}})
        + the VOC test split (ImageSets txt + Annotations/*.xml) the evaluation ran on.
Output: for every GT box, the best-IoU detection (any label) and how it was labelled ->
          unknown GT : detected-as-unknown / detected-as-known (mis-labelled) / missed
          known GT   : detected-as-known(correct class) / detected-as-other-known / detected-as-unknown / missed
        aggregated per relative-size bin and per class, side by side for all runs; plus U-Recall and
        #unknown detections per image. Uses IoU >= 0.5 with the evaluator's 1-based -> 0-based shift.

Usage:
  python tools/analyze_unknown_recall.py --split datasets/ImageSets/Main/<SUPERSPLIT>/test.txt \
      --ann-dir datasets/Annotations --run R0=output/.../R0_predictions.json --run R3=output/.../R3_predictions.json \
      --out docs/results/unknown_recall_decomp.md
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

ALL_CLASS_NAMES = set()
S_BINS = [0.0, 0.01, 0.03, 0.10, 9.0]
S_LABELS = ["s<0.01", "0.01-0.03", "0.03-0.10", "s>=0.10"]


def load_gt(split_txt, ann_dir):
    """image_id -> dict(W, H, boxes xyxy (0-based, like core/pascal_voc.py), names)"""
    gts = {}
    with open(split_txt) as f:
        ids = [ln.strip().split(".")[0] for ln in f if ln.strip()]
    for i in ids:
        p = os.path.join(ann_dir, i + ".xml")
        if not os.path.exists(p):
            continue
        t = ET.parse(p)
        W = float(t.findall("./size/width")[0].text); H = float(t.findall("./size/height")[0].text)
        boxes, names, diff = [], [], []
        for o in t.findall("object"):
            bb = o.find("bndbox")
            x0, y0, x1, y1 = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
            boxes.append([x0 - 1.0, y0 - 1.0, x1, y1]); names.append(o.find("name").text)
            d = o.find("difficult"); diff.append(int(d.text) if d is not None and d.text is not None else 0)
        gts[i] = dict(W=W, H=H, boxes=np.asarray(boxes, np.float64).reshape(-1, 4), names=names,
                      difficult=np.asarray(diff, bool))
    return gts


def evaluator_unknown_recall(gts, per_img, unk_idx, known_names, thr):
    """Exact replica of core/pascal_voc_evaluation.py::voc_eval for classname='unknown':
    every GT whose name is not a known class counts as unknown (including names outside the class list, e.g.
    'ignored regions'), difficult GT are excluded from npos and never TP/FP, VOC +1 IoU, greedy in global score order.
    Returns dict(npos, tp, recall, n_unknown_names_outside_class_list, n_difficult)."""
    recs = {}
    n_outside = 0; n_diff = 0; npos = 0
    for img, g in gts.items():
        keep = [j for j, nm in enumerate(g["names"]) if nm not in known_names]
        n_outside += sum(1 for j in keep if g["names"][j] not in ALL_CLASS_NAMES)
        bbox = g["boxes"][keep] + np.array([1.0, 1.0, 0.0, 0.0])   # back to the evaluator's 1-based xmin/ymin frame
        difficult = g["difficult"][keep]
        n_diff += int(difficult.sum()); npos += int((~difficult).sum())
        recs[img] = dict(bbox=bbox, difficult=difficult, det=[False] * len(keep),
                         det_gen=[False] * len(keep), ndet=np.zeros(len(keep), int))
    dets = [(sc, img, x0 + 1.0, y0 + 1.0, x1, y1) for img, L in per_img.items() for (c, sc, x0, y0, x1, y1) in L if c == unk_idx]
    dets.sort(key=lambda d: -d[0])
    tp = 0; tp_gen = 0; n_multi = 0
    for sc, img, x0, y0, x1, y1 in dets:
        R = recs.get(img)
        if R is None or R["bbox"].size == 0:
            continue
        G = R["bbox"]
        iw = np.maximum(np.minimum(G[:, 2], x1) - np.maximum(G[:, 0], x0) + 1.0, 0.0)
        ih = np.maximum(np.minimum(G[:, 3], y1) - np.maximum(G[:, 1], y0) + 1.0, 0.0)
        inter = iw * ih
        uni = (x1 - x0 + 1.0) * (y1 - y0 + 1.0) + (G[:, 2] - G[:, 0] + 1.0) * (G[:, 3] - G[:, 1] + 1.0) - inter
        ov = inter / uni
        hit = np.where(ov > thr)[0]
        R["ndet"][hit] += 1
        if len(hit) > 1:
            n_multi += 1
        j = int(np.argmax(ov))
        if ov[j] > thr and not R["difficult"][j] and not R["det"][j]:
            tp += 1; R["det"][j] = True
        # generous variant: the det may claim any still-unmatched GT it overlaps (>thr), not only its argmax GT
        free = [k for k in hit if not R["det_gen"][k] and not R["difficult"][k]]
        if free:
            k = free[int(np.argmax(ov[free]))]; R["det_gen"][k] = True; tp_gen += 1
    ndet_all = np.concatenate([R["ndet"] for R in recs.values()]) if recs else np.zeros(0, int)
    # GT-GT overlap among unknown GT of the same image (annotation overlap -> inherent greedy loss)
    n_pair = 0
    for R in recs.values():
        G = R["bbox"]
        if len(G) < 2:
            continue
        ix1 = np.maximum(G[:, None, 0], G[None, :, 0]); iy1 = np.maximum(G[:, None, 1], G[None, :, 1])
        ix2 = np.minimum(G[:, None, 2], G[None, :, 2]); iy2 = np.minimum(G[:, None, 3], G[None, :, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        a = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])
        iou = inter / np.maximum(a[:, None] + a[None, :] - inter, 1e-12); np.fill_diagonal(iou, 0)
        n_pair += int((iou.max(1) > 0.3).sum())
    return dict(npos=npos, tp=tp, recall=tp / max(npos, 1), n_outside=n_outside, n_difficult=n_diff, n_unk_dets=len(dets),
                recall_generous=tp_gen / max(npos, 1), n_dets_multi_gt=n_multi,
                gt_with_0_dets=float((ndet_all == 0).mean()) if len(ndet_all) else 0.0,
                dets_per_gt_p50=float(np.percentile(ndet_all, 50)) if len(ndet_all) else 0.0,
                dets_per_gt_p90=float(np.percentile(ndet_all, 90)) if len(ndet_all) else 0.0,
                gt_overlap_frac=n_pair / max(npos + n_diff, 1))


def load_preds(path):
    d = json.load(open(path))
    per_img = defaultdict(list)  # image_id -> list of (cls_idx, score, x0, y0, x1, y1) 0-based
    for cls, lines in d["predictions"].items():
        c = int(cls)
        for ln in lines:
            img, sc, x0, y0, x1, y1 = ln.split()
            per_img[img].append((c, float(sc), float(x0) - 1.0, float(y0) - 1.0, float(x1), float(y1)))
    return d, per_img


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


def analyze(run_name, pred_path, gts, iou_thr):
    global ALL_CLASS_NAMES
    meta, per_img = load_preds(pred_path)
    names = meta["class_names"]; K = int(meta["num_seen_classes"]); unk_idx = len(names) - 1
    ALL_CLASS_NAMES = set(names)
    name2idx = {n: i for i, n in enumerate(names)}
    rows = []  # per GT: (is_unknown, cls_name, s, outcome, any_unknown)
    n_unk_det_per_img = []
    ev = evaluator_unknown_recall(gts, per_img, unk_idx, set(names[:K]), iou_thr)
    for img, g in gts.items():
        dets = per_img.get(img, [])
        db = np.asarray([d[2:] for d in dets], np.float64).reshape(-1, 4)
        dl = np.asarray([d[0] for d in dets], int)
        n_unk_det_per_img.append(int((dl == unk_idx).sum()))
        if len(g["boxes"]) == 0:
            continue
        ious = iou_matrix(g["boxes"], db)
        s_rel = np.sqrt((g["boxes"][:, 2] - g["boxes"][:, 0]) * (g["boxes"][:, 3] - g["boxes"][:, 1]) / (g["W"] * g["H"]))
        for gi, nm in enumerate(g["names"]):
            ci = name2idx.get(nm, None)
            if ci is None:
                continue
            is_unk = ci >= K  # not a currently-known class -> counts as unknown GT
            any_unk = bool(ious.shape[1] and ((ious[gi] >= iou_thr) & (dl == unk_idx)).any())
            if ious.shape[1] == 0 or ious[gi].max() < iou_thr:
                outcome = "missed"
            else:
                # among detections with IoU>=thr, prefer the highest-scoring one (what NMS/eval would keep)
                cand = np.where(ious[gi] >= iou_thr)[0]
                best = cand[np.argmax([dets[j][1] for j in cand])]
                lab = dl[best]
                if is_unk:
                    outcome = "as_unknown" if lab == unk_idx else "as_known"
                else:
                    outcome = "as_known_correct" if lab == ci else ("as_unknown" if lab == unk_idx else "as_other_known")
            rows.append((is_unk, nm, float(s_rel[gi]), outcome, any_unk))
    return dict(run=run_name, rows=rows, n_unk_det_per_img=np.asarray(n_unk_det_per_img), names=names, K=K, ev=ev)


def table(results, is_unknown, group_key):
    outcomes = ["as_unknown", "as_known", "missed"] if is_unknown else ["as_known_correct", "as_other_known", "as_unknown", "missed"]
    groups = sorted({group_key(r) for res in results for r in res["rows"] if r[0] == is_unknown},
                    key=lambda g: (S_LABELS.index(g) if g in S_LABELS else 99, g))
    hdr = "| group | n_gt | " + " | ".join(f"{res['run']} {o}" for res in results for o in outcomes) + " |"
    lines = [hdr, "|---|---|" + "---|" * (len(results) * len(outcomes))]
    for grp in groups + ["ALL"]:
        cells = []; n_gt = None
        for res in results:
            sel = [r for r in res["rows"] if r[0] == is_unknown and (grp == "ALL" or group_key(r) == grp)]
            n_gt = len(sel)
            for o in outcomes:
                cells.append(f"{(sum(r[3] == o for r in sel) / max(n_gt, 1)):.3f}")
        lines.append(f"| {grp} | {n_gt} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="ImageSets txt of the evaluated test split")
    ap.add_argument("--ann-dir", default="datasets/Annotations")
    ap.add_argument("--run", action="append", required=True, help="NAME=path/to/predictions.json (repeatable)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gts = load_gt(args.split, args.ann_dir)
    results = [analyze(kv.split("=", 1)[0], kv.split("=", 1)[1], gts, args.iou) for kv in args.run]

    sbin = lambda r: S_LABELS[min(int(np.digitize(r[2], S_BINS[1:-1])), 3)]
    md = [f"# Unknown / known recall decomposition (IoU>={args.iou}, {len(gts)} images)\n",
          f"known classes (K={results[0]['K']}): {results[0]['names'][:results[0]['K']]}\n",
          "Outcome = label of the highest-scoring detection with IoU>=thr to the GT; 'missed' = no such detection.\n",
          "## Unknown GT by size\n", table(results, True, sbin),
          "\n## Unknown GT by class\n", table(results, True, lambda r: r[1]),
          "\n## Known GT by size\n", table(results, False, sbin),
          "\n## Known GT by class\n", table(results, False, lambda r: r[1]),
          "\n## Unknown detections and recall reconciliation\n",
          "| run | unk dets/img mean | median | p90 | total unk dets | top-det-is-unknown (this tool) | any-unknown-det (>=1, any rank) "
          "| **evaluator-replica U-Recall** | eval npos | eval TP | unknown GT w/ name outside class list | difficult unknown GT |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        u = [r for r in res["rows"] if r[0]]
        ur = sum(r[3] == "as_unknown" for r in u) / max(len(u), 1)
        anyu = sum(r[4] for r in u) / max(len(u), 1)
        d = res["n_unk_det_per_img"]; ev = res["ev"]
        md.append(f"| {res['run']} | {d.mean():.1f} | {np.median(d):.0f} | {np.percentile(d, 90):.0f} | {int(d.sum())} | {ur:.4f} | {anyu:.4f} "
                  f"| **{ev['recall']:.4f}** | {ev['npos']} | {ev['tp']} | {ev['n_outside']} | {ev['n_difficult']} |")
    md += ["\n## Why greedy recall is lower than coverage\n",
           "| run | replica U-Recall | generous U-Recall (det may claim any unmatched GT it overlaps) | unk GT with 0 unk dets (>thr, eval frame) "
           "| unk dets per GT p50 / p90 | unk dets overlapping >1 GT | unk GT with another unk GT at IoU>0.3 |",
           "|---|---|---|---|---|---|---|"]
    for res in results:
        ev = res["ev"]
        md.append(f"| {res['run']} | {ev['recall']:.4f} | {ev['recall_generous']:.4f} | {ev['gt_with_0_dets']:.4f} "
                  f"| {ev['dets_per_gt_p50']:.0f} / {ev['dets_per_gt_p90']:.0f} | {ev['n_dets_multi_gt']} | {ev['gt_overlap_frac']:.4f} |")
    md.append("\nIf the evaluator-replica U-Recall differs from the training log's 'Unknown Recall50', the dump / split / "
              "annotation dir do not match the evaluation run. 'top-det-is-unknown' is this tool's outcome definition; "
              "'any-unknown-det' is an upper bound for greedy recall.")
    text = "\n".join(md)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(text)
        print("->", args.out)


if __name__ == "__main__":
    main()
