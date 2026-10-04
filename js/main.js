document.addEventListener('DOMContentLoaded', function () {
  // Toggle "see more" / "see less" text for collapsible sections
  document.querySelectorAll('.news-toggle').forEach(function (toggle) {
    toggle.addEventListener('click', function () {
      this.textContent = this.textContent === 'see more' ? 'see less' : 'see more';
    });
  });

  // Set sidebar sticky top to exactly match its natural position
  var navbar = document.querySelector('.navbar-custom');
  var sidebar = document.querySelector('.sidebar-col');
  if (navbar && sidebar) {
    var container = sidebar.closest('.container');
    var paddingTop = parseFloat(getComputedStyle(container).paddingTop);
    sidebar.style.top = (navbar.offsetHeight + paddingTop) + 'px';
  }
});
