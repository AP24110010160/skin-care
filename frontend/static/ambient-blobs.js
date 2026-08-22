/*
  Scroll-linked parallax for the ambient background blobs.
  Each .blob-parallax element moves vertically as the page scrolls, at a
  rate set by its data-speed attribute -- different speeds per blob create
  a subtle sense of depth (some blobs appear to drift "faster" than others
  relative to scroll, like they're at different distances).

  Positive speed = moves down as you scroll down (feels closer/heavier).
  Negative speed = moves up as you scroll down (feels farther/lighter).

  Respects prefers-reduced-motion: if the user's OS has that set, this
  script does nothing and the blobs simply stay in their CSS-defined
  position (the idle drift animation is separately disabled in the CSS).
*/

(function () {
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;
  if (prefersReducedMotion) return;

  const blobs = document.querySelectorAll(".blob-parallax");
  if (!blobs.length) return;

  let ticking = false;

  function updateBlobPositions() {
    const scrollY = window.scrollY || window.pageYOffset;
    blobs.forEach((el) => {
      const speed = parseFloat(el.dataset.speed || "0");
      el.style.transform = `translateY(${scrollY * speed}px)`;
    });
    ticking = false;
  }

  function onScroll() {
    if (!ticking) {
      window.requestAnimationFrame(updateBlobPositions);
      ticking = true;
    }
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  updateBlobPositions(); // set initial position on load
})();
