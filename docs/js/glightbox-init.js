(function () {
  function enhanceImages() {
    if (typeof GLightbox !== "function") {
      return;
    }
    document.querySelectorAll("article img").forEach(function (img) {
      if (img.closest("a.glightbox")) {
        return;
      }
      var link = document.createElement("a");
      link.href = img.src;
      link.className = "glightbox";
      link.setAttribute("data-gallery", "docs-images");
      img.parentNode.insertBefore(link, img);
      link.appendChild(img);
    });
    GLightbox({ selector: "a.glightbox" });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enhanceImages);
  } else {
    enhanceImages();
  }
})();

