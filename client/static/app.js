const fileInput = document.getElementById("image");
const preview = document.getElementById("preview");

function renderPreviewFromSrc(src) {
  if (!preview || !src) {
    return;
  }
  preview.innerHTML = `<img src="${src}" alt="Selected image preview" id="preview-image">`;
}

if (fileInput && preview) {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      renderPreviewFromSrc(reader.result);
    };
    reader.readAsDataURL(file);
  });
}
