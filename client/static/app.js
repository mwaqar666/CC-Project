const sampleSelect = document.getElementById("sample");
const fileInput = document.getElementById("image");
const preview = document.getElementById("preview");

function renderEmptyPreview() {
  if (!preview) {
    return;
  }

  preview.innerHTML = '<div class="empty-state">Choose a sample or upload a file to preview it here.</div>';
}

function renderPreviewFromSrc(src) {
  if (!preview || !src) {
    return;
  }

  preview.innerHTML = `<img src="${src}" alt="Selected image preview" id="preview-image">`;
}

if (sampleSelect && preview) {
  sampleSelect.addEventListener("change", () => {
    const selectedSample = sampleSelect.value;

    if (!selectedSample) {
      renderEmptyPreview();
      return;
    }

    renderPreviewFromSrc(`/samples/${selectedSample}`);
  });
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
