const fileInput = document.getElementById("image");
const preview = document.getElementById("preview");
const sampleInput = document.getElementById("sample");
const pickerRoot = document.getElementById("sample-picker");
const pickerTrigger = document.getElementById("picker-trigger");
const pickerMenu = document.getElementById("picker-menu");

function sampleToUrl(samplePath) {
  if (!samplePath) {
    return "";
  }

  const encoded = samplePath
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
  return `/samples/${encoded}`;
}

function renderPreviewFromSrc(src) {
  if (!preview || !src) {
    return;
  }
  preview.innerHTML = `<img src="${src}" alt="Selected image preview" id="preview-image">`;
}

function closePicker() {
  if (!pickerMenu || !pickerTrigger) {
    return;
  }
  pickerMenu.hidden = true;
  pickerTrigger.setAttribute("aria-expanded", "false");
}

function openPicker() {
  if (!pickerMenu || !pickerTrigger) {
    return;
  }
  pickerMenu.hidden = false;
  pickerTrigger.setAttribute("aria-expanded", "true");
}

function createNodeElement(node) {
  if (node.type === "file") {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "picker-item picker-file";
    item.textContent = node.name;
    item.dataset.path = node.path;
    item.addEventListener("click", () => {
      if (sampleInput) {
        sampleInput.value = node.path;
      }
      if (pickerTrigger) {
        pickerTrigger.textContent = node.path;
      }
      if (fileInput) {
        fileInput.value = "";
      }
      renderPreviewFromSrc(sampleToUrl(node.path));
      closePicker();
    });
    return item;
  }

  const group = document.createElement("div");
  group.className = "picker-item picker-dir";

  const dirLabel = document.createElement("div");
  dirLabel.className = "picker-dir-label";
  dirLabel.innerHTML = `<span>${node.name}</span><span class="picker-caret">▶</span>`;
  group.appendChild(dirLabel);

  const submenu = document.createElement("div");
  submenu.className = "picker-submenu";

  (node.children || []).forEach((child) => {
    submenu.appendChild(createNodeElement(child));
  });

  if (!node.children || node.children.length === 0) {
    const empty = document.createElement("div");
    empty.className = "picker-empty";
    empty.textContent = "No images";
    submenu.appendChild(empty);
  }

  group.appendChild(submenu);
  return group;
}

if (pickerRoot && pickerMenu && pickerTrigger) {
  let sampleTree = [];
  try {
    sampleTree = JSON.parse(pickerRoot.dataset.sampleTree || "[]");
  } catch (err) {
    sampleTree = [];
  }

  pickerMenu.innerHTML = "";

  if (!sampleTree.length) {
    const empty = document.createElement("div");
    empty.className = "picker-empty";
    empty.textContent = "No sample images found";
    pickerMenu.appendChild(empty);
  } else {
    sampleTree.forEach((node) => {
      pickerMenu.appendChild(createNodeElement(node));
    });
  }

  pickerTrigger.addEventListener("click", () => {
    if (pickerMenu.hidden) {
      openPicker();
    } else {
      closePicker();
    }
  });

  document.addEventListener("click", (event) => {
    if (!pickerRoot.contains(event.target)) {
      closePicker();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closePicker();
    }
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