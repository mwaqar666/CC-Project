import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, send_from_directory


BASE_IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "/app/images"))
MODEL_SERVER_HOST = os.getenv("MODEL_SERVER_HOST", "localhost")
MODEL_SERVER_PORT = os.getenv("MODEL_SERVER_PORT", "8001")

app = Flask(__name__)


def discover_samples() -> list:
    if not BASE_IMAGE_DIR.exists():
        return []

    return [str(p.relative_to(BASE_IMAGE_DIR)) for p in BASE_IMAGE_DIR.rglob("*.jpg") if p.is_file()]


def build_sample_tree(samples: list[str]) -> list[dict]:
    root: dict = {}

    for sample in samples:
        parts = Path(sample).parts
        node = root

        for part in parts[:-1]:
            node = node.setdefault(part, {"__type__": "dir", "__children__": {}})["__children__"]

        node[parts[-1]] = {"__type__": "file", "__path__": sample}

    def serialize(children: dict, parent_path: str = "") -> list[dict]:
        items = []

        for name in sorted(children.keys()):
            entry = children[name]
            path = f"{parent_path}/{name}" if parent_path else name

            if entry["__type__"] == "dir":
                items.append(
                    {
                        "name": name,
                        "type": "dir",
                        "path": path,
                        "children": serialize(entry["__children__"], path),
                    }
                )
            else:
                items.append(
                    {
                        "name": name,
                        "type": "file",
                        "path": entry["__path__"],
                    }
                )

        return items

    return serialize(root)

def load_image_from_sample(sample_path: str):
    absolute_path = BASE_IMAGE_DIR / sample_path
    image = cv2.imread(str(absolute_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {absolute_path}")
    return image, str(absolute_path)


def load_image_from_upload(uploaded_file):
    raw_bytes = uploaded_file.read()
    image_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image")
    return image, raw_bytes, uploaded_file.mimetype or "image/jpeg"


def prepare_payload(image):
    resized = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    return base64.b64encode(cv2.imencode(".jpeg", resized)[1].tobytes()).decode("utf-8")


def infer_image(image):
    payload = prepare_payload(image)
    url = f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}/infer"
    started_at = time.perf_counter()
    response = requests.post(url, json={"data": payload}, timeout=30)
    elapsed = round(time.perf_counter() - started_at, 3)
    response.raise_for_status()
    return response.json(), elapsed, url


@app.route("/samples/<path:sample_path>")
def sample_file(sample_path):
    return send_from_directory(str(BASE_IMAGE_DIR), sample_path)


@app.route("/", methods=["GET"])
def index():
    samples = discover_samples()
    sample_tree = build_sample_tree(samples)
    
    selected_sample = samples[0] if samples else ""
    preview_url = f"/samples/{selected_sample}" if selected_sample else ""
    return render_template(
        "index.html",
        samples=samples,
        sample_tree=sample_tree,
        selected_sample=selected_sample,
        preview_url=preview_url,
        labels=None,
        elapsed=None,
        error=None,
        server_url=f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}/infer",
    )


@app.route("/infer", methods=["POST"])
def infer():
    samples = discover_samples()
    sample_tree = build_sample_tree(samples)
    selected_sample = request.form.get("sample", samples[0] if samples else "")
    uploaded_file = request.files.get("image")
    preview_url = ""

    try:
        if uploaded_file and uploaded_file.filename:
            image, raw_bytes, mime_type = load_image_from_upload(uploaded_file)
            preview_url = f"data:{mime_type};base64," + base64.b64encode(raw_bytes).decode("utf-8")
            selected_sample = uploaded_file.filename
        else:
            if not selected_sample:
                raise FileNotFoundError("No sample image selected")
            image, _ = load_image_from_sample(selected_sample)
            preview_url = f"/samples/{selected_sample}"

        labels, elapsed, server_url = infer_image(image)
        return render_template(
            "index.html",
            samples=samples,
            sample_tree=sample_tree,
            selected_sample=selected_sample,
            preview_url=preview_url,
            labels=labels,
            elapsed=elapsed,
            error=None,
            server_url=server_url,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            samples=samples,
            sample_tree=sample_tree,
            selected_sample=selected_sample,
            preview_url=preview_url,
            labels=None,
            elapsed=None,
            error=str(exc),
            server_url=f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}/infer",
        ), 400
