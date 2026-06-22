import io
import base64
import time
import torch
import asyncio
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2 as transforms
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app, Gauge, Histogram

# Configure strict 1-CPU limits to prevent thread thrashing
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

app = FastAPI()

# Add Prometheus metrics to replicate what Ray was doing automatically
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Custom metrics trackers
INFERENCE_LATENCY = Histogram("ml_inference_latency_seconds", "Time spent performing model inference")
QUEUE_LENGTH = Gauge("dispatcher_queue_length", "Number of queries currently waiting or processing")

# The core concurrency constraint lock
# This forces the pod to process EXACTLY 1 request at a time
concurrency_lock = asyncio.Lock()

# Initialize Model on CPU
device = torch.device("cpu")
preprocessor = ResNet18_Weights.IMAGENET1K_V1.transforms()
resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
resnet_model.eval()
categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]


@app.post("/infer")
async def infer_handler(request: Request):
    # Increment our custom queue gauge tracker
    QUEUE_LENGTH.inc()

    # The lock forces incoming requests to line up sequentially
    async with concurrency_lock:
        t = time.perf_counter()

        # Parse payload
        d = await request.json()
        decoded = base64.b64decode(d["data"])
        inp = Image.open(io.BytesIO(decoded))

        # Core transformations
        inp_tensor = transforms.functional.to_image(inp)
        inp_tensor = transforms.functional.to_dtype(inp_tensor, torch.float32, scale=True)
        inp = preprocessor(inp_tensor).unsqueeze(0).to(device)

        # Core ML Inference execution
        with torch.no_grad():
            preds = resnet_model(inp)

        # Extract top 5 classification categories
        labels = []
        top5_indices = preds[0].argsort(descending=True)[:5].tolist()
        for idx in top5_indices:
            labels.append(categories[idx])

        duration = time.perf_counter() - t
        INFERENCE_LATENCY.observe(duration)
        QUEUE_LENGTH.dec()

        print(f"Inference complete. Server-side latency: {round(duration, 3)}s")
        return labels
