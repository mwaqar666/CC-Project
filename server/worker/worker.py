import asyncio
import base64
from io import BytesIO
import os
import time

import httpx
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from prometheus_client import Histogram, start_http_server

torch.set_num_interop_threads(1)
torch.set_num_threads(1)


class MLWorker:
    def __init__(self):
        self.dispatcher_url = os.getenv("DISPATCHER_URL", "http://dispatcher-service.ml-apps.svc.cluster.local:8000")
        self.inference_latency = Histogram("ml_inference_latency_seconds", "Time spent performing model inference inside the PyTorch worker engine")
        self.device = torch.device("cpu")
        self.preprocessor = None
        self.model = None
        self.categories = None

    def initialize_model(self):
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.preprocessor = weights.transforms()

        self.model = resnet18(weights=weights).to(self.device)
        self.model.eval()
        self.categories = weights.meta["categories"]

    def process_image(self, encoded_data: str) -> list:
        if self.preprocessor is None or self.model is None or self.categories is None:
            raise RuntimeError("Model has not been initialized")

        decoded = base64.b64decode(encoded_data)
        image = Image.open(BytesIO(decoded)).convert("RGB")
        tensor = self.preprocessor(image).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            preds = self.model(tensor)

        duration = time.perf_counter() - start_time
        self.inference_latency.observe(duration)

        top5_indices = preds[0].argsort(descending=True)[:5].tolist()
        return [self.categories[idx] for idx in top5_indices]

    async def run_loop(self):
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

        async with httpx.AsyncClient(base_url=self.dispatcher_url, limits=limits, timeout=None) as client:
            while True:
                try:
                    response = await client.get("/get_job")
                    if response.status_code == 204:
                        await asyncio.sleep(1)
                        continue
                    response.raise_for_status()

                    task = response.json()
                    job_id = task["job_id"]
                    data = task["data"]

                    labels = self.process_image(data)

                    await client.post(f"/submit_result/{job_id}", json={"labels": labels})
                except httpx.HTTPStatusError:
                    await asyncio.sleep(2)
                except httpx.RequestError:
                    await asyncio.sleep(2)
                except Exception:
                    await asyncio.sleep(1)


if __name__ == "__main__":
    worker = MLWorker()
    worker.initialize_model()
    start_http_server(8000)

    asyncio.run(worker.run_loop())
