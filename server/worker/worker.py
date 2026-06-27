import io
import os
import base64
import time
import torch
import asyncio
import httpx
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2 as transforms
from prometheus_client import start_http_server, Histogram

# Enforce resource boundaries before loading heavy sub-modules
torch.set_num_interop_threads(1)
torch.set_num_threads(1)


class MLWorker:
    def __init__(self):
        # Configuration parameters
        self.dispatcher_url = os.getenv("DISPATCHER_SERVICE_URL", "http://dispatcher-service:8000")
        self.metrics_port = 8000

        # Instantiate Prometheus Histograms
        self.inference_latency = Histogram("ml_inference_latency_seconds", "Time spent performing model inference inside the PyTorch worker engine")

        # Runtime device configuration
        self.device = torch.device("cpu")
        self.preprocessor = None
        self.model = None
        self.categories = None

    def initialize_model(self):
        """Loads weights and preps the vision model on the CPU architecture."""
        print("Loading ResNet18 weights and setting up inference graph...")
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.preprocessor = weights.transforms()

        self.model = resnet18(weights=weights).to(self.device)
        self.model.eval()
        self.categories = weights.meta["categories"]
        print("Model configuration loaded successfully.")

    def start_metrics_daemon(self):
        """Starts the isolated Prometheus scraping route background handler."""
        print(f"Starting Prometheus endpoint scraping listener on port {self.metrics_port}...")
        start_http_server(self.metrics_port)

    def process_image(self, encoded_data: str) -> list:
        """Executes the core tensor pipeline transforms and model evaluation."""
        decoded = base64.b64decode(encoded_data)
        img = Image.open(io.BytesIO(decoded))

        # Tensor conversions
        tensor = transforms.functional.to_image(img)
        tensor = transforms.functional.to_dtype(tensor, torch.float32, scale=True)
        tensor = self.preprocessor(tensor).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()

        # Inference processing pass bounded by execution clock metric tracking
        with torch.no_grad():
            preds = self.model(tensor)

        duration = time.perf_counter() - start_time
        self.inference_latency.observe(duration)

        # Map top-5 classification tags
        top5_indices = preds[0].argsort(descending=True)[:5].tolist()
        return [self.categories[idx] for idx in top5_indices]

    async def run_loop(self):
        """The main polling loop engine executing transactions via httpx."""
        print("ML Worker Engine lifecycle activated. Initiating worker pool...")

        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

        async with httpx.AsyncClient(base_url=self.dispatcher_url, limits=limits, timeout=None) as client:
            while True:
                try:
                    # Pull step
                    response = await client.get("/get_job")
                    if response.status_code != 200:
                        await asyncio.sleep(1)
                        continue

                    task = response.json()
                    job_id = task["job_id"]

                    # Execution step
                    print(f"Acquired job {job_id}. Processing classification...")
                    labels = self.process_image(task["data"])

                    # Submission step
                    await client.post(f"/submit_result/{job_id}", json=labels)
                    print(f"Job {job_id} successfully synchronized back to dispatcher.")

                except httpx.RequestError as exc:
                    print(f"Connection boundary transmission issue with dispatcher target: {exc}")
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"Internal runtime engine error detected inside processing loop: {e}")
                    await asyncio.sleep(1)


if __name__ == "__main__":
    worker = MLWorker()
    worker.initialize_model()
    worker.start_metrics_daemon()

    # Enter async event loop processing block
    asyncio.run(worker.run_loop())
