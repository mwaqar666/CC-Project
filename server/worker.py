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

# Configure strict 1-CPU limits to prevent thread thrashing
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

# Start standard standalone metrics daemon on port 8000 for cAdvisor scraping path (Tasks 4 & 5)
start_http_server(8000)
INFERENCE_LATENCY = Histogram("ml_inference_latency_seconds", "Time spent performing model inference")

# Core internal networking connection configuration pointing to the dispatcher cluster location
DISPATCHER_URL = os.getenv("DISPATCHER_SERVICE_URL", "http://dispatcher-service:8000")

# Initialize Model on CPU
device = torch.device("cpu")
preprocessor = ResNet18_Weights.IMAGENET1K_V1.transforms()
resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
resnet_model.eval()
categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]

async def main_worker_loop():
    print("ML Worker Engine initiated. Entering job pooling loop...")
    
    # Configure a long-polling persistent client connection pool
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(base_url=DISPATCHER_URL, limits=limits, timeout=None) as client:
        while True:
            try:
                # Step 1: Active pulling pull-request pattern handshake invocation
                response = await client.get("/get_job")
                if response.status_code != 200:
                    await asyncio.sleep(1)
                    continue
                
                task = response.json()
                job_id = task["job_id"]
                encoded_data = task["data"]
                
                # Step 2: Core extraction execution processing loop
                t = time.perf_counter()
                
                decoded = base64.b64decode(encoded_data)
                inp = Image.open(io.BytesIO(decoded))

                # Core tensor image transformations execution
                inp_tensor = transforms.functional.to_image(inp)
                inp_tensor = transforms.functional.to_dtype(inp_tensor, torch.float32, scale=True)
                inp = preprocessor(inp_tensor).unsqueeze(0).to(device)

                # Core ML Inference engine evaluation pass execution
                with torch.no_grad():
                    preds = resnet_model(inp)

                # Extract top 5 classification categories
                labels = []
                top5_indices = preds[0].argsort(descending=True)[:5].tolist()
                for idx in top5_indices:
                    labels.append(categories[idx])

                duration = time.perf_counter() - t
                INFERENCE_LATENCY.observe(duration)
                
                print(f"Job {job_id} complete. Processing Core Latency: {round(duration, 3)}s")
                
                # Step 3: Handshake loop transaction results callback return sequence
                await client.post(f"/submit_result/{job_id}", json=labels)
                
            except httpx.RequestError as exc:
                print(f"Connection boundary transmission issue with dispatcher target: {exc}")
                await asyncio.sleep(2)
            except Exception as e:
                print(f"Internal runtime engine error detected inside processing matrix pipeline: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main_worker_loop())