import io
import base64
import time
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2 as transforms
from fastapi import FastAPI, Request
from ray import serve

# Configure strict CPU limits before the model loads, as required by your pods
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

# Initialize FastAPI framework
app = FastAPI()


@serve.deployment(
    num_replicas=1,  # Altered dynamically by your custom Autoscaler
    max_ongoing_requests=1,  # Replicas process exactly 1 query at a time
    ray_actor_options={"num_cpus": 1},  # Restricts worker replica to 1 CPU core
)
@serve.ingress(app)
class ResNet18Deployment:
    def __init__(self):
        # Force CPU execution as required by your project guidelines
        self.device = torch.device("cpu")

        # Load model and preprocessor using your specific IMAGENET1K_V1 configuration
        self.preprocessor = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(self.device)
        self.resnet_model.eval()

        self.categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]

    @app.post("/infer")
    async def infer_handler(self, request: Request):
        """
        Receives Base64 JSON payloads from the load tester,
        queues them centrally via Ray Proxy, and processes them one-by-one.
        """
        t = time.perf_counter()

        # Parse the JSON payload exactly like your original handler
        d = await request.json()
        decoded = base64.b64decode(d["data"])
        inp = Image.open(io.BytesIO(decoded))

        # Apply your exact tensor conversions and transformations
        inp_tensor = transforms.functional.to_image(inp)
        inp_tensor = transforms.functional.to_dtype(inp_tensor, torch.float32, scale=True)
        inp = self.preprocessor(inp_tensor).unsqueeze(0).to(self.device)

        # Execute Top-5 extraction logic
        with torch.no_grad():
            preds = self.resnet_model(inp)

        labels = []
        top5_indices = preds[0].argsort(descending=True)[:5].tolist()
        for idx in top5_indices:
            labels.append(self.categories[idx])

        print("Server-side processing took:", round(time.perf_counter() - t, 3))
        return labels


# Bind the deployment for Ray Serve execution
entrypoint = ResNet18Deployment.bind()
