import io
import base64
import time
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2 as transforms
from aiohttp import web

preprocessor = ResNet18_Weights.IMAGENET1K_V1.transforms()

# These two lines are important, as your pods will have CPU request and CPU limit of "1" (for memory also use "1G" for both request and limit)
torch.set_num_interop_threads(1)
torch.set_num_threads(1)


resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet_model.eval()


def infer(d):
    t = time.perf_counter()
    decoded = base64.b64decode(d["data"])
    inp = Image.open(io.BytesIO(decoded))

    inp_tensor = transforms.functional.to_image(inp)
    inp_tensor = transforms.functional.to_dtype(inp_tensor, torch.float32, scale=True)
    inp = preprocessor(inp_tensor).unsqueeze(0)

    preds = resnet_model(inp)
    labels = []

    top5_indices = preds[0].argsort(descending=True)[:5].tolist()
    for idx in top5_indices:
        labels.append(ResNet18_Weights.IMAGENET1K_V1.meta["categories"][idx])

    print("Server-side processing took:", round(time.perf_counter() - t, 3))
    return labels


app = web.Application()


async def infer_handler(request):
    req = await request.json()
    return web.json_response(infer(req))


app.add_routes(
    [
        web.post("/infer", infer_handler),
    ]
)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8001, access_log=None)
