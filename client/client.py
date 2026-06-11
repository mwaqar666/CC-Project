import asyncio
import base64
import random
import time
import aiohttp
import os
from pathlib import Path

# Grab host address variables defined by our Kubernetes deployment file
SERVER_HOST = os.getenv("SERVER_HOST", "server")
SERVER_PORT = os.getenv("SERVER_PORT", "8000")
URL = f"http://{SERVER_HOST}:{SERVER_PORT}/infer"
BASE_IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "/app/images"))


def discover_samples():
    if not BASE_IMAGE_DIR.exists():
        return []

    return [str(p.relative_to(BASE_IMAGE_DIR)) for p in BASE_IMAGE_DIR.rglob("*.jpg") if p.is_file()]


def load_image_from_sample(sample_path):
    absolute_path = BASE_IMAGE_DIR / sample_path
    if not absolute_path.is_file():
        raise FileNotFoundError(f"Could not read image at {absolute_path}")
    return absolute_path.read_bytes()


def get_random_payload():
    """Loads a random image from the images folder and returns it as base64 payload."""
    samples = discover_samples()
    if not samples:
        raise FileNotFoundError(f"No .jpg images found in {BASE_IMAGE_DIR}")

    image_bytes = load_image_from_sample(random.choice(samples))
    return {"data": base64.b64encode(image_bytes).decode("utf-8")}


async def send_single_request(session, req_id):
    """Sends one asynchronous POST request and returns elapsed duration."""
    payload = get_random_payload()
    start_time = time.perf_counter()
    try:
        async with session.post(URL, json=payload, timeout=30) as response:
            if response.status == 200:
                await response.json()
                return time.perf_counter() - start_time
            else:
                return "BAD_STATUS"
    except Exception:
        return "TIMEOUT_OR_ERROR"


async def run_one_second_wave(qps_target):
    """Fires exactly N parallel connections simultaneously inside a 1-second block."""
    start_wave = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        # Construct and launch all requests at the exact same moment
        tasks = [send_single_request(session, i) for i in range(qps_target)]
        results = await asyncio.gather(*tasks)

    valid_latencies = [r for r in results if isinstance(r, float)]
    failed_count = len(results) - len(valid_latencies)

    # Measure time spent on processing this entire second block
    elapsed_wave = time.perf_counter() - start_wave

    # Pause if tasks finished in less than 1 full second
    time_to_sleep = 1.0 - elapsed_wave
    if time_to_sleep > 0:
        await asyncio.sleep(time_to_sleep)

    return valid_latencies, failed_count


async def main():
    # Load your experimental QPS integers from the text file
    if not os.path.exists("workload.txt"):
        print("Error: workload.txt file not found in client context.")
        return

    with open("workload.txt", "r") as f:
        content = f.read()

    workload_steps = [int(x) for x in content.split()]
    print(f"Beginning workload execution simulation. Total timeline: {len(workload_steps)} seconds.")

    all_latencies = []
    total_failed = 0

    for sec_idx, qps in enumerate(workload_steps):
        print(f"[Timeline Second {sec_idx + 1}] Target load: {qps} requests/sec")

        latencies, failed = await run_one_second_wave(qps)
        all_latencies.extend(latencies)
        total_failed += failed

    print("\n================ Experiment Finished ================")
    if all_latencies:
        all_latencies.sort()
        p99_idx = int(len(all_latencies) * 0.99)
        p99_latency = all_latencies[p99_idx]
        print(f"Successful request transactions: {len(all_latencies)}")
        print(f"Dropped or failed requests: {total_failed}")
        print(f"99th Percentile (P99) Latency Result: {round(p99_latency, 3)} seconds")


if __name__ == "__main__":
    asyncio.run(main())
