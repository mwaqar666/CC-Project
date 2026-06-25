import asyncio
import base64
import random
import time
import aiohttp
import os
from pathlib import Path

# Grab host address variables defined by our Kubernetes deployment file
SERVER_HOST = os.getenv("SERVER_HOST", "dispatcher-service")
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


async def send_single_request(session, req_id, latencies_list):
    """Sends one asynchronous POST request and records elapsed duration directly."""
    payload = get_random_payload()
    start_time = time.perf_counter()
    try:
        async with session.post(URL, json=payload, timeout=30) as response:
            if response.status == 200:
                await response.json()
                latency = time.perf_counter() - start_time
                latencies_list.append(latency)
            else:
                latencies_list.append("FAILED")
    except Exception:
        latencies_list.append("FAILED")


async def run_detached_wave(session, qps_target, latencies_list):
    """Fires parallel connections in the background using the shared session."""
    tasks = [send_single_request(session, i, latencies_list) for i in range(qps_target)]
    await asyncio.gather(*tasks)


async def main():
    # Load your experimental QPS integers from the text file
    if not os.path.exists("workload.txt"):
        print("Error: workload.txt file not found in client context.")
        return

    with open("workload.txt", "r") as f:
        content = f.read()

    workload_steps = [int(x) for x in content.split()]
    print(f"Beginning workload execution simulation. Total timeline: {len(workload_steps)} seconds.")

    # A shared list where background tasks safely append results as they finish
    all_results = []

    # Maintain ONE single persistent connection pool session for the entire test run
    async with aiohttp.ClientSession() as session:
        for sec_idx, qps in enumerate(workload_steps):
            print(f"[Timeline Second {sec_idx + 1}] Target load: {qps} requests/sec")

            # Pass the persistent session down to the background wave task
            asyncio.create_task(run_detached_wave(session, qps, all_results))

            # Rigidly freeze the loop main timeline tick for exactly 1 second
            await asyncio.sleep(1.0)

        print("\nWaiting for any final tail requests to wrap up processing...")
        await asyncio.sleep(15.0)  # Grace period to let lagging worker transactions clear out

    # Extract valid floats and separate failures
    valid_latencies = [r for r in all_results if isinstance(r, float)]
    total_failed = len(all_results) - len(valid_latencies)

    print("\n================ Experiment Finished ================")
    if valid_latencies:
        valid_latencies.sort()
        p99_idx = min(int(len(valid_latencies) * 0.99), len(valid_latencies) - 1)
        p99_latency = valid_latencies[p99_idx]
        print(f"Successful request transactions: {len(valid_latencies)}")
        print(f"Dropped or failed requests: {total_failed}")
        print(f"99th Percentile (P99) Latency Result: {round(p99_latency, 3)} seconds")
    else:
        print("No successful transactions processed.")


if __name__ == "__main__":
    asyncio.run(main())