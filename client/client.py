import asyncio
import base64
import random
import time
import aiohttp
import os
import sys
from pathlib import Path
from prometheus_client import start_http_server, Counter, Gauge


class MLClusterClient:
    def __init__(self):
        # 1. Cluster Networking Configs
        self.server_host = os.getenv("SERVER_HOST", "dispatcher-service")
        self.server_port = os.getenv("SERVER_PORT", "8000")
        self.url = f"http://{self.server_host}:{self.server_port}/infer"
        self.base_image_dir = Path(os.getenv("IMAGE_DIR", "/app/images"))

        # 2. Prometheus Client Metrics Trackers
        self.client_requests_total = Counter(
            "client_requests_total",
            "Total number of requests dispatched by this test runner",
            ["status"],  # Labels: 'success' or 'failed'
        )
        self.client_latency_p99 = Gauge("client_latency_p99", "The 99th percentile response latency from the last completed workload step run")
        self.client_active_qps = Gauge("client_current_target_qps", "The currently running target queries-per-second setting injected from workload config")

    def discover_samples(self):
        try:
            return [str(p.relative_to(self.base_image_dir)) for p in self.base_image_dir.rglob("*.jpg") if p.is_file()]
        except Exception:
            return []

    def load_image_from_sample(self, sample_path):
        absolute_path = self.base_image_dir / sample_path
        try:
            return absolute_path.read_bytes()
        except Exception:
            raise FileNotFoundError(f"Could not read image payload at {absolute_path}")

    def get_random_payload(self):
        samples = self.discover_samples()
        if not samples:
            raise FileNotFoundError(f"No valid sample .jpg images located in directory {self.base_image_dir}")
        image_bytes = self.load_image_from_sample(random.choice(samples))
        return {"data": base64.b64encode(image_bytes).decode("utf-8")}

    async def send_single_request(self, session, latencies_list):
        try:
            payload = self.get_random_payload()
            start_time = time.perf_counter()

            async with session.post(self.url, json=payload, timeout=30) as response:
                if response.status == 200:
                    await response.json()
                    duration = time.perf_counter() - start_time
                    latencies_list.append(duration)
                    self.client_requests_total.labels(status="success").inc()
                else:
                    latencies_list.append("FAILED")
                    self.client_requests_total.labels(status="failed").inc()
        except Exception:
            latencies_list.append("FAILED")
            self.client_requests_total.labels(status="failed").inc()

    async def run_detached_wave(self, session, qps_target, latencies_list):
        tasks = [self.send_single_request(session, latencies_list) for _ in range(qps_target)]
        await asyncio.gather(*tasks)

    async def execute_experiment(self):
        if not os.path.exists("workload.txt"):
            print("Error: workload.txt matrix file missing from context root configuration path.")
            return

        with open("workload.txt", "r") as f:
            workload_steps = [int(x) for x in f.read().split()]

        print(f"Beginning workload execution simulation. Total timeline: {len(workload_steps)} seconds.")
        all_results = []

        async with aiohttp.ClientSession() as session:
            for sec_idx, qps in enumerate(workload_steps):
                print(f"[Timeline Second {sec_idx + 1}] Target load: {qps} requests/sec")
                self.client_active_qps.set(qps)

                asyncio.create_task(self.run_detached_wave(session, qps, all_results))
                await asyncio.sleep(1.0)

            print("\nWaiting for any final tail requests to wrap up processing...")
            await asyncio.sleep(15.0)

        # Process and record results
        valid_latencies = [r for r in all_results if isinstance(r, float)]
        total_failed = len(all_results) - len(valid_latencies)

        print("\n================ Experiment Finished ================")
        self.client_active_qps.set(0)  # Reset load indicator

        if valid_latencies:
            valid_latencies.sort()
            p99_idx = min(int(len(valid_latencies) * 0.99), len(valid_latencies) - 1)
            p99_latency = valid_latencies[p99_idx]

            # Expose the final P99 outcome directly to Prometheus
            self.client_latency_p99.set(p99_latency)

            print(f"Successful request transactions: {len(valid_latencies)}")
            print(f"Dropped or failed requests: {total_failed}")
            print(f"99th Percentile (P99) Latency Result: {round(p99_latency, 3)} seconds")
        else:
            print("No successful transactions processed.")


if __name__ == "__main__":
    client_runner = MLClusterClient()

    # Start the Prometheus scraping endpoint server on background port 8001
    start_http_server(8000)

    # Check CLI arguments: If no explicit run argument is given, keep the pod sleeping
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        asyncio.run(client_runner.execute_experiment())
    else:
        print("Client Pod initialized in STANDBY mode. Persistent endpoint port 8001 open.")
        print("To trigger the experiment run, execute: python client.py --run")

        # Keep the process sleeping infinitely so the Kubernetes Pod doesn't die
        loop = asyncio.get_event_loop()
        loop.run_forever()
