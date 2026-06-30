import asyncio
import base64
from contextlib import suppress
from enum import Enum
import random
import time
import aiohttp
import os
from pathlib import Path
from prometheus_client import Counter, Gauge, make_asgi_app
from fastapi import FastAPI, HTTPException


class ExperimentState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class MLClusterClient:
    def __init__(self):
        self.server_url = os.getenv("SERVER_URL", "http://dispatcher-service.ml-apps.svc.cluster.local:8000")
        self.server_url = f"{self.server_url}/infer"
        self.base_image_dir = Path(os.getenv("IMAGE_DIR", "/app/images"))

        self.client_requests_total = Counter(
            "client_requests_total",
            "Total number of requests dispatched by this test runner",
            ["status"],
        )
        self.client_latency_p99 = Gauge("client_latency_p99", "The 99th percentile response latency from the last completed workload step run")
        self.client_active_qps = Gauge("client_current_target_qps", "The currently running target queries-per-second setting injected from workload config")

        self.state = ExperimentState.IDLE
        self.total_steps = 0
        self.current_step = 0
        self.last_error = None
        self.last_summary = None
        self._task = None
        self._pause_event = asyncio.Event()
        self._lock = asyncio.Lock()

    def _load_workload(self):
        workload_path = Path("workload.txt")
        if not workload_path.exists():
            raise FileNotFoundError("workload.txt matrix file missing from context root configuration path.")
        return [int(x) for x in workload_path.read_text().split() if x.isdigit() and int(x) >= 0]

    def _load_samples(self):
        samples = [p for p in self.base_image_dir.rglob("*.jpg") if p.is_file()]
        if not samples:
            raise FileNotFoundError(f"No valid sample .jpg images located in directory {self.base_image_dir}")
        return samples

    async def _send_single_request(self, session, samples, latencies_list):
        try:
            image_bytes = random.choice(samples).read_bytes()
            payload = {"data": base64.b64encode(image_bytes).decode("utf-8")}
            start_time = time.perf_counter()

            async with session.post(self.server_url, json=payload, timeout=30) as response:
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

    async def _run_wave(self, session, qps_target, samples, latencies_list):
        tasks = [self._send_single_request(session, samples, latencies_list) for _ in range(qps_target)]
        await asyncio.gather(*tasks)

    def _summarize_results(self, results):
        valid_latencies = [r for r in results if isinstance(r, float)]
        total_failed = len(results) - len(valid_latencies)

        summary = {
            "successful_requests": len(valid_latencies),
            "failed_requests": total_failed,
            "p99_latency_seconds": None,
        }

        if valid_latencies:
            valid_latencies.sort()
            p99_idx = min(int(len(valid_latencies) * 0.99), len(valid_latencies) - 1)
            p99_latency = valid_latencies[p99_idx]
            self.client_latency_p99.set(p99_latency)
            summary["p99_latency_seconds"] = round(p99_latency, 3)

        self.last_summary = summary
        return summary

    async def _execute_experiment(self):
        workload_steps = self._load_workload()
        samples = self._load_samples()

        self.total_steps = len(workload_steps)
        self.current_step = 0
        self.last_error = None
        self.last_summary = None
        all_results = []
        wave_tasks = []

        print(f"Beginning workload execution simulation. Total timeline: {self.total_steps} seconds.")

        try:
            async with aiohttp.ClientSession() as session:
                while self.current_step < self.total_steps:
                    await self._pause_event.wait()

                    qps = workload_steps[self.current_step]
                    print(f"[Timeline Second {self.current_step + 1}] Target load: {qps} requests/sec")
                    self.client_active_qps.set(qps)

                    wave_task = asyncio.create_task(self._run_wave(session, qps, samples, all_results))
                    wave_tasks.append(wave_task)
                    self.current_step += 1
                    await asyncio.sleep(1.0)

                print("\nWaiting for all dispatched request waves to finish...")
                if wave_tasks:
                    await asyncio.gather(*wave_tasks)

            summary = self._summarize_results(all_results)
            print("\n================ Experiment Finished ================")
            print(f"Successful request transactions: {summary['successful_requests']}")
            print(f"Dropped or failed requests: {summary['failed_requests']}")
            if summary["p99_latency_seconds"] is not None:
                print(f"99th Percentile (P99) Latency Result: {summary['p99_latency_seconds']} seconds")
            else:
                print("No successful transactions processed.")
        finally:
            self.client_active_qps.set(0)

    async def _run_experiment(self):
        try:
            await self._execute_experiment()
            self.state = ExperimentState.COMPLETED
        except asyncio.CancelledError:
            self.state = ExperimentState.IDLE
            raise
        except Exception as exc:
            self.state = ExperimentState.ERROR
            self.last_error = str(exc)
            print(f"Experiment execution error: {exc}")

    async def start_experiment(self):
        async with self._lock:
            if self._task and not self._task.done():
                raise HTTPException(status_code=409, detail=f"Experiment already {self.state.value}")

            self._pause_event.set()
            self.state = ExperimentState.RUNNING
            self._task = asyncio.create_task(self._run_experiment())

    async def pause_experiment(self):
        async with self._lock:
            if self.state != ExperimentState.RUNNING:
                raise HTTPException(status_code=409, detail="Experiment is not running")
            self.state = ExperimentState.PAUSED
            self._pause_event.clear()
            self.client_active_qps.set(0)

    async def resume_experiment(self):
        async with self._lock:
            if self.state != ExperimentState.PAUSED:
                raise HTTPException(status_code=409, detail="Experiment is not paused")
            self.state = ExperimentState.RUNNING
            self._pause_event.set()

    async def reset_experiment(self):
        async with self._lock:
            task = self._task
            self._pause_event.set()
            self.state = ExperimentState.IDLE

        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        async with self._lock:
            self._task = None
            self.current_step = 0
            self.total_steps = 0
            self.last_error = None
            self.last_summary = None
            self.client_active_qps.set(0)

    def status(self):
        return {
            "state": self.state.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "last_error": self.last_error,
            "last_summary": self.last_summary,
        }


client_runner = MLClusterClient()
app = FastAPI(title="ML Cluster Client Controller")
app.mount("/metrics", make_asgi_app())


@app.post("/start")
async def start_experiment():
    await client_runner.start_experiment()
    return client_runner.status()


@app.post("/pause")
async def pause_experiment():
    await client_runner.pause_experiment()
    return client_runner.status()


@app.post("/resume")
async def resume_experiment():
    await client_runner.resume_experiment()
    return client_runner.status()


@app.post("/reset")
async def reset_experiment():
    await client_runner.reset_experiment()
    return client_runner.status()


@app.get("/status")
async def get_status():
    return client_runner.status()
