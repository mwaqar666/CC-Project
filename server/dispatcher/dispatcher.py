import asyncio
import time
import uuid
from fastapi import FastAPI, Request, HTTPException
from prometheus_client import make_asgi_app, Gauge, Counter, Histogram


class MLDispatcher:
    def __init__(self):
        # 1. Core State Initialization
        self.active_jobs = {}
        self.task_queue = asyncio.Queue()

        # 2. Prometheus Metric Trackers
        self.queue_length = Gauge("dispatcher_queue_length", "Number of queries currently waiting in the backlog queue")
        self.incoming_reqs = Counter("dispatcher_requests_incoming_total", "Total request throughput entering dispatcher queue")
        self.outgoing_reqs = Counter("dispatcher_requests_outgoing_total", "Total request throughput pulled by ML workers")
        self.active_jobs_gauge = Gauge("dispatcher_active_jobs", "Number of jobs currently being actively processed inside an ML worker container")
        self.client_latency = Histogram(
            "dispatcher_client_latency_seconds",
            "Total end-to-end transaction latency from client request ingress to egress",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 5.0, float("inf")),
        )

        # 3. FastAPI App Initialization
        self.app = FastAPI()

        # Mount the Prometheus ASGI metrics endpoints
        metrics_asgi = make_asgi_app()
        self.app.mount("/metrics", metrics_asgi)

        # 4. Explicit Route Registrations
        self._register_routes()

    def _register_routes(self):
        """Binds the HTTP route endpoints to their respective class methods."""
        self.app.add_api_route("/infer", self.infer_handler, methods=["POST"])
        self.app.add_api_route("/get_job", self.get_job_handler, methods=["GET"])
        self.app.add_api_route("/submit_result/{job_id}", self.submit_result_handler, methods=["POST"])

    async def infer_handler(self, request: Request):
        """
        Client Route: Accepts image data from clients, places it onto the
        centralized queue backlog, and waits until an ML worker completes it.
        """
        self.incoming_reqs.inc()
        self.queue_length.inc()

        start_time = time.perf_counter()
        payload = await request.json()
        job_id = str(uuid.uuid4())

        # Create a future token to track the result of this unique job execution
        loop = asyncio.get_running_loop()
        job_future = loop.create_future()
        self.active_jobs[job_id] = job_future

        # Enqueue task context for workers to pull
        task_context = {"job_id": job_id, "data": payload["data"]}
        await self.task_queue.put(task_context)

        try:
            # Halt execution branch here until an ML worker delivers results
            result_labels = await job_future
            return result_labels
        finally:
            # Clean memory store trace boundaries
            self.active_jobs.pop(job_id, None)

            # Record the overall processing latency for this specific request run
            duration = time.perf_counter() - start_time
            self.client_latency.observe(duration)

    async def get_job_handler(self):
        """
        Worker Pull Route: Long-polls until a task lands in the queue,
        then returns the task configuration payload back to the requesting worker.
        """
        # Awaits atomically until an item becomes available in the queue
        task_context = await self.task_queue.get()

        self.queue_length.dec()
        self.outgoing_reqs.inc()
        self.active_jobs_gauge.inc()

        return task_context

    async def submit_result_handler(self, job_id: str, result: list):
        """
        Worker Return Route: Workers post their computed PyTorch predictions
        here. This resolves the client's original waiting future token.
        """
        if job_id in self.active_jobs:
            job_future = self.active_jobs[job_id]
            if not job_future.done():
                job_future.set_result(result)
                self.active_jobs_gauge.dec()
                return {"status": "accepted"}

        raise HTTPException(status_code=404, detail="Job entry expired or not found")


# ==================== UVICORN ENTRYPOINT ====================
# Instantiating the class exposes the underlying inner `.app` target
dispatcher_instance = MLDispatcher()
app = dispatcher_instance.app
