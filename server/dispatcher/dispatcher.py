import asyncio
import time
import uuid
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import make_asgi_app, Gauge, Counter, Histogram


class MLDispatcher:
    def __init__(self):
        self.active_jobs = {}
        self.task_queue = asyncio.Queue()

        self.queue_length = Gauge("dispatcher_queue_length", "Number of queries currently waiting in the backlog queue")
        self.incoming_reqs = Counter("dispatcher_requests_incoming_total", "Total request throughput entering dispatcher queue")
        self.outgoing_reqs = Counter("dispatcher_requests_outgoing_total", "Total request throughput pulled by ML workers")
        self.active_jobs_gauge = Gauge("dispatcher_active_jobs", "Number of jobs currently being actively processed inside an ML worker container")
        self.client_latency = Histogram(
            "dispatcher_client_latency_seconds",
            "Total end-to-end transaction latency from client request ingress to egress",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 5.0, float("inf")),
        )

    async def infer(self, payload: dict):
        self.incoming_reqs.inc()
        self.queue_length.inc()

        start_time = time.perf_counter()
        data = payload.get("data")
        if not data:
            raise HTTPException(status_code=400, detail="Missing data payload")

        job_id = str(uuid.uuid4())

        loop = asyncio.get_running_loop()
        job_future = loop.create_future()
        self.active_jobs[job_id] = job_future

        task_context = {"job_id": job_id, "data": data}
        await self.task_queue.put(task_context)

        try:
            return await job_future
        finally:
            self.active_jobs.pop(job_id, None)
            duration = time.perf_counter() - start_time
            self.client_latency.observe(duration)

    async def get_job(self):
        try:
            task_context = self.task_queue.get_nowait()
        except asyncio.QueueEmpty:
            return Response(status_code=204)

        self.queue_length.dec()
        self.outgoing_reqs.inc()
        self.active_jobs_gauge.inc()

        return task_context

    async def submit_result(self, job_id: str, result: dict):
        if job_id in self.active_jobs:
            job_future = self.active_jobs[job_id]
            if not job_future.done():
                job_future.set_result(result)
                self.active_jobs_gauge.dec()
                return {"status": "accepted"}

        raise HTTPException(status_code=404, detail="Job entry expired or not found")


dispatcher_instance = MLDispatcher()
app = FastAPI()
app.mount("/metrics", make_asgi_app())


@app.post("/infer")
async def infer(payload: dict):
    return await dispatcher_instance.infer(payload)


@app.get("/get_job")
async def get_job():
    return await dispatcher_instance.get_job()


@app.post("/submit_result/{job_id}")
async def submit_result(job_id: str, result: dict):
    return await dispatcher_instance.submit_result(job_id, result)
