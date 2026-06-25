import asyncio
import uuid
from fastapi import FastAPI, Request, HTTPException
from prometheus_client import make_asgi_app, Gauge, Counter

app = FastAPI()

# Mount metrics endpoint for Prometheus tracking
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Prometheus Trackers (Tasks 6 & 7)
QUEUE_LENGTH = Gauge("dispatcher_queue_length", "Number of queries currently waiting in the backlog")
INCOMING_REQS = Counter("dispatcher_requests_incoming_total", "Total request throughput entering dispatcher queue")
OUTGOING_REQS = Counter("dispatcher_requests_outgoing_total", "Total request throughput pulled by ML workers")

# Memory store for active job processing tracking
# Structure: { job_id: asyncio.Future }
active_jobs = {}

# The centralized atomic queue backlog
task_queue = asyncio.Queue()

@app.post("/infer")
async def infer_handler(request: Request):
    """
    Client Route: Accepts image data from clients, places it onto the 
    centralized queue backlog, and waits until an ML worker completes it.
    """
    INCOMING_REQS.inc()
    QUEUE_LENGTH.inc()

    # Read incoming client data payload
    payload = await request.json()
    job_id = str(uuid.uuid4())

    # Create a future token to track the result of this unique job execution
    loop = asyncio.get_running_loop()
    job_future = loop.create_future()
    active_jobs[job_id] = job_future

    # Enqueue task context for workers to pull
    task_context = {"job_id": job_id, "data": payload["data"]}
    await task_queue.put(task_context)

    try:
        # Halt execution branch here until an ML worker delivers results
        result_labels = await job_future
        return result_labels
    finally:
        # Clean memory store trace boundaries
        active_jobs.pop(job_id, None)

@app.get("/get_job")
async def get_job_handler():
    """
    Worker Pull Route: Long-polls until a task lands in the queue, 
    then returns the task configuration payload back to the requesting worker.
    """
    # Awaits atomically until an item becomes available in the queue
    task_context = await task_queue.get()
    
    QUEUE_LENGTH.dec()
    OUTGOING_REQS.inc()
    return task_context

@app.post("/submit_result/{job_id}")
async def submit_result_handler(job_id: str, result: list):
    """
    Worker Return Route: Workers post their computed PyTorch predictions 
    here. This resolves the client's original waiting future token.
    """
    if job_id in active_jobs:
        job_future = active_jobs[job_id]
        if not job_future.done():
            job_future.set_result(result)
            return {"status": "accepted"}
    
    raise HTTPException(status_code=404, detail="Job entry expired or not found")