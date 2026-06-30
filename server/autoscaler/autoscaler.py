import asyncio
import contextlib
import math
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from kubernetes import client, config
from prometheus_client import Counter, Gauge, make_asgi_app
import uvicorn


class PredictiveAutoscaler:
    def __init__(self):
        self.apps_v1 = self._initialize_kube_client()
        self.prom_url = os.getenv("PROM_URL", "http://prometheus-service.monitoring.svc.cluster.local:9090/api/v1/query")
        self.namespace = os.getenv("NAMESPACE", "ml-apps")
        self.deployment_name = os.getenv("DEPLOYMENT_NAME", "worker-deployment")
        self.min_replicas = int(os.getenv("MIN_REPLICAS", "1"))
        self.max_replicas = int(os.getenv("MAX_REPLICAS", "3"))
        self.scale_down_cooldown = int(os.getenv("SCALE_DOWN_COOLDOWN", "5"))
        self.target_latency_seconds = float(os.getenv("TARGET_LATENCY_SECONDS", "0.5"))
        self.min_infer_time = float(os.getenv("DEFAULT_AVG_INFER_SECONDS", "0.15"))
        self.last_scale_time = time.monotonic()
        self.running = False
        self.task = None

        self.suggested_replicas_gauge = Gauge("autoscaler_suggested_replicas", "Raw replicas suggested by the autoscaler")
        self.applied_target_replicas_gauge = Gauge("autoscaler_applied_target_replicas", "Clamped replicas the autoscaler is trying to apply")
        self.current_replicas_gauge = Gauge("autoscaler_current_replicas", "Current replicas on the worker deployment")
        self.scale_attempts_total = Counter("autoscaler_scale_attempts_total", "Scale attempts made by the autoscaler")
        self.scale_success_total = Counter("autoscaler_scale_success_total", "Successful scale operations")
        self.scale_failure_total = Counter("autoscaler_scale_failure_total", "Failed scale operations")

    def _initialize_kube_client(self):
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return client.AppsV1Api()

    async def _query_prometheus(self, query: str) -> float:
        try:
            async with httpx.AsyncClient(timeout=5) as http_client:
                response = await http_client.get(self.prom_url, params={"query": query})
                response.raise_for_status()
                payload = response.json()
                result = payload.get("data", {}).get("result", [])
                if not result:
                    return 0.0
                return float(result[0]["value"][1])
        except Exception:
            return 0.0

    async def _read_current_replicas(self) -> int:
        try:
            scale = self.apps_v1.read_namespaced_deployment_scale(self.deployment_name, self.namespace)
            return int(scale.spec.replicas or 0)
        except Exception:
            return 0

    async def _calculate_desired_replicas(self) -> int:
        q_length = await self._query_prometheus("dispatcher_queue_length")
        inbound_rps = await self._query_prometheus("sum(rate(dispatcher_requests_incoming_total[5s]))")
        avg_infer_time = await self._query_prometheus("sum(rate(ml_inference_latency_seconds_sum[1m])) / sum(rate(ml_inference_latency_seconds_count[1m]))")
        if avg_infer_time <= 0:
            avg_infer_time = self.min_infer_time

        # Heuristic: target more replicas when arrival rate and latency push toward the 0.5s budget.
        capacity_pressure = math.ceil((inbound_rps * avg_infer_time) / self.target_latency_seconds)
        queue_boost = math.ceil(q_length / 3)
        desired = capacity_pressure + queue_boost

        self.suggested_replicas_gauge.set(desired)

        return max(self.min_replicas, min(self.max_replicas, desired))

    def _scale_deployment(self, replicas: int):
        body = {"spec": {"replicas": replicas}}
        self.apps_v1.patch_namespaced_deployment_scale(self.deployment_name, self.namespace, body)

    async def _run_loop(self):
        self.running = True
        try:
            while self.running:
                try:
                    current_replicas = await self._read_current_replicas()
                    desired_replicas = await self._calculate_desired_replicas()
                    self.current_replicas_gauge.set(current_replicas)
                    self.applied_target_replicas_gauge.set(desired_replicas)

                    should_scale_up = desired_replicas > current_replicas
                    should_scale_down = desired_replicas < current_replicas
                    cooldown_passed = (time.monotonic() - self.last_scale_time) > self.scale_down_cooldown

                    if should_scale_up or (should_scale_down and cooldown_passed):
                        self.scale_attempts_total.inc()
                        self._scale_deployment(desired_replicas)
                        self.scale_success_total.inc()
                        self.last_scale_time = time.monotonic()

                except Exception:
                    self.scale_failure_total.inc()

                await asyncio.sleep(2)
        finally:
            self.running = False
            self.task = None

    async def apply(self):
        if self.running and self.task and not self.task.done():
            raise HTTPException(status_code=409, detail="Autoscaler is already applied")

        self.task = asyncio.create_task(self._run_loop())
        return {"status": "applied"}

    async def remove(self):
        self.running = False
        task = self.task
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return {"status": "removed"}


autoscaler_instance = PredictiveAutoscaler()
app = FastAPI()
app.mount("/metrics", make_asgi_app())


@app.post("/apply")
async def apply_autoscaler():
    return await autoscaler_instance.apply()


@app.post("/remove")
async def remove_autoscaler():
    return await autoscaler_instance.remove()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
