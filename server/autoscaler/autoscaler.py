import math
import os
import time
import requests
from kubernetes import client, config


class PredictiveAutoscaler:
    def __init__(self):
        self.apps_v1 = self.initialize_kube_client()
        self.prom_url = os.getenv("PROM_URL", "http://prometheus-service.monitoring.svc.cluster.local:9090/api/v1/query")
        self.namespace = os.getenv("NAMESPACE", "ml-apps")
        self.deployment_name = os.getenv("DEPLOYMENT_NAME", "worker-deployment")
        self.min_replicas = int(os.getenv("MIN_REPLICAS", "1"))
        self.max_replicas = int(os.getenv("MAX_REPLICAS", "3"))
        self.scale_down_cooldown = int(os.getenv("SCALE_DOWN_COOLDOWN", "5"))
        self.last_scale_time = time.time()

    def initialize_kube_client(self):
        try:
            # 1. Try to authenticate using internal pod infrastructure
            config.load_incluster_config()
            print("Authenticated successfully via In-Cluster ServiceAccount.")
        except config.ConfigException:
            # 2. Fall back to your local development machine config
            try:
                config.load_kube_config()
                print("Authenticated successfully via local ~/.kube/config.")
            except config.ConfigException:
                raise RuntimeError("Could not find any valid Kubernetes configuration profile.")

        # Now you can initialize your target API endpoints safely
        return client.AppsV1Api()

    def get_prometheus_metric(self, query):
        try:
            response = requests.get(self.prom_url, params={"query": query}).json()
            return float(response["data"]["result"][0]["value"][1])
        except Exception:
            return 0.0

    def calculate_desired_replicas(self):
        # 1. Fetch live operational telemetry
        q_length = self.get_prometheus_metric("dispatcher_queue_length")
        inbound_rps = self.get_prometheus_metric("sum(rate(dispatcher_requests_incoming_total[5s]))")
        avg_infer_time = self.get_prometheus_metric("sum(rate(ml_inference_latency_seconds_sum[1m])) / sum(rate(ml_inference_latency_seconds_count[1m]))") or 0.15

        # 2. Predictive Math: How many workers do we need to stay under 0.5s?
        # Target capacity required based purely on arrival speed
        needed_by_rps = math.ceil(inbound_rps * avg_infer_time)

        # Aggressive Boost: If a queue is actively piling up, inject extra workers immediately
        queue_boost = math.ceil(q_length / 3)

        desired = needed_by_rps + queue_boost
        return max(self.min_replicas, min(self.max_replicas, desired))

    def scale_deployment(self, replicas):
        print(f"Scaling worker infrastructure to: {replicas} replicas.")
        body = {"spec": {"replicas": replicas}}
        self.apps_v1.patch_namespaced_deployment_scale(self.deployment_name, self.namespace, body)
        self.last_scale_time = time.time()

    def run(self):
        while True:
            try:
                current_scale = self.apps_v1.read_namespaced_deployment_scale(self.deployment_name, self.namespace).spec.replicas
                target_scale = self.calculate_desired_replicas()

                if target_scale > current_scale:
                    # Scale Up instantly!
                    self.scale_deployment(target_scale)
                elif target_scale < current_scale and (time.time() - self.last_scale_time) > self.scale_down_cooldown:
                    # Scale Down cautiously only after cooldown clears
                    self.scale_deployment(target_scale)

            except Exception as e:
                print(f"Error in control loop: {e}")

            time.sleep(2)  # High-resolution execution cadence


if __name__ == "__main__":
    autoscaler = PredictiveAutoscaler()
    autoscaler.run()
