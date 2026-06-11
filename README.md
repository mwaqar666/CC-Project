# Cloud Computing Project: Automated Machine Learning Scaling Cluster

This repository contains an automated cloud computing platform. It serves a ResNet18 machine learning model using Ray Serve inside a local Kubernetes (Minikube) cluster. The system monitors live application performance using Prometheus to enable dynamic, automated service scaling based on incoming traffic.

## 🏗️ System Architecture

- __Server Component:__ Hosts a FastAPI web gateway and processes incoming request images using an isolated CPU-bound ResNet18 model execution worker.
- __Monitoring Component:__ A Prometheus timeseries database server that continuously pulls system metrics directly from the active Ray instances.
- __Client Component:__ An automated load testing engine and user interface (UI) web portal that generates traffic patterns based on defined workloads.

## 🚀 Deployment Instructions

Follow these steps in your terminal to build, launch, and run the entire project infrastructure locally.

### Step 1: Initialize the Cluster Env

Start up your local Kubernetes sandbox cluster environment:

```bash
minikube start
```

### Step 2: Configure the Local Container Registry

Point your active terminal session variables directly into the internal Minikube runtime manager. This ensures your computer saves built images inside the cluster instead of online:

```bash
eval $(minikube docker-env)
```

### Step 3: Build Custom Images Locally

Assemble your local Docker container image packages from source directories:

```bash
docker build -t ray-server-image:latest ./server
docker build -t client-image:latest ./client
```

### Step 4: Apply Kubernetes Manifests

Instruct Kubernetes to spin up your deployments, configurations, and core network routing services:

```bash
kubectl apply -f kubernetes/server-deployment.yml
kubectl apply -f kubernetes/prometheus-deployment.yml
kubectl apply -f kubernetes/client-deployment.yml
```

### Step 5: Check Deployment Status

Verify that all your system pods have initialized properly, are ready, and display a stable `Running` status:

```bash
kubectl get pods
```

## 🔍 Accessing Dashboards and Testing

Because the applications run in an isolated local cluster, you must open local access tunnels (port-forwarding links) in your terminal to view them in your desktop web browser.

### Open the Ray Management Dashboard

Run this command to view the live status of your model actors:

```bash
kubectl port-forward deployment/ray-server 8265:8265
```

- __URL:__ http://localhost:8265

### Open the Prometheus Data Interface

Run this command to inspect performance graphs and metrics:

```bash
kubectl port-forward deployment/prometheus 9090:9090
```

- __URL:__ http://localhost:9090
- __Tip:__ Navigate to __Status -> Targets__ to ensure the server scraping state is green and __UP__.

### Open the Client Web UI Portal

Run this command to access your manual image upload interface:

```bash
kubectl port-forward deployment/client 5001:5001
```

- __URL:__ http://localhost:5001
