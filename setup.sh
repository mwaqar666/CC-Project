#!/bin/bash

# Exit immediately if any individual command fails
set -e

echo "🔌 Enabling Minikube addons..."
minikube addons addons enable metrics-server
minikube addons addons enable dashboard

echo "🔌 Connecting terminal session to Minikube's Docker daemon..."
eval $(minikube docker-env)

# Automatically create or fix the buildx builder if it doesn't respond
if ! docker buildx inspect minikube-builder >/dev/null 2>&1; then
    echo "🏗️  Creating a fresh BuildKit engine instance inside Minikube..."
    docker buildx rm minikube-builder >/dev/null 2>&1
    docker buildx create --name minikube-builder --use --bootstrap
else
    echo "🔄 Switching active context to existing minikube-builder..."
    docker buildx use minikube-builder
fi

echo "🚀 Starting optimized container builds via BuildKit..."

# Build and load images natively into the local cluster registry
docker buildx build --load -t dispatcher-image:latest ./server/dispatcher
docker buildx build --load -t worker-image:latest ./server/worker
docker buildx build --load -t autoscaler-image:latest ./server/autoscaler
docker buildx build --load -t client-image:latest ./client

echo "📦 Applying out Kubernetes manifests..."

# Apply your Kubernetes manifests
kubectl apply -k ./kubernetes

echo "✅ Entire stack successfully compiled, containerized, and deployed to Minikube!"
