#!/bin/bash

# Exit immediately if any individual command fails
set -ex

# 1. Safely inspect the builder state
inspect_builder() {
    echo "🔍 Inspecting minikube-builder status..."

    if docker buildx inspect minikube-builder >/dev/null 2>&1; then
        echo "✅ SUCCESS: minikube-builder profile is present and responding."
        return 0
    else
        echo "❌ ERROR: minikube-builder profile is missing or corrupted."
        return 1
    fi
}

# 2. Safely remove the builder profile
remove_builder() {
    echo "🔄 Attempting to remove any existing minikube-builder profile..."

    if docker buildx rm minikube-builder >/dev/null 2>&1; then
        echo "✅ SUCCESS: Old builder profile removed successfully."
    else
        echo "❌ ERROR: No existing builder profile to remove (or it was already gone)."
    fi
    
    return 0
}

# 3. Safely initialize a new builder engine instance
create_builder() {
    echo "🔄 Attempting to bootstrap a new BuildKit engine instance..."
    
    if docker buildx create --name minikube-builder --use --bootstrap minikube-context 2>/dev/null; then
        echo "✅ SUCCESS: BuildKit engine successfully provisioned and booted!"
        return 0
    else
        echo "⚠️ WARNING: Bootstrap failed. Attempting lazy setup fallback..."
        docker buildx create --name minikube-builder --use minikube-context >/dev/null 2>&1 || true
        return 1
    fi
}

# 4. Safely toggle the active terminal context
use_builder() {
    echo "🔄 Attempting to select minikube-builder..."

    if docker buildx use minikube-builder >/dev/null 2>&1; then
        echo "✅ SUCCESS: Switched active context to minikube-builder."
        return 0
    else
        echo "❌ ERROR: Could not switch to minikube-builder context."
        return 1
    fi
}

# 5. Safely build and load Docker images with retry logic
build_image() {
    local TAG=$1
    local PATH_DIR=$2
    local RETRY_COUNT=0
    local MAX_RETRIES=2

    echo "🔄 Starting compilation pass: [$TAG]"
    
    until docker buildx build --load -t "$TAG" "$PATH_DIR"; do
        RETRY_COUNT=$((RETRY_COUNT+1))
        if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
            echo "❌ ERROR: Build failed permanently for $TAG after $MAX_RETRIES retries."
            return 1
        fi
        
        echo "⚠️ WARNING: Build failed for $TAG. [Attempt $RETRY_COUNT/$MAX_RETRIES]"
        docker builder prune -f >/dev/null 2>&1 || true
        sleep 3
    done
    echo "✅ SUCCESS: Successfully built and loaded: [$TAG]"
}


echo "🔌 Enabling Minikube addons..."

minikube addons enable metrics-server
minikube addons enable dashboard

echo "🔌 Connecting terminal session to Minikube's Docker daemon..."
eval $(minikube docker-env)

echo "🌐 Registering Minikube endpoints inside Docker contexts..."
docker context rm minikube-context >/dev/null 2>&1 || true

docker context create minikube-context \
  --description "Minikube Docker Engine" \
  --docker "host=$DOCKER_HOST,ca=$DOCKER_CERT_PATH/ca.pem,cert=$DOCKER_CERT_PATH/cert.pem,key=$DOCKER_CERT_PATH/key.pem" >/dev/null 2>&1 || true

# Automatically create or fix the buildx builder if it doesn't respond
if inspect_builder; then
    echo "🏗️  Creating a fresh BuildKit engine instance inside Minikube..."

    use_builder || (remove_builder && create_builder)
else
    echo "🔄 Switching active context to existing minikube-builder..."
    remove_builder
    create_builder
fi

echo "🚀 Starting optimized container builds via BuildKit..."

# Build and load images natively into the local cluster registry
build_image "client-image:latest" "./client"
build_image "autoscaler-image:latest" "./server/autoscaler"
build_image "dispatcher-image:latest" "./server/dispatcher"
build_image "worker-image:latest" "./server/worker"

echo "📦 Applying out Kubernetes manifests..."

# Apply your Kubernetes manifests
kubectl apply -k ./kubernetes

echo "✅ Entire stack successfully compiled, containerized, and deployed to Minikube!"
