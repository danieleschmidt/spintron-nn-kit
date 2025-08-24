#!/bin/bash
set -euo pipefail

NAMESPACE="spintron-production"
APP_NAME="spintron-nn-kit"

echo "🚀 Deploying to Kubernetes..."

# Create namespace if it doesn't exist
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Apply all manifests
echo "📋 Applying Kubernetes manifests..."
kubectl apply -f k8s/ -n "$NAMESPACE"

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment "$APP_NAME" -n "$NAMESPACE" --timeout=300s

# Verify deployment
echo "🔍 Verifying deployment..."
kubectl get pods -n "$NAMESPACE" -l app="$APP_NAME"

# Check service endpoints
echo "🌐 Service endpoints:"
kubectl get svc -n "$NAMESPACE"

echo "✅ Deployment completed successfully"
