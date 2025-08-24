#!/bin/bash
set -euo pipefail

NAMESPACE="spintron-production"
APP_NAME="spintron-nn-kit"

echo "🏥 Performing health checks..."

# Check pod status
echo "📋 Pod status:"
kubectl get pods -n "$NAMESPACE" -l app="$APP_NAME"

# Check service status
echo "🌐 Service status:"
kubectl get svc -n "$NAMESPACE" -l app="$APP_NAME"

# Check ingress status
echo "🚪 Ingress status:"
kubectl get ingress -n "$NAMESPACE" -l app="$APP_NAME" || echo "No ingress found"

# Port forward for local testing
echo "🔌 Setting up port forwarding for local testing..."
kubectl port-forward -n "$NAMESPACE" svc/"$APP_NAME" 8080:80 &
PF_PID=$!

sleep 2

# Test health endpoints
echo "🧪 Testing health endpoints..."
curl -f http://localhost:8080/health/live || echo "❌ Liveness check failed"
curl -f http://localhost:8080/health/ready || echo "❌ Readiness check failed"
curl -f http://localhost:8080/health/startup || echo "❌ Startup check failed"

# Cleanup
kill $PF_PID 2>/dev/null || true

echo "✅ Health checks completed"
