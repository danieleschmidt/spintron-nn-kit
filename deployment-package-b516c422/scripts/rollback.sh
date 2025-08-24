#!/bin/bash
set -euo pipefail

NAMESPACE="spintron-production"
APP_NAME="spintron-nn-kit"

echo "🔄 Rolling back deployment..."

# Get current revision
CURRENT_REVISION=$(kubectl rollout history deployment "$APP_NAME" -n "$NAMESPACE" --output=jsonpath='{.metadata.generation}')

if [ "$CURRENT_REVISION" -gt 1 ]; then
    echo "Rolling back to previous revision..."
    kubectl rollout undo deployment "$APP_NAME" -n "$NAMESPACE"
    
    echo "⏳ Waiting for rollback to complete..."
    kubectl rollout status deployment "$APP_NAME" -n "$NAMESPACE" --timeout=300s
    
    echo "✅ Rollback completed successfully"
else
    echo "❌ No previous revision available for rollback"
    exit 1
fi
