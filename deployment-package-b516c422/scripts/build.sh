#!/bin/bash
set -euo pipefail

# Build configuration
IMAGE_NAME="spintron-nn-kit"
IMAGE_TAG="${VERSION:-v1.0.0}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "🐳 Building Docker image..."
docker build \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg VCS_REF="$VCS_REF" \
    --build-arg VERSION="$IMAGE_TAG" \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    -t "$IMAGE_NAME:latest" \
    .

echo "✅ Build completed successfully"
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
