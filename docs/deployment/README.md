# Deployment Guide

## Overview

SpinTron-NN-Kit supports multiple deployment scenarios from development environments to production edge devices and cloud platforms.

## Deployment Options

### 1. Development Environment

#### Local Installation
```bash
# Install from source
git clone https://github.com/danieleschmidt/spintron-nn-kit.git
cd spintron-nn-kit
pip install -e ".[dev]"

# Or install from PyPI
pip install spintron-nn-kit[dev]
```

#### Docker Development
```bash
# Build development image
docker build --target development -t spintron-nn-kit:dev .

# Run with Docker Compose
docker-compose up spintron-dev

# Interactive development
docker run -it --rm -v $(pwd):/workspace spintron-nn-kit:dev bash
```

#### DevContainer (VS Code)
```bash
# Open in VS Code with Remote-Containers extension
code .
# Select "Reopen in Container" when prompted
```

### 2. Production Deployment

#### Docker Production
```bash
# Build production image
docker build --target production -t spintron-nn-kit:latest .

# Run production container
docker run -d \
  --name spintron-production \
  -v /path/to/models:/app/models \
  -v /path/to/data:/app/data \
  -e ENVIRONMENT=production \
  spintron-nn-kit:latest
```

#### Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spintron-nn-kit
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spintron-nn-kit
  template:
    metadata:
      labels:
        app: spintron-nn-kit
    spec:
      containers:
      - name: spintron-nn-kit
        image: spintron-nn-kit:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: spintron-nn-kit-service
spec:
  selector:
    app: spintron-nn-kit
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Cloud Deployment

##### AWS ECS
```json
{
  "family": "spintron-nn-kit",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "spintron-nn-kit",
      "image": "your-registry/spintron-nn-kit:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/spintron-nn-kit",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

##### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy spintron-nn-kit \
  --image gcr.io/your-project/spintron-nn-kit:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10
```

##### Azure Container Instances
```bash
# Deploy to Azure
az container create \
  --resource-group spintron-rg \
  --name spintron-nn-kit \
  --image your-registry/spintron-nn-kit:latest \
  --cpu 1 \
  --memory 1 \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production
```

### 3. Edge Device Deployment

#### NVIDIA Jetson
```bash
# Install with CUDA support
pip install spintron-nn-kit[gpu]

# Or use Jetson-optimized Docker image
docker pull spintron-nn-kit:jetson
docker run --runtime nvidia -d spintron-nn-kit:jetson
```

#### Raspberry Pi
```bash
# ARM-compatible installation
pip install spintron-nn-kit --extra-index-url https://www.piwheels.org/simple

# Lightweight Docker image
docker pull spintron-nn-kit:arm64
```

#### FPGA Deployment
```bash
# Generate FPGA bitstream
spintron-convert \
  --input model.pth \
  --target fpga \
  --board zcu104 \
  --output fpga_design/

# Program FPGA
cd fpga_design/
vivado -mode batch -source program_fpga.tcl
```

## Hardware Simulation Deployment

### SPICE Simulation Server
```bash
# Build simulation image
docker build --target simulation -t spintron-nn-kit:simulation .

# Run simulation server
docker run -d \
  --name spintron-simulation \
  -v /path/to/circuits:/simulation \
  -e SPICE_SIMULATOR=ngspice \
  spintron-nn-kit:simulation
```

### Distributed Simulation
```yaml
# docker-compose-simulation.yml
version: '3.8'
services:
  simulation-master:
    image: spintron-nn-kit:simulation
    environment:
      - SIMULATION_MODE=master
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  simulation-worker:
    image: spintron-nn-kit:simulation
    environment:
      - SIMULATION_MODE=worker
      - REDIS_URL=redis://redis:6379
    deploy:
      replicas: 4
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
```

## Performance Monitoring

### Prometheus Integration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'spintron-nn-kit'
    static_configs:
      - targets: ['spintron-nn-kit:8000']
    metrics_path: /metrics
```

### Grafana Dashboards
```bash
# Import pre-built dashboard
curl -X POST \
  -H "Content-Type: application/json" \
  -d @monitoring/spintron-dashboard.json \
  http://admin:admin@grafana:3000/api/dashboards/db
```

## Configuration Management

### Environment Variables
```bash
# Production configuration
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export SPICE_SIMULATOR=ngspice
export HARDWARE_BACKEND=fpga
export DATABASE_URL=postgresql://user:pass@db:5432/spintron
export REDIS_URL=redis://redis:6379
export WANDB_API_KEY=your_wandb_key
```

### Configuration Files
```yaml
# config/production.yaml
environment: production
logging:
  level: INFO
  format: json
  
hardware:
  backend: fpga
  spice_simulator: ngspice
  
database:
  url: postgresql://user:pass@db:5432/spintron
  pool_size: 10
  
monitoring:
  enabled: true
  prometheus_port: 9090
  
security:
  secret_key: ${SECRET_KEY}
  allowed_hosts:
    - spintron.example.com
```

## Scaling Strategies

### Horizontal Scaling
```bash
# Scale with Docker Swarm
docker service scale spintron-nn-kit=5

# Scale with Kubernetes
kubectl scale deployment spintron-nn-kit --replicas=5
```

### Auto-scaling
```yaml
# HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: spintron-nn-kit-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: spintron-nn-kit
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Security Considerations

### Container Security
```dockerfile
# Security-hardened Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash spintron \
    && mkdir -p /app \
    && chown spintron:spintron /app

# Install security updates
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER spintron

# Read-only root filesystem
VOLUME ["/tmp"]
```

### Network Security
```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: spintron-nn-kit-netpol
spec:
  podSelector:
    matchLabels:
      app: spintron-nn-kit
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8000
```

## Backup and Recovery

### Data Backup
```bash
# Backup models and data
docker run --rm \
  -v spintron_models:/data \
  -v $(pwd)/backup:/backup \
  alpine:latest \
  tar czf /backup/models_$(date +%Y%m%d).tar.gz -C /data .
```

### Database Backup
```bash
# PostgreSQL backup
kubectl exec -t postgres-pod -- pg_dump -U spintron spintron_metrics > backup.sql

# Restore
kubectl exec -i postgres-pod -- psql -U spintron spintron_metrics < backup.sql
```

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Check memory usage
docker stats spintron-nn-kit

# Increase memory limits
docker run --memory=2g spintron-nn-kit:latest
```

#### GPU Access
```bash
# Verify GPU access
docker run --gpus all spintron-nn-kit:latest nvidia-smi

# Check CUDA version
docker run --gpus all spintron-nn-kit:latest python -c "import torch; print(torch.cuda.is_available())"
```

#### Network Connectivity
```bash
# Test connectivity
kubectl exec -it spintron-pod -- curl -I http://external-service:8080

# Check DNS resolution
kubectl exec -it spintron-pod -- nslookup external-service
```

### Logging and Debugging

#### Container Logs
```bash
# View logs
docker logs spintron-nn-kit
kubectl logs deployment/spintron-nn-kit

# Follow logs
docker logs -f spintron-nn-kit
kubectl logs -f deployment/spintron-nn-kit
```

#### Debug Mode
```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG spintron-nn-kit:latest

# Interactive debugging
docker run -it --entrypoint /bin/bash spintron-nn-kit:latest
```

## Performance Optimization

### Resource Allocation
```yaml
# Optimized resource requests
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### Caching
```bash
# Enable Redis caching
docker run -d --name redis redis:alpine
docker run --link redis:redis -e REDIS_URL=redis://redis:6379 spintron-nn-kit:latest
```

### Load Balancing
```yaml
# HAProxy configuration
backend spintron_backend
    balance roundrobin
    server spintron1 spintron-1:8000 check
    server spintron2 spintron-2:8000 check
    server spintron3 spintron-3:8000 check
```

## CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build and push Docker image
      run: |
        docker build -t spintron-nn-kit:${{ github.ref_name }} .
        docker push spintron-nn-kit:${{ github.ref_name }}
    - name: Deploy to production
      run: |
        kubectl set image deployment/spintron-nn-kit \
          spintron-nn-kit=spintron-nn-kit:${{ github.ref_name }}
```

### GitLab CI/CD
```yaml
# .gitlab-ci.yml
stages:
  - build
  - deploy

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG

deploy:
  stage: deploy
  script:
    - kubectl set image deployment/spintron-nn-kit \
        spintron-nn-kit=$CI_REGISTRY_IMAGE:$CI_COMMIT_TAG
  only:
    - tags
```

This comprehensive deployment guide covers all major deployment scenarios for SpinTron-NN-Kit, from development environments to production edge devices and cloud platforms.