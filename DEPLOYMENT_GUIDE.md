# SpinTron-NN-Kit Production Deployment Guide

## 🚀 Quick Start

SpinTron-NN-Kit supports multiple deployment scenarios from edge devices to large-scale cloud deployments.

### Prerequisites

- **Python 3.8+** with pip
- **Docker** (for containerized deployment)
- **Kubernetes** (for orchestrated deployment)
- **Node.js 16+** (for JavaScript tooling)

## 📦 Installation Methods

### 1. Package Installation (Recommended)

```bash
# Install from PyPI (when available)
pip install spintron-nn-kit

# Install with all optional dependencies
pip install spintron-nn-kit[all]

# Install specific feature sets
pip install spintron-nn-kit[simulation,visualization]
```

### 2. Development Installation

```bash
# Clone repository
git clone https://github.com/danieleschmidt/spintron-nn-kit.git
cd spintron-nn-kit

# Install in development mode
pip install -e .[dev,all]

# Verify installation
python3 run_minimal_validation.py
```

### 3. Docker Deployment

```bash
# Build production image
docker build --target production -t spintron-nn-kit:production .

# Run basic container
docker run -p 8080:8080 spintron-nn-kit:production

# Use docker-compose for full stack
docker-compose -f docker-compose.production.yml up -d
```

### 4. Kubernetes Deployment

```bash
# Deploy to Kubernetes cluster
kubectl apply -f deployment/kubernetes/spintron-deployment.yaml

# Check deployment status
kubectl get pods -n spintron-nn

# Access logs
kubectl logs -f deployment/spintron-api -n spintron-nn
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SpinTron-NN-Kit                          │
├─────────────────────────────────────────────────────────────┤
│  Edge Device    │    Cloud Service    │   HPC Cluster      │
│  ──────────     │    ─────────────    │   ─────────────    │
│  • Single core │    • Auto-scaling   │   • Multi-node     │
│  • 1GB RAM     │    • Load balancing │   • GPU clusters   │
│  • Basic model │    • Full features  │   • Quantum accel  │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Deployment Scenarios

### Scenario 1: Edge Device (Minimal)

**Target**: IoT devices, embedded systems, edge computing

```bash
# Minimal installation
pip install spintron-nn-kit[core]

# Docker edge deployment
docker build --target edge -t spintron-edge .
docker run --memory=512m spintron-edge
```

**Configuration:**
- CPU: 1-2 cores
- RAM: 512MB - 2GB
- Storage: 1GB minimum
- Features: Core functionality only

### Scenario 2: Single Server (Standard)

**Target**: Development, small production, proof-of-concept

```bash
# Full installation
pip install spintron-nn-kit[all]

# Docker deployment with monitoring
docker-compose up -d
```

**Configuration:**
- CPU: 4-8 cores
- RAM: 8-16GB
- Storage: 50GB
- Features: Full framework with monitoring

### Scenario 3: Cloud Service (Production)

**Target**: Production workloads, auto-scaling, high availability

```bash
# Kubernetes deployment
kubectl apply -f deployment/kubernetes/
```

**Configuration:**
- Multi-node cluster
- Auto-scaling (3-20 replicas)
- Load balancing
- Persistent storage
- Monitoring and alerting

### Scenario 4: HPC Cluster (Research)

**Target**: Large-scale research, quantum acceleration, distributed training

```bash
# HPC-specific configuration
pip install spintron-nn-kit[all,quantum,hpc]

# SLURM job submission
sbatch deployment/hpc/spintron-job.slurm
```

**Configuration:**
- Multi-node HPC cluster
- GPU/Quantum acceleration
- Distributed computing
- High-performance storage

## ⚙️ Configuration

### Environment Variables

```bash
# Core configuration
export SPINTRON_ENV=production
export SPINTRON_LOG_LEVEL=INFO
export SPINTRON_WORKERS=4
export SPINTRON_MAX_MEMORY_GB=8

# Database configuration
export POSTGRES_URL=postgresql://user:pass@host:5432/db
export REDIS_URL=redis://host:6379

# Security configuration
export SPINTRON_SECRET_KEY=your-secret-key
export SPINTRON_API_KEY=your-api-key

# Performance tuning
export SPINTRON_CACHE_SIZE=1000
export SPINTRON_BATCH_SIZE=32
export SPINTRON_QUANTUM_ENABLED=true
```

### Configuration Files

Create `spintron_config.yaml`:

```yaml
# spintron_config.yaml
environment: production

database:
  url: postgresql://localhost:5432/spintron
  pool_size: 10
  max_overflow: 20

redis:
  url: redis://localhost:6379
  db: 0

security:
  enable_auth: true
  enable_encryption: true
  differential_privacy:
    epsilon: 1.0
    delta: 1e-5

performance:
  workers: 4
  cache_size: 1000
  enable_quantum: true
  
logging:
  level: INFO
  format: json
  file: /var/log/spintron/app.log

monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30
```

## 🔐 Security Configuration

### 1. Authentication Setup

```python
from spintron_nn.security import SecurityConfig, SecurityLevel

security_config = SecurityConfig(
    security_level=SecurityLevel.HIGH,
    enable_differential_privacy=True,
    enable_encryption=True,
    dp_epsilon=1.0
)
```

### 2. Network Security

```bash
# Enable firewall
ufw enable
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 8080/tcp  # API

# TLS certificate setup
certbot --nginx -d api.spintron.ai
```

### 3. Container Security

```yaml
# Kubernetes security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
```

## 📊 Monitoring and Observability

### Metrics Collection

```bash
# Prometheus metrics endpoint
curl http://localhost:8080/metrics

# Health check endpoint
curl http://localhost:8080/health

# Ready check endpoint
curl http://localhost:8080/ready
```

### Log Aggregation

```bash
# View application logs
docker logs spintron-api

# Kubernetes logs
kubectl logs -f deployment/spintron-api -n spintron-nn

# Log aggregation with ELK stack
docker-compose -f docker-compose.logging.yml up -d
```

### Alerting Rules

```yaml
# prometheus-alerts.yml
groups:
- name: spintron-alerts
  rules:
  - alert: SpintronHighMemoryUsage
    expr: container_memory_usage_bytes{container="spintron-api"} / container_spec_memory_limit_bytes > 0.85
    for: 5m
    annotations:
      summary: "SpinTron API high memory usage"
```

## 🔧 Performance Tuning

### 1. Memory Optimization

```python
# Optimize memory usage
import torch
torch.set_num_threads(4)  # Limit CPU threads
torch.cuda.empty_cache()  # Clear GPU cache
```

### 2. CPU Optimization

```bash
# Set CPU affinity
taskset -c 0-3 python3 -m spintron_nn.api

# Enable CPU optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 3. I/O Optimization

```bash
# Use SSD storage for better performance
mount -o noatime /dev/ssd1 /app/data

# Enable memory mapped files
echo 'vm.max_map_count=262144' >> /etc/sysctl.conf
```

## 🌐 Load Balancing

### NGINX Configuration

```nginx
# /etc/nginx/sites-available/spintron
upstream spintron_backend {
    least_conn;
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8081 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8082 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.spintron.ai;
    
    location / {
        proxy_pass http://spintron_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://spintron_backend;
    }
}
```

### Traefik Configuration

```yaml
# traefik.yml
api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"

providers:
  docker:
    exposedByDefault: false

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@spintron.ai
      storage: acme.json
      httpChallenge:
        entryPoint: web
```

## 📈 Scaling Strategies

### Horizontal Scaling

```bash
# Docker Swarm scaling
docker service scale spintron-api=5

# Kubernetes scaling
kubectl scale deployment spintron-api --replicas=10 -n spintron-nn

# Auto-scaling based on metrics
kubectl autoscale deployment spintron-api --cpu-percent=70 --min=3 --max=20 -n spintron-nn
```

### Vertical Scaling

```yaml
# Increase resource limits
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

## 🔄 Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U spintron spintron > backup_$(date +%Y%m%d).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h postgres -U spintron spintron | gzip > "$BACKUP_DIR/spintron_$DATE.sql.gz"
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete
```

### Configuration Backup

```bash
# Backup configuration
kubectl get all,pvc,secrets,configmaps -n spintron-nn -o yaml > backup/k8s-backup.yaml

# Backup Docker volumes
docker run --rm -v spintron_data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz /data
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Check memory usage
   kubectl top pods -n spintron-nn
   
   # Increase memory limits
   kubectl patch deployment spintron-api -n spintron-nn -p '{"spec":{"template":{"spec":{"containers":[{"name":"spintron-api","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
   ```

2. **Database Connection Issues**
   ```bash
   # Check database connectivity
   kubectl exec -it pod/spintron-postgres-0 -n spintron-nn -- psql -U spintron -d spintron -c "SELECT 1;"
   
   # Reset database connection
   kubectl rollout restart deployment spintron-api -n spintron-nn
   ```

3. **Performance Issues**
   ```bash
   # Check API performance
   curl -w "@curl-format.txt" -o /dev/null -s "http://api.spintron.ai/health"
   
   # Profile application
   py-spy top --pid $(pgrep python)
   ```

### Debug Mode

```bash
# Enable debug logging
export SPINTRON_LOG_LEVEL=DEBUG

# Run with profiler
python3 -m cProfile -o profile.stats -m spintron_nn.api

# Analyze profile
python3 -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

## 📊 Health Checks

### Application Health

```python
# health_check.py
import requests
import sys

def check_health():
    try:
        response = requests.get('http://localhost:8080/health', timeout=10)
        if response.status_code == 200:
            print("✅ Application healthy")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    sys.exit(0 if check_health() else 1)
```

### System Health

```bash
#!/bin/bash
# system_health.sh

echo "🔍 System Health Check"
echo "====================="

# Check disk space
df -h | grep -E "/$|/app"

# Check memory usage
free -h

# Check CPU usage
top -bn1 | grep "Cpu(s)"

# Check network connectivity
ping -c 1 8.8.8.8 > /dev/null && echo "✅ Network OK" || echo "❌ Network issues"

# Check Docker containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## 🔧 Maintenance

### Regular Maintenance Tasks

1. **Daily**
   - Check application logs
   - Monitor resource usage
   - Verify backup completion

2. **Weekly**
   - Update security patches
   - Clean up old logs
   - Review performance metrics

3. **Monthly**
   - Update dependencies
   - Review and rotate secrets
   - Performance optimization

### Upgrade Procedure

```bash
# 1. Backup current deployment
kubectl create backup spintron-backup-$(date +%Y%m%d)

# 2. Update image
kubectl set image deployment/spintron-api spintron-api=spintron-nn-kit:v2.0.0 -n spintron-nn

# 3. Monitor rollout
kubectl rollout status deployment/spintron-api -n spintron-nn

# 4. Verify health
kubectl exec deployment/spintron-api -n spintron-nn -- python3 -c "import spintron_nn; print('Version:', spintron_nn.__version__)"

# 5. Rollback if needed
kubectl rollout undo deployment/spintron-api -n spintron-nn
```

## 📞 Support

### Getting Help

- **Documentation**: [docs.spintron.ai](https://docs.spintron.ai)
- **GitHub Issues**: [github.com/danieleschmidt/spintron-nn-kit/issues](https://github.com/danieleschmidt/spintron-nn-kit/issues)
- **Community**: [community.spintron.ai](https://community.spintron.ai)
- **Enterprise Support**: [support@spintron.ai](mailto:support@spintron.ai)

### Reporting Issues

When reporting issues, include:

1. **Environment Information**
   ```bash
   python3 -c "import spintron_nn; print('Version:', spintron_nn.__version__)"
   kubectl version --client
   docker --version
   ```

2. **Error Logs**
   ```bash
   kubectl logs deployment/spintron-api -n spintron-nn --since=1h
   ```

3. **System Information**
   ```bash
   kubectl top nodes
   kubectl top pods -n spintron-nn
   ```

---

## 📈 Success Metrics

Track these metrics to ensure successful deployment:

- **Availability**: > 99.9% uptime
- **Response Time**: < 100ms p95 latency
- **Error Rate**: < 0.1% error rate
- **Resource Usage**: < 80% CPU/Memory utilization
- **Scalability**: Handle 10x traffic spikes

For detailed deployment assistance, consult the specific deployment guides in the `deployment/` directory.