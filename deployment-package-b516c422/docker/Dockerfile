# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set labels for metadata
LABEL maintainer="SpinTron-NN-Kit Team" \
      version="$VERSION" \
      description="Ultra-low-power neural inference framework" \
      build-date="$BUILD_DATE" \
      vcs-ref="$VCS_REF"

# Create non-root user
RUN groupadd -r spintron && useradd --no-log-init -r -g spintron spintron

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=spintron:spintron spintron_nn/ ./spintron_nn/
COPY --chown=spintron:spintron examples/ ./examples/
COPY --chown=spintron:spintron *.py ./
COPY --chown=spintron:spintron pyproject.toml ./

# Install application in development mode
RUN pip install -e .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/cache /app/data && \
    chown -R spintron:spintron /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER spintron

# Expose ports
EXPOSE 8080 8081

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    SPINTRON_ENV=production \
    SPINTRON_LOG_LEVEL=INFO

# Default command
CMD ["python", "-m", "spintron_nn.cli.main", "--serve", "--host", "0.0.0.0", "--port", "8080"]
