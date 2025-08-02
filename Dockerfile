# SpinTron-NN-Kit Docker Image
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /build
COPY requirements*.txt pyproject.toml setup.cfg ./
COPY spintron_nn/ ./spintron_nn/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -e .

# Development stage (includes dev tools)
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Install hardware simulation tools (if available)
RUN apt-get update && apt-get install -y \
    ngspice \
    gtkwave \
    iverilog \
    verilator \
    && rm -rf /var/lib/apt/lists/* || true

# Install Node.js for JavaScript tooling
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set up development environment
WORKDIR /workspace
COPY . .
RUN pip install --no-cache-dir -e ".[all]"

# Set up pre-commit hooks
RUN pre-commit install || true

# Expose ports for development services
EXPOSE 8888 6006 8000

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage (optimized for deployment)
FROM python:3.11-slim as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash spintron

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built package from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set up application directory
WORKDIR /app
COPY spintron_nn/ ./spintron_nn/
COPY examples/ ./examples/
COPY README.md LICENSE ./

# Change ownership to non-root user
RUN chown -R spintron:spintron /app

# Switch to non-root user
USER spintron

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import spintron_nn; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "spintron_nn.cli", "--help"]

# Hardware simulation stage (includes EDA tools)
FROM development as simulation

USER root

# Install additional hardware simulation tools
RUN apt-get update && apt-get install -y \
    octave \
    octave-signal \
    python3-matplotlib \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set up X11 forwarding for GUI tools
ENV DISPLAY=:99

# Install PySpice dependencies
RUN pip install --no-cache-dir ngspice-shared

# Create simulation workspace
RUN mkdir -p /simulation && chown spintron:spintron /simulation
WORKDIR /simulation

USER spintron

# Default command for simulation
CMD ["python", "-m", "spintron_nn.simulation.runner"]

# Benchmarking stage (optimized for performance testing)
FROM production as benchmarking

USER root

# Install performance monitoring tools
RUN apt-get update && apt-get install -y \
    htop \
    iotop \
    sysstat \
    perf-tools-unstable \
    && rm -rf /var/lib/apt/lists/* || true

# Install Python performance tools
RUN pip install --no-cache-dir \
    memory-profiler \
    py-spy \
    psutil \
    pytest-benchmark

USER spintron

# Set up benchmarking environment
ENV PYTHONPATH=/app
ENV BENCHMARK_OUTPUT_DIR=/app/benchmarks

# Default command for benchmarking
CMD ["python", "-m", "spintron_nn.benchmarks.runner"]