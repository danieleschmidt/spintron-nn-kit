# SpinTron-NN-Kit Production Container
FROM python:3.10-slim

LABEL version="1.0.0"
LABEL description="SpinTron-NN-Kit: Ultra-low-power neural inference framework"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Copy configuration
COPY config/production.json config.json

# Create non-root user
RUN useradd --create-home --shell /bin/bash spintron
RUN chown -R spintron:spintron /app
USER spintron

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import spintron_nn; print('OK')" || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "-m", "spintron_nn.cli", "serve", "--port", "8080"]
