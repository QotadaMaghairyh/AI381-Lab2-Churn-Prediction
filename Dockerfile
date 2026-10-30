# Stage 1: Build dependencies
FROM python:3.14.4-slim AS builder

WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv

# Install dependencies
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime
FROM python:3.14.4-slim AS runtime

WORKDIR /app

# Use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy installed dependencies from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application files and trained model
COPY src/ ./src/
COPY model.joblib .

# Create a non-root user
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --uid 10001 appuser && \
    chown -R appuser:appgroup /app

# Run as non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Check application health
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]

# Start the API
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
