# Stage 1: Build dependencies
FROM python:3.12.16-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS builder

WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv

# Install dependencies
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt



# Stage 2: Runtime
FROM python:3.12.16-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS runtime

# Update OS packages to include security fixes
RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy installed dependencies from builder
COPY --from=builder /opt/venv /opt/venv

# Package the trusted champion exported by the promotion gate, with its provenance.
# GitHub Actions downloads these files into artifacts/ before building.
COPY src/ ./src/
COPY artifacts/model.joblib artifacts/metadata.json artifacts/champion.json ./artifacts/
RUN sha256sum artifacts/model.joblib > artifacts/model.sha256

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
