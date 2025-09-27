# Multi-stage build for optimized Docker image
FROM python:3.11-slim as builder

# Set environment variables for build optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_TIMEOUT=1000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements-railway.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --prefer-binary --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app directory
WORKDIR /app

# Copy application code
COPY . .

# Note: Context documents should be in the memory service directory if needed
# Railway cannot access parent directories outside build context

# Debug: List what was actually copied
RUN echo "=== DEBUGGING FILE COPY ===" && \
    echo "Contents of /app:" && \
    ls -la /app && \
    echo "Contents of /app/luki_memory:" && \
    ls -la /app/luki_memory && \
    echo "Contents of /app/luki_memory/storage:" && \
    ls -la /app/luki_memory/storage || echo "Storage directory does not exist" && \
    echo "Note: _context and context-documents not copied (outside build context)"

# Create verification script
RUN echo 'import sys, os' > /tmp/verify.py && \
    echo 'print(f"Python version: {sys.version}")' >> /tmp/verify.py && \
    echo 'sys.path.insert(0, "/app")' >> /tmp/verify.py && \
    echo 'import luki_memory' >> /tmp/verify.py && \
    echo 'print(f"✓ luki_memory imported from: {luki_memory.__file__}")' >> /tmp/verify.py && \
    echo 'storage_path = os.path.join(luki_memory.__path__[0], "storage")' >> /tmp/verify.py && \
    echo 'print(f"Storage exists: {os.path.exists(storage_path)}")' >> /tmp/verify.py && \
    echo 'if os.path.exists(storage_path):' >> /tmp/verify.py && \
    echo '    print("Storage directory contents:")' >> /tmp/verify.py && \
    echo '    for f in os.listdir(storage_path): print(f"  {f}")' >> /tmp/verify.py && \
    echo '    knowledge_store_file = os.path.join(storage_path, "knowledge_store.py")' >> /tmp/verify.py && \
    echo '    print(f"knowledge_store.py exists: {os.path.exists(knowledge_store_file)}")' >> /tmp/verify.py && \
    echo '    if os.path.exists(knowledge_store_file):' >> /tmp/verify.py && \
    echo '        try:' >> /tmp/verify.py && \
    echo '            from luki_memory.storage.knowledge_store import get_project_knowledge_store, initialize_project_knowledge' >> /tmp/verify.py && \
    echo '            print("✓ knowledge_store module imported successfully as package module")' >> /tmp/verify.py && \
    echo '            print(f"✓ get_project_knowledge_store function available: {callable(get_project_knowledge_store)}")' >> /tmp/verify.py && \
    echo '            print(f"✓ initialize_project_knowledge function available: {callable(initialize_project_knowledge)}")' >> /tmp/verify.py && \
    echo '        except ImportError as e:' >> /tmp/verify.py && \
    echo '            print(f"✗ Failed to import knowledge_store as package module: {e}")' >> /tmp/verify.py && \
    echo '            sys.exit(1)' >> /tmp/verify.py && \
    echo '    else:' >> /tmp/verify.py && \
    echo '        print("✗ knowledge_store.py file missing - cannot verify module loading")' >> /tmp/verify.py && \
    echo '        sys.exit(1)' >> /tmp/verify.py && \
    echo 'else:' >> /tmp/verify.py && \
    echo '    print("✗ Storage directory does not exist")' >> /tmp/verify.py && \
    echo '    sys.exit(1)' >> /tmp/verify.py

# Install the package in editable mode and verify critical modules can be imported
RUN pip install -e . && \
    echo "=== VERIFYING PACKAGE INSTALLATION ===" && \
    python /tmp/verify.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "luki_memory.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
