# OpenGenAI Production Dockerfile
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash opengenai
WORKDIR /app

# Install Python dependencies
COPY requirements/prod.txt requirements/base.txt ./requirements/
RUN pip install --no-cache-dir -r requirements/prod.txt

# Copy application code
COPY --chown=opengenai:opengenai . .

# Install the application
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R opengenai:opengenai /app

# Switch to non-root user
USER opengenai

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "opengenai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


# Development stage
FROM base as development

USER root

# Install development dependencies
COPY requirements/dev.txt requirements/test.txt ./requirements/
RUN pip install --no-cache-dir -r requirements/dev.txt -r requirements/test.txt

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER opengenai

# Override command for development
CMD ["python", "-m", "uvicorn", "opengenai.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Testing stage
FROM development as testing

USER root

# Copy test files
COPY --chown=opengenai:opengenai opengenai/tests/ ./opengenai/tests/
COPY --chown=opengenai:opengenai pyproject.toml tox.ini ./

USER opengenai

# Run tests
CMD ["python", "-m", "pytest", "opengenai/tests/", "-v", "--cov=opengenai", "--cov-report=xml", "--cov-report=html"] 