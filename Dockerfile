# RAG Application Docker Configuration
#
# This Dockerfile creates a containerized environment for the RAG (Retrieval-Augmented Generation)
# diagnosis application. It implements a multi-stage build process optimized for Python applications
# and is designed for production deployment with Streamlit web interface and Poetry dependency management.
#
# Key components:
# - Base image: Python 3.11 slim for optimal size and compatibility
# - Dependency management: Poetry for reproducible builds and dependency resolution
# - Environment optimization: Python environment variables for performance
# - System dependencies: Essential build tools and libraries for RAG components
# - Port configuration: Streamlit web server on port 8501
# - Security configuration: Streamlit server settings optimized for containerized deployment
#
# Dependencies: Python 3.11, Poetry 1.8.3, Streamlit, system build tools
# Usage: docker build -t diagnosis-app . && docker run -p 8501:8501 diagnosis-app

# Use lightweight Python image compatible with multi-architecture deployment
FROM python:3.11-slim

# Environment variables for Python optimization and performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Add Python path for the project module resolution
ENV PYTHONPATH=/app

# Install system dependencies required for RAG project components
# Includes build tools for Python packages with native extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory in the container
WORKDIR /app

# Copy Poetry configuration files first for Docker layer caching optimization
# This allows dependency installation to be cached when source code changes
COPY pyproject.toml poetry.lock* ./

# Install Poetry with specified version for reproducible builds
RUN pip install poetry==1.8.3

# Configure Poetry to not create virtual environment (optimal for Docker containers)
# Virtual environments are unnecessary in isolated container environments
RUN poetry config virtualenvs.create false

# Install project dependencies via Poetry
# Configuration flags:
# --no-dev: exclude development dependencies for production deployment
# --no-interaction: non-interactive mode for automated builds
# --no-ansi: disable color output in build logs
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy remaining project files after dependency installation
# This layer will be rebuilt when source code changes but dependencies remain cached
COPY . .

# Create necessary directories for data storage and configuration
# Ensures proper directory structure for application runtime
RUN mkdir -p /app/data /app/config /app/core

# Expose standard Streamlit port for web interface
EXPOSE 8501

# Streamlit server configuration optimized for Docker deployment
# Configuration parameters:
# --server.address=0.0.0.0: listen on all network interfaces for container access
# --server.port=8501: standard Streamlit port for web interface
# --server.headless=true: headless mode appropriate for containerized deployment
# --server.enableCORS=false: disable CORS for simplified Docker networking
# --server.enableXsrfProtection=false: disable XSRF protection for container environment
CMD ["poetry", "run", "streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]