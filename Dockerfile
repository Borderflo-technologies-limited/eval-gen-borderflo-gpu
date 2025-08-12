# Evaluation Agent Service Dockerfile (GPU-enabled for Whisper)
# Base on NVIDIA CUDA runtime to enable GPU for Faster-Whisper (ctranslate2)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python 3.10 and essentials
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimized pip configuration
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Create temp directory
RUN mkdir -p temp

# Create environment file template
COPY env.example .env

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget -qO- http://localhost:8002/health || exit 1

# Run the FastAPI application
CMD ["python3.10", "app/main.py"]