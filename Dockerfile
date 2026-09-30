FROM python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Native dependencies for OpenMVS + OpenCV + raster/3D processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    libgomp1 \
    libomp-dev \
    libboost-all-dev \
    libeigen3-dev \
    libopencv-dev \
    libnanoflann-dev \
    libcgal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel

# CPU PyTorch for Railway
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining Python dependencies
RUN grep -vE '^(torch|torchvision)([<>=].*)?$' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application
COPY . .

# Build OpenMVS
RUN chmod +x scripts/build_openmvs_linux.sh \
    && CMAKE_BUILD_PARALLEL_LEVEL=2 \
       OPENMVS_BUILD_DIR=/tmp/openmvs-build \
       OPENMVS_INSTALL_DIR=/opt/openmvs \
       bash scripts/build_openmvs_linux.sh

# Runtime directories
RUN mkdir -p \
    /app/data/input \
    /app/data/uploads \
    /app/data/output \
    /app/data/workspace \
    /app/data/job_state

ENV PATH="/app/.local/bin:/opt/openmvs/bin:${PATH}"

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]