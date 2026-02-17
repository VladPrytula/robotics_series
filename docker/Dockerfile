FROM nvcr.io/nvidia/pytorch:25.12-py3

ARG DEBIAN_FRONTEND=noninteractive

# System deps:
# - OpenGL/EGL/OSMesa + GLFW for MuJoCo rendering (headless + windowed)
# - ffmpeg for video writing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglfw3 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Default to headless EGL on DGX; can override at runtime.
ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir -U pip wheel \
  && python -m pip install --no-cache-dir \
    gymnasium \
    gymnasium-robotics \
    mujoco \
    stable-baselines3 \
    tensorboard \
    imageio
