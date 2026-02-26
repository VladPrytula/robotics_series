#!/usr/bin/env bash
set -euo pipefail

# Isaac Lab developer entrypoint.
#
# Usage:
#   bash docker/dev-isaac.sh                              # interactive shell
#   bash docker/dev-isaac.sh python scripts/isaac_proof_of_life.py all --headless
#
# Differences from dev.sh (MuJoCo pipeline):
#   - GPU-only: fails fast on Mac or missing nvidia-smi
#   - No venv: NGC image has its own Python env
#   - Runs as root: Omniverse requires OMNI_KIT_ALLOW_ROOT=1
#   - Named volumes: persist Kit/GL/compute caches across runs
#   - --network=host: Isaac Sim uses network for Omniverse services
#   - Mounts repo at /workspace/project (not /workspace) to avoid
#     shadowing Isaac Lab's internal /workspace/isaaclab

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- platform gate ----------

os_name="$(uname -s)"
if [ "$os_name" = "Darwin" ]; then
    echo "ERROR: Isaac Lab requires an NVIDIA GPU. Mac is not supported." >&2
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Isaac Lab requires an NVIDIA GPU with drivers installed." >&2
    exit 1
fi

# ---------- image ----------

image="${IMAGE:-robotics-rl:isaac}"

if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "Docker image ${image} not found; building from Dockerfile.isaac ..." >&2
    if ! docker build -t "${image}" -f "${repo_root}/docker/Dockerfile.isaac" "${repo_root}"; then
        echo "WARNING: build failed; falling back to raw NGC image (frame saving may be unavailable)." >&2
        image="nvcr.io/nvidia/isaac-lab:2.3.2"
    fi
fi

# ---------- command ----------

cmd=( "$@" )
if [ ${#cmd[@]} -eq 0 ]; then
    cmd=( bash )
fi

# ---------- inner script ----------
# Isaac Sim's Python needs PYTHONPATH + LD_LIBRARY_PATH from setup_python_env.sh.
# We source that inside the container so "python3" finds all Omniverse modules.

inner_script='
set -euo pipefail
export CARB_APP_PATH=/isaac-sim/kit
export ISAAC_PATH=/isaac-sim
export EXP_PATH=/isaac-sim/apps
# Initialize before sourcing -- setup_python_env.sh appends to these.
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /isaac-sim/setup_python_env.sh
export PATH="/isaac-sim/kit/python/bin:${PATH}"
# python.sh sets this; needed for Omniverse Kit plugin loading.
export LD_PRELOAD=/isaac-sim/kit/libcarb.so
export RESOURCE_NAME="IsaacSim"
exec "$@"
'

# ---------- docker run ----------

# Use -it only when stdin is a terminal (allows non-interactive use from scripts/CI).
tty_flags=()
if [ -t 0 ]; then
    tty_flags=(-it)
fi

# Mount host Vulkan ICD so the container sees the actual driver's Vulkan version.
vulkan_mounts=()
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    vulkan_mounts=(-v /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro)
fi

docker run --rm "${tty_flags[@]}" \
    --gpus all \
    --ipc=host \
    --network=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e PYTHONUNBUFFERED=1 \
    -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
    "${vulkan_mounts[@]}" \
    -v "${repo_root}":/workspace/project \
    -v isaac-cache-kit:/root/.local/share/ov/kit \
    -v isaac-cache-glcache:/root/.cache/nvidia/GLCache \
    -v isaac-cache-computecache:/root/.nv/ComputeCache \
    -v isaac-cache-logs:/root/.nvidia-omniverse/logs \
    -v isaac-cache-data:/root/.local/share/ov/data \
    -w /workspace/project \
    "${image}" \
    bash -c "${inner_script}" bash "${cmd[@]}"
