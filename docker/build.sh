#!/usr/bin/env bash
set -euo pipefail

# Platform-aware Docker image build.
# Automatically selects the correct Dockerfile for Mac M4 vs DGX/NVIDIA.

# Platform detection
os="$(uname -s)"

if [ "$os" = "Darwin" ]; then
    default_image="robotics-rl:mac"
    dockerfile="docker/Dockerfile.mac"
else
    default_image="robotics-rl:latest"
    dockerfile="docker/Dockerfile"
fi

image="${1:-$default_image}"

docker build -t "${image}" -f "${dockerfile}" .
