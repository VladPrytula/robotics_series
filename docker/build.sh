#!/usr/bin/env bash
set -euo pipefail

image="${1:-robotics-rl:latest}"

docker build -t "${image}" -f docker/Dockerfile .

