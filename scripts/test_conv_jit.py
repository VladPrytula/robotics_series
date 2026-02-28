#!/usr/bin/env python3
"""Test PyTorch Conv2d JIT compilation time on this GPU.

Runs a minimal NatureCNN-like forward pass to trigger CUDA kernel compilation.
If this takes minutes, the bottleneck in pixel training is JIT compilation
(expected on CUDA 12.1 / GB10 which is beyond PyTorch's supported 12.0).
If it's instant, the bottleneck is elsewhere.

Usage:
    python3 scripts/test_conv_jit.py
    # Or inside Isaac container:
    bash docker/dev-isaac.sh python3 scripts/test_conv_jit.py
"""
import time
import torch
import torch.nn as nn


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {cap[0]}.{cap[1]}")
        print(f"CUDA runtime: {torch.version.cuda}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NatureCNN architecture (same as SB3's default for image inputs)
    cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
    ).to(device)

    # Input: batch of 1, 3 channels, 84x84 (standard pixel obs size)
    x = torch.randn(1, 3, 84, 84, device=device)

    print(f"\n--- Forward pass (first call triggers JIT compilation) ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = cnn(x)
    torch.cuda.synchronize()
    dt1 = time.perf_counter() - t0
    print(f"First forward:  {dt1:.3f}s  (output shape: {out.shape})")

    # Second call should be fast (kernels cached)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = cnn(x)
    torch.cuda.synchronize()
    dt2 = time.perf_counter() - t0
    print(f"Second forward: {dt2:.3f}s")

    # Backward pass (also triggers kernel compilation)
    x_grad = torch.randn(1, 3, 84, 84, device=device, requires_grad=False)
    cnn.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = cnn(x_grad)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    dt3 = time.perf_counter() - t0
    print(f"Backward pass:  {dt3:.3f}s")

    # Also test with batch_size=256 (typical training batch)
    x_batch = torch.randn(256, 3, 84, 84, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = cnn(x_batch)
    torch.cuda.synchronize()
    dt4 = time.perf_counter() - t0
    print(f"Batch=256 fwd:  {dt4:.3f}s  (output shape: {out.shape})")

    if dt1 > 10:
        print(f"\n⚠  First forward took {dt1:.1f}s -- JIT compilation is the bottleneck!")
        print("   Subsequent runs will be faster (kernels are now cached).")
    else:
        print(f"\n✓  JIT compilation is fast ({dt1:.3f}s). Bottleneck is elsewhere.")


if __name__ == "__main__":
    main()
