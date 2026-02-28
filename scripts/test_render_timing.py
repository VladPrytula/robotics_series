#!/usr/bin/env python3
"""Benchmark Isaac Lab rendering speed: viewport render() vs stepping.

Measures per-step time with and without render() to quantify the rendering
bottleneck for pixel-based RL training.

Usage (inside Isaac container):
    bash docker/dev-isaac.sh python3 scripts/test_render_timing.py --headless --enable_cameras
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# Isaac Lab boot sequence (must happen before any other Isaac imports)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

# Now safe to import Isaac Lab + tasks
import isaaclab_tasks  # noqa: F401
import gymnasium as gym

from isaaclab_rl.sb3 import Sb3VecEnvWrapper


def make_env(env_id, num_envs=1, render_mode=None):
    """Create Isaac Lab env using the same pattern as appendix_e_isaac_peg.py."""
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(env_id, device="cuda:0", num_envs=num_envs)

    # Disable curriculum
    if hasattr(env_cfg, "curriculum"):
        curriculum_cfg = env_cfg.curriculum
        if curriculum_cfg is not None:
            for attr_name in list(vars(curriculum_cfg)):
                if not attr_name.startswith("_"):
                    setattr(curriculum_cfg, attr_name, None)

    kwargs = {"cfg": env_cfg}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)


def main():
    env_id = "Isaac-Lift-Cube-Franka-v0"
    n_steps = 50

    print(f"[bench] Creating env: {env_id} with render_mode=rgb_array, num_envs=1")
    isaac_env = make_env(env_id, num_envs=1, render_mode="rgb_array")
    sb3_env = Sb3VecEnvWrapper(isaac_env)

    obs = sb3_env.reset()
    print(f"[bench] obs shape: {obs.shape}, dtype: {obs.dtype}")

    action = sb3_env.action_space.sample()

    # Warm up renderer (first few frames may be blank / shader compilation)
    print("[bench] Warming up renderer (10 frames)...")
    for i in range(10):
        obs, rew, done, info = sb3_env.step(np.expand_dims(action, 0))
        isaac_env.render()
    print("[bench] Warmup done.")

    # Benchmark 1: step WITHOUT render
    print(f"\n[bench] === {n_steps} steps WITHOUT render ===")
    t0 = time.perf_counter()
    for _ in range(n_steps):
        obs, rew, done, info = sb3_env.step(np.expand_dims(action, 0))
    dt_no_render = time.perf_counter() - t0
    print(f"[bench] Total: {dt_no_render:.3f}s, per-step: {dt_no_render/n_steps*1000:.1f}ms, fps: {n_steps/dt_no_render:.0f}")

    # Benchmark 2: render() alone (no step)
    print(f"\n[bench] === {n_steps} renders WITHOUT step ===")
    t0 = time.perf_counter()
    for _ in range(n_steps):
        frame = isaac_env.render()
    dt_render_only = time.perf_counter() - t0
    print(f"[bench] Total: {dt_render_only:.3f}s, per-render: {dt_render_only/n_steps*1000:.1f}ms")
    if frame is not None:
        frame = np.asarray(frame)
        print(f"[bench] frame: shape={frame.shape}, dtype={frame.dtype}")

    # Benchmark 3: step + render (the pixel training path)
    print(f"\n[bench] === {n_steps} step+render cycles ===")
    from PIL import Image
    t0 = time.perf_counter()
    for _ in range(n_steps):
        obs, rew, done, info = sb3_env.step(np.expand_dims(action, 0))
        frame = isaac_env.render()
        if frame is not None:
            frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            pil_img = Image.fromarray(frame.astype(np.uint8) if frame.dtype != np.uint8 else frame)
            pil_img = pil_img.resize((84, 84), Image.Resampling.BILINEAR)
            pixels = np.array(pil_img, dtype=np.uint8).transpose(2, 0, 1)
    dt_step_render = time.perf_counter() - t0
    print(f"[bench] Total: {dt_step_render:.3f}s, per-cycle: {dt_step_render/n_steps*1000:.1f}ms, fps: {n_steps/dt_step_render:.1f}")

    # Summary
    render_overhead = dt_step_render - dt_no_render
    print(f"\n[bench] === Summary ===")
    print(f"[bench] Step only:       {dt_no_render/n_steps*1000:.1f} ms/step  ({n_steps/dt_no_render:.0f} fps)")
    print(f"[bench] Render only:     {dt_render_only/n_steps*1000:.1f} ms/render")
    print(f"[bench] Step + render:   {dt_step_render/n_steps*1000:.1f} ms/cycle ({n_steps/dt_step_render:.1f} fps)")
    print(f"[bench] Render overhead: {render_overhead/n_steps*1000:.1f} ms/step")
    print(f"[bench] At 2000 steps:   {2000*dt_step_render/n_steps:.0f}s ({2000*dt_step_render/n_steps/60:.1f} min)")
    print(f"[bench] At 2M steps:     {2e6*dt_step_render/n_steps/3600:.1f} hours")

    sb3_env.close()
    print("[bench] DONE")


if __name__ == "__main__":
    main()
