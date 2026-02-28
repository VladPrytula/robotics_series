#!/usr/bin/env python3
"""Test SAC + MultiInputPolicy initialization time with pixel Dict obs.

Isolates whether the stall is in SB3 model creation or Isaac Lab env.

Usage (inside Isaac container):
    bash docker/dev-isaac.sh python3 scripts/test_sac_pixel_init.py
"""
import time
import sys
import numpy as np

# Minimal imports to test SB3
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

class FakePixelEnv(gym.Env):
    """Minimal gym env with Dict(pixels=CHW, state=flat) obs space."""
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict({
            "pixels": gym.spaces.Box(0, 255, (3, 84, 84), dtype=np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, (36,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        return {
            "pixels": np.zeros((3, 84, 84), dtype=np.uint8),
            "state": np.zeros(36, dtype=np.float32),
        }, {}

    def step(self, action):
        self._step += 1
        obs = {
            "pixels": np.random.randint(0, 256, (3, 84, 84), dtype=np.uint8),
            "state": np.random.randn(36).astype(np.float32),
        }
        return obs, 0.0, self._step >= 250, False, {}


def main():
    # Step 1: Create VecEnv
    print("[test] Creating DummyVecEnv with pixel Dict obs...", flush=True)
    t0 = time.perf_counter()
    env = DummyVecEnv([FakePixelEnv])
    dt = time.perf_counter() - t0
    print(f"[test] VecEnv created: {dt:.3f}s", flush=True)
    print(f"[test] obs_space: {env.observation_space}", flush=True)

    # Step 2: Create SAC model
    print("[test] Creating SAC(MultiInputPolicy, buffer_size=10000)...", flush=True)
    sys.stdout.flush()
    t0 = time.perf_counter()
    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        device="auto",
        buffer_size=10_000,
        batch_size=64,
        learning_starts=100,
    )
    dt = time.perf_counter() - t0
    print(f"[test] SAC created: {dt:.3f}s", flush=True)

    # Step 3: Run a few steps
    print("[test] Starting model.learn(500)...", flush=True)
    sys.stdout.flush()
    t0 = time.perf_counter()
    model.learn(total_timesteps=500)
    dt = time.perf_counter() - t0
    print(f"[test] model.learn(500) done: {dt:.3f}s", flush=True)

    print("[test] ALL TESTS PASSED", flush=True)
    env.close()


if __name__ == "__main__":
    main()
