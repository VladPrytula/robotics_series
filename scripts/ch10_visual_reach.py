#!/usr/bin/env python3
"""Chapter 10: Pixels, No Cheating -- Visual SAC on FetchReachDense

Week 10 goals:
1. Demonstrate state vs pixel observation trade-off on the same task
2. Train SAC on FetchReachDense with privileged state (baseline from Ch3)
3. Train SAC on FetchReachDense from pixels only (no goal vectors in obs)
4. Measure the sample-efficiency gap quantitatively
5. Train SAC with DrQ augmentation (Kostrikov et al. 2020) to close the gap
6. Three-way comparison: state vs pixel vs pixel+DrQ

Usage:
    # Full pipeline: state + pixel + pixel+DrQ training + comparison
    python scripts/ch10_visual_reach.py all --seed 0

    # Train state-based SAC (baseline)
    python scripts/ch10_visual_reach.py train-state --seed 0

    # Train pixel-based SAC
    python scripts/ch10_visual_reach.py train-pixel --seed 0

    # Train pixel-based SAC with DrQ augmentation
    python scripts/ch10_visual_reach.py train-pixel-drq --seed 0

    # Evaluate a checkpoint (use --pixel for pixel checkpoints)
    python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_state_FetchReachDense-v4_seed0.zip
    python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_pixel_FetchReachDense-v4_seed0.zip --pixel
    python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_drq_FetchReachDense-v4_seed0.zip --pixel

    # Compare results (2-way or 3-way depending on available evals)
    python scripts/ch10_visual_reach.py compare
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `from scripts.labs...` imports work
# when this script is invoked as `python scripts/ch10_visual_reach.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Ch10Config:
    """Centralized configuration for Chapter 10 experiments.

    Two training modes share most hyperparameters but differ in:
    - Observation type (state vector vs pixel image)
    - Total steps (pixel needs more samples)
    - Buffer size (pixel buffers are memory-constrained)
    - Number of parallel envs (pixel rendering is expensive)
    """
    # Environment
    env: str = "FetchReachDense-v4"
    seed: int = 0
    device: str = "auto"

    # State-based training
    state_n_envs: int = 8
    state_total_steps: int = 500_000
    state_buffer_size: int = 1_000_000

    # Pixel-based training
    pixel_n_envs: int = 4
    pixel_total_steps: int = 2_000_000
    pixel_buffer_size: int = 200_000
    image_size: int = 84
    pixel_goal_mode: str = "none"  # {"none", "desired", "both"} (see PixelObservationWrapper)

    # DrQ (pixel + augmentation)
    drq_pad: int = 4
    drq_total_steps: int = 2_000_000

    # Shared SAC hyperparameters
    batch_size: int = 256
    learning_starts: int = 1_000
    learning_rate: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005

    # Paths
    log_dir: str = "runs"
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"

    # Evaluation
    n_eval_episodes: int = 100
    eval_deterministic: bool = True


DEFAULT_CONFIG = Ch10Config()


# =============================================================================
# Utilities
# =============================================================================

def _gather_versions() -> dict[str, str]:
    """Collect package versions for reproducibility."""
    versions: dict[str, str] = {"python": sys.version.replace("\n", " ")}
    try:
        import torch
        versions["torch"] = getattr(torch, "__version__", "unknown")
        versions["torch_cuda"] = str(getattr(torch.version, "cuda", "unknown"))
    except Exception:
        pass
    for module_name in ["gymnasium", "gymnasium_robotics", "mujoco", "stable_baselines3"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            continue
    return versions


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ckpt_path(cfg: Ch10Config, mode: str) -> Path:
    """Generate checkpoint path for state or pixel mode."""
    name = f"sac_{mode}_{cfg.env}_seed{cfg.seed}"
    return _ensure_dir(cfg.checkpoints_dir) / f"{name}.zip"


def _meta_path(ckpt: Path) -> Path:
    """Derive metadata path from checkpoint path."""
    return ckpt.parent / (ckpt.stem + ".meta.json")


def _result_path(cfg: Ch10Config, mode: str) -> Path:
    """Generate result JSON path for state or pixel mode."""
    return _ensure_dir(cfg.results_dir) / f"ch10_{mode}_eval.json"


def _comparison_path(cfg: Ch10Config) -> Path:
    return _ensure_dir(cfg.results_dir) / "ch10_comparison.json"


# =============================================================================
# Commands
# =============================================================================

def _pixel_mode_tag(cfg: Ch10Config) -> str:
    """Filename tag for pixel-mode runs (encodes goal exposure)."""
    return "pixel" if cfg.pixel_goal_mode == "none" else f"pixel_{cfg.pixel_goal_mode}"


def _drq_mode_tag(cfg: Ch10Config) -> str:
    """Filename tag for DrQ pixel-mode runs (encodes goal exposure)."""
    return "drq" if cfg.pixel_goal_mode == "none" else f"drq_{cfg.pixel_goal_mode}"


def cmd_train_state(cfg: Ch10Config) -> int:
    """Train SAC on FetchReachDense with privileged state observations."""
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    device = _resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    print(f"[ch10] Training STATE SAC on {cfg.env}")
    print(f"[ch10] seed={cfg.seed}, n_envs={cfg.state_n_envs}, "
          f"total_steps={cfg.state_total_steps}, device={device}")

    env = make_vec_env(cfg.env, n_envs=cfg.state_n_envs, seed=cfg.seed)

    log_dir = _ensure_dir(cfg.log_dir)
    run_id = f"sac_state/{cfg.env}/seed{cfg.seed}"

    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
            batch_size=cfg.batch_size,
            buffer_size=cfg.state_buffer_size,
            learning_starts=cfg.learning_starts,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            tau=cfg.tau,
            ent_coef="auto",
        )

        t0 = time.perf_counter()
        model.learn(total_timesteps=cfg.state_total_steps, tb_log_name=run_id)
        elapsed = time.perf_counter() - t0
        fps = cfg.state_total_steps / elapsed

        ckpt = _ckpt_path(cfg, "state")
        model.save(str(ckpt))
        print(f"[ch10] Saved: {ckpt}")
        print(f"[ch10] Training time: {elapsed:.1f}s ({fps:.0f} steps/sec)")

    finally:
        env.close()

    # Write metadata
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 10,
        "mode": "state",
        "algo": "sac",
        "env_id": cfg.env,
        "seed": cfg.seed,
        "device": device,
        "n_envs": cfg.state_n_envs,
        "total_steps": cfg.state_total_steps,
        "training_time_sec": elapsed,
        "steps_per_sec": fps,
        "checkpoint": str(ckpt),
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.state_buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": "auto",
        },
        "versions": _gather_versions(),
    }
    _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Wrote metadata: {_meta_path(ckpt)}")

    return 0


def cmd_train_pixel(cfg: Ch10Config) -> int:
    """Train SAC on FetchReachDense with pixel observations.

    Uses PixelObservationWrapper from our lab module to replace the flat
    state vector with rendered 84x84 RGB images. By default we expose
    pixels only (goal vectors removed from the observation dict), so the
    policy must infer both current state and target from the image.

    SB3's MultiInputPolicy auto-detects the image space and uses
    CombinedExtractor -> NatureCNN for the "pixels" key.
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    device = _resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    print(f"[ch10] Training PIXEL SAC on {cfg.env}")
    print(f"[ch10] seed={cfg.seed}, n_envs={cfg.pixel_n_envs}, "
          f"total_steps={cfg.pixel_total_steps}, device={device}")
    print(f"[ch10] image_size={cfg.image_size}x{cfg.image_size}, "
          f"buffer_size={cfg.pixel_buffer_size}")
    print(f"[ch10] pixel_goal_mode={cfg.pixel_goal_mode} (none=fully pixels-only)")

    def make_pixel_env():
        env = gym.make(cfg.env, render_mode="rgb_array")
        return PixelObservationWrapper(
            env,
            image_size=(cfg.image_size, cfg.image_size),
            goal_mode=cfg.pixel_goal_mode,
        )

    env = make_vec_env(make_pixel_env, n_envs=cfg.pixel_n_envs, seed=cfg.seed)

    log_dir = _ensure_dir(cfg.log_dir)
    run_id = f"sac_pixel/{cfg.env}/seed{cfg.seed}"

    try:
        # SB3 auto-detects image space -> CombinedExtractor -> NatureCNN for pixels
        # No custom policy_kwargs needed
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
            batch_size=cfg.batch_size,
            buffer_size=cfg.pixel_buffer_size,
            learning_starts=cfg.learning_starts,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            tau=cfg.tau,
            ent_coef="auto",
        )

        t0 = time.perf_counter()
        model.learn(total_timesteps=cfg.pixel_total_steps, tb_log_name=run_id)
        elapsed = time.perf_counter() - t0
        fps = cfg.pixel_total_steps / elapsed

        ckpt = _ckpt_path(cfg, _pixel_mode_tag(cfg))
        model.save(str(ckpt))
        print(f"[ch10] Saved: {ckpt}")
        print(f"[ch10] Training time: {elapsed:.1f}s ({fps:.0f} steps/sec)")

    finally:
        env.close()

    # Write metadata
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 10,
        "mode": _pixel_mode_tag(cfg),
        "algo": "sac",
        "env_id": cfg.env,
        "seed": cfg.seed,
        "device": device,
        "n_envs": cfg.pixel_n_envs,
        "total_steps": cfg.pixel_total_steps,
        "training_time_sec": elapsed,
        "steps_per_sec": fps,
        "checkpoint": str(ckpt),
        "image_size": cfg.image_size,
        "pixel_goal_mode": cfg.pixel_goal_mode,
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.pixel_buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": "auto",
        },
        "versions": _gather_versions(),
    }
    _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Wrote metadata: {_meta_path(ckpt)}")

    return 0


def cmd_train_pixel_drq(cfg: Ch10Config) -> int:
    """Train SAC on FetchReachDense with pixel observations + DrQ augmentation.

    Identical to cmd_train_pixel except the replay buffer applies random
    shift augmentation (Kostrikov et al. 2020) to pixel observations at
    sample time. This regularizes the Q-function against overfitting to
    pixel-level details, closing part of the state-vs-pixel gap.

    The only change from naive pixel SAC is the replay buffer class --
    everything else (architecture, hyperparameters) stays the same. This
    isolates the effect of augmentation.
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    from scripts.labs.image_augmentation import DrQDictReplayBuffer, RandomShiftAug
    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    device = _resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    print(f"[ch10] Training PIXEL+DrQ SAC on {cfg.env}")
    print(f"[ch10] seed={cfg.seed}, n_envs={cfg.pixel_n_envs}, "
          f"total_steps={cfg.drq_total_steps}, device={device}")
    print(f"[ch10] image_size={cfg.image_size}x{cfg.image_size}, "
          f"drq_pad={cfg.drq_pad}, buffer_size={cfg.pixel_buffer_size}")
    print(f"[ch10] pixel_goal_mode={cfg.pixel_goal_mode} (none=fully pixels-only)")

    def make_pixel_env():
        env = gym.make(cfg.env, render_mode="rgb_array")
        return PixelObservationWrapper(
            env,
            image_size=(cfg.image_size, cfg.image_size),
            goal_mode=cfg.pixel_goal_mode,
        )

    env = make_vec_env(make_pixel_env, n_envs=cfg.pixel_n_envs, seed=cfg.seed)

    log_dir = _ensure_dir(cfg.log_dir)
    run_id = f"sac_drq/{cfg.env}/seed{cfg.seed}"

    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
            batch_size=cfg.batch_size,
            buffer_size=cfg.pixel_buffer_size,
            learning_starts=cfg.learning_starts,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            tau=cfg.tau,
            ent_coef="auto",
            replay_buffer_class=DrQDictReplayBuffer,
            replay_buffer_kwargs={
                "aug_fn": RandomShiftAug(pad=cfg.drq_pad),
                "image_key": "pixels",
            },
        )

        t0 = time.perf_counter()
        model.learn(total_timesteps=cfg.drq_total_steps, tb_log_name=run_id)
        elapsed = time.perf_counter() - t0
        fps = cfg.drq_total_steps / elapsed

        ckpt = _ckpt_path(cfg, _drq_mode_tag(cfg))
        model.save(str(ckpt))
        print(f"[ch10] Saved: {ckpt}")
        print(f"[ch10] Training time: {elapsed:.1f}s ({fps:.0f} steps/sec)")

    finally:
        env.close()

    # Write metadata
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 10,
        "mode": _drq_mode_tag(cfg),
        "algo": "sac",
        "env_id": cfg.env,
        "seed": cfg.seed,
        "device": device,
        "n_envs": cfg.pixel_n_envs,
        "total_steps": cfg.drq_total_steps,
        "training_time_sec": elapsed,
        "steps_per_sec": fps,
        "checkpoint": str(ckpt),
        "image_size": cfg.image_size,
        "drq_pad": cfg.drq_pad,
        "pixel_goal_mode": cfg.pixel_goal_mode,
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.pixel_buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": "auto",
            "drq_pad": cfg.drq_pad,
        },
        "versions": _gather_versions(),
    }
    _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Wrote metadata: {_meta_path(ckpt)}")

    return 0


def cmd_eval(cfg: Ch10Config, ckpt: str, pixel: bool = False, json_out: str | None = None) -> int:
    """Evaluate a checkpoint using eval.py subprocess.

    For pixel checkpoints, we need to recreate the wrapper environment.
    We do this by writing a small env registration snippet that eval.py
    can use. However, the simpler approach is to invoke eval.py with
    the correct env -- for pixel models, SB3 saves the observation space
    in the checkpoint, so it knows what to expect. We just need the
    wrapper at env creation time.

    For pixel checkpoints, we use a custom eval loop instead of subprocess
    eval.py, because eval.py doesn't know about our wrapper.
    """
    if not ckpt:
        print("[ch10] Error: --ckpt required")
        return 1

    if pixel:
        ckpt_name = Path(ckpt).name
        inferred_mode = _drq_mode_tag(cfg) if "sac_drq" in ckpt_name else _pixel_mode_tag(cfg)
        if json_out is None:
            json_out = str(_result_path(cfg, inferred_mode))
        return _eval_pixel(cfg, ckpt, json_out, mode=inferred_mode)
    else:
        return _eval_state(cfg, ckpt, json_out)


def _eval_state(cfg: Ch10Config, ckpt: str, json_out: str | None) -> int:
    """Evaluate a state-based checkpoint via eval.py subprocess."""
    if json_out is None:
        json_out = str(_result_path(cfg, "state"))

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)

    seeds_arg = f"0-{cfg.n_eval_episodes - 1}"
    cmd = [
        sys.executable, "eval.py",
        "--ckpt", ckpt,
        "--env", cfg.env,
        "--algo", "sac",
        "--device", _resolve_device(cfg.device),
        "--n-episodes", str(cfg.n_eval_episodes),
        "--seeds", seeds_arg,
        "--json-out", json_out,
    ]
    if cfg.eval_deterministic:
        cmd.append("--deterministic")

    print(f"[ch10] Evaluating state checkpoint: {ckpt}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        print(f"[ch10] State evaluation saved: {json_out}")
    return result.returncode


def _eval_pixel(cfg: Ch10Config, ckpt: str, json_out: str | None, *, mode: str) -> int:
    """Evaluate a pixel-based checkpoint.

    We can't use eval.py directly because it doesn't know about our
    PixelObservationWrapper. Instead, we run a custom eval loop that:
    1. Creates the wrapped environment
    2. Loads the SB3 model
    3. Runs episodes and collects metrics
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import numpy as np
    from stable_baselines3 import SAC

    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    if json_out is None:
        json_out = str(_result_path(cfg, _pixel_mode_tag(cfg)))

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    print(f"[ch10] Evaluating pixel checkpoint: {ckpt}")
    print(f"[ch10] device={device}, n_episodes={cfg.n_eval_episodes}")

    # Load model
    model = SAC.load(ckpt, device=device)

    # Create pixel environment
    base_env = gym.make(cfg.env, render_mode="rgb_array")
    env = PixelObservationWrapper(
        base_env,
        image_size=(cfg.image_size, cfg.image_size),
        goal_mode=cfg.pixel_goal_mode,
    )

    # Validate observation space matches the loaded model
    env_keys = set(env.observation_space.spaces.keys())
    model_keys = set(model.observation_space.spaces.keys())
    if env_keys != model_keys:
        print(f"[ch10] ERROR: Observation space mismatch.")
        print(f"[ch10]   Model expects keys: {sorted(model_keys)}")
        print(f"[ch10]   Environment provides: {sorted(env_keys)}")
        print(f"[ch10]   Check that --pixel-goal-mode matches training.")
        env.close()
        return 1

    episode_returns = []
    episode_successes = []
    episode_lengths = []
    final_distances = []

    for ep in range(cfg.n_eval_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=cfg.eval_deterministic)
            obs, reward, term, trunc, info = env.step(action)
            ep_return += reward
            ep_len += 1
            done = term or trunc

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)

        is_success = info.get("is_success", False)
        episode_successes.append(float(is_success))

        # Final distance: achieved_goal vs desired_goal.
        # In pixels-only mode these are not exposed in `obs`; we read them
        # from the wrapper's stored raw observation.
        if "achieved_goal" in obs and "desired_goal" in obs:
            achieved = obs["achieved_goal"]
            desired = obs["desired_goal"]
        else:
            assert env.last_raw_obs is not None, "PixelObservationWrapper did not store last_raw_obs"
            achieved = env.last_raw_obs["achieved_goal"]
            desired = env.last_raw_obs["desired_goal"]
        dist = np.linalg.norm(achieved - desired)
        final_distances.append(float(dist))

    env.close()

    # Compute aggregate metrics
    success_rate = float(np.mean(episode_successes))
    return_mean = float(np.mean(episode_returns))
    return_std = float(np.std(episode_returns))
    distance_mean = float(np.mean(final_distances))
    distance_std = float(np.std(final_distances))
    length_mean = float(np.mean(episode_lengths))

    print(f"[ch10] Pixel eval: success_rate={success_rate:.1%}, "
          f"return={return_mean:.3f} +/- {return_std:.3f}, "
          f"final_dist={distance_mean:.4f}")

    # Save results in same format as eval.py
    report = {
        "checkpoint": ckpt,
        "env_id": cfg.env,
        "mode": mode,
        "image_size": cfg.image_size,
        "pixel_goal_mode": cfg.pixel_goal_mode,
        "n_episodes": cfg.n_eval_episodes,
        "deterministic": cfg.eval_deterministic,
        "aggregate": {
            "success_rate": success_rate,
            "return_mean": return_mean,
            "return_std": return_std,
            "final_distance_mean": distance_mean,
            "final_distance_std": distance_std,
            "ep_len_mean": length_mean,
        },
        "per_episode": [
            {
                "seed": ep,
                "return": episode_returns[ep],
                "success": int(episode_successes[ep]),
                "final_distance": final_distances[ep],
                "length": episode_lengths[ep],
            }
            for ep in range(cfg.n_eval_episodes)
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": _gather_versions(),
    }

    Path(json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Pixel evaluation saved: {json_out}")

    return 0


def cmd_compare(cfg: Ch10Config) -> int:
    """Compare state vs pixel vs pixel+DrQ evaluation results.

    Prints a 3-column table when DrQ results are available, otherwise
    falls back to 2-column (state vs pixel) comparison.
    """
    state_path = _result_path(cfg, "state")
    pixel_path = _result_path(cfg, _pixel_mode_tag(cfg))
    drq_path = _result_path(cfg, _drq_mode_tag(cfg))

    if not state_path.exists():
        print(f"[ch10] State results not found: {state_path}")
        print("[ch10] Run: python scripts/ch10_visual_reach.py train-state && eval")
        return 1

    if not pixel_path.exists():
        print(f"[ch10] Pixel results not found: {pixel_path}")
        print("[ch10] Run: python scripts/ch10_visual_reach.py train-pixel && eval")
        return 1

    with open(state_path) as f:
        state_data = json.load(f)
    with open(pixel_path) as f:
        pixel_data = json.load(f)

    has_drq = drq_path.exists()
    drq_data = None
    if has_drq:
        with open(drq_path) as f:
            drq_data = json.load(f)

    s = state_data["aggregate"]
    p = pixel_data["aggregate"]
    d = drq_data["aggregate"] if drq_data else None

    # Load metadata for training steps
    state_ckpt = _ckpt_path(cfg, "state")
    pixel_ckpt = _ckpt_path(cfg, _pixel_mode_tag(cfg))
    drq_ckpt = _ckpt_path(cfg, _drq_mode_tag(cfg))
    state_steps = cfg.state_total_steps
    pixel_steps = cfg.pixel_total_steps
    drq_steps = cfg.drq_total_steps

    if _meta_path(state_ckpt).exists():
        with open(_meta_path(state_ckpt)) as f:
            state_meta = json.load(f)
            state_steps = state_meta.get("total_steps", state_steps)

    if _meta_path(pixel_ckpt).exists():
        with open(_meta_path(pixel_ckpt)) as f:
            pixel_meta = json.load(f)
            pixel_steps = pixel_meta.get("total_steps", pixel_steps)

    if has_drq and _meta_path(drq_ckpt).exists():
        with open(_meta_path(drq_ckpt)) as f:
            drq_meta = json.load(f)
            drq_steps = drq_meta.get("total_steps", drq_steps)

    print()
    print("=" * 92)
    if has_drq:
        print("Chapter 10: State vs Pixel vs Pixel+DrQ Comparison")
    else:
        print("Chapter 10: State vs Pixel Observation Comparison")
    print(f"Environment: {cfg.env}")
    print("=" * 92)

    if has_drq and d is not None:
        # 3-column table
        print(f"\n{'Metric':<25} | {'State':>20} | {'Pixel':>20} | {'Pixel+DrQ':>20}")
        print("-" * 92)
        print(f"{'Training steps':<25} | {state_steps:>20,} | {pixel_steps:>20,} | {drq_steps:>20,}")
        print(f"{'Success rate':<25} | {s['success_rate']:>19.1%} | {p['success_rate']:>19.1%} | {d['success_rate']:>19.1%}")
        print(f"{'Return (mean)':<25} | {s['return_mean']:>20.3f} | {p['return_mean']:>20.3f} | {d['return_mean']:>20.3f}")
        print(f"{'Return (std)':<25} | {s['return_std']:>20.3f} | {p['return_std']:>20.3f} | {d['return_std']:>20.3f}")
        print(f"{'Final distance (mean)':<25} | {s['final_distance_mean']:>20.4f} | {p['final_distance_mean']:>20.4f} | {d['final_distance_mean']:>20.4f}")
        print(f"{'Episode length (mean)':<25} | {s.get('ep_len_mean', 0):>20.1f} | {p.get('ep_len_mean', 0):>20.1f} | {d.get('ep_len_mean', 0):>20.1f}")
        print("-" * 92)

        # Sample efficiency ratios
        print()
        if s['success_rate'] > 0:
            if p['success_rate'] > 0:
                pixel_ratio = pixel_steps / state_steps
                print(f"Sample efficiency: pixel needs {pixel_ratio:.1f}x more steps than state")
            else:
                pixel_ratio = float("inf")
                print("Sample efficiency: pixel did not succeed")

            if d['success_rate'] > 0:
                drq_ratio = drq_steps / state_steps
                print(f"Sample efficiency: pixel+DrQ needs {drq_ratio:.1f}x more steps than state")
            else:
                drq_ratio = float("inf")
                print("Sample efficiency: pixel+DrQ did not succeed")

            # Gap closure
            if p['success_rate'] > 0 and d['success_rate'] > 0:
                state_sr = s['success_rate']
                pixel_sr = p['success_rate']
                drq_sr = d['success_rate']
                gap = state_sr - pixel_sr
                if gap > 0.01:
                    closure = (drq_sr - pixel_sr) / gap
                    print(f"\nDrQ closes {closure:.0%} of the state-vs-pixel success rate gap")
                    print(f"  State: {state_sr:.1%} -> Pixel: {pixel_sr:.1%} -> DrQ: {drq_sr:.1%}")
        else:
            print("Sample efficiency: state agent did not succeed")

        # Summary
        print()
        drq_sr = d['success_rate']
        pixel_sr = p['success_rate']
        state_sr = s['success_rate']
        if state_sr > 0.8 and drq_sr > pixel_sr:
            print("RESULT: DrQ augmentation improves pixel-based learning.")
            print("This demonstrates Q-function regularization via data augmentation")
            print("(Kostrikov et al. 2020).")
        elif state_sr > 0.8 and drq_sr <= pixel_sr:
            print("RESULT: DrQ did not improve over naive pixel SAC.")
            print("This may indicate insufficient training or task-specific effects.")
        else:
            print("RESULT: Check training duration and hyperparameters.")

        # Save 3-way comparison report
        report = {
            "env_id": cfg.env,
            "seed": cfg.seed,
            "state": {"total_steps": state_steps, **s},
            "pixel": {"total_steps": pixel_steps, "image_size": cfg.image_size, "pixel_goal_mode": cfg.pixel_goal_mode, **p},
            "drq": {
                "total_steps": drq_steps,
                "image_size": cfg.image_size,
                "drq_pad": cfg.drq_pad,
                "pixel_goal_mode": cfg.pixel_goal_mode,
                **d,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    else:
        # 2-column table (no DrQ results)
        print(f"\n{'Metric':<25} | {'State':>20} | {'Pixel':>20}")
        print("-" * 70)
        print(f"{'Training steps':<25} | {state_steps:>20,} | {pixel_steps:>20,}")
        print(f"{'Success rate':<25} | {s['success_rate']:>19.1%} | {p['success_rate']:>19.1%}")
        print(f"{'Return (mean +/- std)':<25} | {s['return_mean']:>8.3f} +/- {s['return_std']:<8.3f} | {p['return_mean']:>8.3f} +/- {p['return_std']:<8.3f}")
        print(f"{'Final distance (mean)':<25} | {s['final_distance_mean']:>20.4f} | {p['final_distance_mean']:>20.4f}")
        print(f"{'Episode length (mean)':<25} | {s.get('ep_len_mean', 0):>20.1f} | {p.get('ep_len_mean', 0):>20.1f}")
        print("-" * 70)

        if p['success_rate'] > 0 and s['success_rate'] > 0:
            ratio = pixel_steps / state_steps
            print(f"\nSample efficiency ratio: {ratio:.1f}x (pixel needs {ratio:.1f}x more steps)")
        else:
            ratio = float("inf")
            print("\nSample efficiency ratio: N/A (one or both agents did not succeed)")

        print()
        if s["success_rate"] > 0.8 and p["success_rate"] > 0.5:
            print("RESULT: Both agents solve the task. Pixel SAC requires ~{:.0f}x more samples.".format(ratio))
            print("This demonstrates the cost of learning from raw pixels vs privileged state.")
        elif s["success_rate"] > 0.8 and p["success_rate"] <= 0.5:
            print("RESULT: State agent solves the task; pixel agent still learning.")
            print("Pixel SAC may need more training steps or hyperparameter tuning.")
        else:
            print("RESULT: Neither agent fully converged. Check training duration and hyperparameters.")

        if not has_drq:
            print("\n[ch10] Note: DrQ results not found. Run train-pixel-drq + eval for 3-way comparison.")

        report = {
            "env_id": cfg.env,
            "seed": cfg.seed,
            "state": {"total_steps": state_steps, **s},
            "pixel": {"total_steps": pixel_steps, "image_size": cfg.image_size, "pixel_goal_mode": cfg.pixel_goal_mode, **p},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    comp_path = _comparison_path(cfg)
    comp_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch10] Comparison report saved: {comp_path}")

    return 0


def cmd_all(cfg: Ch10Config) -> int:
    """Full pipeline: state + pixel + pixel+DrQ training, eval, comparison."""
    print("=" * 72)
    print("Chapter 10: Full Pipeline -- State vs Pixel vs Pixel+DrQ SAC")
    print(f"Environment: {cfg.env}, seed={cfg.seed}")
    print("=" * 72)

    # 1. Train state baseline
    print("\n[1/7] Training state-based SAC...")
    ret = cmd_train_state(cfg)
    if ret != 0:
        print("[ch10] State training failed")
        return ret

    # 2. Evaluate state
    print("\n[2/7] Evaluating state checkpoint...")
    state_ckpt = str(_ckpt_path(cfg, "state"))
    ret = cmd_eval(cfg, state_ckpt, pixel=False)
    if ret != 0:
        print("[ch10] State evaluation failed")
        return ret

    # 3. Train pixel
    print("\n[3/7] Training pixel-based SAC...")
    ret = cmd_train_pixel(cfg)
    if ret != 0:
        print("[ch10] Pixel training failed")
        return ret

    # 4. Evaluate pixel
    print("\n[4/7] Evaluating pixel checkpoint...")
    pixel_ckpt = str(_ckpt_path(cfg, _pixel_mode_tag(cfg)))
    ret = cmd_eval(cfg, pixel_ckpt, pixel=True)
    if ret != 0:
        print("[ch10] Pixel evaluation failed")
        return ret

    # 5. Train pixel + DrQ
    print("\n[5/7] Training pixel+DrQ SAC...")
    ret = cmd_train_pixel_drq(cfg)
    if ret != 0:
        print("[ch10] DrQ training failed")
        return ret

    # 6. Evaluate pixel + DrQ
    print("\n[6/7] Evaluating pixel+DrQ checkpoint...")
    drq_ckpt = str(_ckpt_path(cfg, _drq_mode_tag(cfg)))
    ret = cmd_eval(cfg, drq_ckpt, pixel=True, json_out=str(_result_path(cfg, _drq_mode_tag(cfg))))
    if ret != 0:
        print("[ch10] DrQ evaluation failed")
        return ret

    # 7. Compare (3-way)
    print("\n[7/7] Comparing results...")
    return cmd_compare(cfg)


# =============================================================================
# CLI
# =============================================================================

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by multiple subcommands."""
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed,
                        help="Random seed")
    parser.add_argument("--device", default=DEFAULT_CONFIG.device,
                        help="Training device: auto, cpu, cuda")
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CONFIG.checkpoints_dir,
                        help="Directory for checkpoint outputs")
    parser.add_argument("--results-dir", default=DEFAULT_CONFIG.results_dir,
                        help="Directory for evaluation outputs")
    parser.add_argument("--log-dir", default=DEFAULT_CONFIG.log_dir,
                        help="TensorBoard log directory")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add training-specific arguments."""
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size,
                        help="Batch size for SAC updates")
    parser.add_argument("--learning-starts", type=int, default=DEFAULT_CONFIG.learning_starts,
                        help="Steps before learning starts")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG.gamma,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=DEFAULT_CONFIG.tau,
                        help="Soft update coefficient")


def _config_from_args(args: argparse.Namespace) -> Ch10Config:
    """Build Ch10Config from parsed args."""
    cfg = Ch10Config()
    for attr in vars(cfg):
        arg_name = attr.replace("-", "_")
        if hasattr(args, arg_name):
            setattr(cfg, attr, getattr(args, arg_name))
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter 10: Pixels, No Cheating -- Visual SAC on FetchReachDense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (state + pixel + pixel+DrQ + comparison)
  python scripts/ch10_visual_reach.py all --seed 0

  # State baseline only
  python scripts/ch10_visual_reach.py train-state --seed 0 --state-total-steps 500000

  # Pixel training only
  python scripts/ch10_visual_reach.py train-pixel --seed 0 --pixel-total-steps 2000000

  # Pixel + DrQ augmentation
  python scripts/ch10_visual_reach.py train-pixel-drq --seed 0 --drq-total-steps 2000000

  # Evaluate checkpoints
  python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_state_FetchReachDense-v4_seed0.zip
  python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_pixel_FetchReachDense-v4_seed0.zip --pixel
  python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_drq_FetchReachDense-v4_seed0.zip --pixel

  # Compare results (3-way if DrQ eval exists)
  python scripts/ch10_visual_reach.py compare
""",
    )
    parser.add_argument("--mujoco-gl", default=None,
                        help="Override MUJOCO_GL (default: 'egl' for pixel, 'disable' for state)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train-state
    p_state = sub.add_parser("train-state", help="Train SAC with state observations",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_state)
    _add_train_args(p_state)
    p_state.add_argument("--state-n-envs", type=int, default=DEFAULT_CONFIG.state_n_envs,
                          help="Number of parallel envs for state training")
    p_state.add_argument("--state-total-steps", type=int, default=DEFAULT_CONFIG.state_total_steps,
                          help="Total training steps for state")
    p_state.add_argument("--state-buffer-size", type=int, default=DEFAULT_CONFIG.state_buffer_size,
                          help="Replay buffer size for state")

    # train-pixel
    p_pixel = sub.add_parser("train-pixel", help="Train SAC with pixel observations",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_pixel)
    _add_train_args(p_pixel)
    p_pixel.add_argument("--pixel-n-envs", type=int, default=DEFAULT_CONFIG.pixel_n_envs,
                          help="Number of parallel envs for pixel training")
    p_pixel.add_argument("--pixel-total-steps", type=int, default=DEFAULT_CONFIG.pixel_total_steps,
                          help="Total training steps for pixel")
    p_pixel.add_argument("--pixel-buffer-size", type=int, default=DEFAULT_CONFIG.pixel_buffer_size,
                          help="Replay buffer size for pixel (memory-conscious)")
    p_pixel.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size,
                          help="Image size for pixel observations")
    p_pixel.add_argument(
        "--pixel-goal-mode",
        choices=["none", "desired", "both"],
        default=DEFAULT_CONFIG.pixel_goal_mode,
        help="Which goal vectors (if any) are included in the observation dict",
    )

    # train-pixel-drq
    p_drq = sub.add_parser("train-pixel-drq",
                            help="Train SAC with pixel observations + DrQ augmentation",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_drq)
    _add_train_args(p_drq)
    p_drq.add_argument("--pixel-n-envs", type=int, default=DEFAULT_CONFIG.pixel_n_envs,
                        help="Number of parallel envs for pixel training")
    p_drq.add_argument("--drq-total-steps", type=int, default=DEFAULT_CONFIG.drq_total_steps,
                        help="Total training steps for pixel+DrQ")
    p_drq.add_argument("--pixel-buffer-size", type=int, default=DEFAULT_CONFIG.pixel_buffer_size,
                        help="Replay buffer size for pixel (memory-conscious)")
    p_drq.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size,
                        help="Image size for pixel observations")
    p_drq.add_argument("--drq-pad", type=int, default=DEFAULT_CONFIG.drq_pad,
                        help="DrQ augmentation pad size (pixels)")
    p_drq.add_argument(
        "--pixel-goal-mode",
        choices=["none", "desired", "both"],
        default=DEFAULT_CONFIG.pixel_goal_mode,
        help="Which goal vectors (if any) are included in the observation dict",
    )

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_eval)
    p_eval.add_argument("--ckpt", required=True, help="Checkpoint path")
    p_eval.add_argument("--pixel", action="store_true",
                         help="Use pixel wrapper for evaluation (required for pixel checkpoints)")
    p_eval.add_argument("--json-out", default=None, help="Output JSON path")
    p_eval.add_argument("--n-eval-episodes", type=int, default=DEFAULT_CONFIG.n_eval_episodes,
                         help="Number of evaluation episodes")
    p_eval.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size,
                         help="Image size (must match training)")
    p_eval.add_argument(
        "--pixel-goal-mode",
        choices=["none", "desired", "both"],
        default=DEFAULT_CONFIG.pixel_goal_mode,
        help="Pixel observation design (must match training for pixel checkpoints)",
    )

    # compare
    p_cmp = sub.add_parser("compare", help="Compare state vs pixel vs pixel+DrQ results",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_cmp)
    p_cmp.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size,
                        help="Image size used in pixel training")
    p_cmp.add_argument("--state-total-steps", type=int, default=DEFAULT_CONFIG.state_total_steps,
                        help="State training steps (for ratio calculation)")
    p_cmp.add_argument("--pixel-total-steps", type=int, default=DEFAULT_CONFIG.pixel_total_steps,
                        help="Pixel training steps (for ratio calculation)")
    p_cmp.add_argument("--drq-total-steps", type=int, default=DEFAULT_CONFIG.drq_total_steps,
                        help="DrQ training steps (for ratio calculation)")
    p_cmp.add_argument("--drq-pad", type=int, default=DEFAULT_CONFIG.drq_pad,
                        help="DrQ augmentation pad size (for reporting)")
    p_cmp.add_argument(
        "--pixel-goal-mode",
        choices=["none", "desired", "both"],
        default=DEFAULT_CONFIG.pixel_goal_mode,
        help="Which goal vectors (if any) were included during pixel training",
    )

    # all
    p_all = sub.add_parser("all",
                            help="Full pipeline: train-state, train-pixel, train-pixel-drq, eval, compare",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_all)
    _add_train_args(p_all)
    p_all.add_argument("--state-n-envs", type=int, default=DEFAULT_CONFIG.state_n_envs)
    p_all.add_argument("--state-total-steps", type=int, default=DEFAULT_CONFIG.state_total_steps)
    p_all.add_argument("--state-buffer-size", type=int, default=DEFAULT_CONFIG.state_buffer_size)
    p_all.add_argument("--pixel-n-envs", type=int, default=DEFAULT_CONFIG.pixel_n_envs)
    p_all.add_argument("--pixel-total-steps", type=int, default=DEFAULT_CONFIG.pixel_total_steps)
    p_all.add_argument("--pixel-buffer-size", type=int, default=DEFAULT_CONFIG.pixel_buffer_size)
    p_all.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size)
    p_all.add_argument("--drq-total-steps", type=int, default=DEFAULT_CONFIG.drq_total_steps)
    p_all.add_argument("--drq-pad", type=int, default=DEFAULT_CONFIG.drq_pad)
    p_all.add_argument("--n-eval-episodes", type=int, default=DEFAULT_CONFIG.n_eval_episodes)
    p_all.add_argument(
        "--pixel-goal-mode",
        choices=["none", "desired", "both"],
        default=DEFAULT_CONFIG.pixel_goal_mode,
        help="Which goal vectors (if any) are included in the pixel observation dict",
    )

    args = parser.parse_args()

    # Set MUJOCO_GL
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    elif args.cmd in ("train-pixel", "train-pixel-drq", "all"):
        # Pixel training needs rendering
        os.environ.setdefault("MUJOCO_GL", "egl")
    elif args.cmd == "eval" and getattr(args, "pixel", False):
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("MUJOCO_GL", "disable")

    cfg = _config_from_args(args)

    if args.cmd == "train-state":
        return cmd_train_state(cfg)
    elif args.cmd == "train-pixel":
        return cmd_train_pixel(cfg)
    elif args.cmd == "train-pixel-drq":
        return cmd_train_pixel_drq(cfg)
    elif args.cmd == "eval":
        return cmd_eval(cfg, args.ckpt, pixel=args.pixel, json_out=args.json_out)
    elif args.cmd == "compare":
        return cmd_compare(cfg)
    elif args.cmd == "all":
        return cmd_all(cfg)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
