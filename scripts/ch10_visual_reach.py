#!/usr/bin/env python3
"""Chapter 10: Pixels, No Cheating -- Visual SAC on FetchReachDense

Week 10 goals:
1. Demonstrate state vs pixel observation trade-off on the same task
2. Train SAC on FetchReachDense with privileged state (baseline from Ch3)
3. Train SAC on FetchReachDense with pixel observations (CNN encoder)
4. Measure the sample-efficiency gap quantitatively
5. Prove that pixel-based agents can solve the task (just slower)

Usage:
    # Full pipeline: state baseline + pixel training + comparison
    python scripts/ch10_visual_reach.py all --seed 0

    # Train state-based SAC (baseline)
    python scripts/ch10_visual_reach.py train-state --seed 0

    # Train pixel-based SAC
    python scripts/ch10_visual_reach.py train-pixel --seed 0

    # Evaluate a checkpoint (use --pixel for pixel checkpoints)
    python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_state_FetchReachDense-v4_seed0.zip
    python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_pixel_FetchReachDense-v4_seed0.zip --pixel

    # Compare state vs pixel results
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
    state vector with rendered 84x84 RGB images. SB3's MultiInputPolicy
    auto-detects the image space and uses CombinedExtractor (NatureCNN
    for pixels, MLP for goal vectors).
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

    def make_pixel_env():
        env = gym.make(cfg.env, render_mode="rgb_array")
        return PixelObservationWrapper(env, image_size=(cfg.image_size, cfg.image_size))

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

        ckpt = _ckpt_path(cfg, "pixel")
        model.save(str(ckpt))
        print(f"[ch10] Saved: {ckpt}")
        print(f"[ch10] Training time: {elapsed:.1f}s ({fps:.0f} steps/sec)")

    finally:
        env.close()

    # Write metadata
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 10,
        "mode": "pixel",
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
        return _eval_pixel(cfg, ckpt, json_out)
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


def _eval_pixel(cfg: Ch10Config, ckpt: str, json_out: str | None) -> int:
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
        json_out = str(_result_path(cfg, "pixel"))

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    print(f"[ch10] Evaluating pixel checkpoint: {ckpt}")
    print(f"[ch10] device={device}, n_episodes={cfg.n_eval_episodes}")

    # Load model
    model = SAC.load(ckpt, device=device)

    # Create pixel environment
    env = gym.make(cfg.env, render_mode="rgb_array")
    env = PixelObservationWrapper(env, image_size=(cfg.image_size, cfg.image_size))

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

        # Final distance: achieved_goal vs desired_goal
        dist = np.linalg.norm(
            obs["achieved_goal"] - obs["desired_goal"]
        )
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
        "mode": "pixel",
        "image_size": cfg.image_size,
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
        "per_episode": {
            "returns": episode_returns,
            "successes": episode_successes,
            "final_distances": final_distances,
            "lengths": episode_lengths,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": _gather_versions(),
    }

    Path(json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Pixel evaluation saved: {json_out}")

    return 0


def cmd_compare(cfg: Ch10Config) -> int:
    """Compare state vs pixel evaluation results."""
    import math

    state_path = _result_path(cfg, "state")
    pixel_path = _result_path(cfg, "pixel")

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

    s = state_data["aggregate"]
    p = pixel_data["aggregate"]

    # Load metadata for training steps
    state_ckpt = _ckpt_path(cfg, "state")
    pixel_ckpt = _ckpt_path(cfg, "pixel")
    state_steps = cfg.state_total_steps
    pixel_steps = cfg.pixel_total_steps

    if _meta_path(state_ckpt).exists():
        with open(_meta_path(state_ckpt)) as f:
            state_meta = json.load(f)
            state_steps = state_meta.get("total_steps", state_steps)

    if _meta_path(pixel_ckpt).exists():
        with open(_meta_path(pixel_ckpt)) as f:
            pixel_meta = json.load(f)
            pixel_steps = pixel_meta.get("total_steps", pixel_steps)

    print()
    print("=" * 72)
    print("Chapter 10: State vs Pixel Observation Comparison")
    print(f"Environment: {cfg.env}")
    print("=" * 72)

    print(f"\n{'Metric':<25} | {'State':>20} | {'Pixel':>20}")
    print("-" * 70)
    print(f"{'Training steps':<25} | {state_steps:>20,} | {pixel_steps:>20,}")
    print(f"{'Success rate':<25} | {s['success_rate']:>19.1%} | {p['success_rate']:>19.1%}")
    print(f"{'Return (mean +/- std)':<25} | {s['return_mean']:>8.3f} +/- {s['return_std']:<8.3f} | {p['return_mean']:>8.3f} +/- {p['return_std']:<8.3f}")
    print(f"{'Final distance (mean)':<25} | {s['final_distance_mean']:>20.4f} | {p['final_distance_mean']:>20.4f}")
    print(f"{'Episode length (mean)':<25} | {s.get('ep_len_mean', 0):>20.1f} | {p.get('ep_len_mean', 0):>20.1f}")
    print("-" * 70)

    # Sample efficiency ratio
    if p['success_rate'] > 0 and s['success_rate'] > 0:
        ratio = pixel_steps / state_steps
        print(f"\nSample efficiency ratio: {ratio:.1f}x (pixel needs {ratio:.1f}x more steps)")
    else:
        ratio = float("inf")
        print("\nSample efficiency ratio: N/A (one or both agents did not succeed)")

    # Summary
    print()
    if s["success_rate"] > 0.8 and p["success_rate"] > 0.5:
        print("RESULT: Both agents solve the task. Pixel SAC requires ~{:.0f}x more samples.".format(ratio))
        print("This demonstrates the cost of learning from raw pixels vs privileged state.")
    elif s["success_rate"] > 0.8 and p["success_rate"] <= 0.5:
        print("RESULT: State agent solves the task; pixel agent still learning.")
        print("Pixel SAC may need more training steps or hyperparameter tuning.")
    else:
        print("RESULT: Neither agent fully converged. Check training duration and hyperparameters.")

    # Save comparison report
    report = {
        "env_id": cfg.env,
        "seed": cfg.seed,
        "state": {
            "total_steps": state_steps,
            **s,
        },
        "pixel": {
            "total_steps": pixel_steps,
            "image_size": cfg.image_size,
            **p,
        },
        "sample_efficiency_ratio": ratio if math.isfinite(ratio) else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    comp_path = _comparison_path(cfg)
    comp_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch10] Comparison report saved: {comp_path}")

    return 0


def cmd_all(cfg: Ch10Config) -> int:
    """Full pipeline: state baseline + pixel training + eval + comparison."""
    print("=" * 72)
    print("Chapter 10: Full Pipeline -- State vs Pixel SAC")
    print(f"Environment: {cfg.env}, seed={cfg.seed}")
    print("=" * 72)

    # 1. Train state baseline
    print("\n[1/5] Training state-based SAC...")
    ret = cmd_train_state(cfg)
    if ret != 0:
        print("[ch10] State training failed")
        return ret

    # 2. Evaluate state
    print("\n[2/5] Evaluating state checkpoint...")
    state_ckpt = str(_ckpt_path(cfg, "state"))
    ret = cmd_eval(cfg, state_ckpt, pixel=False)
    if ret != 0:
        print("[ch10] State evaluation failed")
        return ret

    # 3. Train pixel
    print("\n[3/5] Training pixel-based SAC...")
    ret = cmd_train_pixel(cfg)
    if ret != 0:
        print("[ch10] Pixel training failed")
        return ret

    # 4. Evaluate pixel
    print("\n[4/5] Evaluating pixel checkpoint...")
    pixel_ckpt = str(_ckpt_path(cfg, "pixel"))
    ret = cmd_eval(cfg, pixel_ckpt, pixel=True)
    if ret != 0:
        print("[ch10] Pixel evaluation failed")
        return ret

    # 5. Compare
    print("\n[5/5] Comparing results...")
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
  # Full pipeline
  python scripts/ch10_visual_reach.py all --seed 0

  # State baseline only
  python scripts/ch10_visual_reach.py train-state --seed 0 --state-total-steps 500000

  # Pixel training only
  python scripts/ch10_visual_reach.py train-pixel --seed 0 --pixel-total-steps 2000000

  # Evaluate checkpoints
  python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_state_FetchReachDense-v4_seed0.zip
  python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_pixel_FetchReachDense-v4_seed0.zip --pixel

  # Compare results
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

    # compare
    p_cmp = sub.add_parser("compare", help="Compare state vs pixel results",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_cmp)
    p_cmp.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size,
                        help="Image size used in pixel training")
    p_cmp.add_argument("--state-total-steps", type=int, default=DEFAULT_CONFIG.state_total_steps,
                        help="State training steps (for ratio calculation)")
    p_cmp.add_argument("--pixel-total-steps", type=int, default=DEFAULT_CONFIG.pixel_total_steps,
                        help="Pixel training steps (for ratio calculation)")

    # all
    p_all = sub.add_parser("all", help="Full pipeline: train-state, eval, train-pixel, eval, compare",
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
    p_all.add_argument("--n-eval-episodes", type=int, default=DEFAULT_CONFIG.n_eval_episodes)

    args = parser.parse_args()

    # Set MUJOCO_GL
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    elif args.cmd in ("train-pixel", "all"):
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
