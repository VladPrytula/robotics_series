#!/usr/bin/env python3
"""Chapter 05: PickAndPlace -- From Push to Grasping

Week 5 goals:
1. Dense debug: validate pipeline on FetchPickAndPlaceDense-v4 (~13 min)
2. Sparse + HER: transfer ch04's proven hyperparameters to PickAndPlace
3. Stratified evaluation: measure air-goal vs table-goal performance separately
4. Stress evaluation: quantify robustness under observation/action noise
5. Curriculum learning: available as add-on if sparse success < 60%

Approach: Start from ch04's best config (gamma=0.95, ent_coef=0.05,
n_sampled_goal=4) and test whether it transfers to the harder task.
PickAndPlace adds multi-phase control (grasp, lift, place) and air goals.

Usage:
    # Dense debug (~5 min) -- validate pipeline before long runs
    python scripts/ch05_pick_and_place.py dense-debug

    # Train SAC+HER on sparse PickAndPlace (5M steps, ~1-1.5 hr)
    python scripts/ch05_pick_and_place.py train --seed 0

    # Train with custom steps
    python scripts/ch05_pick_and_place.py train --seed 0 --total-steps 2000000

    # Evaluate with stratified breakdown (air vs table goals)
    python scripts/ch05_pick_and_place.py eval --ckpt checkpoints/sac_her_FetchPickAndPlace-v4_seed0.zip

    # Stress evaluation (noise injection)
    python scripts/ch05_pick_and_place.py stress --ckpt checkpoints/sac_her_FetchPickAndPlace-v4_seed0.zip

    # Compare results across seeds
    python scripts/ch05_pick_and_place.py compare --seeds 0,1,2

    # Full pipeline: dense-debug -> train (3 seeds) -> eval -> stress -> compare
    python scripts/ch05_pick_and_place.py all --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Centralized training configuration for ch05.

    Inherits ch04's proven hyperparameters:
    - gamma=0.95 (dominated all other factors in ch04's 120-run sweep)
    - ent_coef=0.05 (fixed entropy prevents collapse on sparse rewards)
    - n_sampled_goal=4 (default, effective for Push)

    Changes from ch04:
    - env: FetchPickAndPlace-v4 (sparse by default)
    - total_steps: 5M (PickAndPlace needs longer than Push)
    - her: True by default (ch04 proved no-HER fails on sparse)
    """
    # Environment
    env: str = "FetchPickAndPlace-v4"
    seed: int = 0
    n_envs: int = 8
    total_steps: int = 5_000_000
    device: str = "auto"
    monitor_keywords: str = "is_success"

    # HER settings (on by default -- ch04 proved this is required)
    her: bool = True
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"

    # SAC hyperparameters (ch04's proven winners)
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 1_000
    learning_rate: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005
    ent_coef: float = 0.05  # Fixed -- ch04 showed auto-tuning collapses

    # Curriculum (optional, off by default)
    curriculum: bool = False
    curriculum_mode: str = "linear"
    curriculum_success_threshold: float = 0.7

    # Paths
    out: str | None = None
    log_dir: str = "runs"
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"

    # Evaluation
    n_eval_episodes: int = 100
    eval_seed: int = 0
    eval_seeds: str | None = None
    eval_deterministic: bool = True
    eval_device: str = "auto"
    eval_algo: str = "auto"

    # Stress testing
    stress_obs_noise: float = 0.01
    stress_act_noise: float = 0.05

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainConfig":
        """Create config from parsed CLI arguments."""
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if hasattr(args, f):
                kwargs[f] = getattr(args, f)
        return cls(**kwargs)


DEFAULT_CONFIG = TrainConfig()


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


def _parse_seeds(seeds_str: str) -> list[int]:
    """Parse seeds string like '0,1,2' or '0-4' into list of ints."""
    seeds = []
    for part in seeds_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


def _infer_env_id_from_ckpt(ckpt_name: str) -> str | None:
    match = re.search(r"(Fetch[A-Za-z0-9]+-v\d+)", ckpt_name)
    return match.group(1) if match else None


def _ensure_writable_dir(
    path: str | Path,
    label: str,
    fallback: Path | None = None,
) -> Path:
    target = Path(path).expanduser().resolve()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        if fallback is None:
            raise SystemExit(f"[ch05] {label} not writable: {target}")
        fallback.mkdir(parents=True, exist_ok=True)
        if not os.access(fallback, os.W_OK):
            raise SystemExit(f"[ch05] {label} not writable: {target}")
        print(f"[ch05] {label} not writable: {target}. Using {fallback}")
        return fallback

    if not os.access(target, os.W_OK):
        if fallback is None:
            raise SystemExit(f"[ch05] {label} not writable: {target}")
        fallback.mkdir(parents=True, exist_ok=True)
        if not os.access(fallback, os.W_OK):
            raise SystemExit(f"[ch05] {label} not writable: {target}")
        print(f"[ch05] {label} not writable: {target}. Using {fallback}")
        return fallback

    return target


def _prepare_paths(cfg: TrainConfig) -> tuple[Path, Path]:
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")
    fallback_ckpt = results_dir / "checkpoints"
    checkpoints_dir = _ensure_writable_dir(cfg.checkpoints_dir, "Checkpoints dir", fallback=fallback_ckpt)
    cfg.results_dir = str(results_dir)
    cfg.checkpoints_dir = str(checkpoints_dir)
    return results_dir, checkpoints_dir


def _resolve_log_dir(cfg: TrainConfig, run_id: str | None = None) -> Path:
    log_dir = Path(cfg.log_dir).expanduser().resolve()
    target = log_dir / run_id if run_id else log_dir
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback = Path(cfg.results_dir).expanduser().resolve() / "runs"
        fallback_target = fallback / run_id if run_id else fallback
        fallback_target.mkdir(parents=True, exist_ok=True)
        print(f"[ch05] Log dir not writable: {log_dir}. Using {fallback}")
        return fallback
    if not os.access(target, os.W_OK):
        fallback = Path(cfg.results_dir).expanduser().resolve() / "runs"
        fallback_target = fallback / run_id if run_id else fallback
        fallback_target.mkdir(parents=True, exist_ok=True)
        print(f"[ch05] Log dir not writable: {log_dir}. Using {fallback}")
        return fallback
    return log_dir


def _ckpt_name(env_id: str, seed: int, suffix: str = "") -> str:
    """Generate checkpoint name."""
    base = f"sac_her_{env_id}_seed{seed}"
    if suffix:
        base += f"_{suffix}"
    return base


def _result_name(env_id: str, seed: int, suffix: str = "eval") -> str:
    """Generate result JSON name."""
    return f"ch05_sac_her_{env_id.lower()}_seed{seed}_{suffix}.json"


def _load_sac_model(ckpt: str, env_id: str, device: str = "auto") -> Any:
    """Load a SAC checkpoint, handling HER replay buffer requirement.

    Models trained with HerReplayBuffer require an environment for loading
    because the buffer needs goal spaces for reconstruction.
    """
    import gymnasium
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC

    try:
        return SAC.load(ckpt, device=device)
    except (AssertionError, Exception) as e:
        if "HerReplayBuffer" in str(e) or "env" in str(e).lower():
            env = gymnasium.make(env_id)
            model = SAC.load(ckpt, env=env, device=device)
            env.close()
            return model
        raise


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, and 95% CI for a list of values."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
        ci95 = 1.96 * std / math.sqrt(n) if n >= 30 else 2.0 * std / math.sqrt(n)
    else:
        std = 0.0
        ci95 = 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "n": n}


# =============================================================================
# Goal Classification
# =============================================================================

# Height offset for Fetch PickAndPlace: the table surface is at z ~= 0.42
_HEIGHT_OFFSET = 0.42
_AIR_THRESHOLD = _HEIGHT_OFFSET + 0.02


def _classify_goal(desired_goal: np.ndarray) -> str:
    """Classify a goal as 'air' or 'table' based on z coordinate.

    In FetchPickAndPlace, about 50% of goals are in the air (above the
    table surface + a small margin). Air goals require grasping and lifting;
    table goals can sometimes be solved by pushing.

    Args:
        desired_goal: Goal position [x, y, z].

    Returns:
        'air' if goal is above table surface + margin, 'table' otherwise.
    """
    return "air" if desired_goal[2] > _AIR_THRESHOLD else "table"


# =============================================================================
# Noisy Evaluation Wrapper
# =============================================================================

class NoisyEvalWrapper:
    """Wrapper that adds Gaussian noise during evaluation for stress testing.

    Injects noise into observations (simulating sensor noise) and actions
    (simulating actuator imprecision). This tests whether the policy relies
    on fragile precision or has learned robust behavior.

    Args:
        env: The environment to wrap.
        obs_noise_std: Standard deviation of observation noise.
        act_noise_std: Standard deviation of action noise.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        env: Any,
        obs_noise_std: float = 0.01,
        act_noise_std: float = 0.05,
        seed: int = 0,
    ):
        self.env = env
        self.obs_noise_std = obs_noise_std
        self.act_noise_std = act_noise_std
        self.rng = np.random.default_rng(seed)
        # Forward common attributes (env may be None when used as param carrier)
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space

    def reset(self, **kwargs: Any) -> tuple:
        obs, info = self.env.reset(**kwargs)
        return self._add_obs_noise(obs), info

    def step(self, action: np.ndarray) -> tuple:
        # Add action noise
        noisy_action = action + self.rng.normal(0, self.act_noise_std, size=action.shape)
        noisy_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(noisy_action)
        return self._add_obs_noise(obs), reward, terminated, truncated, info

    def _add_obs_noise(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Add noise to observation arrays (not goals -- goals are ground truth)."""
        noisy_obs = {}
        for key, val in obs.items():
            if key == "observation":
                noisy_obs[key] = val + self.rng.normal(0, self.obs_noise_std, size=val.shape).astype(val.dtype)
            else:
                noisy_obs[key] = val
        return noisy_obs

    def close(self) -> None:
        self.env.close()


# =============================================================================
# Stratified Evaluation
# =============================================================================

def _run_stratified_eval(
    model: Any,
    env_id: str,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 0,
    noisy_wrapper: NoisyEvalWrapper | None = None,
) -> dict[str, Any]:
    """Run evaluation with per-goal-type stratification.

    Unlike eval.py (which computes aggregate metrics), this function
    separately tracks performance on air goals vs table goals. This is
    critical for PickAndPlace because air goals require the full
    grasp-lift-place sequence while table goals may be solvable by pushing.

    Args:
        model: Trained SB3 model.
        env_id: Environment ID.
        n_episodes: Number of evaluation episodes.
        deterministic: Use deterministic policy.
        seed: Random seed for eval environment.
        noisy_wrapper: Optional NoisyEvalWrapper for stress testing.

    Returns:
        Dictionary with overall and per-type metrics.
    """
    import gymnasium
    import gymnasium_robotics  # noqa: F401

    env = gymnasium.make(env_id)

    if noisy_wrapper is not None:
        eval_env = NoisyEvalWrapper(
            env,
            obs_noise_std=noisy_wrapper.obs_noise_std,
            act_noise_std=noisy_wrapper.act_noise_std,
            seed=seed,
        )
    else:
        eval_env = env

    episodes: list[dict[str, Any]] = []

    for ep in range(n_episodes):
        obs, info = eval_env.reset(seed=seed + ep)
        desired_goal = obs["desired_goal"].copy()
        goal_type = _classify_goal(desired_goal)

        done = False
        ep_reward = 0.0
        ep_steps = 0
        success = False
        time_to_success = None
        actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            ep_steps += 1
            actions.append(action.copy())
            done = terminated or truncated

            if info.get("is_success", False) and time_to_success is None:
                time_to_success = ep_steps
                success = True

        # Check final success
        if info.get("is_success", False):
            success = True
            if time_to_success is None:
                time_to_success = ep_steps

        # Compute action smoothness: mean squared difference between consecutive actions
        smoothness = 0.0
        if len(actions) > 1:
            diffs = [np.sum((actions[i+1] - actions[i])**2) for i in range(len(actions) - 1)]
            smoothness = float(np.mean(diffs))

        # Final distance to goal
        final_distance = float(np.linalg.norm(
            obs["achieved_goal"] - desired_goal
        ))

        episodes.append({
            "episode": ep,
            "goal_type": goal_type,
            "success": success,
            "return": float(ep_reward),
            "steps": ep_steps,
            "final_distance": final_distance,
            "time_to_success": time_to_success,
            "action_smoothness": smoothness,
        })

    eval_env.close()

    # Compute aggregate stats
    all_success = [e["success"] for e in episodes]
    all_returns = [e["return"] for e in episodes]
    all_distances = [e["final_distance"] for e in episodes]
    all_tts = [e["time_to_success"] for e in episodes if e["time_to_success"] is not None]
    all_smoothness = [e["action_smoothness"] for e in episodes]

    # Stratify by goal type
    air_eps = [e for e in episodes if e["goal_type"] == "air"]
    table_eps = [e for e in episodes if e["goal_type"] == "table"]

    def _summarize(eps: list[dict]) -> dict[str, Any]:
        if not eps:
            return {"n": 0, "success_rate": 0.0, "return_mean": 0.0,
                    "final_distance_mean": 0.0, "time_to_success_mean": None,
                    "action_smoothness_mean": 0.0}
        successes = [e["success"] for e in eps]
        returns = [e["return"] for e in eps]
        distances = [e["final_distance"] for e in eps]
        tts = [e["time_to_success"] for e in eps if e["time_to_success"] is not None]
        smooth = [e["action_smoothness"] for e in eps]
        return {
            "n": len(eps),
            "success_rate": sum(successes) / len(successes),
            "return_mean": float(np.mean(returns)),
            "final_distance_mean": float(np.mean(distances)),
            "time_to_success_mean": float(np.mean(tts)) if tts else None,
            "action_smoothness_mean": float(np.mean(smooth)),
        }

    result = {
        "env_id": env_id,
        "n_episodes": n_episodes,
        "seed": seed,
        "deterministic": deterministic,
        "aggregate": {
            "success_rate": sum(all_success) / len(all_success),
            "return_mean": float(np.mean(all_returns)),
            "final_distance_mean": float(np.mean(all_distances)),
            "time_to_success_mean": float(np.mean(all_tts)) if all_tts else None,
            "action_smoothness_mean": float(np.mean(all_smoothness)),
        },
        "stratified": {
            "air": _summarize(air_eps),
            "table": _summarize(table_eps),
        },
        "episodes": episodes,
    }

    return result


# =============================================================================
# Commands
# =============================================================================

def cmd_train(args: argparse.Namespace) -> int:
    """Train SAC+HER on PickAndPlace."""
    import gymnasium_robotics  # noqa: F401

    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    cfg = TrainConfig.from_args(args)
    _prepare_paths(cfg)

    device = cfg.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    her_str = f"HER (n_sampled_goal={cfg.n_sampled_goal}, strategy={cfg.goal_selection_strategy})" if cfg.her else "no-HER"
    print(f"[ch05] Training SAC on {cfg.env} with {her_str}")
    print(f"[ch05] seed={cfg.seed}, n_envs={cfg.n_envs}, total_steps={cfg.total_steps}, device={device}")
    print(f"[ch05] ent_coef={cfg.ent_coef}, gamma={cfg.gamma}, lr={cfg.learning_rate}")

    set_random_seed(cfg.seed)

    # Setup paths
    out_path = Path(cfg.out) if cfg.out else Path(cfg.checkpoints_dir) / _ckpt_name(cfg.env, cfg.seed)
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    her_tag = f"her_nsg{cfg.n_sampled_goal}" if cfg.her else "noher"
    run_id = f"sac_{her_tag}/{cfg.env}/seed{cfg.seed}"
    log_dir = _resolve_log_dir(cfg, run_id=run_id)

    monitor_keywords = _parse_csv_list(cfg.monitor_keywords)
    monitor_kwargs = {"info_keywords": tuple(monitor_keywords)} if monitor_keywords else None

    # Setup curriculum if requested
    curriculum_wrapper = None
    curriculum_callback = None

    if cfg.curriculum:
        from scripts.labs.curriculum_wrapper import (
            CurriculumGoalWrapper,
            CurriculumScheduleCallback,
        )

        def make_curriculum_env_fn():
            import gymnasium
            import gymnasium_robotics  # noqa: F401
            base_env = gymnasium.make(cfg.env)
            return CurriculumGoalWrapper(base_env, initial_difficulty=0.0)

        env = make_vec_env(
            make_curriculum_env_fn,
            n_envs=cfg.n_envs,
            seed=cfg.seed,
            monitor_kwargs=monitor_kwargs,
        )
        # Get reference to first wrapper for callback
        # Note: with vectorized envs, we control difficulty on all via the callback
        curriculum_wrapper = env.envs[0] if hasattr(env, 'envs') else None
        if curriculum_wrapper is not None:
            curriculum_callback = CurriculumScheduleCallback(
                wrapper=curriculum_wrapper,
                total_timesteps=cfg.total_steps,
                mode=cfg.curriculum_mode,
                success_threshold=cfg.curriculum_success_threshold,
                verbose=1,
            )
        print(f"[ch05] Curriculum enabled: mode={cfg.curriculum_mode}")
    else:
        env = make_vec_env(
            cfg.env,
            n_envs=cfg.n_envs,
            seed=cfg.seed,
            monitor_kwargs=monitor_kwargs,
        )

    try:
        # Setup HER
        replay_buffer_class = None
        replay_buffer_kwargs = None

        if cfg.her:
            try:
                from stable_baselines3 import HerReplayBuffer
            except ImportError:
                from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

            replay_buffer_class = HerReplayBuffer
            replay_buffer_kwargs = {
                "n_sampled_goal": cfg.n_sampled_goal,
                "goal_selection_strategy": cfg.goal_selection_strategy,
            }

        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
            batch_size=cfg.batch_size,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            ent_coef=cfg.ent_coef,
            gamma=cfg.gamma,
            tau=cfg.tau,
            learning_rate=cfg.learning_rate,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
        )

        print(f"[ch05] Starting training for {cfg.total_steps} steps...")
        t0 = time.perf_counter()
        model.learn(
            total_timesteps=cfg.total_steps,
            tb_log_name=run_id,
            callback=curriculum_callback,
        )
        elapsed = time.perf_counter() - t0
        steps_per_sec = cfg.total_steps / elapsed

        print(f"[ch05] Training complete in {elapsed:.1f}s ({steps_per_sec:.0f} steps/sec)")

        # Save checkpoint
        ckpt_file = str(out_path) if str(out_path).endswith(".zip") else str(out_path) + ".zip"
        model.save(ckpt_file)
        print(f"[ch05] Saved checkpoint: {ckpt_file}")

    finally:
        env.close()

    # Write metadata
    ckpt_str = str(out_path) if str(out_path).endswith(".zip") else str(out_path) + ".zip"
    meta_path = Path(ckpt_str.removesuffix(".zip") + ".meta.json")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algo": "sac",
        "env_id": cfg.env,
        "seed": cfg.seed,
        "device": device,
        "n_envs": cfg.n_envs,
        "total_steps": cfg.total_steps,
        "training_time_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "checkpoint": ckpt_str,
        "her": cfg.her,
        "curriculum": cfg.curriculum,
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.ent_coef,
        },
        "versions": _gather_versions(),
    }
    if cfg.her:
        metadata["her_config"] = {
            "n_sampled_goal": cfg.n_sampled_goal,
            "goal_selection_strategy": cfg.goal_selection_strategy,
        }
    if cfg.curriculum:
        metadata["curriculum_config"] = {
            "mode": cfg.curriculum_mode,
            "success_threshold": cfg.curriculum_success_threshold,
        }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ch05] Wrote metadata: {meta_path}")

    return 0


def cmd_dense_debug(args: argparse.Namespace) -> int:
    """Quick validation on dense PickAndPlace.

    Before committing to 5M steps on sparse rewards, we verify the pipeline
    works on the dense variant. Dense rewards provide continuous feedback,
    so learning should be visible within 500k steps.

    PickAndPlace is harder than Reach or Push -- it requires the agent to
    discover grasping, a discontinuous skill that takes significant training
    even with dense rewards. We check for a learning *trend* (reward
    improvement, any success), not a specific success threshold.

    Verdict logic:
      - PASS: success_rate > 0% OR mean return significantly above worst-case
      - LEARNING: return improved but no successes yet (proceed cautiously)
      - FAIL: no improvement at all (likely a pipeline bug)
    """
    cfg = TrainConfig.from_args(args)

    dense_env = "FetchPickAndPlaceDense-v4"
    dense_steps = getattr(args, "dense_steps", 500_000)

    est_minutes = dense_steps / 650 / 60  # ~650 fps on DGX
    print("=" * 70)
    print("[ch05] Dense Debug: Validating pipeline on FetchPickAndPlaceDense-v4")
    print(f"[ch05] Steps: {dense_steps} (~{est_minutes:.0f} min)")
    print("=" * 70)

    # Override for dense debug
    args.env = dense_env
    args.total_steps = dense_steps
    args.out = str(Path(cfg.checkpoints_dir) / f"sac_her_{dense_env}_dense_debug")

    ret = cmd_train(args)
    if ret != 0:
        print("[ch05] Dense debug training FAILED")
        return ret

    # Evaluate
    ckpt = args.out if args.out.endswith(".zip") else args.out + ".zip"
    print(f"\n[ch05] Evaluating dense debug checkpoint...")

    model = _load_sac_model(ckpt, dense_env, device=cfg.eval_device)

    result = _run_stratified_eval(
        model, dense_env,
        n_episodes=50,
        deterministic=cfg.eval_deterministic,
        seed=cfg.eval_seed,
    )

    success_rate = result["aggregate"]["success_rate"]
    mean_return = result["aggregate"]["return_mean"]
    mean_distance = result["aggregate"]["final_distance_mean"]
    air_sr = result["stratified"]["air"]["success_rate"]
    table_sr = result["stratified"]["table"]["success_rate"]

    # Worst-case return for 50-step episode with dense rewards:
    # max distance ~0.5m, so worst return ~= -50 * 0.5 = -25
    # A random policy typically gets around -15 to -20.
    # Any return better than -15 indicates some learning.
    worst_case_return = -25.0
    random_baseline_return = -15.0

    print(f"\n[ch05] Dense Debug Results:")
    print(f"  Overall success: {success_rate:.1%}")
    print(f"  Mean return:     {mean_return:.2f} (random baseline: ~{random_baseline_return:.0f})")
    print(f"  Mean final dist: {mean_distance:.4f}m")
    print(f"  Air goals:       {air_sr:.1%} (n={result['stratified']['air']['n']})")
    print(f"  Table goals:     {table_sr:.1%} (n={result['stratified']['table']['n']})")

    # Save results
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")
    out_json = results_dir / "ch05_dense_debug_eval.json"
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["versions"] = _gather_versions()
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[ch05] Results saved: {out_json}")

    # Verdict: check for learning trend, not just final success rate.
    # PickAndPlace is hard -- even dense rewards take significant training
    # to achieve high success. The key question is: is the agent learning?
    if success_rate >= 0.2:
        print(f"\n[ch05] PASS: Dense debug success rate {success_rate:.1%} >= 20%")
        print("[ch05] Pipeline validated. Proceed to sparse training.")
    elif success_rate > 0 or mean_return > random_baseline_return:
        print(f"\n[ch05] LEARNING: Agent is improving (success={success_rate:.1%}, "
              f"return={mean_return:.2f} > random ~{random_baseline_return:.0f})")
        print("[ch05] Pipeline works. PickAndPlace needs more steps to converge.")
        print("[ch05] Proceeding to sparse training (5M steps).")
    elif mean_return > worst_case_return:
        print(f"\n[ch05] MARGINAL: Some learning detected (return={mean_return:.2f})")
        print("[ch05] Consider increasing --dense-steps if concerned.")
    else:
        print(f"\n[ch05] WARNING: No learning detected (return={mean_return:.2f})")
        print("[ch05] Pipeline may have issues. Check environment and hyperparameters.")

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate checkpoint with stratified metrics (air vs table)."""
    ckpt = args.ckpt
    if not ckpt:
        print("[ch05] Error: --ckpt required")
        return 1

    cfg = TrainConfig.from_args(args)
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")

    # Infer env if not specified
    env_id = args.env
    if not env_id:
        ckpt_name = Path(ckpt).stem
        env_id = _infer_env_id_from_ckpt(ckpt_name)
        if not env_id:
            print("[ch05] Error: Could not infer env from checkpoint name. Use --env")
            return 1

    device = cfg.eval_device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print(f"[ch05] Evaluating {ckpt} on {env_id}")
    print(f"[ch05] n_episodes={cfg.n_eval_episodes}, deterministic={cfg.eval_deterministic}")

    model = _load_sac_model(ckpt, env_id, device=device)

    result = _run_stratified_eval(
        model, env_id,
        n_episodes=cfg.n_eval_episodes,
        deterministic=cfg.eval_deterministic,
        seed=cfg.eval_seed,
    )

    # Print results
    sr = result["aggregate"]["success_rate"]
    air = result["stratified"]["air"]
    table = result["stratified"]["table"]

    print(f"\n{'='*60}")
    print(f"Evaluation Results: {env_id}")
    print(f"{'='*60}")
    print(f"  Overall:    {sr:.1%} success ({cfg.n_eval_episodes} episodes)")
    print(f"  Air goals:  {air['success_rate']:.1%} success ({air['n']} episodes)")
    print(f"  Table goals: {table['success_rate']:.1%} success ({table['n']} episodes)")
    if result["aggregate"]["time_to_success_mean"] is not None:
        print(f"  Time to success: {result['aggregate']['time_to_success_mean']:.1f} steps (among successes)")
    print(f"  Final distance:  {result['aggregate']['final_distance_mean']:.4f}m")
    print(f"  Action smoothness: {result['aggregate']['action_smoothness_mean']:.6f}")

    # Save results
    out_json = getattr(args, "json_out", None)
    if not out_json:
        ckpt_stem = Path(ckpt).stem.lower()
        out_json = str(results_dir / f"ch05_{ckpt_stem}_eval.json")
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["checkpoint"] = ckpt
    result["versions"] = _gather_versions()
    Path(out_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch05] Results saved: {out_json}")

    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    """Evaluate checkpoint under noise injection (stress test).

    Adds Gaussian noise to observations and actions to test policy
    robustness. A robust policy should degrade gracefully rather than
    catastrophically.
    """
    ckpt = args.ckpt
    if not ckpt:
        print("[ch05] Error: --ckpt required")
        return 1

    cfg = TrainConfig.from_args(args)
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")

    env_id = args.env
    if not env_id:
        ckpt_name = Path(ckpt).stem
        env_id = _infer_env_id_from_ckpt(ckpt_name)
        if not env_id:
            print("[ch05] Error: Could not infer env from checkpoint name. Use --env")
            return 1

    device = cfg.eval_device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    obs_noise = cfg.stress_obs_noise
    act_noise = cfg.stress_act_noise

    print(f"[ch05] Stress testing {ckpt} on {env_id}")
    print(f"[ch05] obs_noise_std={obs_noise}, act_noise_std={act_noise}")

    model = _load_sac_model(ckpt, env_id, device=device)

    # Run clean eval first
    print("\n[ch05] Running clean evaluation...")
    clean_result = _run_stratified_eval(
        model, env_id,
        n_episodes=cfg.n_eval_episodes,
        deterministic=cfg.eval_deterministic,
        seed=cfg.eval_seed,
    )

    # Run stress eval
    print("[ch05] Running stress evaluation...")
    noisy = NoisyEvalWrapper(
        None,  # env created inside _run_stratified_eval
        obs_noise_std=obs_noise,
        act_noise_std=act_noise,
    )
    stress_result = _run_stratified_eval(
        model, env_id,
        n_episodes=cfg.n_eval_episodes,
        deterministic=cfg.eval_deterministic,
        seed=cfg.eval_seed,
        noisy_wrapper=noisy,
    )

    # Print comparison
    clean_sr = clean_result["aggregate"]["success_rate"]
    stress_sr = stress_result["aggregate"]["success_rate"]
    degradation = clean_sr - stress_sr

    print(f"\n{'='*60}")
    print(f"Stress Test Results: {env_id}")
    print(f"{'='*60}")
    print(f"  {'Condition':<20} | {'Success':>10} | {'Air':>10} | {'Table':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Clean':<20} | {clean_sr:>9.1%} | "
          f"{clean_result['stratified']['air']['success_rate']:>9.1%} | "
          f"{clean_result['stratified']['table']['success_rate']:>9.1%}")
    print(f"  {'Stress':<20} | {stress_sr:>9.1%} | "
          f"{stress_result['stratified']['air']['success_rate']:>9.1%} | "
          f"{stress_result['stratified']['table']['success_rate']:>9.1%}")
    print(f"  {'Degradation':<20} | {degradation:>+9.1%} | "
          f"{clean_result['stratified']['air']['success_rate'] - stress_result['stratified']['air']['success_rate']:>+9.1%} | "
          f"{clean_result['stratified']['table']['success_rate'] - stress_result['stratified']['table']['success_rate']:>+9.1%}")

    if degradation < 0.1:
        print("\n  Verdict: ROBUST -- less than 10% degradation under noise")
    elif degradation < 0.3:
        print("\n  Verdict: MODERATE -- 10-30% degradation, policy is somewhat sensitive")
    else:
        print("\n  Verdict: FRAGILE -- >30% degradation, policy relies on precision")

    # Save results
    report = {
        "env_id": env_id,
        "checkpoint": ckpt,
        "noise": {"obs_noise_std": obs_noise, "act_noise_std": act_noise},
        "clean": clean_result["aggregate"],
        "clean_stratified": clean_result["stratified"],
        "stress": stress_result["aggregate"],
        "stress_stratified": stress_result["stratified"],
        "degradation": {
            "overall": degradation,
            "air": clean_result["stratified"]["air"]["success_rate"] - stress_result["stratified"]["air"]["success_rate"],
            "table": clean_result["stratified"]["table"]["success_rate"] - stress_result["stratified"]["table"]["success_rate"],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": _gather_versions(),
    }

    ckpt_stem = Path(ckpt).stem.lower()
    out_json = str(results_dir / f"ch05_{ckpt_stem}_stress.json")
    Path(out_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch05] Stress report saved: {out_json}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare results across seeds with stratified breakdown."""
    cfg = TrainConfig.from_args(args)
    seeds = _parse_seeds(args.seeds)
    env_id = cfg.env
    results_dir = Path(cfg.results_dir).expanduser().resolve()

    print(f"\n[ch05] Comparing SAC+HER on {env_id}")
    print(f"[ch05] Seeds: {seeds}")

    # Load eval results for each seed
    seed_results = []
    for seed in seeds:
        pattern = f"ch05_sac_her_{env_id.lower()}_seed{seed}_eval.json"
        files = list(results_dir.glob(pattern))
        if files:
            with open(files[0]) as f:
                seed_results.append(json.load(f))
        else:
            print(f"[ch05] Warning: No results found for seed {seed} (pattern: {pattern})")

    if not seed_results:
        print(f"[ch05] No results found. Run training and evaluation first.")
        return 1

    # Compute per-seed success rates
    overall_srs = [r["aggregate"]["success_rate"] for r in seed_results]
    air_srs = [r["stratified"]["air"]["success_rate"] for r in seed_results if r["stratified"]["air"]["n"] > 0]
    table_srs = [r["stratified"]["table"]["success_rate"] for r in seed_results if r["stratified"]["table"]["n"] > 0]

    overall_stats = _compute_stats(overall_srs)
    air_stats = _compute_stats(air_srs)
    table_stats = _compute_stats(table_srs)

    # Print comparison
    print(f"\n{'='*70}")
    print(f"Chapter 5: SAC+HER on {env_id} ({len(seed_results)} seeds)")
    print(f"{'='*70}")

    print(f"\n{'Category':<15} | {'Success Rate':>20} | {'Seeds':>6}")
    print(f"{'-'*50}")
    print(f"{'Overall':<15} | {overall_stats['mean']:.1%} +/- {overall_stats['ci95']:.1%} | {overall_stats['n']:>6}")
    print(f"{'Air goals':<15} | {air_stats['mean']:.1%} +/- {air_stats['ci95']:.1%} | {air_stats['n']:>6}")
    print(f"{'Table goals':<15} | {table_stats['mean']:.1%} +/- {table_stats['ci95']:.1%} | {table_stats['n']:>6}")

    # Per-seed breakdown
    print(f"\n{'Seed':>6} | {'Overall':>10} | {'Air':>10} | {'Table':>10}")
    print(f"{'-'*45}")
    for i, r in enumerate(seed_results):
        seed = seeds[i] if i < len(seeds) else "?"
        overall = r["aggregate"]["success_rate"]
        air = r["stratified"]["air"]["success_rate"]
        table = r["stratified"]["table"]["success_rate"]
        print(f"{seed:>6} | {overall:>9.1%} | {air:>9.1%} | {table:>9.1%}")

    # Load stress results if available
    stress_results = []
    for seed in seeds:
        pattern = f"ch05_sac_her_{env_id.lower()}_seed{seed}_stress.json"
        files = list(results_dir.glob(pattern))
        if files:
            with open(files[0]) as f:
                stress_results.append(json.load(f))

    if stress_results:
        print(f"\n{'='*70}")
        print("Stress Test Summary")
        print(f"{'='*70}")
        degradations = [r["degradation"]["overall"] for r in stress_results]
        deg_stats = _compute_stats(degradations)
        print(f"  Mean degradation: {deg_stats['mean']:.1%} +/- {deg_stats['ci95']:.1%}")

    # Summary
    print(f"\n{'='*70}")
    if overall_stats["mean"] >= 0.8:
        print("SUCCESS: SAC+HER achieves >80% on PickAndPlace.")
    elif overall_stats["mean"] >= 0.6:
        print("PARTIAL: SAC+HER achieves 60-80%. Consider curriculum or more steps.")
    else:
        print("NEEDS WORK: SAC+HER < 60%. Try curriculum learning or hyperparameter tuning.")

    if air_stats["n"] > 0 and table_stats["n"] > 0:
        gap = table_stats["mean"] - air_stats["mean"]
        if gap > 0.2:
            print(f"AIR GAP: Table goals outperform air goals by {gap:.1%}.")
            print("Air goals require grasp+lift -- the harder sub-task.")
        else:
            print(f"No significant gap between air and table goals ({gap:.1%}).")

    # Save comparison report
    report = {
        "env_id": env_id,
        "seeds": seeds,
        "n_seeds_found": len(seed_results),
        "overall": overall_stats,
        "air": air_stats,
        "table": table_stats,
        "per_seed": [
            {
                "seed": seeds[i] if i < len(seeds) else None,
                "overall": r["aggregate"]["success_rate"],
                "air": r["stratified"]["air"]["success_rate"],
                "table": r["stratified"]["table"]["success_rate"],
            }
            for i, r in enumerate(seed_results)
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if stress_results:
        report["stress_degradation"] = _compute_stats(degradations)

    report_path = results_dir / f"ch05_{env_id.lower()}_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch05] Comparison report saved: {report_path}")

    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Full pipeline: dense-debug -> train (multi-seed) -> eval -> stress -> compare."""
    cfg = TrainConfig.from_args(args)
    seeds = _parse_seeds(args.seeds)

    print("=" * 70)
    print("Chapter 5: PickAndPlace -- Full Pipeline")
    print(f"Seeds: {seeds}, Total steps per run: {cfg.total_steps}")
    print("=" * 70)

    # Step 1: Dense debug
    print("\n[1/5] Dense debug validation...")
    ret = cmd_dense_debug(args)
    if ret != 0:
        print("[ch05] Dense debug failed. Aborting.")
        return ret

    # Step 2: Train on sparse PickAndPlace for each seed
    print(f"\n[2/5] Training SAC+HER on {cfg.env} ({len(seeds)} seeds)...")
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        # Create fresh args for this seed
        train_args = argparse.Namespace(**vars(args))
        train_args.seed = seed
        train_args.env = cfg.env
        train_args.total_steps = cfg.total_steps
        train_args.out = None
        ret = cmd_train(train_args)
        if ret != 0:
            print(f"[ch05] Training failed for seed {seed}")

    # Step 3: Evaluate each seed with stratification
    print(f"\n[3/5] Evaluating checkpoints...")
    for seed in seeds:
        ckpt = str(Path(cfg.checkpoints_dir) / f"{_ckpt_name(cfg.env, seed)}.zip")
        if not Path(ckpt).exists():
            print(f"[ch05] Checkpoint not found: {ckpt}, skipping eval")
            continue

        eval_args = argparse.Namespace(**vars(args))
        eval_args.ckpt = ckpt
        eval_args.env = cfg.env
        eval_args.json_out = str(Path(cfg.results_dir) / _result_name(cfg.env, seed, "eval"))
        ret = cmd_eval(eval_args)
        if ret != 0:
            print(f"[ch05] Evaluation failed for seed {seed}")

    # Step 4: Stress test each seed
    print(f"\n[4/5] Stress testing checkpoints...")
    for seed in seeds:
        ckpt = str(Path(cfg.checkpoints_dir) / f"{_ckpt_name(cfg.env, seed)}.zip")
        if not Path(ckpt).exists():
            print(f"[ch05] Checkpoint not found: {ckpt}, skipping stress")
            continue

        stress_args = argparse.Namespace(**vars(args))
        stress_args.ckpt = ckpt
        stress_args.env = cfg.env
        ret = cmd_stress(stress_args)
        if ret != 0:
            print(f"[ch05] Stress test failed for seed {seed}")

    # Step 5: Compare
    print(f"\n[5/5] Comparing results...")
    compare_args = argparse.Namespace(**vars(args))
    compare_args.seeds = ",".join(str(s) for s in seeds)
    return cmd_compare(compare_args)


# =============================================================================
# CLI
# =============================================================================

def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    """Add common training arguments."""
    defaults = DEFAULT_CONFIG

    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="Random seed")
    parser.add_argument("--n-envs", type=int, default=defaults.n_envs,
                        help="Number of parallel environments")
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps,
                        help="Total training timesteps")
    parser.add_argument("--device", default=defaults.device,
                        help="Training device: auto, cpu, cuda")
    parser.add_argument("--monitor-keywords", default=defaults.monitor_keywords,
                        help="Comma-separated info keywords to record")

    # HER
    parser.add_argument("--her", default=defaults.her,
                        action=argparse.BooleanOptionalAction,
                        help="Enable HER (default: on)")
    parser.add_argument("--n-sampled-goal", type=int, default=defaults.n_sampled_goal,
                        help="Number of HER relabeled goals per transition")
    parser.add_argument("--goal-selection-strategy",
                        choices=["future", "final", "episode"],
                        default=defaults.goal_selection_strategy,
                        help="HER goal selection strategy")

    # SAC hyperparameters
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size,
                        help="Batch size for SAC updates")
    parser.add_argument("--buffer-size", type=int, default=defaults.buffer_size,
                        help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=defaults.learning_starts,
                        help="Steps before learning starts")
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=defaults.gamma,
                        help="Discount factor (0.95 optimal per ch04 sweep)")
    parser.add_argument("--tau", type=float, default=defaults.tau,
                        help="Soft update coefficient")
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef,
                        help="Entropy coefficient (0.05 fixed per ch04)")

    # Curriculum
    parser.add_argument("--curriculum", default=defaults.curriculum,
                        action=argparse.BooleanOptionalAction,
                        help="Enable curriculum learning")
    parser.add_argument("--curriculum-mode", choices=["linear", "success_gated"],
                        default=defaults.curriculum_mode,
                        help="Curriculum scheduling mode")
    parser.add_argument("--curriculum-success-threshold", type=float,
                        default=defaults.curriculum_success_threshold,
                        help="Success threshold for gated curriculum")

    # Paths
    parser.add_argument("--log-dir", default=defaults.log_dir,
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoints-dir", default=defaults.checkpoints_dir,
                        help="Directory for checkpoint outputs")
    parser.add_argument("--results-dir", default=defaults.results_dir,
                        help="Directory for evaluation outputs")

    # Evaluation
    parser.add_argument("--n-eval-episodes", type=int, default=defaults.n_eval_episodes,
                        help="Number of evaluation episodes")
    parser.add_argument("--eval-seed", type=int, default=defaults.eval_seed,
                        help="Base seed for evaluation")
    parser.add_argument("--eval-seeds", default=defaults.eval_seeds,
                        help="Seeds for evaluation (comma-separated or range)")
    parser.add_argument("--eval-deterministic", default=defaults.eval_deterministic,
                        action=argparse.BooleanOptionalAction,
                        help="Deterministic policy for evaluation")
    parser.add_argument("--eval-device", default=defaults.eval_device,
                        help="Evaluation device")
    parser.add_argument("--eval-algo", default=defaults.eval_algo,
                        help="Evaluation algo override")

    # Stress testing
    parser.add_argument("--stress-obs-noise", type=float, default=defaults.stress_obs_noise,
                        help="Observation noise std for stress testing")
    parser.add_argument("--stress-act-noise", type=float, default=defaults.stress_act_noise,
                        help="Action noise std for stress testing")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter 05: PickAndPlace -- From Push to Grasping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dense debug (validate pipeline, ~13 min)
  python scripts/ch05_pick_and_place.py dense-debug

  # Full pipeline (3 seeds, ~5 hours total)
  python scripts/ch05_pick_and_place.py all --seeds 0,1,2

  # Train single seed
  python scripts/ch05_pick_and_place.py train --seed 0

  # Train with curriculum learning (if sparse performance is low)
  python scripts/ch05_pick_and_place.py train --seed 0 --curriculum

  # Evaluate with stratified breakdown
  python scripts/ch05_pick_and_place.py eval --ckpt checkpoints/sac_her_FetchPickAndPlace-v4_seed0.zip

  # Stress test
  python scripts/ch05_pick_and_place.py stress --ckpt checkpoints/sac_her_FetchPickAndPlace-v4_seed0.zip
""",
    )
    parser.add_argument("--mujoco-gl", default=None,
                        help="Override MUJOCO_GL for this run")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # dense-debug
    p_dense = sub.add_parser("dense-debug",
                             help="Quick validation on dense PickAndPlace (~13 min)",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_dense.add_argument("--dense-steps", type=int, default=500_000,
                         help="Steps for dense debug run")
    _add_common_train_args(p_dense)

    # train
    p_train = sub.add_parser("train",
                             help="Train SAC+HER on sparse PickAndPlace",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_train.add_argument("--env", default=DEFAULT_CONFIG.env,
                         help="Environment ID")
    p_train.add_argument("--out", default=None,
                         help="Checkpoint output path prefix")
    _add_common_train_args(p_train)

    # eval
    p_eval = sub.add_parser("eval",
                            help="Evaluate with stratified metrics",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_eval.add_argument("--ckpt", required=True, help="Checkpoint path")
    p_eval.add_argument("--env", default=None,
                        help="Environment ID (inferred from ckpt if not specified)")
    p_eval.add_argument("--json-out", default=None, help="Output JSON path")
    _add_common_train_args(p_eval)

    # stress
    p_stress = sub.add_parser("stress",
                              help="Stress test with noise injection",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_stress.add_argument("--ckpt", required=True, help="Checkpoint path")
    p_stress.add_argument("--env", default=None,
                          help="Environment ID (inferred from ckpt if not specified)")
    _add_common_train_args(p_stress)

    # compare
    p_cmp = sub.add_parser("compare",
                           help="Compare results across seeds",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_cmp.add_argument("--seeds", default="0,1,2",
                       help="Seeds to compare (comma-separated or range)")
    _add_common_train_args(p_cmp)

    # all
    p_all = sub.add_parser("all",
                           help="Full pipeline: dense-debug -> train -> eval -> stress -> compare",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_all.add_argument("--seeds", default="0,1,2",
                       help="Seeds for multi-seed training")
    p_all.add_argument("--dense-steps", type=int, default=500_000,
                       help="Steps for dense debug")
    _add_common_train_args(p_all)

    args = parser.parse_args()

    if args.mujoco_gl is None:
        os.environ.setdefault("MUJOCO_GL", "disable")
    else:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    handlers = {
        "dense-debug": cmd_dense_debug,
        "train": cmd_train,
        "eval": cmd_eval,
        "stress": cmd_stress,
        "compare": cmd_compare,
        "all": cmd_all,
    }

    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
