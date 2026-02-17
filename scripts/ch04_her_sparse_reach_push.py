#!/usr/bin/env python3
"""Chapter 04: Sparse Reach + Push -- Introduce HER Where It Matters

Week 4 goals:
1. Demonstrate that HER is the difference-maker on sparse goals
2. Train SAC on sparse Reach/Push WITHOUT HER (baseline difficulty)
3. Train SAC + HER on sparse Reach/Push (the solution)
4. Compare with clear separation in success rates
5. Multi-seed experiments for statistical validity

Usage:
    # Full pipeline for any Fetch env (train no-HER, train HER, compare)
    python scripts/ch04_her_sparse_reach_push.py env-all --env FetchReach-v4 --seeds 0,1,2
    python scripts/ch04_her_sparse_reach_push.py env-all --env FetchPush-v4 --seeds 0,1,2 --ent-coef 0.05

    # Train single experiment
    python scripts/ch04_her_sparse_reach_push.py train --env FetchReach-v4 --her --seed 0

    # Train with entropy floor (prevents collapse while allowing adaptation)
    python scripts/ch04_her_sparse_reach_push.py train --env FetchPush-v4 --her --ent-coef auto-floor --ent-coef-min 0.05

    # Train with scheduled entropy decay
    python scripts/ch04_her_sparse_reach_push.py train --env FetchPush-v4 --her --ent-coef schedule --ent-coef-max 0.3 --ent-coef-min 0.05

    # Train with adaptive entropy based on success rate
    python scripts/ch04_her_sparse_reach_push.py train --env FetchPush-v4 --her --ent-coef adaptive --ent-coef-adaptive-target 0.2 --ent-coef-adaptive-window 200

    # Evaluate checkpoint
    python scripts/ch04_her_sparse_reach_push.py eval --ckpt checkpoints/sac_her_FetchReach-v4_seed0.zip

    # Compare HER vs no-HER results
    python scripts/ch04_her_sparse_reach_push.py compare --env FetchReach-v4 --seeds 0,1,2

    # Ablation: sweep n_sampled_goal values
    python scripts/ch04_her_sparse_reach_push.py ablation --env FetchPush-v4 --seed 0 --ent-coef 0.05

Entropy coefficient modes:
    --ent-coef <float>       Fixed value (e.g., 0.05) - recommended for sparse Push
    --ent-coef auto          SB3's default auto-tuning (may collapse on sparse rewards)
    --ent-coef auto-floor    Auto-tune with minimum floor (use --ent-coef-min)
    --ent-coef schedule      Linear decay from --ent-coef-max to --ent-coef-min
    --ent-coef adaptive      Adjust entropy based on success rate
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
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Centralized training configuration with CLI-exposable defaults.

    This dataclass serves as the single source of truth for all training
    hyperparameters. CLI arguments populate this, and all subcommands use it.
    """
    # Environment
    env: str = "FetchReach-v4"
    seed: int = 0
    n_envs: int = 8
    total_steps: int = 1_000_000
    device: str = "auto"
    monitor_keywords: str = "is_success"

    # HER settings
    her: bool = False
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"

    # SAC hyperparameters
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 1_000
    learning_rate: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005

    # Entropy coefficient settings
    # ent_coef can be: float, "auto", "auto-floor", or "schedule"
    ent_coef: str | float | None = None  # None -> "auto"
    ent_coef_min: float = 0.01  # Floor for auto-floor or schedule end
    ent_coef_max: float = 0.3   # Schedule start value
    target_entropy: float | None = None  # None -> "auto" (-dim(A))
    ent_coef_check_freq: int = 1000
    ent_coef_warmup_frac: float = 0.1
    ent_coef_log_pct: int = 10
    ent_coef_adaptive_start: float | None = None
    ent_coef_adaptive_target: float = 0.2
    ent_coef_adaptive_tolerance: float = 0.05
    ent_coef_adaptive_rate: float = 0.1
    ent_coef_adaptive_window: int = 200
    ent_coef_adaptive_warmup: float = 0.05  # Fraction of total_steps before adaptive adjustments begin
    ent_coef_adaptive_key: str = "is_success"

    # Paths
    out: str | None = None
    log_dir: str = "runs"
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"
    run_tag: str | None = None

    # Evaluation
    n_eval_episodes: int = 100
    eval_seed: int = 0
    eval_seeds: str | None = None
    eval_deterministic: bool = True
    eval_device: str = "auto"
    eval_algo: str = "auto"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainConfig":
        """Create config from parsed CLI arguments."""
        return cls(
            env=getattr(args, "env", cls.env),
            seed=getattr(args, "seed", cls.seed),
            n_envs=getattr(args, "n_envs", cls.n_envs),
            total_steps=getattr(args, "total_steps", cls.total_steps),
            device=getattr(args, "device", cls.device),
            monitor_keywords=getattr(args, "monitor_keywords", cls.monitor_keywords),
            her=getattr(args, "her", cls.her),
            n_sampled_goal=getattr(args, "n_sampled_goal", cls.n_sampled_goal),
            goal_selection_strategy=getattr(args, "goal_selection_strategy", cls.goal_selection_strategy),
            batch_size=getattr(args, "batch_size", cls.batch_size),
            buffer_size=getattr(args, "buffer_size", cls.buffer_size),
            learning_starts=getattr(args, "learning_starts", cls.learning_starts),
            learning_rate=getattr(args, "learning_rate", cls.learning_rate),
            gamma=getattr(args, "gamma", cls.gamma),
            tau=getattr(args, "tau", cls.tau),
            ent_coef=getattr(args, "ent_coef", cls.ent_coef),
            ent_coef_min=getattr(args, "ent_coef_min", cls.ent_coef_min),
            ent_coef_max=getattr(args, "ent_coef_max", cls.ent_coef_max),
            target_entropy=getattr(args, "target_entropy", cls.target_entropy),
            ent_coef_check_freq=getattr(args, "ent_coef_check_freq", cls.ent_coef_check_freq),
            ent_coef_warmup_frac=getattr(args, "ent_coef_warmup_frac", cls.ent_coef_warmup_frac),
            ent_coef_log_pct=getattr(args, "ent_coef_log_pct", cls.ent_coef_log_pct),
            ent_coef_adaptive_start=getattr(args, "ent_coef_adaptive_start", cls.ent_coef_adaptive_start),
            ent_coef_adaptive_target=getattr(args, "ent_coef_adaptive_target", cls.ent_coef_adaptive_target),
            ent_coef_adaptive_tolerance=getattr(args, "ent_coef_adaptive_tolerance", cls.ent_coef_adaptive_tolerance),
            ent_coef_adaptive_rate=getattr(args, "ent_coef_adaptive_rate", cls.ent_coef_adaptive_rate),
            ent_coef_adaptive_window=getattr(args, "ent_coef_adaptive_window", cls.ent_coef_adaptive_window),
            ent_coef_adaptive_warmup=getattr(args, "ent_coef_adaptive_warmup", cls.ent_coef_adaptive_warmup),
            ent_coef_adaptive_key=getattr(args, "ent_coef_adaptive_key", cls.ent_coef_adaptive_key),
            out=getattr(args, "out", cls.out),
            log_dir=getattr(args, "log_dir", cls.log_dir),
            checkpoints_dir=getattr(args, "checkpoints_dir", cls.checkpoints_dir),
            results_dir=getattr(args, "results_dir", cls.results_dir),
            run_tag=getattr(args, "run_tag", cls.run_tag),
            n_eval_episodes=getattr(args, "n_eval_episodes", cls.n_eval_episodes),
            eval_seed=getattr(args, "eval_seed", cls.eval_seed),
            eval_seeds=getattr(args, "eval_seeds", cls.eval_seeds),
            eval_deterministic=getattr(args, "eval_deterministic", cls.eval_deterministic),
            eval_device=getattr(args, "eval_device", cls.eval_device),
            eval_algo=getattr(args, "eval_algo", cls.eval_algo),
        )


# Default config instance for reference
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


def _normalize_run_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    tag = tag.strip()
    if not tag:
        return None
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tag)


def _is_under_default_root(path: str | Path, default_root: str) -> bool:
    path_parts = Path(path).parts
    default_parts = Path(default_root).parts
    if len(path_parts) < len(default_parts):
        return False
    return path_parts[: len(default_parts)] == default_parts


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
            raise SystemExit(f"[ch04] {label} not writable: {target}")
        fallback.mkdir(parents=True, exist_ok=True)
        if not os.access(fallback, os.W_OK):
            raise SystemExit(f"[ch04] {label} not writable: {target}")
        print(f"[ch04] {label} not writable: {target}. Using {fallback}")
        return fallback

    if not os.access(target, os.W_OK):
        if fallback is None:
            raise SystemExit(f"[ch04] {label} not writable: {target}")
        fallback.mkdir(parents=True, exist_ok=True)
        if not os.access(fallback, os.W_OK):
            raise SystemExit(f"[ch04] {label} not writable: {target}")
        print(f"[ch04] {label} not writable: {target}. Using {fallback}")
        return fallback

    return target


def _prepare_paths(cfg: TrainConfig) -> tuple[Path, Path]:
    run_tag = _normalize_run_tag(cfg.run_tag)
    cfg.run_tag = run_tag

    if run_tag:
        if cfg.results_dir == DEFAULT_CONFIG.results_dir:
            cfg.results_dir = str(Path(cfg.results_dir) / run_tag)
        if cfg.checkpoints_dir == DEFAULT_CONFIG.checkpoints_dir:
            cfg.checkpoints_dir = str(Path(cfg.checkpoints_dir) / run_tag)
        if cfg.log_dir == DEFAULT_CONFIG.log_dir:
            cfg.log_dir = str(Path(cfg.log_dir) / run_tag)

    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")
    fallback_ckpt = None
    if _is_under_default_root(cfg.checkpoints_dir, DEFAULT_CONFIG.checkpoints_dir):
        fallback_ckpt = results_dir / "checkpoints"
    checkpoints_dir = _ensure_writable_dir(cfg.checkpoints_dir, "Checkpoints dir", fallback=fallback_ckpt)

    cfg.results_dir = str(results_dir)
    cfg.checkpoints_dir = str(checkpoints_dir)

    return results_dir, checkpoints_dir


def _resolve_log_dir(cfg: TrainConfig, run_id: str | None = None) -> Path:
    def ensure_writable(root: Path) -> bool:
        target = root / run_id if run_id else root
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        return os.access(target, os.W_OK)

    log_dir = Path(cfg.log_dir).expanduser().resolve()
    if ensure_writable(log_dir):
        return log_dir

    if _is_under_default_root(cfg.log_dir, DEFAULT_CONFIG.log_dir):
        fallback = Path(cfg.results_dir).expanduser().resolve() / "runs"
        if ensure_writable(fallback):
            print(f"[ch04] Log dir not writable: {log_dir}. Using {fallback}")
            return fallback

    raise SystemExit(
        f"[ch04] Log dir not writable: {log_dir}. "
        "Set --log-dir to a writable path."
    )


def _build_train_args(cfg: TrainConfig, **overrides: Any) -> argparse.Namespace:
    data = cfg.__dict__.copy()
    data.update(overrides)
    return argparse.Namespace(**data)


def _build_eval_args(
    ckpt: str,
    env_id: str,
    cfg: TrainConfig,
    json_out: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        ckpt=ckpt,
        env=env_id,
        n_episodes=cfg.n_eval_episodes,
        json_out=json_out,
        results_dir=cfg.results_dir,
        seeds=cfg.eval_seeds,
        seed=cfg.eval_seed,
        deterministic=cfg.eval_deterministic,
        device=cfg.eval_device,
        algo=cfg.eval_algo,
    )


def _ckpt_name(env_id: str, her: bool, seed: int, n_sampled_goal: int | None = None) -> str:
    """Generate checkpoint name."""
    base = f"sac_{'her_' if her else ''}{env_id}_seed{seed}"
    if her and n_sampled_goal is not None and n_sampled_goal != 4:
        base += f"_nsg{n_sampled_goal}"
    return base


def _result_name(env_id: str, her: bool, seed: int, n_sampled_goal: int | None = None) -> str:
    """Generate result JSON name."""
    base = f"ch04_sac_{'her_' if her else ''}{env_id.lower()}_seed{seed}"
    if her and n_sampled_goal is not None and n_sampled_goal != 4:
        base += f"_nsg{n_sampled_goal}"
    return base + "_eval.json"


def _parse_ent_coef(value: str | float | None) -> tuple[str, float | None]:
    """Parse entropy coefficient argument.

    Returns:
        (mode, value) where mode is one of: "auto", "auto-floor", "schedule", "adaptive", "fixed"
        and value is the float value for "fixed" mode, None otherwise.
    """
    if value is None:
        return ("auto", None)

    if isinstance(value, (int, float)):
        return ("fixed", float(value))

    value_str = str(value).lower().strip()

    if value_str == "auto":
        return ("auto", None)
    elif value_str == "auto-floor":
        return ("auto-floor", None)
    elif value_str == "schedule":
        return ("schedule", None)
    elif value_str == "adaptive":
        return ("adaptive", None)
    else:
        # Try to parse as float
        try:
            return ("fixed", float(value_str))
        except ValueError:
            raise ValueError(
                f"Invalid --ent-coef value: {value}. "
                "Must be a float, 'auto', 'auto-floor', 'schedule', or 'adaptive'."
            )


# =============================================================================
# Entropy Coefficient Management
# =============================================================================

class EntropyFloorCallback:
    """Callback to enforce minimum entropy coefficient during training.

    This prevents the entropy collapse problem observed with SAC's auto-tuning
    on sparse reward tasks. When ent_coef drops below the floor, it's clamped.
    """

    def __init__(self, ent_coef_min: float = 0.01, check_freq: int = 1000, verbose: int = 1):
        self.ent_coef_min = ent_coef_min
        self.check_freq = check_freq
        self.verbose = verbose
        self.n_calls = 0
        self.clamp_count = 0

    def __call__(self, locals_dict: dict, globals_dict: dict) -> bool:
        """Called at each training step."""
        self.n_calls += 1

        if self.n_calls % self.check_freq != 0:
            return True

        model = locals_dict.get("self")
        if model is None:
            return True

        # Access the log_ent_coef parameter (SAC stores log of entropy coef)
        if hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
            import torch
            with torch.no_grad():
                current_ent_coef = model.log_ent_coef.exp().item()

                if current_ent_coef < self.ent_coef_min:
                    # Clamp to minimum
                    new_log_ent_coef = math.log(self.ent_coef_min)
                    model.log_ent_coef.fill_(new_log_ent_coef)
                    self.clamp_count += 1

                    if self.verbose > 0 and self.clamp_count <= 10:
                        print(f"[EntropyFloor] Clamped ent_coef from {current_ent_coef:.6f} to {self.ent_coef_min:.4f}")
                        if self.clamp_count == 10:
                            print("[EntropyFloor] (further clamp messages suppressed)")

        return True


class EntropyScheduleCallback:
    """Callback to implement scheduled entropy coefficient decay.

    Linearly decays entropy coefficient from ent_coef_max to ent_coef_min
    over the course of training. This provides high exploration early
    and exploitation later, without risk of premature collapse.
    """

    def __init__(
        self,
        total_timesteps: int,
        ent_coef_max: float = 0.3,
        ent_coef_min: float = 0.05,
        warmup_fraction: float = 0.1,
        log_pct: int = 10,
        verbose: int = 1,
    ):
        self.total_timesteps = total_timesteps
        self.ent_coef_max = ent_coef_max
        self.ent_coef_min = ent_coef_min
        self.warmup_steps = int(total_timesteps * warmup_fraction)
        self.decay_steps = total_timesteps - self.warmup_steps
        self.verbose = verbose
        self.n_calls = 0
        self.log_pct = int(log_pct)
        self.last_logged_pct = -self.log_pct if self.log_pct > 0 else 0

    def __call__(self, locals_dict: dict, globals_dict: dict) -> bool:
        """Called at each training step."""
        self.n_calls += 1

        model = locals_dict.get("self")
        if model is None:
            return True

        # Compute scheduled entropy coefficient
        if self.n_calls < self.warmup_steps:
            # Warmup: stay at max
            target_ent_coef = self.ent_coef_max
        else:
            # Linear decay
            progress = (self.n_calls - self.warmup_steps) / max(self.decay_steps, 1)
            progress = min(1.0, progress)
            target_ent_coef = self.ent_coef_max - progress * (self.ent_coef_max - self.ent_coef_min)

        # Set the entropy coefficient
        if hasattr(model, "ent_coef") and not hasattr(model, "log_ent_coef"):
            # Fixed ent_coef mode - just update the value
            model.ent_coef = target_ent_coef
        elif hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
            # Auto mode with learnable log_ent_coef - override it
            import torch
            with torch.no_grad():
                model.log_ent_coef.fill_(math.log(target_ent_coef))
                # Also update ent_coef tensor if it exists
                if hasattr(model, "ent_coef_tensor"):
                    model.ent_coef_tensor.fill_(target_ent_coef)

        # Log progress periodically
        if self.verbose > 0 and self.log_pct > 0:
            pct = int(100 * self.n_calls / self.total_timesteps)
            if pct >= self.last_logged_pct + self.log_pct:
                self.last_logged_pct = pct
                print(f"[EntropySchedule] {pct}% complete, ent_coef={target_ent_coef:.4f}")

        return True


class AdaptiveEntropyCallback:
    """Callback that adapts entropy based on success rate.

    The adaptive controller works as follows:
    1. Collect success signals from completed episodes into a sliding window
    2. After warmup period, every check_freq steps:
       - If success_rate < target - tolerance: increase entropy (more exploration)
       - If success_rate > target + tolerance: decrease entropy (more exploitation)
    3. Clamp entropy to [ent_coef_min, ent_coef_max]

    The warmup period allows the agent to gather initial experience before
    the controller starts making adjustments, avoiding noisy early adjustments.
    """

    def __init__(
        self,
        ent_coef_min: float,
        ent_coef_max: float,
        target_success: float,
        tolerance: float,
        adjust_rate: float,
        window: int,
        success_key: str,
        total_timesteps: int,
        warmup_fraction: float = 0.05,
        check_freq: int = 1000,
        verbose: int = 1,
    ):
        self.ent_coef_min = min(ent_coef_min, ent_coef_max)
        self.ent_coef_max = max(ent_coef_min, ent_coef_max)
        self.target_success = min(1.0, max(0.0, target_success))
        self.tolerance = max(0.0, tolerance)
        self.adjust_rate = max(0.0, adjust_rate)
        self.window = max(1, int(window))
        self.success_key = success_key
        self.total_timesteps = total_timesteps
        self.warmup_steps = int(total_timesteps * max(0.0, min(1.0, warmup_fraction)))
        self.check_freq = max(1, int(check_freq))
        self.verbose = verbose
        self.n_calls = 0
        self.success_window: deque[float] = deque(maxlen=self.window)
        self.warned_missing = False
        self.warmup_logged = False

    def _extract_success(self, info: dict) -> float | None:
        if self.success_key in info:
            return float(info[self.success_key])
        episode_info = info.get("episode", {})
        if isinstance(episode_info, dict) and self.success_key in episode_info:
            return float(episode_info[self.success_key])
        return None

    def _get_ent_coef(self, model: Any) -> float | None:
        if hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
            return float(model.log_ent_coef.exp().item())
        if hasattr(model, "ent_coef_tensor") and model.ent_coef_tensor is not None:
            return float(model.ent_coef_tensor.mean().item())
        if hasattr(model, "ent_coef"):
            try:
                return float(model.ent_coef)
            except Exception:
                return None
        return None

    def _set_ent_coef(self, model: Any, value: float) -> None:
        value = float(max(self.ent_coef_min, min(self.ent_coef_max, value)))
        if hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
            import torch
            with torch.no_grad():
                model.log_ent_coef.fill_(math.log(value))
                if hasattr(model, "ent_coef_tensor") and model.ent_coef_tensor is not None:
                    model.ent_coef_tensor.fill_(value)
            return
        if hasattr(model, "ent_coef_tensor") and model.ent_coef_tensor is not None:
            model.ent_coef_tensor.fill_(value)
        if hasattr(model, "ent_coef"):
            model.ent_coef = value

    def __call__(self, locals_dict: dict, globals_dict: dict) -> bool:
        self.n_calls += 1

        infos = locals_dict.get("infos") or []
        dones = locals_dict.get("dones", None)

        if isinstance(infos, list):
            for idx, info in enumerate(infos):
                if not isinstance(info, dict):
                    continue
                done = False
                if dones is not None:
                    try:
                        done = bool(dones[idx])
                    except Exception:
                        done = bool(dones)
                if "episode" in info:
                    done = True
                if not done:
                    continue

                success = self._extract_success(info)
                if success is None:
                    if not self.warned_missing:
                        print(f"[EntropyAdaptive] Missing '{self.success_key}' in info; adaptive updates disabled.")
                        self.warned_missing = True
                    continue
                self.success_window.append(success)

        if self.n_calls % self.check_freq != 0:
            return True

        # Skip adjustments during warmup period
        if self.n_calls < self.warmup_steps:
            if not self.warmup_logged and self.verbose > 0:
                pct = int(100 * self.warmup_steps / self.total_timesteps)
                print(f"[EntropyAdaptive] Warmup active until {self.warmup_steps} steps ({pct}% of training)")
                self.warmup_logged = True
            return True

        if len(self.success_window) < self.window:
            return True

        model = locals_dict.get("self")
        if model is None:
            return True

        current = self._get_ent_coef(model)
        if current is None:
            return True

        mean_success = sum(self.success_window) / len(self.success_window)
        if mean_success < self.target_success - self.tolerance:
            new_value = current * (1.0 + self.adjust_rate)
        elif mean_success > self.target_success + self.tolerance:
            new_value = current * (1.0 - self.adjust_rate)
        else:
            return True

        new_value = float(max(self.ent_coef_min, min(self.ent_coef_max, new_value)))
        if abs(new_value - current) > 1e-9:
            self._set_ent_coef(model, new_value)
            if self.verbose > 0:
                print(
                    "[EntropyAdaptive] "
                    f"success={mean_success:.3f} target={self.target_success:.3f} "
                    f"ent_coef {current:.4f} -> {new_value:.4f}"
                )

        return True


def create_entropy_callback(
    mode: str,
    total_timesteps: int,
    ent_coef_min: float = 0.01,
    ent_coef_max: float = 0.3,
    ent_coef_check_freq: int = 1000,
    ent_coef_warmup_frac: float = 0.1,
    ent_coef_log_pct: int = 10,
    ent_coef_adaptive_target: float = 0.2,
    ent_coef_adaptive_tolerance: float = 0.05,
    ent_coef_adaptive_rate: float = 0.1,
    ent_coef_adaptive_window: int = 200,
    ent_coef_adaptive_warmup: float = 0.05,
    ent_coef_adaptive_key: str = "is_success",
    verbose: int = 1,
):
    """Create the appropriate entropy management callback.

    Args:
        mode: One of "auto", "auto-floor", "schedule", "adaptive", "fixed"
        total_timesteps: Total training steps (for schedule mode)
        ent_coef_min: Minimum/floor value
        ent_coef_max: Maximum/start value (for schedule mode)
        verbose: Verbosity level

    Returns:
        Callback function or None
    """
    if mode == "auto-floor":
        return EntropyFloorCallback(
            ent_coef_min=ent_coef_min,
            check_freq=ent_coef_check_freq,
            verbose=verbose,
        )
    elif mode == "schedule":
        return EntropyScheduleCallback(
            total_timesteps=total_timesteps,
            ent_coef_max=ent_coef_max,
            ent_coef_min=ent_coef_min,
            warmup_fraction=ent_coef_warmup_frac,
            log_pct=ent_coef_log_pct,
            verbose=verbose,
        )
    elif mode == "adaptive":
        return AdaptiveEntropyCallback(
            ent_coef_min=ent_coef_min,
            ent_coef_max=ent_coef_max,
            target_success=ent_coef_adaptive_target,
            tolerance=ent_coef_adaptive_tolerance,
            adjust_rate=ent_coef_adaptive_rate,
            window=ent_coef_adaptive_window,
            success_key=ent_coef_adaptive_key,
            total_timesteps=total_timesteps,
            warmup_fraction=ent_coef_adaptive_warmup,
            check_freq=ent_coef_check_freq,
            verbose=verbose,
        )
    else:
        # "auto" or "fixed" don't need callbacks
        return None


# =============================================================================
# Commands
# =============================================================================

def cmd_train(args: argparse.Namespace) -> int:
    """Train SAC on sparse env, optionally with HER."""
    import gymnasium_robotics  # noqa: F401

    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    # Build config from args
    cfg = TrainConfig.from_args(args)
    _prepare_paths(cfg)
    _prepare_paths(cfg)

    # Parse entropy coefficient mode
    ent_mode, ent_fixed_value = _parse_ent_coef(cfg.ent_coef)

    device = cfg.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Format entropy info for logging
    if ent_mode == "fixed":
        ent_str = f"{ent_fixed_value}"
    elif ent_mode == "auto-floor":
        ent_str = f"auto-floor (min={cfg.ent_coef_min})"
    elif ent_mode == "schedule":
        ent_str = f"schedule ({cfg.ent_coef_max} -> {cfg.ent_coef_min})"
    elif ent_mode == "adaptive":
        ent_start = cfg.ent_coef_adaptive_start
        if ent_start is None:
            ent_start = cfg.ent_coef_max
        warmup_pct = int(cfg.ent_coef_adaptive_warmup * 100)
        ent_str = (
            "adaptive "
            f"(start={ent_start}, target_success={cfg.ent_coef_adaptive_target}, "
            f"window={cfg.ent_coef_adaptive_window}, warmup={warmup_pct}%)"
        )
    else:
        ent_str = "auto"

    her_str = f"HER (n_sampled_goal={cfg.n_sampled_goal}, strategy={cfg.goal_selection_strategy})" if cfg.her else "no-HER"
    target_ent_str = "auto" if cfg.target_entropy is None else f"{cfg.target_entropy}"

    print(f"[ch04] Training SAC on {cfg.env} with {her_str}")
    print(f"[ch04] seed={cfg.seed}, n_envs={cfg.n_envs}, total_steps={cfg.total_steps}, device={device}")
    print(f"[ch04] ent_coef={ent_str}, target_entropy={target_ent_str}, gamma={cfg.gamma}")
    print(f"[ch04] learning_rate={cfg.learning_rate}, batch_size={cfg.batch_size}, tau={cfg.tau}")

    set_random_seed(cfg.seed)

    # Setup paths
    out_path = Path(cfg.out) if cfg.out else Path(cfg.checkpoints_dir) / _ckpt_name(
        cfg.env, cfg.her, cfg.seed, cfg.n_sampled_goal if cfg.her else None
    )
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    her_tag = f"her_nsg{cfg.n_sampled_goal}" if cfg.her else "noher"
    run_id = f"sac_{her_tag}/{cfg.env}/seed{cfg.seed}"
    log_dir = _resolve_log_dir(cfg, run_id=run_id)

    monitor_keywords = _parse_csv_list(cfg.monitor_keywords)
    if ent_mode == "adaptive" and cfg.ent_coef_adaptive_key not in monitor_keywords:
        monitor_keywords.append(cfg.ent_coef_adaptive_key)
    monitor_kwargs = {"info_keywords": tuple(monitor_keywords)} if monitor_keywords else None

    env = make_vec_env(
        cfg.env,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        monitor_kwargs=monitor_kwargs,
    )

    try:
        # Setup HER if requested
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

        # Determine entropy coefficient for SAC
        # For "auto", "auto-floor", "schedule" modes, we start with "auto" and use callbacks
        # For "fixed" mode, we use the specified value
        if ent_mode == "fixed":
            sac_ent_coef = ent_fixed_value
        elif ent_mode == "schedule":
            # Schedule mode: start with max value, callback will adjust
            sac_ent_coef = cfg.ent_coef_max
        elif ent_mode == "adaptive":
            sac_ent_coef = cfg.ent_coef_adaptive_start
            if sac_ent_coef is None:
                sac_ent_coef = cfg.ent_coef_max
        else:
            # "auto" or "auto-floor": use SB3's auto-tuning
            sac_ent_coef = "auto"

        target_ent = "auto" if cfg.target_entropy is None else cfg.target_entropy

        # Create SAC model
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
            batch_size=cfg.batch_size,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            ent_coef=sac_ent_coef,
            target_entropy=target_ent,
            gamma=cfg.gamma,
            tau=cfg.tau,
            learning_rate=cfg.learning_rate,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
        )

        # Create entropy management callback if needed
        entropy_callback = create_entropy_callback(
            mode=ent_mode,
            total_timesteps=cfg.total_steps,
            ent_coef_min=cfg.ent_coef_min,
            ent_coef_max=cfg.ent_coef_max,
            ent_coef_check_freq=cfg.ent_coef_check_freq,
            ent_coef_warmup_frac=cfg.ent_coef_warmup_frac,
            ent_coef_log_pct=cfg.ent_coef_log_pct,
            ent_coef_adaptive_target=cfg.ent_coef_adaptive_target,
            ent_coef_adaptive_tolerance=cfg.ent_coef_adaptive_tolerance,
            ent_coef_adaptive_rate=cfg.ent_coef_adaptive_rate,
            ent_coef_adaptive_window=cfg.ent_coef_adaptive_window,
            ent_coef_adaptive_warmup=cfg.ent_coef_adaptive_warmup,
            ent_coef_adaptive_key=cfg.ent_coef_adaptive_key,
            verbose=1,
        )

        # Train
        print(f"[ch04] Starting training for {cfg.total_steps} steps...")
        t0 = time.perf_counter()
        model.learn(total_timesteps=cfg.total_steps, tb_log_name=run_id, callback=entropy_callback)
        elapsed = time.perf_counter() - t0
        steps_per_sec = cfg.total_steps / elapsed

        print(f"[ch04] Training complete in {elapsed:.1f}s ({steps_per_sec:.0f} steps/sec)")

        # Save checkpoint
        # Save checkpoint (pass path with .zip so SB3 doesn't misparse dotted stems)
        ckpt_file = str(out_path) if str(out_path).endswith(".zip") else str(out_path) + ".zip"
        model.save(ckpt_file)
        print(f"[ch04] Saved checkpoint: {ckpt_file}")

    finally:
        env.close()

    # Write metadata
    # Normalize: ensure out_path ends with .zip, then derive meta_path from it.
    # Avoid pathlib .with_suffix() -- it misparses dotted stems like "g0.95_old_seed77".
    ckpt_str = str(out_path) if str(out_path).endswith(".zip") else str(out_path) + ".zip"
    meta_path = Path(ckpt_str.removesuffix(".zip") + ".meta.json")

    # Build entropy config for metadata
    ent_coef_config = {"mode": ent_mode}
    if ent_mode == "fixed":
        ent_coef_config["value"] = ent_fixed_value
    elif ent_mode == "auto-floor":
        ent_coef_config["min"] = cfg.ent_coef_min
    elif ent_mode == "schedule":
        ent_coef_config["max"] = cfg.ent_coef_max
        ent_coef_config["min"] = cfg.ent_coef_min
    elif ent_mode == "adaptive":
        ent_coef_config["start"] = (
            cfg.ent_coef_adaptive_start if cfg.ent_coef_adaptive_start is not None else cfg.ent_coef_max
        )
        ent_coef_config["min"] = cfg.ent_coef_min
        ent_coef_config["max"] = cfg.ent_coef_max
        ent_coef_config["target_success"] = cfg.ent_coef_adaptive_target
        ent_coef_config["tolerance"] = cfg.ent_coef_adaptive_tolerance
        ent_coef_config["rate"] = cfg.ent_coef_adaptive_rate
        ent_coef_config["window"] = cfg.ent_coef_adaptive_window
        ent_coef_config["warmup"] = cfg.ent_coef_adaptive_warmup
        ent_coef_config["key"] = cfg.ent_coef_adaptive_key

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
        "run_tag": cfg.run_tag,
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": ent_coef_config,
            "target_entropy": "auto" if cfg.target_entropy is None else cfg.target_entropy,
        },
        "entropy_callback": {
            "check_freq": cfg.ent_coef_check_freq,
            "warmup_frac": cfg.ent_coef_warmup_frac,
            "log_pct": cfg.ent_coef_log_pct,
            "adaptive_target": cfg.ent_coef_adaptive_target,
            "adaptive_tolerance": cfg.ent_coef_adaptive_tolerance,
            "adaptive_rate": cfg.ent_coef_adaptive_rate,
            "adaptive_window": cfg.ent_coef_adaptive_window,
            "adaptive_warmup": cfg.ent_coef_adaptive_warmup,
            "adaptive_key": cfg.ent_coef_adaptive_key,
        },
        "monitor_keywords": monitor_keywords,
        "paths": {
            "checkpoints_dir": str(Path(cfg.checkpoints_dir).expanduser().resolve()),
            "results_dir": str(Path(cfg.results_dir).expanduser().resolve()),
            "log_dir": str(log_dir),
        },
        "versions": _gather_versions(),
    }
    if cfg.her:
        metadata["her_config"] = {
            "n_sampled_goal": cfg.n_sampled_goal,
            "goal_selection_strategy": cfg.goal_selection_strategy,
        }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ch04] Wrote metadata: {meta_path}")

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate checkpoint on sparse env."""
    ckpt = args.ckpt
    if not ckpt:
        print("[ch04] Error: --ckpt required")
        return 1

    # Infer env from checkpoint name if not specified
    env_id = args.env
    if not env_id:
        ckpt_name = Path(ckpt).stem
        env_id = _infer_env_id_from_ckpt(ckpt_name)
        if not env_id:
            print("[ch04] Error: Could not infer env from checkpoint name. Use --env")
            return 1

    n_episodes = args.n_episodes
    out_json = args.json_out
    run_tag = _normalize_run_tag(getattr(args, "run_tag", None))
    results_dir = Path(args.results_dir)
    if run_tag and args.results_dir == DEFAULT_CONFIG.results_dir:
        results_dir = results_dir / run_tag
    results_dir = _ensure_writable_dir(results_dir, "Results dir")
    if not out_json:
        # Generate from checkpoint name
        ckpt_stem = Path(ckpt).stem
        out_json = str(results_dir / f"ch04_{ckpt_stem.lower()}_eval.json")

    Path(out_json).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    seeds_arg = args.seeds or f"{args.seed}-{args.seed + n_episodes - 1}"

    cmd = [
        sys.executable,
        "eval.py",
        "--ckpt",
        ckpt,
        "--env",
        env_id,
        "--algo",
        args.algo,
        "--device",
        args.device,
        "--n-episodes",
        str(n_episodes),
        "--seeds",
        seeds_arg,
        "--json-out",
        out_json,
    ]
    if args.deterministic:
        cmd.append("--deterministic")

    print(f"[ch04] Evaluating: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        print(f"[ch04] Evaluation complete: {out_json}")
    return result.returncode


def _load_eval_results(pattern: str, results_dir: str | Path = "results") -> list[dict[str, Any]]:
    """Load all eval JSON files matching pattern."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(pattern))
    results = []
    for f in files:
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except Exception as e:
            print(f"[ch04] Warning: Could not load {f}: {e}")
    return results


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, and 95% CI for a list of values."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}

    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
        # 95% CI using t-distribution approximation (for small n, use 2.0 as rough multiplier)
        ci95 = 1.96 * std / math.sqrt(n) if n >= 30 else 2.0 * std / math.sqrt(n)
    else:
        std = 0.0
        ci95 = 0.0

    return {"mean": mean, "std": std, "ci95": ci95, "n": n}


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare HER vs no-HER results across seeds."""
    env_id = args.env
    seeds = _parse_seeds(args.seeds)
    run_tag = _normalize_run_tag(getattr(args, "run_tag", None))
    results_dir = Path(args.results_dir)
    if run_tag and args.results_dir == DEFAULT_CONFIG.results_dir:
        results_dir = results_dir / run_tag
    results_dir = results_dir.expanduser().resolve()

    env_lower = env_id.lower()

    print(f"\n[ch04] Comparing HER vs no-HER on {env_id}")
    print(f"[ch04] Seeds: {seeds}")

    # Load no-HER results
    noher_results = []
    for seed in seeds:
        pattern = f"ch04_sac_{env_lower}_seed{seed}_eval.json"
        files = list(results_dir.glob(pattern))
        if files:
            with open(files[0]) as f:
                noher_results.append(json.load(f))

    # Load HER results
    her_results = []
    for seed in seeds:
        pattern = f"ch04_sac_her_{env_lower}_seed{seed}_eval.json"
        files = list(results_dir.glob(pattern))
        if files:
            with open(files[0]) as f:
                her_results.append(json.load(f))

    if not noher_results and not her_results:
        print(f"[ch04] No results found for {env_id}. Run training first.")
        return 1

    print("\n" + "=" * 70)
    print(f"Week 4: HER vs No-HER Comparison ({env_id})")
    print("=" * 70)

    # Compute statistics
    def get_success_rates(results: list[dict]) -> list[float]:
        return [r["aggregate"]["success_rate"] for r in results]

    def get_returns(results: list[dict]) -> list[float]:
        return [r["aggregate"]["return_mean"] for r in results]

    def get_distances(results: list[dict]) -> list[float]:
        return [r["aggregate"]["final_distance_mean"] for r in results]

    noher_sr = _compute_stats(get_success_rates(noher_results)) if noher_results else None
    her_sr = _compute_stats(get_success_rates(her_results)) if her_results else None

    noher_ret = _compute_stats(get_returns(noher_results)) if noher_results else None
    her_ret = _compute_stats(get_returns(her_results)) if her_results else None

    noher_dist = _compute_stats(get_distances(noher_results)) if noher_results else None
    her_dist = _compute_stats(get_distances(her_results)) if her_results else None

    # Print comparison table
    print(f"\n{'Metric':<25} | {'No-HER':>20} | {'HER':>20} | {'Delta':>12}")
    print("-" * 85)

    def fmt_stat(stat: dict | None, fmt: str = ".1%") -> str:
        if stat is None:
            return "N/A"
        if fmt == ".1%":
            return f"{stat['mean']:.1%} +/- {stat['ci95']:.1%}"
        elif fmt == ".3f":
            return f"{stat['mean']:.3f} +/- {stat['ci95']:.3f}"
        elif fmt == ".4f":
            return f"{stat['mean']:.4f} +/- {stat['ci95']:.4f}"
        return f"{stat['mean']} +/- {stat['ci95']}"

    # Success rate
    noher_str = fmt_stat(noher_sr, ".1%")
    her_str = fmt_stat(her_sr, ".1%")
    if noher_sr and her_sr:
        delta = her_sr["mean"] - noher_sr["mean"]
        delta_str = f"{delta:+.1%}"
    else:
        delta_str = "N/A"
    print(f"{'Success Rate':<25} | {noher_str:>20} | {her_str:>20} | {delta_str:>12}")

    # Return
    noher_str = fmt_stat(noher_ret, ".3f")
    her_str = fmt_stat(her_ret, ".3f")
    if noher_ret and her_ret:
        delta = her_ret["mean"] - noher_ret["mean"]
        delta_str = f"{delta:+.3f}"
    else:
        delta_str = "N/A"
    print(f"{'Return (mean)':<25} | {noher_str:>20} | {her_str:>20} | {delta_str:>12}")

    # Final distance
    noher_str = fmt_stat(noher_dist, ".4f")
    her_str = fmt_stat(her_dist, ".4f")
    if noher_dist and her_dist:
        delta = her_dist["mean"] - noher_dist["mean"]
        delta_str = f"{delta:+.4f}"
    else:
        delta_str = "N/A"
    print(f"{'Final Distance (mean)':<25} | {noher_str:>20} | {her_str:>20} | {delta_str:>12}")

    print("-" * 85)
    print(f"{'Seeds evaluated':<25} | {noher_sr['n'] if noher_sr else 0:>20} | {her_sr['n'] if her_sr else 0:>20} |")

    # Summary
    print("\n" + "=" * 70)
    if her_sr and noher_sr:
        improvement = her_sr["mean"] - noher_sr["mean"]
        if improvement > 0.5:  # >50% improvement
            print("CLEAR SEPARATION: HER dramatically outperforms no-HER on sparse rewards.")
            print(f"Success rate improvement: {improvement:.1%}")
        elif improvement > 0.1:
            print("MODERATE SEPARATION: HER shows improvement over no-HER.")
            print(f"Success rate improvement: {improvement:.1%}")
        else:
            print("WEAK/NO SEPARATION: HER and no-HER perform similarly.")
            print("This may indicate the task is too easy or training was insufficient.")

    # Save comparison report
    report = {
        "env_id": env_id,
        "seeds": seeds,
        "no_her": {
            "n_seeds": noher_sr["n"] if noher_sr else 0,
            "success_rate": noher_sr if noher_sr else None,
            "return": noher_ret if noher_ret else None,
            "final_distance": noher_dist if noher_dist else None,
        },
        "her": {
            "n_seeds": her_sr["n"] if her_sr else 0,
            "success_rate": her_sr if her_sr else None,
            "return": her_ret if her_ret else None,
            "final_distance": her_dist if her_dist else None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = results_dir / f"ch04_{env_lower}_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch04] Comparison report saved: {report_path}")

    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """Ablation study: sweep n_sampled_goal values."""
    cfg = TrainConfig.from_args(args)
    nsg_values = [int(value) for value in _parse_csv_list(args.nsg_values)]

    print(f"\n[ch04] Ablation study: n_sampled_goal on {cfg.env}")
    print(f"[ch04] Values: {nsg_values}, seed={cfg.seed}, total_steps={cfg.total_steps}")

    results = []

    for nsg in nsg_values:
        print(f"\n{'='*60}")
        print(f"[ch04] Training with n_sampled_goal={nsg}")
        print("=" * 60)

        train_args = _build_train_args(
            cfg,
            her=True,
            n_sampled_goal=nsg,
            out=None,
        )
        ret = cmd_train(train_args)
        if ret != 0:
            print(f"[ch04] Training failed for nsg={nsg}")
            continue

        # Eval
        ckpt = str(Path(cfg.checkpoints_dir) / f"{_ckpt_name(cfg.env, True, cfg.seed, nsg)}.zip")
        out_json = str(Path(cfg.results_dir) / _result_name(cfg.env, True, cfg.seed, nsg))
        eval_args = _build_eval_args(ckpt=ckpt, env_id=cfg.env, cfg=cfg, json_out=out_json)
        ret = cmd_eval(eval_args)
        if ret != 0:
            print(f"[ch04] Evaluation failed for nsg={nsg}")
            continue

        # Load results
        with open(out_json) as f:
            eval_data = json.load(f)

        results.append({
            "n_sampled_goal": nsg,
            "success_rate": eval_data["aggregate"]["success_rate"],
            "return_mean": eval_data["aggregate"]["return_mean"],
            "final_distance_mean": eval_data["aggregate"]["final_distance_mean"],
        })

    # Print summary
    print("\n" + "=" * 70)
    print(f"Ablation Results: n_sampled_goal on {cfg.env}")
    print("=" * 70)
    print(f"\n{'n_sampled_goal':>15} | {'Success Rate':>15} | {'Return':>12} | {'Final Dist':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['n_sampled_goal']:>15} | {r['success_rate']:>14.1%} | {r['return_mean']:>12.3f} | {r['final_distance_mean']:>12.4f}")

    # Save ablation report
    report = {
        "env_id": cfg.env,
        "seed": cfg.seed,
        "total_steps": cfg.total_steps,
        "config": {
            "ent_coef": str(cfg.ent_coef),
            "gamma": cfg.gamma,
            "learning_rate": cfg.learning_rate,
        },
        "results": results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = Path(cfg.results_dir) / f"ch04_{cfg.env.lower()}_ablation_nsg.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch04] Ablation report saved: {report_path}")

    return 0


def cmd_env_all(args: argparse.Namespace) -> int:
    """Full pipeline for any Fetch env: no-HER baseline + HER + comparison.

    This replaces the separate reach-all and push-all commands with a unified
    interface that passes through all hyperparameters.
    """
    cfg = TrainConfig.from_args(args)
    _prepare_paths(cfg)
    seeds = _parse_seeds(args.seeds)

    print("=" * 70)
    print(f"Week 4: Sparse {cfg.env} - Full Pipeline")
    print(f"Seeds: {seeds}, Total steps per run: {cfg.total_steps}")
    if cfg.ent_coef is not None:
        print(f"Entropy config: {cfg.ent_coef}")
    print("=" * 70)

    # Train no-HER baselines
    print("\n[1/3] Training SAC WITHOUT HER (baseline)...")
    for seed in seeds:
        print(f"\n--- Seed {seed} (no-HER) ---")
        train_args = _build_train_args(cfg, seed=seed, her=False, out=None)
        ret = cmd_train(train_args)
        if ret != 0:
            print(f"[ch04] Training failed for seed {seed}")

        # Eval
        ckpt = str(Path(cfg.checkpoints_dir) / f"{_ckpt_name(cfg.env, False, seed)}.zip")
        out_json = str(Path(cfg.results_dir) / _result_name(cfg.env, False, seed))
        eval_args = _build_eval_args(ckpt=ckpt, env_id=cfg.env, cfg=cfg, json_out=out_json)
        cmd_eval(eval_args)

    # Train HER
    print("\n[2/3] Training SAC WITH HER...")
    for seed in seeds:
        print(f"\n--- Seed {seed} (HER) ---")
        train_args = _build_train_args(cfg, seed=seed, her=True, out=None)
        ret = cmd_train(train_args)
        if ret != 0:
            print(f"[ch04] Training failed for seed {seed}")

        # Eval
        ckpt = str(Path(cfg.checkpoints_dir) / f"{_ckpt_name(cfg.env, True, seed)}.zip")
        out_json = str(Path(cfg.results_dir) / _result_name(cfg.env, True, seed))
        eval_args = _build_eval_args(ckpt=ckpt, env_id=cfg.env, cfg=cfg, json_out=out_json)
        cmd_eval(eval_args)

    # Compare
    print("\n[3/3] Comparing results...")
    compare_args = argparse.Namespace(env=cfg.env, seeds=args.seeds, results_dir=cfg.results_dir)
    return cmd_compare(compare_args)


def _add_common_train_args(parser: argparse.ArgumentParser, defaults: TrainConfig = DEFAULT_CONFIG) -> None:
    """Add common training arguments to a parser.

    This is the single source of truth for training hyperparameters.
    All subcommands that train models use this.
    """
    # Environment
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="Random seed")
    parser.add_argument("--n-envs", type=int, default=defaults.n_envs,
                        help="Number of parallel environments")
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps,
                        help="Total training timesteps")
    parser.add_argument("--device", default=defaults.device,
                        help="Training device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--monitor-keywords", default=defaults.monitor_keywords,
                        help="Comma-separated info keywords to record (empty to disable)")

    # HER
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
                        help="Learning rate for SAC optimizer")
    parser.add_argument("--gamma", type=float, default=defaults.gamma,
                        help="Discount factor (0.95 optimal for 50-step Fetch Push; see Ch04 sweep)")
    parser.add_argument("--tau", type=float, default=defaults.tau,
                        help="Soft update coefficient for target networks")

    # Entropy coefficient (the key parameter for sparse rewards)
    parser.add_argument("--ent-coef", type=str, default=None,
                        help="Entropy coefficient: float value, 'auto', 'auto-floor', 'schedule', or 'adaptive'. "
                             "For sparse Push, use 0.05 (fixed). See Ch04 sweep results.")
    parser.add_argument("--ent-coef-min", type=float, default=defaults.ent_coef_min,
                        help="Minimum entropy coefficient (floor for 'auto-floor', end value for 'schedule')")
    parser.add_argument("--ent-coef-max", type=float, default=defaults.ent_coef_max,
                        help="Maximum entropy coefficient (start value for 'schedule' mode)")
    parser.add_argument("--target-entropy", type=float, default=None,
                        help="SAC target entropy (None -> auto=-dim(A)=-4; try -2.0 for Push)")
    parser.add_argument("--ent-coef-check-freq", type=int, default=defaults.ent_coef_check_freq,
                        help="Check frequency for auto-floor entropy clamp")
    parser.add_argument("--ent-coef-warmup-frac", type=float, default=defaults.ent_coef_warmup_frac,
                        help="Warmup fraction for scheduled entropy decay")
    parser.add_argument("--ent-coef-log-pct", type=int, default=defaults.ent_coef_log_pct,
                        help="Log entropy schedule every N percent (0 to disable)")
    parser.add_argument("--ent-coef-adaptive-start", type=float, default=defaults.ent_coef_adaptive_start,
                        help="Initial entropy coefficient for adaptive mode (defaults to ent-coef-max)")
    parser.add_argument("--ent-coef-adaptive-target", type=float, default=defaults.ent_coef_adaptive_target,
                        help="Target success rate for adaptive entropy control")
    parser.add_argument("--ent-coef-adaptive-tolerance", type=float, default=defaults.ent_coef_adaptive_tolerance,
                        help="Success rate tolerance before adjusting entropy")
    parser.add_argument("--ent-coef-adaptive-rate", type=float, default=defaults.ent_coef_adaptive_rate,
                        help="Multiplicative adjustment rate for adaptive entropy")
    parser.add_argument("--ent-coef-adaptive-window", type=int, default=defaults.ent_coef_adaptive_window,
                        help="Episode window size for adaptive success rate")
    parser.add_argument("--ent-coef-adaptive-warmup", type=float, default=defaults.ent_coef_adaptive_warmup,
                        help="Fraction of training steps before adaptive adjustments begin (0.0-1.0)")
    parser.add_argument("--ent-coef-adaptive-key", type=str, default=defaults.ent_coef_adaptive_key,
                        help="Info key used for success in adaptive mode")

    # Paths
    parser.add_argument("--log-dir", default=defaults.log_dir,
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoints-dir", default=defaults.checkpoints_dir,
                        help="Directory for checkpoint outputs")
    parser.add_argument("--results-dir", default=defaults.results_dir,
                        help="Directory for evaluation outputs and reports")
    parser.add_argument("--run-tag", default=defaults.run_tag,
                        help="Optional run tag (used to namespace outputs)")

    # Evaluation
    parser.add_argument("--n-eval-episodes", type=int, default=defaults.n_eval_episodes,
                        help="Number of episodes for evaluation")
    parser.add_argument("--eval-seed", type=int, default=defaults.eval_seed,
                        help="Base seed for evaluation (used when --eval-seeds is empty)")
    parser.add_argument("--eval-seeds", default=defaults.eval_seeds,
                        help="Seeds for evaluation (comma-separated or range)")
    parser.add_argument("--eval-deterministic", default=defaults.eval_deterministic,
                        action=argparse.BooleanOptionalAction,
                        help="Use deterministic policy for evaluation")
    parser.add_argument("--eval-device", default=defaults.eval_device,
                        help="Evaluation device passed to eval.py")
    parser.add_argument("--eval-algo", default=defaults.eval_algo,
                        help="Evaluation algo override (auto, ppo, sac, td3)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter 04: Sparse Reach + Push -- Introduce HER Where It Matters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Entropy coefficient modes:
  --ent-coef <float>       Fixed value (e.g., 0.1) - recommended for sparse Push
  --ent-coef auto          SB3's default auto-tuning (may collapse on sparse rewards)
  --ent-coef auto-floor    Auto-tune with minimum floor (use --ent-coef-min)
  --ent-coef schedule      Linear decay from --ent-coef-max to --ent-coef-min
  --ent-coef adaptive      Adjust entropy based on success rate

Examples:
  # Full pipeline for Reach (default settings work)
  python scripts/ch04_her_sparse_reach_push.py env-all --env FetchReach-v4 --seeds 0,1,2

  # Full pipeline for Push (needs fixed entropy for reliable learning)
  python scripts/ch04_her_sparse_reach_push.py env-all --env FetchPush-v4 --seeds 0,1,2 \\
      --ent-coef 0.05 --total-steps 2000000

  # Push with adaptive entropy floor (experimental)
  python scripts/ch04_her_sparse_reach_push.py env-all --env FetchPush-v4 --seeds 0,1,2 \\
      --ent-coef auto-floor --ent-coef-min 0.05 --total-steps 2000000

  # Push with scheduled entropy decay
  python scripts/ch04_her_sparse_reach_push.py train --env FetchPush-v4 --her \\
      --ent-coef schedule --ent-coef-max 0.3 --ent-coef-min 0.05 --total-steps 2000000

  # Push with adaptive entropy (success-rate feedback)
  python scripts/ch04_her_sparse_reach_push.py train --env FetchPush-v4 --her \\
      --ent-coef adaptive --ent-coef-adaptive-target 0.2 --ent-coef-adaptive-window 200 --total-steps 2000000
""",
    )
    parser.add_argument("--mujoco-gl", default=None,
                        help="Override MUJOCO_GL for this run (default: disable unless already set)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train SAC on sparse env (with or without HER)",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_train.add_argument("--env", required=True,
                         help="Environment ID (e.g., FetchReach-v4, FetchPush-v4)")
    p_train.add_argument("--her", action="store_true", help="Enable HER")
    p_train.add_argument("--out", default=None, help="Checkpoint output path prefix")
    _add_common_train_args(p_train)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate checkpoint",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_eval.add_argument("--ckpt", required=True, help="Checkpoint path")
    p_eval.add_argument("--env", default=None,
                        help="Environment ID (inferred from ckpt if not specified)")
    p_eval.add_argument("--algo", default=DEFAULT_CONFIG.eval_algo,
                        help="Algorithm override for eval.py (auto, ppo, sac, td3)")
    p_eval.add_argument("--device", default=DEFAULT_CONFIG.eval_device,
                        help="Evaluation device passed to eval.py")
    p_eval.add_argument("--n-episodes", type=int, default=DEFAULT_CONFIG.n_eval_episodes,
                        help="Number of evaluation episodes")
    p_eval.add_argument("--seed", type=int, default=DEFAULT_CONFIG.eval_seed,
                        help="Base seed for evaluation (used when --seeds is empty)")
    p_eval.add_argument("--seeds", default=DEFAULT_CONFIG.eval_seeds,
                        help="Seeds to evaluate (comma-separated or range)")
    p_eval.add_argument("--deterministic", default=DEFAULT_CONFIG.eval_deterministic,
                        action=argparse.BooleanOptionalAction,
                        help="Use deterministic policy for evaluation")
    p_eval.add_argument("--json-out", default=None, help="Output JSON path")
    p_eval.add_argument("--results-dir", default=DEFAULT_CONFIG.results_dir,
                        help="Directory for evaluation outputs")
    p_eval.add_argument("--run-tag", default=DEFAULT_CONFIG.run_tag,
                        help="Optional run tag (used to select results subdir when defaulting)")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare HER vs no-HER results",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_cmp.add_argument("--env", required=True, help="Environment ID")
    p_cmp.add_argument("--seeds", default="0,1,2",
                       help="Seeds to compare (comma-separated or range)")
    p_cmp.add_argument("--results-dir", default=DEFAULT_CONFIG.results_dir,
                       help="Directory containing evaluation outputs")
    p_cmp.add_argument("--run-tag", default=DEFAULT_CONFIG.run_tag,
                       help="Optional run tag (used to select results subdir when defaulting)")

    # ablation
    p_abl = sub.add_parser("ablation", help="Ablation study: sweep n_sampled_goal",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_abl.add_argument("--env", required=True, help="Environment ID")
    p_abl.add_argument("--nsg-values", default="2,4,8",
                       help="n_sampled_goal values to test (comma-separated)")
    _add_common_train_args(p_abl)

    # env-all (unified pipeline, replaces reach-all and push-all)
    p_all = sub.add_parser("env-all",
                           help="Full pipeline for any Fetch env: no-HER baseline + HER + comparison",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_all.add_argument("--env", required=True,
                       help="Environment ID (e.g., FetchReach-v4, FetchPush-v4)")
    p_all.add_argument("--seeds", default="0,1,2",
                       help="Seeds (comma-separated or range)")
    _add_common_train_args(p_all)

    # Legacy aliases for backwards compatibility
    p_reach = sub.add_parser("reach-all", help="(Legacy) Full pipeline for sparse Reach")
    p_reach.add_argument("--seeds", default="0,1,2")
    _add_common_train_args(p_reach)

    p_push = sub.add_parser("push-all", help="(Legacy) Full pipeline for sparse Push")
    p_push.add_argument("--seeds", default="0,1,2")
    _add_common_train_args(p_push)

    args = parser.parse_args()

    if args.mujoco_gl is None:
        os.environ.setdefault("MUJOCO_GL", "disable")
    else:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    # Handle legacy commands by converting to env-all
    if args.cmd == "reach-all":
        args.env = "FetchReach-v4"
        args.cmd = "env-all"
    elif args.cmd == "push-all":
        args.env = "FetchPush-v4"
        args.cmd = "env-all"

    handlers = {
        "train": cmd_train,
        "eval": cmd_eval,
        "compare": cmd_compare,
        "ablation": cmd_ablation,
        "env-all": cmd_env_all,
    }

    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
