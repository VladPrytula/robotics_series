#!/usr/bin/env python3
"""Chapter 9/10: Pixels, No Cheating -- Visual SAC from Reach to Push

Three-act narrative:
  Act 1 (Reach): Measure the pixel penalty -- state vs pixel vs pixel+DrQ
  Act 2 (Push diagnostic): Show Push from state (no HER) fails (~2% success)
  Act 3 (Push synthesis): HER + pixels + DrQ solve Push from raw images

Usage:
    # --- Reach experiments (Act 1) ---
    python scripts/ch10_visual_reach.py all --seed 0
    python scripts/ch10_visual_reach.py train-state --seed 0
    python scripts/ch10_visual_reach.py train-pixel --seed 0
    python scripts/ch10_visual_reach.py train-pixel-drq --seed 0
    python scripts/ch10_visual_reach.py compare

    # --- Push experiments (Acts 2-3) ---
    python scripts/ch10_visual_reach.py push-all --seed 0
    python scripts/ch10_visual_reach.py train-push-state --seed 0
    python scripts/ch10_visual_reach.py train-push-her --seed 0
    python scripts/ch10_visual_reach.py train-push-pixel-her --seed 0
    python scripts/ch10_visual_reach.py train-push-pixel-her-drq --seed 0
    python scripts/ch10_visual_reach.py push-compare

    # --- Evaluation ---
    python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_state_FetchReachDense-v4_seed0.zip
    python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_pixel_FetchReachDense-v4_seed0.zip --pixel
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

    # Push experiments
    push_env: str = "FetchPush-v4"             # Sparse Push (HER target)
    push_dense_env: str = "FetchPushDense-v4"  # Dense Push (no-HER diagnostic)
    push_state_total_steps: int = 1_000_000
    push_her_total_steps: int = 2_000_000
    push_pixel_total_steps: int = 4_000_000
    push_pixel_buffer_size: int = 200_000
    push_her_n_sampled_goal: int = 4
    push_her_goal_strategy: str = "future"
    push_ent_coef: str = "0.05"  # Fixed for sparse Push (auto-tuning collapses)

    # Feature normalization (DrQ-v2 trunk pattern)
    norm: bool = False           # --norm: use NormalizedCombinedExtractor
    cnn_dim: int = 50            # CNN output dim (50 = DrQ-v2 default, SB3 default = 256)
    frame_stack: int = 1         # Number of frames to stack (1 = no stacking, 4 = standard)

    # Fast mode -- bundles all speed optimizations
    fast: bool = False           # --fast: enable native render + SubprocVecEnv + more envs
    native_render: bool = False  # render at 84x84 natively (skip PIL resize)
    gradient_steps: int = 1      # SAC gradient steps per env step
    use_subproc: bool = False    # use SubprocVecEnv instead of DummyVecEnv

    # Shared SAC hyperparameters
    batch_size: int = 256
    learning_starts: int = 1_000
    learning_rate: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005
    ent_coef: str = "auto"  # "auto" or float-as-string (e.g. "0.05")

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


def _parse_ent_coef(val: str) -> str | float:
    """Parse ent_coef from config: 'auto' stays as string, numbers become float."""
    if val == "auto":
        return "auto"
    return float(val)


def _env_suffix(cfg: Ch10Config) -> str:
    """Non-empty env suffix for result filenames when using non-default env."""
    if cfg.env == DEFAULT_CONFIG.env:
        return ""
    return f"_{cfg.env}"


def _result_path(cfg: Ch10Config, mode: str) -> Path:
    """Generate result JSON path for state or pixel mode."""
    return _ensure_dir(cfg.results_dir) / f"ch10_{mode}{_env_suffix(cfg)}_eval.json"


def _comparison_path(cfg: Ch10Config) -> Path:
    return _ensure_dir(cfg.results_dir) / f"ch10{_env_suffix(cfg)}_comparison.json"


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
            ent_coef=_parse_ent_coef(cfg.ent_coef),
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
            "ent_coef": cfg.ent_coef,
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
    if cfg.fast:
        print(f"[ch10] FAST MODE: native_render={cfg.native_render}, "
              f"use_subproc={cfg.use_subproc}, gradient_steps={cfg.gradient_steps}")

    # Build env factory -- native_render passes width/height to MuJoCo
    # so it renders at 84x84 directly, skipping PIL resize entirely.
    # gymnasium_robotics is imported inside the factory because SubprocVecEnv
    # runs each factory in a child process that needs the Fetch envs registered.
    img_sz = cfg.image_size

    def make_pixel_env():
        import gymnasium_robotics  # noqa: F401 -- register Fetch envs in subprocess
        from scripts.labs.pixel_wrapper import PixelObservationWrapper as _Wrapper

        make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
        if cfg.native_render:
            make_kwargs["width"] = img_sz
            make_kwargs["height"] = img_sz
        env = gym.make(cfg.env, **make_kwargs)
        return _Wrapper(
            env,
            image_size=(img_sz, img_sz),
            goal_mode=cfg.pixel_goal_mode,
        )

    vec_env_kwargs: dict[str, Any] = {}
    if cfg.use_subproc:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env_kwargs["vec_env_cls"] = SubprocVecEnv

    env = make_vec_env(make_pixel_env, n_envs=cfg.pixel_n_envs, seed=cfg.seed,
                       **vec_env_kwargs)

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
            ent_coef=_parse_ent_coef(cfg.ent_coef),
            gradient_steps=cfg.gradient_steps,
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
        "fast_mode": cfg.fast,
        "native_render": cfg.native_render,
        "gradient_steps": cfg.gradient_steps,
        "use_subproc": cfg.use_subproc,
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.pixel_buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.ent_coef,
            "gradient_steps": cfg.gradient_steps,
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
    if cfg.fast:
        print(f"[ch10] FAST MODE: native_render={cfg.native_render}, "
              f"use_subproc={cfg.use_subproc}, gradient_steps={cfg.gradient_steps}")

    # gymnasium_robotics imported inside factory for SubprocVecEnv subprocess
    # registration (same pattern as cmd_train_pixel).
    img_sz = cfg.image_size

    def make_pixel_env():
        import gymnasium_robotics  # noqa: F401 -- register Fetch envs in subprocess
        from scripts.labs.pixel_wrapper import PixelObservationWrapper as _Wrapper

        make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
        if cfg.native_render:
            make_kwargs["width"] = img_sz
            make_kwargs["height"] = img_sz
        env = gym.make(cfg.env, **make_kwargs)
        return _Wrapper(
            env,
            image_size=(img_sz, img_sz),
            goal_mode=cfg.pixel_goal_mode,
        )

    vec_env_kwargs: dict[str, Any] = {}
    if cfg.use_subproc:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env_kwargs["vec_env_cls"] = SubprocVecEnv

    env = make_vec_env(make_pixel_env, n_envs=cfg.pixel_n_envs, seed=cfg.seed,
                       **vec_env_kwargs)

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
            ent_coef=_parse_ent_coef(cfg.ent_coef),
            gradient_steps=cfg.gradient_steps,
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
        "fast_mode": cfg.fast,
        "native_render": cfg.native_render,
        "gradient_steps": cfg.gradient_steps,
        "use_subproc": cfg.use_subproc,
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.pixel_buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.ent_coef,
            "gradient_steps": cfg.gradient_steps,
            "drq_pad": cfg.drq_pad,
        },
        "versions": _gather_versions(),
    }
    _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Wrote metadata: {_meta_path(ckpt)}")

    return 0


# =============================================================================
# Push Commands (Act 2-3: the real test)
# =============================================================================

def cmd_train_push_state(cfg: Ch10Config) -> int:
    """Train SAC on FetchPushDense with state observations, NO HER.

    This is the diagnostic baseline: dense reward + state obs + no HER.
    Expected result: ~2% success. The "deceptively dense" reward provides
    distance-to-goal signal, but Push requires object contact first --
    before contact, the reward landscape is nearly flat (the gripper can
    approach the goal, but the object stays put).
    """
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    device = _resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    env_id = cfg.push_dense_env
    print(f"[ch10] Training PUSH STATE (no HER) on {env_id}")
    print(f"[ch10] seed={cfg.seed}, n_envs={cfg.state_n_envs}, "
          f"total_steps={cfg.push_state_total_steps}, device={device}")

    env = make_vec_env(env_id, n_envs=cfg.state_n_envs, seed=cfg.seed)

    log_dir = _ensure_dir(cfg.log_dir)
    run_id = f"sac_push_state/{env_id}/seed{cfg.seed}"

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
            ent_coef=_parse_ent_coef(cfg.ent_coef),
        )

        t0 = time.perf_counter()
        model.learn(total_timesteps=cfg.push_state_total_steps, tb_log_name=run_id)
        elapsed = time.perf_counter() - t0
        fps = cfg.push_state_total_steps / elapsed

        ckpt = _ensure_dir(cfg.checkpoints_dir) / f"sac_push_state_{env_id}_seed{cfg.seed}.zip"
        model.save(str(ckpt))
        print(f"[ch10] Saved: {ckpt}")
        print(f"[ch10] Training time: {elapsed:.1f}s ({fps:.0f} steps/sec)")

    finally:
        env.close()

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 10,
        "mode": "push_state",
        "algo": "sac",
        "env_id": env_id,
        "seed": cfg.seed,
        "device": device,
        "n_envs": cfg.state_n_envs,
        "total_steps": cfg.push_state_total_steps,
        "training_time_sec": elapsed,
        "steps_per_sec": fps,
        "her": False,
        "checkpoint": str(ckpt),
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.state_buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.ent_coef,
        },
        "versions": _gather_versions(),
    }
    _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Wrote metadata: {_meta_path(ckpt)}")

    return 0


def cmd_train_push_her(cfg: Ch10Config) -> int:
    """Train SAC + HER on FetchPush (sparse) with state observations.

    This demonstrates that HER solves Push from state -- the baseline before
    we attempt pixels. Uses fixed ent_coef=0.05 (from Ch5: auto-tuning
    collapses on sparse Push because the entropy target is calibrated for
    the initial near-zero reward regime).
    """
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    try:
        from stable_baselines3 import HerReplayBuffer
    except ImportError:
        from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    device = _resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    env_id = cfg.push_env
    print(f"[ch10] Training PUSH STATE + HER on {env_id}")
    print(f"[ch10] seed={cfg.seed}, n_envs={cfg.state_n_envs}, "
          f"total_steps={cfg.push_her_total_steps}, device={device}")
    print(f"[ch10] HER: strategy={cfg.push_her_goal_strategy}, "
          f"n_sampled_goal={cfg.push_her_n_sampled_goal}, "
          f"ent_coef={cfg.push_ent_coef}")

    env = make_vec_env(env_id, n_envs=cfg.state_n_envs, seed=cfg.seed)

    log_dir = _ensure_dir(cfg.log_dir)
    run_id = f"sac_push_her/{env_id}/seed{cfg.seed}"

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
            ent_coef=_parse_ent_coef(cfg.push_ent_coef),
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": cfg.push_her_n_sampled_goal,
                "goal_selection_strategy": cfg.push_her_goal_strategy,
            },
        )

        t0 = time.perf_counter()
        model.learn(total_timesteps=cfg.push_her_total_steps, tb_log_name=run_id)
        elapsed = time.perf_counter() - t0
        fps = cfg.push_her_total_steps / elapsed

        ckpt = _ensure_dir(cfg.checkpoints_dir) / f"sac_push_her_{env_id}_seed{cfg.seed}.zip"
        model.save(str(ckpt))
        print(f"[ch10] Saved: {ckpt}")
        print(f"[ch10] Training time: {elapsed:.1f}s ({fps:.0f} steps/sec)")

    finally:
        env.close()

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 10,
        "mode": "push_her",
        "algo": "sac",
        "env_id": env_id,
        "seed": cfg.seed,
        "device": device,
        "n_envs": cfg.state_n_envs,
        "total_steps": cfg.push_her_total_steps,
        "training_time_sec": elapsed,
        "steps_per_sec": fps,
        "her": True,
        "her_n_sampled_goal": cfg.push_her_n_sampled_goal,
        "her_goal_strategy": cfg.push_her_goal_strategy,
        "checkpoint": str(ckpt),
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.state_buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.push_ent_coef,
        },
        "versions": _gather_versions(),
    }
    _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Wrote metadata: {_meta_path(ckpt)}")

    return 0


def cmd_train_push_pixel_her(cfg: Ch10Config) -> int:
    """Train SAC + HER on FetchPush (sparse) with pixel observations.

    Uses goal_mode="both": the policy sees pixels (honest visual learning)
    while HER sees goal vectors (for relabeling). This is the bridge
    between visual learning and goal-conditioned exploration.
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    try:
        from stable_baselines3 import HerReplayBuffer
    except ImportError:
        from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    device = _resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    env_id = cfg.push_env
    print(f"[ch10] Training PUSH PIXEL + HER on {env_id}")
    print(f"[ch10] seed={cfg.seed}, n_envs={cfg.pixel_n_envs}, "
          f"total_steps={cfg.push_pixel_total_steps}, device={device}")
    print(f"[ch10] image_size={cfg.image_size}x{cfg.image_size}, "
          f"goal_mode=both (policy=pixels, HER=vectors)")
    print(f"[ch10] HER: strategy={cfg.push_her_goal_strategy}, "
          f"n_sampled_goal={cfg.push_her_n_sampled_goal}, "
          f"ent_coef={cfg.push_ent_coef}")
    if cfg.norm:
        print(f"[ch10] NORM MODE: NormalizedCombinedExtractor, cnn_dim={cfg.cnn_dim}, "
              f"lr={cfg.learning_rate}")
    if cfg.frame_stack > 1:
        print(f"[ch10] FRAME STACK: {cfg.frame_stack} frames "
              f"({3 * cfg.frame_stack} channels)")
    if cfg.fast:
        print(f"[ch10] FAST MODE: native_render={cfg.native_render}, "
              f"use_subproc={cfg.use_subproc}, gradient_steps={cfg.gradient_steps}")

    img_sz = cfg.image_size
    fstack = cfg.frame_stack

    def make_pixel_env():
        import gymnasium_robotics  # noqa: F401
        from scripts.labs.pixel_wrapper import PixelObservationWrapper as _Wrapper

        make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
        if cfg.native_render:
            make_kwargs["width"] = img_sz
            make_kwargs["height"] = img_sz
        env = gym.make(env_id, **make_kwargs)
        return _Wrapper(env, image_size=(img_sz, img_sz), goal_mode="both",
                        frame_stack=fstack)

    vec_env_kwargs: dict[str, Any] = {}
    if cfg.use_subproc:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env_kwargs["vec_env_cls"] = SubprocVecEnv

    env = make_vec_env(make_pixel_env, n_envs=cfg.pixel_n_envs, seed=cfg.seed,
                       **vec_env_kwargs)

    log_dir = _ensure_dir(cfg.log_dir)
    run_id = f"sac_push_pixel_her/{env_id}/seed{cfg.seed}"

    policy_kwargs: dict[str, Any] = {}
    if cfg.norm:
        from scripts.labs.visual_encoder import NormalizedCombinedExtractor
        policy_kwargs["features_extractor_class"] = NormalizedCombinedExtractor
        policy_kwargs["features_extractor_kwargs"] = {"cnn_output_dim": cfg.cnn_dim}

    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
            batch_size=cfg.batch_size,
            buffer_size=cfg.push_pixel_buffer_size,
            learning_starts=cfg.learning_starts,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            tau=cfg.tau,
            ent_coef=_parse_ent_coef(cfg.push_ent_coef),
            gradient_steps=cfg.gradient_steps,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": cfg.push_her_n_sampled_goal,
                "goal_selection_strategy": cfg.push_her_goal_strategy,
            },
        )

        t0 = time.perf_counter()
        model.learn(total_timesteps=cfg.push_pixel_total_steps, tb_log_name=run_id)
        elapsed = time.perf_counter() - t0
        fps = cfg.push_pixel_total_steps / elapsed

        ckpt = _ensure_dir(cfg.checkpoints_dir) / f"sac_push_pixel_her_{env_id}_seed{cfg.seed}.zip"
        model.save(str(ckpt))
        print(f"[ch10] Saved: {ckpt}")
        print(f"[ch10] Training time: {elapsed:.1f}s ({fps:.0f} steps/sec)")

    finally:
        env.close()

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 10,
        "mode": "push_pixel_her",
        "algo": "sac",
        "env_id": env_id,
        "seed": cfg.seed,
        "device": device,
        "n_envs": cfg.pixel_n_envs,
        "total_steps": cfg.push_pixel_total_steps,
        "training_time_sec": elapsed,
        "steps_per_sec": fps,
        "her": True,
        "her_n_sampled_goal": cfg.push_her_n_sampled_goal,
        "her_goal_strategy": cfg.push_her_goal_strategy,
        "image_size": cfg.image_size,
        "pixel_goal_mode": "both",
        "drq": False,
        "norm": cfg.norm,
        "cnn_dim": cfg.cnn_dim if cfg.norm else 256,
        "frame_stack": cfg.frame_stack,
        "fast_mode": cfg.fast,
        "checkpoint": str(ckpt),
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.push_pixel_buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.push_ent_coef,
            "gradient_steps": cfg.gradient_steps,
        },
        "versions": _gather_versions(),
    }
    _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Wrote metadata: {_meta_path(ckpt)}")

    return 0


def cmd_train_push_pixel_her_drq(cfg: Ch10Config) -> int:
    """Train SAC + HER + DrQ on FetchPush (sparse) with pixel observations.

    The synthesis: pixel wrapper (goal_mode="both") + HER relabeling +
    DrQ augmentation. Uses HerDrQDictReplayBuffer which applies random
    shift augmentation AFTER HER's relabeling merge.

    This is the chapter's resolution: three techniques from three chapters
    combine to solve a problem none could solve alone.
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    from scripts.labs.image_augmentation import HerDrQDictReplayBuffer, RandomShiftAug

    device = _resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    env_id = cfg.push_env
    print(f"[ch10] Training PUSH PIXEL + HER + DrQ on {env_id}")
    print(f"[ch10] seed={cfg.seed}, n_envs={cfg.pixel_n_envs}, "
          f"total_steps={cfg.push_pixel_total_steps}, device={device}")
    print(f"[ch10] image_size={cfg.image_size}x{cfg.image_size}, "
          f"drq_pad={cfg.drq_pad}, goal_mode=both")
    print(f"[ch10] HER: strategy={cfg.push_her_goal_strategy}, "
          f"n_sampled_goal={cfg.push_her_n_sampled_goal}, "
          f"ent_coef={cfg.push_ent_coef}")
    if cfg.norm:
        print(f"[ch10] NORM MODE: NormalizedCombinedExtractor, cnn_dim={cfg.cnn_dim}, "
              f"lr={cfg.learning_rate}")
    if cfg.frame_stack > 1:
        print(f"[ch10] FRAME STACK: {cfg.frame_stack} frames "
              f"({3 * cfg.frame_stack} channels)")
    if cfg.fast:
        print(f"[ch10] FAST MODE: native_render={cfg.native_render}, "
              f"use_subproc={cfg.use_subproc}, gradient_steps={cfg.gradient_steps}")

    img_sz = cfg.image_size
    fstack = cfg.frame_stack

    def make_pixel_env():
        import gymnasium_robotics  # noqa: F401
        from scripts.labs.pixel_wrapper import PixelObservationWrapper as _Wrapper

        make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
        if cfg.native_render:
            make_kwargs["width"] = img_sz
            make_kwargs["height"] = img_sz
        env = gym.make(env_id, **make_kwargs)
        return _Wrapper(env, image_size=(img_sz, img_sz), goal_mode="both",
                        frame_stack=fstack)

    vec_env_kwargs: dict[str, Any] = {}
    if cfg.use_subproc:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env_kwargs["vec_env_cls"] = SubprocVecEnv

    env = make_vec_env(make_pixel_env, n_envs=cfg.pixel_n_envs, seed=cfg.seed,
                       **vec_env_kwargs)

    log_dir = _ensure_dir(cfg.log_dir)
    run_id = f"sac_push_pixel_her_drq/{env_id}/seed{cfg.seed}"

    policy_kwargs: dict[str, Any] = {}
    if cfg.norm:
        from scripts.labs.visual_encoder import NormalizedCombinedExtractor
        policy_kwargs["features_extractor_class"] = NormalizedCombinedExtractor
        policy_kwargs["features_extractor_kwargs"] = {"cnn_output_dim": cfg.cnn_dim}

    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
            batch_size=cfg.batch_size,
            buffer_size=cfg.push_pixel_buffer_size,
            learning_starts=cfg.learning_starts,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            tau=cfg.tau,
            ent_coef=_parse_ent_coef(cfg.push_ent_coef),
            gradient_steps=cfg.gradient_steps,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            replay_buffer_class=HerDrQDictReplayBuffer,
            replay_buffer_kwargs={
                "aug_fn": RandomShiftAug(pad=cfg.drq_pad),
                "image_key": "pixels",
                "n_sampled_goal": cfg.push_her_n_sampled_goal,
                "goal_selection_strategy": cfg.push_her_goal_strategy,
            },
        )

        t0 = time.perf_counter()
        model.learn(total_timesteps=cfg.push_pixel_total_steps, tb_log_name=run_id)
        elapsed = time.perf_counter() - t0
        fps = cfg.push_pixel_total_steps / elapsed

        ckpt = _ensure_dir(cfg.checkpoints_dir) / f"sac_push_pixel_her_drq_{env_id}_seed{cfg.seed}.zip"
        model.save(str(ckpt))
        print(f"[ch10] Saved: {ckpt}")
        print(f"[ch10] Training time: {elapsed:.1f}s ({fps:.0f} steps/sec)")

    finally:
        env.close()

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 10,
        "mode": "push_pixel_her_drq",
        "algo": "sac",
        "env_id": env_id,
        "seed": cfg.seed,
        "device": device,
        "n_envs": cfg.pixel_n_envs,
        "total_steps": cfg.push_pixel_total_steps,
        "training_time_sec": elapsed,
        "steps_per_sec": fps,
        "her": True,
        "her_n_sampled_goal": cfg.push_her_n_sampled_goal,
        "her_goal_strategy": cfg.push_her_goal_strategy,
        "image_size": cfg.image_size,
        "pixel_goal_mode": "both",
        "drq": True,
        "drq_pad": cfg.drq_pad,
        "norm": cfg.norm,
        "cnn_dim": cfg.cnn_dim if cfg.norm else 256,
        "frame_stack": cfg.frame_stack,
        "fast_mode": cfg.fast,
        "checkpoint": str(ckpt),
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.push_pixel_buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.push_ent_coef,
            "gradient_steps": cfg.gradient_steps,
            "drq_pad": cfg.drq_pad,
        },
        "versions": _gather_versions(),
    }
    _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ch10] Wrote metadata: {_meta_path(ckpt)}")

    return 0


def _push_ckpt_path(cfg: Ch10Config, mode: str) -> Path:
    """Generate checkpoint path for Push experiments."""
    env_id = cfg.push_env if "her" in mode else cfg.push_dense_env
    return _ensure_dir(cfg.checkpoints_dir) / f"sac_{mode}_{env_id}_seed{cfg.seed}.zip"


def _push_result_path(cfg: Ch10Config, mode: str) -> Path:
    """Generate result JSON path for Push experiments."""
    return _ensure_dir(cfg.results_dir) / f"ch10_{mode}_eval.json"


def _eval_push_state(cfg: Ch10Config, mode: str) -> int:
    """Evaluate a Push state-based checkpoint."""
    ckpt = _push_ckpt_path(cfg, mode)
    if not ckpt.exists():
        print(f"[ch10] Checkpoint not found: {ckpt}")
        return 1

    env_id = cfg.push_env if "her" in mode else cfg.push_dense_env
    json_out = str(_push_result_path(cfg, mode))
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)

    seeds_arg = f"0-{cfg.n_eval_episodes - 1}"
    cmd = [
        sys.executable, "eval.py",
        "--ckpt", str(ckpt),
        "--env", env_id,
        "--algo", "sac",
        "--device", _resolve_device(cfg.device),
        "--n-episodes", str(cfg.n_eval_episodes),
        "--seeds", seeds_arg,
        "--json-out", json_out,
    ]
    if cfg.eval_deterministic:
        cmd.append("--deterministic")

    print(f"[ch10] Evaluating Push {mode}: {ckpt}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        print(f"[ch10] Push {mode} evaluation saved: {json_out}")
    return result.returncode


def _eval_push_pixel(cfg: Ch10Config, mode: str) -> int:
    """Evaluate a Push pixel-based checkpoint with goal_mode=both."""
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import numpy as np
    from stable_baselines3 import SAC

    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    ckpt = _push_ckpt_path(cfg, mode)
    if not ckpt.exists():
        print(f"[ch10] Checkpoint not found: {ckpt}")
        return 1

    env_id = cfg.push_env
    json_out = str(_push_result_path(cfg, mode))
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    print(f"[ch10] Evaluating Push {mode}: {ckpt}")
    print(f"[ch10] device={device}, n_episodes={cfg.n_eval_episodes}")

    model = SAC.load(str(ckpt), device=device)

    make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
    if cfg.native_render:
        make_kwargs["width"] = cfg.image_size
        make_kwargs["height"] = cfg.image_size
    base_env = gym.make(env_id, **make_kwargs)
    env = PixelObservationWrapper(
        base_env, image_size=(cfg.image_size, cfg.image_size), goal_mode="both",
    )

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
        episode_successes.append(float(info.get("is_success", False)))

        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        dist = np.linalg.norm(achieved - desired)
        final_distances.append(float(dist))

    env.close()

    success_rate = float(np.mean(episode_successes))
    return_mean = float(np.mean(episode_returns))
    return_std = float(np.std(episode_returns))
    distance_mean = float(np.mean(final_distances))
    distance_std = float(np.std(final_distances))
    length_mean = float(np.mean(episode_lengths))

    print(f"[ch10] Push {mode} eval: success_rate={success_rate:.1%}, "
          f"return={return_mean:.3f} +/- {return_std:.3f}, "
          f"final_dist={distance_mean:.4f}")

    report = {
        "checkpoint": str(ckpt),
        "env_id": env_id,
        "mode": mode,
        "image_size": cfg.image_size,
        "pixel_goal_mode": "both",
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
    print(f"[ch10] Push {mode} evaluation saved: {json_out}")

    return 0


def cmd_push_compare(cfg: Ch10Config) -> int:
    """Compare Push experiments across all available configurations.

    Prints a table showing the progression:
    state no-HER (~2%) < state+HER (~95%) < pixel+HER (~80%) < pixel+HER+DrQ (~90%+)
    """
    modes = [
        ("push_state", "State (no HER)"),
        ("push_her", "State + HER"),
        ("push_pixel_her", "Pixel + HER"),
        ("push_pixel_her_drq", "Pixel+HER+DrQ"),
    ]

    results = {}
    for mode, label in modes:
        path = _push_result_path(cfg, mode)
        if path.exists():
            with open(path) as f:
                results[mode] = json.load(f)

    if not results:
        print("[ch10] No Push evaluation results found.")
        print("[ch10] Run push-all first, or individual train + eval commands.")
        return 1

    print()
    print("=" * 100)
    print("Chapter 10: Push Experiment Comparison")
    print("=" * 100)

    # Header
    cols = [(mode, label) for mode, label in modes if mode in results]
    header = f"{'Metric':<25}"
    for _, label in cols:
        header += f" | {label:>18}"
    print(f"\n{header}")
    print("-" * (26 + 21 * len(cols)))

    # Rows
    def row(metric_name, key, fmt=".1%"):
        line = f"{metric_name:<25}"
        for mode, _ in cols:
            val = results[mode]["aggregate"].get(key)
            if val is not None:
                line += f" | {val:>18{fmt}}"
            else:
                line += f" | {'N/A':>18}"
        print(line)

    row("Success rate", "success_rate", ".1%")
    row("Return (mean)", "return_mean", ".3f")
    row("Return (std)", "return_std", ".3f")
    row("Final distance", "final_distance_mean", ".4f")
    row("Episode length", "ep_len_mean", ".1f")

    print("-" * (26 + 21 * len(cols)))

    # Summary interpretation
    print()
    if "push_state" in results and "push_her" in results:
        sr_state = results["push_state"]["aggregate"]["success_rate"]
        sr_her = results["push_her"]["aggregate"]["success_rate"]
        print(f"HER effect: {sr_state:.1%} -> {sr_her:.1%} "
              f"(+{sr_her - sr_state:.1%} from goal relabeling)")

    if "push_her" in results and "push_pixel_her_drq" in results:
        sr_state_her = results["push_her"]["aggregate"]["success_rate"]
        sr_pixel_drq = results["push_pixel_her_drq"]["aggregate"]["success_rate"]
        pixel_penalty = sr_state_her - sr_pixel_drq
        print(f"Pixel penalty (with DrQ): {sr_state_her:.1%} -> {sr_pixel_drq:.1%} "
              f"({pixel_penalty:+.1%})")

    if "push_pixel_her" in results and "push_pixel_her_drq" in results:
        sr_no_drq = results["push_pixel_her"]["aggregate"]["success_rate"]
        sr_drq = results["push_pixel_her_drq"]["aggregate"]["success_rate"]
        print(f"DrQ augmentation effect: {sr_no_drq:.1%} -> {sr_drq:.1%} "
              f"({sr_drq - sr_no_drq:+.1%})")

    # Save comparison report
    report = {
        "experiment": "push_comparison",
        "seed": cfg.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for mode, _ in cols:
        report[mode] = results[mode]["aggregate"]

    comp_path = _ensure_dir(cfg.results_dir) / "ch10_push_comparison.json"
    comp_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch10] Push comparison saved: {comp_path}")

    return 0


def cmd_push_all(cfg: Ch10Config) -> int:
    """Full Push pipeline: 4 training configs + eval + comparison."""
    print("=" * 72)
    print("Chapter 10: Push Pipeline -- State vs HER vs Pixel+HER vs Pixel+HER+DrQ")
    print(f"Seed={cfg.seed}")
    print("=" * 72)

    steps = [
        ("1/8", "Push state (no HER)", lambda: cmd_train_push_state(cfg)),
        ("2/8", "Eval push state", lambda: _eval_push_state(cfg, "push_state")),
        ("3/8", "Push state + HER", lambda: cmd_train_push_her(cfg)),
        ("4/8", "Eval push HER", lambda: _eval_push_state(cfg, "push_her")),
        ("5/8", "Push pixel + HER", lambda: cmd_train_push_pixel_her(cfg)),
        ("6/8", "Eval push pixel+HER", lambda: _eval_push_pixel(cfg, "push_pixel_her")),
        ("7/8", "Push pixel + HER + DrQ", lambda: cmd_train_push_pixel_her_drq(cfg)),
        ("8/8", "Eval push pixel+HER+DrQ", lambda: _eval_push_pixel(cfg, "push_pixel_her_drq")),
    ]

    for step_id, desc, fn in steps:
        print(f"\n[{step_id}] {desc}...")
        ret = fn()
        if ret != 0:
            print(f"[ch10] {desc} failed (exit code {ret})")
            return ret

    print("\n[9/9] Comparing Push results...")
    return cmd_push_compare(cfg)


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
    make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
    if cfg.native_render:
        make_kwargs["width"] = cfg.image_size
        make_kwargs["height"] = cfg.image_size
    base_env = gym.make(cfg.env, **make_kwargs)
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
    parser.add_argument("--env", default=DEFAULT_CONFIG.env,
                        help="Gymnasium environment ID (default: %(default)s)")
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
    parser.add_argument("--ent-coef", default=DEFAULT_CONFIG.ent_coef,
                        help="SAC entropy coefficient: 'auto' or float (e.g. 0.05)")


def _add_fast_args(parser: argparse.ArgumentParser) -> None:
    """Add --fast mode arguments (shared by pixel training and eval)."""
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Enable fast mode: native 84x84 render, SubprocVecEnv, "
                             "more envs (12), gradient_steps=3")
    parser.add_argument("--gradient-steps", type=int, default=None,
                        help="SAC gradient steps per env step (default: 1, fast: 3)")


def _add_norm_args(parser: argparse.ArgumentParser) -> None:
    """Add feature normalization and frame stacking arguments."""
    parser.add_argument("--norm", action="store_true", default=False,
                        help="Use NormalizedCombinedExtractor (LayerNorm+Tanh on CNN output)")
    parser.add_argument("--cnn-dim", type=int, default=None,
                        help="CNN output dimension (default: 50 with --norm, 256 without)")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Number of frames to stack (1=none, 4=standard Atari-style)")


def _add_push_args(parser: argparse.ArgumentParser) -> None:
    """Add Push-specific arguments."""
    parser.add_argument("--push-state-total-steps", type=int,
                        default=DEFAULT_CONFIG.push_state_total_steps,
                        help="Total training steps for Push state (no HER)")
    parser.add_argument("--push-her-total-steps", type=int,
                        default=DEFAULT_CONFIG.push_her_total_steps,
                        help="Total training steps for Push state + HER")
    parser.add_argument("--push-pixel-total-steps", type=int,
                        default=DEFAULT_CONFIG.push_pixel_total_steps,
                        help="Total training steps for Push pixel + HER")
    parser.add_argument("--push-pixel-buffer-size", type=int,
                        default=DEFAULT_CONFIG.push_pixel_buffer_size,
                        help="Replay buffer size for Push pixel modes")
    parser.add_argument("--push-her-n-sampled-goal", type=int,
                        default=DEFAULT_CONFIG.push_her_n_sampled_goal,
                        help="HER n_sampled_goal for Push")
    parser.add_argument("--push-ent-coef", default=DEFAULT_CONFIG.push_ent_coef,
                        help="SAC entropy coef for sparse Push (default: 0.05 fixed)")


def _config_from_args(args: argparse.Namespace) -> Ch10Config:
    """Build Ch10Config from parsed args.

    When --fast is set, we apply speed-optimized defaults for fields the user
    did not explicitly override.  The detection logic:
    - ``--gradient-steps`` uses ``default=None`` as a sentinel.  If None and
      fast, we set it to 3; if None and not fast, we set it to 1.
    - ``--pixel-n-envs`` uses the dataclass default (4).  If the user didn't
      pass it (value still 4) and fast, we bump to 12.
    """
    cfg = Ch10Config()
    for attr in vars(cfg):
        arg_name = attr.replace("-", "_")
        if hasattr(args, arg_name):
            val = getattr(args, arg_name)
            if val is not None:
                setattr(cfg, attr, val)

    # Resolve --fast defaults
    fast = getattr(args, "fast", False)
    if fast:
        cfg.fast = True
        cfg.native_render = True
        cfg.use_subproc = True

        # Bump pixel_n_envs only if user didn't explicitly set it
        pixel_n_envs_from_cli = getattr(args, "pixel_n_envs", None)
        if pixel_n_envs_from_cli == DEFAULT_CONFIG.pixel_n_envs:
            cfg.pixel_n_envs = 12

    # Resolve gradient_steps sentinel (None -> default)
    gs = getattr(args, "gradient_steps", None)
    if gs is not None:
        cfg.gradient_steps = gs
    elif fast:
        cfg.gradient_steps = 3
    # else: keep dataclass default (1)

    # Resolve --norm and --cnn-dim
    norm = getattr(args, "norm", False)
    if norm:
        cfg.norm = True
    cnn_dim = getattr(args, "cnn_dim", None)
    if cnn_dim is not None:
        cfg.cnn_dim = cnn_dim
    elif norm:
        cfg.cnn_dim = 50  # DrQ-v2 default

    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter 10: Pixels, No Cheating -- Visual SAC on Fetch Tasks",
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
    _add_fast_args(p_pixel)

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
    _add_fast_args(p_drq)

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
    _add_fast_args(p_eval)

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
    _add_fast_args(p_all)

    # --- Push subcommands ---

    # train-push-state
    p_push_state = sub.add_parser("train-push-state",
                                   help="Train SAC on FetchPushDense (no HER, diagnostic baseline)",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_push_state)
    _add_train_args(p_push_state)
    _add_push_args(p_push_state)
    p_push_state.add_argument("--state-n-envs", type=int, default=DEFAULT_CONFIG.state_n_envs)
    p_push_state.add_argument("--state-buffer-size", type=int, default=DEFAULT_CONFIG.state_buffer_size)

    # train-push-her
    p_push_her = sub.add_parser("train-push-her",
                                 help="Train SAC+HER on FetchPush (sparse, state obs)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_push_her)
    _add_train_args(p_push_her)
    _add_push_args(p_push_her)
    p_push_her.add_argument("--state-n-envs", type=int, default=DEFAULT_CONFIG.state_n_envs)
    p_push_her.add_argument("--state-buffer-size", type=int, default=DEFAULT_CONFIG.state_buffer_size)

    # train-push-pixel-her
    p_push_pixel = sub.add_parser("train-push-pixel-her",
                                   help="Train SAC+HER on FetchPush from pixels (goal_mode=both)",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_push_pixel)
    _add_train_args(p_push_pixel)
    _add_push_args(p_push_pixel)
    p_push_pixel.add_argument("--pixel-n-envs", type=int, default=DEFAULT_CONFIG.pixel_n_envs)
    p_push_pixel.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size)
    _add_norm_args(p_push_pixel)
    _add_fast_args(p_push_pixel)

    # train-push-pixel-her-drq
    p_push_drq = sub.add_parser("train-push-pixel-her-drq",
                                 help="Train SAC+HER+DrQ on FetchPush from pixels (the synthesis)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_push_drq)
    _add_train_args(p_push_drq)
    _add_push_args(p_push_drq)
    p_push_drq.add_argument("--pixel-n-envs", type=int, default=DEFAULT_CONFIG.pixel_n_envs)
    p_push_drq.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size)
    p_push_drq.add_argument("--drq-pad", type=int, default=DEFAULT_CONFIG.drq_pad)
    _add_norm_args(p_push_drq)
    _add_fast_args(p_push_drq)

    # push-compare
    p_push_cmp = sub.add_parser("push-compare",
                                 help="Compare Push experiments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_push_cmp)
    p_push_cmp.add_argument("--n-eval-episodes", type=int, default=DEFAULT_CONFIG.n_eval_episodes)

    # push-all
    p_push_all = sub.add_parser("push-all",
                                 help="Full Push pipeline: 4 configs + eval + compare",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_push_all)
    _add_train_args(p_push_all)
    _add_push_args(p_push_all)
    p_push_all.add_argument("--state-n-envs", type=int, default=DEFAULT_CONFIG.state_n_envs)
    p_push_all.add_argument("--state-buffer-size", type=int, default=DEFAULT_CONFIG.state_buffer_size)
    p_push_all.add_argument("--pixel-n-envs", type=int, default=DEFAULT_CONFIG.pixel_n_envs)
    p_push_all.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size)
    p_push_all.add_argument("--drq-pad", type=int, default=DEFAULT_CONFIG.drq_pad)
    p_push_all.add_argument("--n-eval-episodes", type=int, default=DEFAULT_CONFIG.n_eval_episodes)
    _add_norm_args(p_push_all)
    _add_fast_args(p_push_all)

    args = parser.parse_args()

    # Set MUJOCO_GL
    pixel_cmds = {
        "train-pixel", "train-pixel-drq", "all",
        "train-push-pixel-her", "train-push-pixel-her-drq", "push-all",
    }
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    elif args.cmd in pixel_cmds:
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
    elif args.cmd == "train-push-state":
        return cmd_train_push_state(cfg)
    elif args.cmd == "train-push-her":
        return cmd_train_push_her(cfg)
    elif args.cmd == "train-push-pixel-her":
        return cmd_train_push_pixel_her(cfg)
    elif args.cmd == "train-push-pixel-her-drq":
        return cmd_train_push_pixel_her_drq(cfg)
    elif args.cmd == "push-compare":
        return cmd_push_compare(cfg)
    elif args.cmd == "push-all":
        return cmd_push_all(cfg)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
