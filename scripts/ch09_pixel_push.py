#!/usr/bin/env python3
"""Chapter 9: Cracking Pixel Push with Manipulation-Appropriate Architecture

After four rounds of 8M-step experiments, every NatureCNN configuration fails
at 5-8% success on FetchPush from pixels, while state+HER reaches 99% at
1.86M steps. Root cause: NatureCNN's 8x8 stride-4 first layer destroys the
spatial information the policy needs.

This script implements the fix: ManipulationCNN (3x3 kernels) + SpatialSoftmax
(explicit coordinate extraction) + proprioception (sensor separation principle)
+ DrQ augmentation + HER.

Usage:
    # Verify all components work together (10K steps)
    python scripts/ch09_pixel_push.py smoke-test --seed 0

    # Full training run (8M steps, overnight)
    python scripts/ch09_pixel_push.py train --seed 0

    # Evaluate checkpoint
    python scripts/ch09_pixel_push.py eval --ckpt checkpoints/ch09_manip_FetchPush-v4_seed0.zip

    # Compare results across configs
    python scripts/ch09_pixel_push.py compare

    # DrQ-v2 gradient routing: critic updates shared encoder
    python scripts/ch09_pixel_push.py train --seed 0 --critic-encoder

    # Full-state control (validate pipeline, should match Ch4 results)
    python scripts/ch09_pixel_push.py train --seed 0 --full-state --total-steps 2000000

    # Resume training from checkpoint (--total-steps = new target total)
    python scripts/ch09_pixel_push.py train --seed 0 --critic-encoder --no-drq \
        --resume checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed0.zip \
        --total-steps 8000000

    # Ablation runs (test which interventions matter)
    python scripts/ch09_pixel_push.py train --seed 0 --no-spatial-softmax
    python scripts/ch09_pixel_push.py train --seed 0 --no-proprio
    python scripts/ch09_pixel_push.py train --seed 0 --no-drq
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `from scripts.labs...` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Configuration
# =============================================================================

# FetchPush-v4 observation vector (25D):
# [0:3]   grip_pos, [3:6] object_pos, [6:9] object_rel_pos,
# [9:11]  gripper_state, [11:14] object_rot,
# [14:17] object_velp, [17:20] object_velr,
# [20:23] grip_velp, [23:25] gripper_vel
#
# Robot-only (proprioception): grip_pos + gripper_state + grip_velp + gripper_vel
PUSH_PROPRIO_INDICES = [0, 1, 2, 9, 10, 20, 21, 22, 23, 24]  # 10D


@dataclass
class Ch09Config:
    """Configuration for Chapter 9 pixel Push experiments.

    Defaults encode the research-backed architecture:
    - ManipulationCNN (3x3 kernels, stride 2) + SpatialSoftmax
    - Proprioception passthrough (10D robot state)
    - DrQ random shift augmentation
    - Frame stacking (4 frames -> 12 channels)
    - HER with future goal relabeling
    """
    # Environment
    env: str = "FetchPush-v4"
    seed: int = 0
    device: str = "auto"

    # Architecture
    spatial_softmax: bool = True       # Use SpatialSoftmax (vs flatten+linear)
    proprio: bool = True               # Include proprioception
    proprio_indices: list[int] = field(default_factory=lambda: list(PUSH_PROPRIO_INDICES))
    num_filters: int = 32              # CNN channels
    flat_features_dim: int = 50        # Output dim when spatial_softmax=False

    # Pixel pipeline
    image_size: int = 84
    frame_stack: int = 4               # 4-frame stack -> 12ch input
    pixels: bool = True                # Pixels enabled (vs vector-only baseline)
    native_render: bool = False        # Render at 84x84 natively (skip PIL resize)
    drq: bool = True                   # DrQ random shift augmentation
    drq_pad: int = 4

    # Training
    n_envs: int = 4
    total_steps: int = 8_000_000
    buffer_size: int = 200_000
    batch_size: int = 256
    learning_starts: int = 1_000
    learning_rate: float = 1e-4        # 1e-4 not 3e-4 (pixel training needs lower lr)
    gamma: float = 0.95
    tau: float = 0.005
    ent_coef: str = "0.05"             # Fixed (auto-tuning collapses on sparse)
    gradient_steps: int = 1
    share_encoder: bool = False        # Share features extractor between actor/critic
    critic_encoder: bool = False       # DrQ-v2 pattern: critic updates shared encoder
    full_state: bool = False           # Full-state control (no pixels, use raw 25D obs)

    # HER
    her_n_sampled_goal: int = 4
    her_goal_strategy: str = "future"

    # Paths
    log_dir: str = "runs"
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"

    # Resume
    resume: str = ""  # Path to checkpoint .zip to resume training from

    # Evaluation
    n_eval_episodes: int = 100
    eval_deterministic: bool = True


DEFAULT_CONFIG = Ch09Config()


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


def _parse_ent_coef(val: str) -> str | float:
    """Parse ent_coef: 'auto' stays as string, numbers become float."""
    if val == "auto":
        return "auto"
    return float(val)


def _meta_path(ckpt: Path) -> Path:
    return ckpt.parent / (ckpt.stem + ".meta.json")


def _config_tag(cfg: Ch09Config) -> str:
    """Short tag encoding config choices for filenames."""
    parts = ["ch09_manip"]
    if not cfg.pixels:
        parts.append("noPix")
    if cfg.pixels and cfg.native_render:
        parts.append("native")
    if cfg.pixels and not cfg.spatial_softmax:
        parts.append("noSS")
    if not cfg.proprio:
        parts.append("noPro")
    if cfg.pixels and not cfg.drq:
        parts.append("noDrQ")
    if cfg.critic_encoder:
        parts.append("criticEnc")
    elif cfg.share_encoder:
        parts.append("shareEnc")
    if cfg.full_state:
        parts.append("fullState")
    if cfg.frame_stack != 4:
        parts.append(f"fs{cfg.frame_stack}")
    return "_".join(parts)


def _ckpt_path(cfg: Ch09Config) -> Path:
    tag = _config_tag(cfg)
    return _ensure_dir(cfg.checkpoints_dir) / f"{tag}_{cfg.env}_seed{cfg.seed}.zip"


def _result_path(cfg: Ch09Config) -> Path:
    tag = _config_tag(cfg)
    return _ensure_dir(cfg.results_dir) / f"{tag}_{cfg.env}_seed{cfg.seed}_eval.json"


# =============================================================================
# Environment Factory
# =============================================================================

def make_pixel_push_env(cfg: Ch09Config):
    """Create a pixel-wrapped FetchPush environment.

    Returns a single (non-vectorized) environment with:
    - PixelObservationWrapper (goal_mode="both", frame_stack, optional proprio)
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
    if cfg.native_render:
        make_kwargs["width"] = cfg.image_size
        make_kwargs["height"] = cfg.image_size
    env = gym.make(cfg.env, **make_kwargs)
    env = PixelObservationWrapper(
        env,
        image_size=(cfg.image_size, cfg.image_size),
        goal_mode="both",
        frame_stack=cfg.frame_stack,
        proprio_indices=cfg.proprio_indices if cfg.proprio else None,
    )
    return env


def make_vector_push_env(cfg: Ch09Config):
    """Create a vector-only Push env: goals + optional proprioception (no pixels)."""
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import numpy as np

    class ProprioGoalWrapper(gym.ObservationWrapper):
        def __init__(self, env: gym.Env, proprio_indices: list[int] | None):
            super().__init__(env)
            self._proprio_indices = proprio_indices

            old_spaces = env.observation_space.spaces
            new_spaces: dict[str, gym.Space] = {
                "desired_goal": old_spaces["desired_goal"],
                "achieved_goal": old_spaces["achieved_goal"],
            }
            if proprio_indices is not None:
                obs_space = old_spaces["observation"]
                low = np.asarray(obs_space.low)[proprio_indices]
                high = np.asarray(obs_space.high)[proprio_indices]
                new_spaces["proprioception"] = gym.spaces.Box(low=low, high=high, dtype=np.float64)

            self.observation_space = gym.spaces.Dict(new_spaces)

        def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {
                "desired_goal": observation["desired_goal"],
                "achieved_goal": observation["achieved_goal"],
            }
            if self._proprio_indices is not None:
                out["proprioception"] = observation["observation"][self._proprio_indices]
            return out

    env = gym.make(cfg.env)
    proprio_idx = cfg.proprio_indices if cfg.proprio else None
    env = ProprioGoalWrapper(env, proprio_idx)
    return env


def make_full_state_env(cfg: Ch09Config):
    """Create a full-state Push env with all 25D + goals (no pixel wrapper).

    This matches Ch4's pipeline exactly: the policy sees the raw observation
    vector, achieved_goal, and desired_goal. Used as a control to validate
    the Ch9 algorithm wiring independently of pixel processing.
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    return gym.make(cfg.env)


# =============================================================================
# Commands
# =============================================================================

def cmd_smoke_test(cfg: Ch09Config) -> int:
    """Quick 10K-step test to verify all components wire together.

    Checks:
    1. Pixel wrapper produces correct observation dict
    2. ManipulationExtractor has expected features_dim
    3. SAC + HER + DrQ training runs without crashes
    4. Q-loss is finite after 10K steps
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    if not cfg.full_state:
        from scripts.labs.manipulation_encoder import ManipulationExtractor

    device = _resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    smoke_steps = 10_000
    print("=" * 60)
    print("Ch09 Pixel Push -- Smoke Test")
    print("=" * 60)
    print(f"  env={cfg.env}, seed={cfg.seed}, device={device}")
    print(f"  pixels={cfg.pixels}" + (f" (native_render={cfg.native_render})" if cfg.pixels else ""))
    print(f"  spatial_softmax={cfg.spatial_softmax}")
    print(f"  proprio={cfg.proprio} ({len(cfg.proprio_indices)}D)" if cfg.proprio else f"  proprio={cfg.proprio}")
    print(f"  drq={cfg.drq and cfg.pixels}, frame_stack={cfg.frame_stack}")
    print(f"  share_encoder={cfg.share_encoder}, critic_encoder={cfg.critic_encoder}")
    print(f"  full_state={cfg.full_state}")
    print(f"  steps={smoke_steps}")
    print()

    # 1. Check single env observation space
    print("[1/4] Checking observation space...")
    if cfg.full_state:
        test_env = make_full_state_env(cfg)
    elif cfg.pixels:
        test_env = make_pixel_push_env(cfg)
    else:
        test_env = make_vector_push_env(cfg)
    obs, _ = test_env.reset()
    obs_keys = sorted(obs.keys())
    print(f"  Obs keys: {obs_keys}")
    for k, v in obs.items():
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
    test_env.close()

    # 2. Check extractor (skip for full-state -- uses SB3's default CombinedExtractor)
    n_params = 0
    if not cfg.full_state:
        print("\n[2/4] Checking ManipulationExtractor...")
        if cfg.pixels:
            test_env2 = make_pixel_push_env(cfg)
        else:
            test_env2 = make_vector_push_env(cfg)
        ext = ManipulationExtractor(
            test_env2.observation_space,
            spatial_softmax=cfg.spatial_softmax,
            num_filters=cfg.num_filters,
            flat_features_dim=cfg.flat_features_dim,
        )
        print(f"  features_dim: {ext.features_dim}")
        n_params = sum(p.numel() for p in ext.parameters())
        print(f"  parameters: {n_params:,}")
        test_env2.close()
    else:
        print("\n[2/4] Full-state mode: using SB3 default extractor (skip ManipulationExtractor)")

    # 3. Build SAC model and train
    print(f"\n[3/4] Training SAC for {smoke_steps} steps...")

    img_sz = cfg.image_size
    fstack = cfg.frame_stack
    proprio_idx = cfg.proprio_indices if cfg.proprio else None

    def _make_env():
        import gymnasium_robotics  # noqa: F401
        from scripts.labs.pixel_wrapper import PixelObservationWrapper as _Wrapper

        if cfg.full_state:
            return make_full_state_env(cfg)
        if not cfg.pixels:
            return make_vector_push_env(cfg)
        make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
        if cfg.native_render:
            make_kwargs["width"] = img_sz
            make_kwargs["height"] = img_sz
        e = gym.make(cfg.env, **make_kwargs)
        return _Wrapper(
            e,
            image_size=(img_sz, img_sz),
            goal_mode="both",
            frame_stack=fstack,
            proprio_indices=proprio_idx,
        )

    env = make_vec_env(_make_env, n_envs=2, seed=cfg.seed)

    # Policy class and kwargs depend on mode
    if cfg.full_state:
        # Full-state: standard MultiInputPolicy, no custom extractor
        policy_class: Any = "MultiInputPolicy"
        policy_kwargs: dict[str, Any] = {}
    elif cfg.critic_encoder:
        # DrQ-v2 pattern: custom policy handles sharing internally
        from scripts.labs.drqv2_sac_policy import DrQv2SACPolicy
        policy_class = DrQv2SACPolicy
        policy_kwargs = {
            "features_extractor_class": ManipulationExtractor,
            "features_extractor_kwargs": {
                "spatial_softmax": cfg.spatial_softmax,
                "num_filters": cfg.num_filters,
                "flat_features_dim": cfg.flat_features_dim,
            },
            # share_features_extractor not needed -- DrQv2SACPolicy forces True
        }
    else:
        policy_class = "MultiInputPolicy"
        policy_kwargs = {
            "features_extractor_class": ManipulationExtractor,
            "features_extractor_kwargs": {
                "spatial_softmax": cfg.spatial_softmax,
                "num_filters": cfg.num_filters,
                "flat_features_dim": cfg.flat_features_dim,
            },
            "share_features_extractor": cfg.share_encoder,
        }

    # Choose replay buffer
    if cfg.pixels and cfg.drq and not cfg.full_state:
        from scripts.labs.image_augmentation import HerDrQDictReplayBuffer, RandomShiftAug
        replay_cls = HerDrQDictReplayBuffer
        replay_kwargs: dict[str, Any] = {
            "aug_fn": RandomShiftAug(pad=cfg.drq_pad),
            "image_key": "pixels",
            "n_sampled_goal": cfg.her_n_sampled_goal,
            "goal_selection_strategy": cfg.her_goal_strategy,
        }
    else:
        try:
            from stable_baselines3 import HerReplayBuffer
        except ImportError:
            from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
        replay_cls = HerReplayBuffer
        replay_kwargs = {
            "n_sampled_goal": cfg.her_n_sampled_goal,
            "goal_selection_strategy": cfg.her_goal_strategy,
        }

    model = SAC(
        policy_class,
        env,
        verbose=1,
        device=device,
        buffer_size=5_000,
        learning_starts=500,
        batch_size=32,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        tau=cfg.tau,
        ent_coef=_parse_ent_coef(cfg.ent_coef),
        gradient_steps=cfg.gradient_steps,
        policy_kwargs=policy_kwargs,
        replay_buffer_class=replay_cls,
        replay_buffer_kwargs=replay_kwargs,
    )

    if cfg.critic_encoder:
        from scripts.labs.drqv2_sac_policy import DrQv2SACPolicy as _DPolicy
        print(f"  Policy: {type(model.policy).__name__}")
        assert isinstance(model.policy, _DPolicy), (
            f"Expected DrQv2SACPolicy, got {type(model.policy).__name__}"
        )

    t0 = time.perf_counter()
    model.learn(total_timesteps=smoke_steps)
    elapsed = time.perf_counter() - t0
    env.close()

    # 4. Summary
    print(f"\n[4/4] Smoke test summary")
    if cfg.full_state:
        print(f"  Mode: full-state (raw 25D observation)")
    elif cfg.critic_encoder:
        print(f"  Mode: critic-encoder (DrQ-v2 gradient routing)")
        print(f"  Encoder in critic optimizer: YES")
        print(f"  Encoder in actor optimizer: NO (features detached)")
    else:
        print(f"  Feature extractor: ManipulationExtractor")
    if n_params > 0:
        print(f"  Extractor params: ~{n_params:,}")
    print(f"  {smoke_steps} steps in {elapsed:.1f}s ({smoke_steps/elapsed:.0f} steps/sec)")
    print()
    print("=" * 60)
    print("[PASS] Smoke test completed successfully")
    print("=" * 60)
    return 0


def cmd_train(cfg: Ch09Config) -> int:
    """Full training run: SAC + HER with configurable encoder and gradient routing."""
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    if not cfg.full_state:
        from scripts.labs.manipulation_encoder import ManipulationExtractor

    device = _resolve_device(cfg.device)
    set_random_seed(cfg.seed)

    tag = _config_tag(cfg)
    print("=" * 60)
    print(f"Ch09 Pixel Push -- Training ({tag})")
    print("=" * 60)
    print(f"  env={cfg.env}, seed={cfg.seed}, device={device}")
    print(f"  total_steps={cfg.total_steps:,}, n_envs={cfg.n_envs}")
    print(f"  pixels={cfg.pixels}" + (f" (native_render={cfg.native_render})" if cfg.pixels else ""))
    print(f"  spatial_softmax={cfg.spatial_softmax}, proprio={cfg.proprio}")
    print(f"  drq={cfg.drq and cfg.pixels} (pad={cfg.drq_pad}), frame_stack={cfg.frame_stack}")
    print(f"  lr={cfg.learning_rate}, ent_coef={cfg.ent_coef}, gamma={cfg.gamma}")
    print(f"  HER: strategy={cfg.her_goal_strategy}, n_sampled_goal={cfg.her_n_sampled_goal}")
    print(f"  buffer_size={cfg.buffer_size:,}, batch_size={cfg.batch_size}")
    print(f"  share_encoder={cfg.share_encoder}, critic_encoder={cfg.critic_encoder}")
    print(f"  full_state={cfg.full_state}")
    print()

    img_sz = cfg.image_size
    fstack = cfg.frame_stack
    proprio_idx = cfg.proprio_indices if cfg.proprio else None

    def _make_env():
        import gymnasium_robotics  # noqa: F401
        from scripts.labs.pixel_wrapper import PixelObservationWrapper as _Wrapper

        if cfg.full_state:
            return make_full_state_env(cfg)
        if not cfg.pixels:
            return make_vector_push_env(cfg)
        make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
        if cfg.native_render:
            make_kwargs["width"] = img_sz
            make_kwargs["height"] = img_sz
        e = gym.make(cfg.env, **make_kwargs)
        return _Wrapper(
            e,
            image_size=(img_sz, img_sz),
            goal_mode="both",
            frame_stack=fstack,
            proprio_indices=proprio_idx,
        )

    env = make_vec_env(_make_env, n_envs=cfg.n_envs, seed=cfg.seed)

    log_dir = _ensure_dir(cfg.log_dir)
    run_id = f"{tag}/{cfg.env}/seed{cfg.seed}"

    # Policy class and kwargs depend on mode
    if cfg.full_state:
        policy_class: Any = "MultiInputPolicy"
        policy_kwargs: dict[str, Any] = {}
    elif cfg.critic_encoder:
        from scripts.labs.drqv2_sac_policy import DrQv2SACPolicy
        policy_class = DrQv2SACPolicy
        policy_kwargs = {
            "features_extractor_class": ManipulationExtractor,
            "features_extractor_kwargs": {
                "spatial_softmax": cfg.spatial_softmax,
                "num_filters": cfg.num_filters,
                "flat_features_dim": cfg.flat_features_dim,
            },
        }
    else:
        policy_class = "MultiInputPolicy"
        policy_kwargs = {
            "features_extractor_class": ManipulationExtractor,
            "features_extractor_kwargs": {
                "spatial_softmax": cfg.spatial_softmax,
                "num_filters": cfg.num_filters,
                "flat_features_dim": cfg.flat_features_dim,
            },
            "share_features_extractor": cfg.share_encoder,
        }

    if cfg.pixels and cfg.drq and not cfg.full_state:
        from scripts.labs.image_augmentation import HerDrQDictReplayBuffer, RandomShiftAug
        replay_cls = HerDrQDictReplayBuffer
        replay_kwargs: dict[str, Any] = {
            "aug_fn": RandomShiftAug(pad=cfg.drq_pad),
            "image_key": "pixels",
            "n_sampled_goal": cfg.her_n_sampled_goal,
            "goal_selection_strategy": cfg.her_goal_strategy,
        }
    else:
        try:
            from stable_baselines3 import HerReplayBuffer
        except ImportError:
            from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
        replay_cls = HerReplayBuffer
        replay_kwargs = {
            "n_sampled_goal": cfg.her_n_sampled_goal,
            "goal_selection_strategy": cfg.her_goal_strategy,
        }

    try:
        if cfg.resume:
            # Resume from checkpoint -- load trained weights, fresh buffer
            load_kwargs: dict[str, Any] = {}
            if cfg.critic_encoder:
                from scripts.labs.drqv2_sac_policy import DrQv2SACPolicy
                load_kwargs["custom_objects"] = {"policy_class": DrQv2SACPolicy}

            model = SAC.load(
                cfg.resume, env=env, device=device,
                tensorboard_log=str(log_dir),
                **load_kwargs,
            )

            start_steps = model.num_timesteps
            remaining = cfg.total_steps - start_steps
            if remaining <= 0:
                print(f"[ch09] Checkpoint already at {start_steps:,} steps, "
                      f"target is {cfg.total_steps:,}. Nothing to do.")
                return 0

            # Fresh buffer needs warm-up: shift learning_starts past current
            # num_timesteps so SB3 collects transitions before training.
            # HER requires at least one complete episode (50 steps × n_envs).
            model.learning_starts = start_steps + cfg.learning_starts

            print(f"[ch09] Resuming from {start_steps:,} steps")
            print(f"[ch09] Target: {cfg.total_steps:,} ({remaining:,} remaining)")
            print(f"[ch09] Buffer warm-up: {cfg.learning_starts:,} steps before training")
            print()

            t0 = time.perf_counter()
            model.learn(
                total_timesteps=remaining,
                tb_log_name=run_id,
                reset_num_timesteps=False,
            )
            elapsed = time.perf_counter() - t0
            trained_steps = model.num_timesteps - start_steps
            fps = trained_steps / elapsed if elapsed > 0 else 0

        else:
            model = SAC(
                policy_class,
                env,
                verbose=1,
                device=device,
                tensorboard_log=str(log_dir),
                batch_size=cfg.batch_size,
                buffer_size=cfg.buffer_size,
                learning_starts=cfg.learning_starts,
                learning_rate=cfg.learning_rate,
                gamma=cfg.gamma,
                tau=cfg.tau,
                ent_coef=_parse_ent_coef(cfg.ent_coef),
                gradient_steps=cfg.gradient_steps,
                policy_kwargs=policy_kwargs,
                replay_buffer_class=replay_cls,
                replay_buffer_kwargs=replay_kwargs,
            )

            t0 = time.perf_counter()
            model.learn(total_timesteps=cfg.total_steps, tb_log_name=run_id)
            elapsed = time.perf_counter() - t0
            trained_steps = cfg.total_steps
            fps = trained_steps / elapsed if elapsed > 0 else 0

        ckpt = _ckpt_path(cfg)
        model.save(str(ckpt))
        print(f"\n[ch09] Saved: {ckpt}")
        print(f"[ch09] Training time: {elapsed:.1f}s ({fps:.0f} steps/sec)")

    finally:
        env.close()

    # Metadata
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 9,
        "config_tag": tag,
        "algo": "sac",
        "env_id": cfg.env,
        "seed": cfg.seed,
        "device": device,
        "n_envs": cfg.n_envs,
        "total_steps": model.num_timesteps,
        "resumed_from": cfg.resume if cfg.resume else None,
        "training_time_sec": elapsed,
        "steps_per_sec": fps,
        "architecture": {
            "encoder": "full_state" if cfg.full_state else "ManipulationCNN",
            "spatial_softmax": cfg.spatial_softmax,
            "num_filters": cfg.num_filters,
            "proprio": cfg.proprio,
            "proprio_dim": len(cfg.proprio_indices) if cfg.proprio else 0,
            "frame_stack": cfg.frame_stack,
            "drq": cfg.drq and not cfg.full_state,
            "drq_pad": cfg.drq_pad if (cfg.drq and not cfg.full_state) else None,
            "critic_encoder": cfg.critic_encoder,
            "full_state": cfg.full_state,
        },
        "her": {
            "n_sampled_goal": cfg.her_n_sampled_goal,
            "goal_strategy": cfg.her_goal_strategy,
        },
        "hyperparams": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.ent_coef,
            "gradient_steps": cfg.gradient_steps,
        },
        "checkpoint": str(ckpt),
        "versions": _gather_versions(),
    }
    meta_file = _meta_path(ckpt)
    meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ch09] Wrote metadata: {meta_file}")

    return 0


def cmd_eval(cfg: Ch09Config, ckpt_path: str, pixel: bool = True) -> int:
    """Evaluate a checkpoint on FetchPush."""
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import numpy as np
    from stable_baselines3 import SAC

    device = _resolve_device(cfg.device)

    print("=" * 60)
    print("Ch09 Pixel Push -- Evaluation")
    print("=" * 60)
    print(f"  ckpt={ckpt_path}")
    print(f"  env={cfg.env}, n_episodes={cfg.n_eval_episodes}")
    print(f"  pixel={pixel}, deterministic={cfg.eval_deterministic}")
    print(f"  critic_encoder={cfg.critic_encoder}, full_state={cfg.full_state}")
    print()

    if cfg.full_state:
        env = make_full_state_env(cfg)
    elif pixel:
        env = make_pixel_push_env(cfg) if cfg.pixels else make_vector_push_env(cfg)
    else:
        env = gym.make(cfg.env)

    # DrQv2SACPolicy checkpoints need custom_objects so SB3 can find the class
    load_kwargs: dict[str, Any] = {}
    if cfg.critic_encoder:
        from scripts.labs.drqv2_sac_policy import DrQv2SACPolicy
        load_kwargs["custom_objects"] = {"policy_class": DrQv2SACPolicy}

    model = SAC.load(ckpt_path, env=env, device=device, **load_kwargs)

    successes = []
    returns = []
    goal_dists = []
    times_to_success = []

    for ep in range(cfg.n_eval_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        done = False
        step_count = 0
        first_success_step = None

        while not done:
            action, _ = model.predict(obs, deterministic=cfg.eval_deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            step_count += 1
            done = terminated or truncated

            if info.get("is_success", False) and first_success_step is None:
                first_success_step = step_count

        successes.append(float(info.get("is_success", False)))
        returns.append(ep_return)

        # Goal distance from last raw obs
        if hasattr(env, "last_raw_obs") and env.last_raw_obs is not None:
            ag = env.last_raw_obs["achieved_goal"]
            dg = env.last_raw_obs["desired_goal"]
            goal_dists.append(float(np.linalg.norm(ag - dg)))

        if first_success_step is not None:
            times_to_success.append(first_success_step)

    env.close()

    success_rate = np.mean(successes)
    mean_return = np.mean(returns)
    mean_goal_dist = np.mean(goal_dists) if goal_dists else float("nan")
    mean_tts = np.mean(times_to_success) if times_to_success else float("nan")

    print(f"  Success rate: {success_rate:.1%} ({int(sum(successes))}/{cfg.n_eval_episodes})")
    print(f"  Mean return:  {mean_return:.2f}")
    print(f"  Mean goal dist: {mean_goal_dist:.4f}m")
    print(f"  Mean time-to-success: {mean_tts:.1f} steps" if times_to_success
          else "  Mean time-to-success: N/A (no successes)")

    # Save results
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chapter": 9,
        "checkpoint": ckpt_path,
        "env_id": cfg.env,
        "n_episodes": cfg.n_eval_episodes,
        "deterministic": cfg.eval_deterministic,
        "pixel": pixel,
        "metrics": {
            "success_rate": success_rate,
            "mean_return": mean_return,
            "mean_goal_distance": mean_goal_dist,
            "mean_time_to_success": mean_tts,
            "n_successes": int(sum(successes)),
        },
        "versions": _gather_versions(),
    }

    out_path = _result_path(cfg)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Saved: {out_path}")

    return 0


def cmd_compare(cfg: Ch09Config) -> int:
    """Compare results from different configs (reads eval JSON files)."""
    print("=" * 60)
    print("Ch09 Pixel Push -- Comparison")
    print("=" * 60)

    results_dir = Path(cfg.results_dir)
    if not results_dir.exists():
        print("  No results directory found. Run eval first.")
        return 1

    result_files = sorted(results_dir.glob("ch09_*_eval.json"))
    if not result_files:
        print("  No ch09 result files found.")
        return 1

    print(f"\n  {'Config':<40} {'Success':>10} {'Return':>10} {'Goal Dist':>12}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*12}")

    for f in result_files:
        data = json.loads(f.read_text())
        m = data["metrics"]
        tag = f.stem.replace("_eval", "")
        print(f"  {tag:<40} {m['success_rate']:>9.1%} {m['mean_return']:>10.2f} "
              f"{m['mean_goal_distance']:>12.4f}m")

    print()
    print("=" * 60)
    return 0


# =============================================================================
# CLI
# =============================================================================

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", default=DEFAULT_CONFIG.env,
                        help="Gymnasium environment ID (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed,
                        help="Random seed")
    parser.add_argument("--device", default=DEFAULT_CONFIG.device,
                        help="Device: auto, cpu, cuda")
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CONFIG.checkpoints_dir)
    parser.add_argument("--results-dir", default=DEFAULT_CONFIG.results_dir)
    parser.add_argument("--log-dir", default=DEFAULT_CONFIG.log_dir)


def _add_arch_args(parser: argparse.ArgumentParser) -> None:
    """Architecture ablation flags."""
    parser.add_argument("--no-pixels", action="store_true",
                        help="Disable pixels entirely (vector-only proprio+goals baseline)")
    parser.add_argument("--no-spatial-softmax", action="store_true",
                        help="Use flatten+linear instead of SpatialSoftmax")
    parser.add_argument("--no-proprio", action="store_true",
                        help="Disable proprioception passthrough")
    parser.add_argument("--no-drq", action="store_true",
                        help="Disable DrQ augmentation")
    parser.add_argument("--native-render", action="store_true",
                        help="Render at 84x84 natively (skip PIL resize) [pixels mode only]")
    parser.add_argument("--share-encoder", action="store_true",
                        help="Share features extractor between actor/critic (SB3 default sharing)")
    parser.add_argument("--critic-encoder", action="store_true",
                        help="DrQ-v2 pattern: critic loss updates shared encoder, actor gets detached features")
    parser.add_argument("--full-state", action="store_true",
                        help="Full-state control: use raw 25D obs (no pixels) to validate pipeline")
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_CONFIG.frame_stack,
                        help="Number of frames to stack (default: %(default)s)")
    parser.add_argument("--num-filters", type=int, default=DEFAULT_CONFIG.num_filters,
                        help="CNN channel count (default: %(default)s)")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--total-steps", type=int, default=DEFAULT_CONFIG.total_steps,
                        help="Total training steps (default: %(default)s)")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_CONFIG.n_envs,
                        help="Number of parallel envs (default: %(default)s)")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_CONFIG.buffer_size,
                        help="Replay buffer size (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size,
                        help="Batch size (default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate,
                        help="Learning rate (default: %(default)s)")
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG.gamma,
                        help="Discount factor (default: %(default)s)")
    parser.add_argument("--ent-coef", default=DEFAULT_CONFIG.ent_coef,
                        help="Entropy coefficient: 'auto' or float (default: %(default)s)")
    parser.add_argument("--gradient-steps", type=int, default=DEFAULT_CONFIG.gradient_steps,
                        help="Gradient steps per env step (default: %(default)s)")
    parser.add_argument("--her-n-sampled-goal", type=int,
                        default=DEFAULT_CONFIG.her_n_sampled_goal,
                        help="HER n_sampled_goal (default: %(default)s)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume training from checkpoint .zip (--total-steps = target total)")


def _config_from_args(args: argparse.Namespace) -> Ch09Config:
    """Build Ch09Config from parsed arguments."""
    cfg = Ch09Config()

    # Map CLI args to config fields
    for attr in ["env", "seed", "device", "frame_stack", "num_filters",
                 "total_steps", "n_envs", "buffer_size", "batch_size",
                 "learning_rate", "gamma", "ent_coef", "gradient_steps",
                 "her_n_sampled_goal", "checkpoints_dir", "results_dir", "log_dir",
                 "native_render", "resume"]:
        arg_name = attr.replace("-", "_")
        if hasattr(args, arg_name):
            val = getattr(args, arg_name)
            if val is not None:
                setattr(cfg, attr, val)

    # Ablation flags (negated)
    if getattr(args, "no_pixels", False):
        cfg.pixels = False
    if getattr(args, "no_spatial_softmax", False):
        cfg.spatial_softmax = False
    if getattr(args, "no_proprio", False):
        cfg.proprio = False
    if getattr(args, "no_drq", False):
        cfg.drq = False
    if getattr(args, "share_encoder", False):
        cfg.share_encoder = True
    if getattr(args, "critic_encoder", False):
        cfg.critic_encoder = True
    if getattr(args, "full_state", False):
        cfg.full_state = True
        cfg.pixels = False  # full-state mode disables pixels

    return cfg


def cmd_feature_stats(cfg: Ch09Config, n_samples: int = 16) -> int:
    """Print per-key feature statistics for the current observation space."""
    import numpy as np
    import torch

    from scripts.labs.manipulation_encoder import ManipulationExtractor, SpatialSoftmax

    env = make_pixel_push_env(cfg) if cfg.pixels else make_vector_push_env(cfg)
    try:
        ext = ManipulationExtractor(
            env.observation_space,
            spatial_softmax=cfg.spatial_softmax,
            num_filters=cfg.num_filters,
            flat_features_dim=cfg.flat_features_dim,
        )

        obs_list: list[dict[str, np.ndarray]] = []
        obs, _ = env.reset(seed=cfg.seed)
        obs_list.append(obs)
        for _ in range(max(0, n_samples - 1)):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
            obs_list.append(obs)

        batch: dict[str, torch.Tensor] = {}
        for key in obs_list[0].keys():
            arr = np.stack([o[key] for o in obs_list], axis=0)
            t = torch.as_tensor(arr)
            if t.dtype != torch.uint8:
                t = t.float()
            batch[key] = t

        print("=" * 60)
        print("Ch09 Pixel Push -- Feature Stats")
        print("=" * 60)
        print(f"  env={cfg.env}, pixels={cfg.pixels}, n_samples={len(obs_list)}")
        if cfg.pixels:
            print(f"  native_render={cfg.native_render}, frame_stack={cfg.frame_stack}, drq={cfg.drq}")
        print(f"  spatial_softmax={cfg.spatial_softmax}, proprio={cfg.proprio}, share_encoder={cfg.share_encoder}")
        print()

        # SpatialSoftmax temperature (if present)
        ssm_mods = [m for m in ext.modules() if isinstance(m, SpatialSoftmax)]
        if ssm_mods:
            temps = [float(m.temperature.detach().cpu().item()) for m in ssm_mods]
            print(f"  SpatialSoftmax temperature: {temps}")

        # Per-key stats (post-subencoder)
        print("\n  Per-key encoder outputs:")
        for key, sub in ext.extractors.items():
            x = batch[key]
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            y = sub(x).detach().cpu()
            y_np = y.numpy()
            print(
                f"    {key:<14} dim={y_np.shape[1]:>4} "
                f"mean={y_np.mean():>8.4f} std={y_np.std():>8.4f} "
                f"min={y_np.min():>8.4f} max={y_np.max():>8.4f}"
            )

        # Full concatenated features
        feats = ext(batch).detach().cpu().numpy()
        print("\n  Concatenated features:")
        print(
            f"    dim={feats.shape[1]} mean={feats.mean():.4f} std={feats.std():.4f} "
            f"min={feats.min():.4f} max={feats.max():.4f}"
        )
        print("=" * 60)
        return 0
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ch09: Pixel Push with Manipulation-Appropriate Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (10K steps, ~2 min)
  python scripts/ch09_pixel_push.py smoke-test --seed 0

  # Full training (8M steps, overnight)
  python scripts/ch09_pixel_push.py train --seed 0

  # Vector-only baseline (no pixels rendered)
  python scripts/ch09_pixel_push.py train --seed 0 --no-pixels

  # Fast pixels (skip PIL resize)
  python scripts/ch09_pixel_push.py train --seed 0 --native-render

  # DrQ-v2 gradient routing: critic updates shared encoder
  python scripts/ch09_pixel_push.py train --seed 0 --critic-encoder

  # Full-state control (validate pipeline, match Ch4 results)
  python scripts/ch09_pixel_push.py train --seed 0 --full-state --total-steps 2000000

  # Ablation: no spatial softmax
  python scripts/ch09_pixel_push.py train --seed 0 --no-spatial-softmax

  # Evaluate checkpoint
  python scripts/ch09_pixel_push.py eval --ckpt checkpoints/ch09_manip_FetchPush-v4_seed0.zip

  # Compare results
  python scripts/ch09_pixel_push.py compare
        """
    )
    subparsers = parser.add_subparsers(dest="cmd")

    # smoke-test
    sp = subparsers.add_parser("smoke-test", help="10K-step wiring test")
    _add_common_args(sp)
    _add_arch_args(sp)

    # train
    sp = subparsers.add_parser("train", help="Full training run")
    _add_common_args(sp)
    _add_arch_args(sp)
    _add_train_args(sp)

    # eval
    sp = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    _add_common_args(sp)
    _add_arch_args(sp)
    sp.add_argument("--ckpt", required=True, help="Path to checkpoint .zip file")
    sp.add_argument("--n-eval-episodes", type=int, default=DEFAULT_CONFIG.n_eval_episodes)
    sp.add_argument("--no-pixel", action="store_true",
                    help="Evaluate as state-based (no pixel wrapper)")

    # compare
    sp = subparsers.add_parser("compare", help="Compare results across configs")
    _add_common_args(sp)

    # feature-stats
    sp = subparsers.add_parser("feature-stats", help="Print feature statistics")
    _add_common_args(sp)
    _add_arch_args(sp)
    sp.add_argument("--n-samples", type=int, default=16,
                    help="Number of random env steps to sample (default: %(default)s)")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return 1

    # Set MUJOCO_GL for pixel rendering
    needs_render = args.cmd in ("smoke-test", "train", "feature-stats") or (
        args.cmd == "eval" and not getattr(args, "no_pixel", False)
    )
    if needs_render and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    cfg = _config_from_args(args)

    if args.cmd == "smoke-test":
        return cmd_smoke_test(cfg)
    elif args.cmd == "train":
        return cmd_train(cfg)
    elif args.cmd == "eval":
        if hasattr(args, "n_eval_episodes"):
            cfg.n_eval_episodes = args.n_eval_episodes
        return cmd_eval(cfg, args.ckpt, pixel=not getattr(args, "no_pixel", False))
    elif args.cmd == "compare":
        return cmd_compare(cfg)
    elif args.cmd == "feature-stats":
        return cmd_feature_stats(cfg, n_samples=getattr(args, "n_samples", 16))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
