#!/usr/bin/env python3
"""Appendix E: Isaac Lab Manipulation Pipeline (book-style Run It script).

This script prepares a reproducible SAC (+ optional HER) pipeline for Appendix E
without coupling to MuJoCo/Gymnasium-Robotics assumptions.

Primary target: Isaac-Lift-Cube-Franka-v0 -- a manager-based manipulation task
that mirrors the book's FetchPush work (approach, grasp, lift, track). The same
SAC methodology from Chapters 3-4 transfers directly, with GPU-parallel physics
providing dramatic speedups (500-800 fps at 256 envs vs 30-50 fps on MuJoCo).

Scope boundary: Factory PegInsert is documented as a case study of method-problem
mismatch (POMDP requiring recurrence; see tutorial for analysis).

Design goals (aligned with the tutorial/book workflow):
1) Dense-first debugging: run a short smoke train on a known-easy Isaac env
   before long manipulation runs.
2) One-command reproducibility: train/eval artifacts are always written to
   checkpoints/*.zip + *.meta.json and results/*.json.
3) Isaac-safe boot order: initialize AppLauncher BEFORE importing/registering
   Isaac tasks, and keep one SimulationApp per process.

SB3 integration:
  We use Isaac Lab's official Sb3VecEnvWrapper (from isaaclab_rl.sb3) which:
    - Subclasses SB3's VecEnv directly (no DummyVecEnv wrapping)
    - Extracts the 'policy' obs key and converts tensors to numpy
    - Clips infinite action bounds to [-100, 100]
    - Supports batched envs (num_envs > 1 is possible)
  This is the same wrapper used by Isaac Lab's own training scripts.

Typical usage (inside Isaac container):
    # Discover available Isaac env IDs
    python3 scripts/appendix_e_isaac_peg.py discover-envs --headless

    # Dense-first smoke test on Lift-Cube
    python3 scripts/appendix_e_isaac_peg.py smoke --headless --seed 0 \
        --dense-env-id Isaac-Lift-Cube-Franka-v0

    # Train SAC on Lift-Cube (primary target)
    python3 scripts/appendix_e_isaac_peg.py train --headless --seed 0 \
        --env-id Isaac-Lift-Cube-Franka-v0 --num-envs 256 --total-steps 2000000

    # Train on any other Isaac env explicitly
    python3 scripts/appendix_e_isaac_peg.py train --headless --env-id Isaac-Reach-Franka-v0

    # Evaluate a checkpoint
    python3 scripts/appendix_e_isaac_peg.py eval --headless \
        --ckpt checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip

    # Record video from a checkpoint
    python3 scripts/appendix_e_isaac_peg.py record --headless \
        --ckpt checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip

    # Pixel-based training (native TiledCamera sensor, multi-env)
    python3 scripts/appendix_e_isaac_peg.py train --headless --pixel \
        --env-id Isaac-Lift-Cube-Franka-v0 --num-envs 16 --total-steps 2000000

Run through wrapper (recommended):
    bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke --headless
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None  # type: ignore[assignment]

RESULTS_DIR = Path("results")
CHECKPOINTS_DIR = Path("checkpoints")

DEFAULT_DENSE_ENV_ID = "Isaac-Reach-Franka-v0"

# Conservative regex set for auto-selecting a likely insertion task.
PEG_ENV_PATTERNS = [
    r"peg",
    r"insert",
    r"insertion",
    r"assembly",
    r"nut",
    r"bolt",
]


@dataclass
class AppendixEConfig:
    env_id: str = ""  # Empty => auto-select peg/insertion env from registry
    dense_env_id: str = DEFAULT_DENSE_ENV_ID
    seed: int = 0
    device: str = "cuda:0"
    num_envs: int = 1

    smoke_steps: int = 10_000
    total_steps: int = 500_000

    batch_size: int = 256
    buffer_size: int = 500_000
    learning_starts: int = 5_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = "auto"

    pixel: bool = False  # Use native TiledCamera sensor for pixel observations
    frame_stack: int = 1  # Number of frames to stack (1 = no stacking)
    net_arch: str = "256,256"  # MLP hidden layer sizes (comma-separated)

    her: str = "auto"  # {auto,on,off}
    her_n_sampled_goal: int = 4
    her_goal_selection_strategy: str = "future"

    eval_episodes: int = 100
    deterministic_eval: bool = True
    success_threshold: float = 0.05

    checkpoint_freq: int = 100_000  # Save every N steps (0 = end only)
    resume_ckpt: str = ""  # Path to checkpoint to resume training from

    log_dir: Path = Path("runs")
    checkpoints_dir: Path = CHECKPOINTS_DIR
    results_dir: Path = RESULTS_DIR


def _require_gym() -> None:
    if gym is None:
        raise SystemExit(
            "[appendix-e] Missing dependency: gymnasium. "
            "Run inside the project container: bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py ..."
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_env(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ckpt_path(cfg: AppendixEConfig, env_id: str) -> Path:
    base = f"appendix_e_sac_{_safe_env(env_id)}_seed{cfg.seed}"
    return _ensure_dir(cfg.checkpoints_dir) / f"{base}.zip"


def _meta_path(ckpt: Path) -> Path:
    return ckpt.parent / f"{ckpt.stem}.meta.json"


def _eval_path(cfg: AppendixEConfig, env_id: str) -> Path:
    base = f"appendix_e_sac_{_safe_env(env_id)}_seed{cfg.seed}_eval.json"
    return _ensure_dir(cfg.results_dir) / base


def _comparison_path(cfg: AppendixEConfig, env_id: str) -> Path:
    base = f"appendix_e_sac_{_safe_env(env_id)}_comparison.json"
    return _ensure_dir(cfg.results_dir) / base


def _gather_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
    }

    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", "unknown")
        versions["torch_cuda"] = str(getattr(torch.version, "cuda", "unknown"))
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
            versions["gpu_count"] = str(torch.cuda.device_count())
    except Exception:
        pass

    for module_name in ["gymnasium", "stable_baselines3", "isaaclab", "isaacsim"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            continue
    return versions


# ---------------------------------------------------------------------------
# Isaac boot + env registry
# ---------------------------------------------------------------------------


def _init_isaac(extra_args: list[str]):
    """Initialize Isaac AppLauncher and return simulation app."""
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[appendix-e] Missing Isaac Lab runtime (isaaclab). "
            "Run this inside the Isaac container via docker/dev-isaac.sh."
        ) from exc

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args(extra_args)
    if unknown:
        print(f"[appendix-e] Warning: ignoring unknown Isaac args: {unknown}")

    app_launcher = AppLauncher(args)
    return app_launcher.app


def _import_isaac_tasks() -> None:
    """Register Isaac task env IDs into gymnasium registry."""
    try:
        import isaaclab_tasks  # noqa: F401
    except ImportError:
        import omni.isaac.lab_tasks  # type: ignore # noqa: F401


def _registered_isaac_env_ids() -> list[str]:
    _require_gym()
    env_ids = [str(env_id) for env_id in gym.envs.registry.keys()]
    return sorted([eid for eid in env_ids if eid.startswith("Isaac-")])


def _find_peg_candidates(env_ids: list[str]) -> list[str]:
    regex = re.compile("|".join(PEG_ENV_PATTERNS), flags=re.IGNORECASE)
    return [eid for eid in env_ids if regex.search(eid)]


def _resolve_train_env_id(cfg: AppendixEConfig) -> str:
    if cfg.env_id:
        return cfg.env_id

    env_ids = _registered_isaac_env_ids()
    candidates = _find_peg_candidates(env_ids)
    if not candidates:
        raise SystemExit(
            "[appendix-e] No peg/insertion-like env ID found automatically. "
            "Run 'discover-envs' and pass --env-id explicitly."
        )

    chosen = candidates[0]
    print(f"[appendix-e] Auto-selected target env: {chosen}")
    return chosen


def _ensure_enable_cameras(extra_args: list[str]) -> list[str]:
    """Inject --enable_cameras if not already present (needed for rendering)."""
    if "--enable_cameras" not in extra_args:
        extra_args = ["--enable_cameras"] + extra_args
    return extra_args


def _create_visuomotor_lift_cfg(
    env_id: str,
    device: str,
    num_envs: int,
    seed: int,
    image_height: int = 84,
    image_width: int = 84,
):
    """Construct visuomotor env config: base env + per-env tiled camera.

    Extends the registered env config (e.g., Isaac-Lift-Cube-Franka-v0) by:
    1. Adding a TiledCamera sensor to the scene (per-env via {ENV_REGEX_NS})
    2. Adding a camera ObsTerm with clip=(0,255) so the obs space has [0,255]
       bounds -- required by Sb3VecEnvWrapper's image validation (Lesson 16)
    3. Setting concatenate_terms=False so obs becomes a Dict (image + state)
    4. Disabling curriculum (Lesson 14: reward scaling poisons off-policy buffers)

    Rather than creating a new observation group class (which would require
    importing task-specific mdp functions like object_position_in_robot_root_frame
    from isaaclab_tasks.manager_based.manipulation.lift.mdp), we modify the
    EXISTING parsed config. This inherits whatever obs terms the base config
    already defines -- no import path fragility.

    TiledCamera renders all envs into a single GPU-tiled image, then slices
    per-env -- this is Isaac Lab's native approach to visuomotor RL, scaling
    to 16-64+ parallel envs on a single GPU.

    The resulting observation space after Sb3VecEnvWrapper is::

        Dict({
            "joint_pos":              Box(9,),
            "joint_vel":              Box(9,),
            "object_position":        Box(3,),
            "target_object_position": Box(7,),
            "actions":                Box(8,),
            "tiled_camera":           Box(3, 84, 84), uint8
        })
    """
    # All Isaac Lab imports are deferred -- only available inside container.
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import TiledCameraCfg
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils

    try:
        from isaaclab_tasks.utils import parse_env_cfg
    except ModuleNotFoundError:
        from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore

    # Start from the registered env config (preserves rewards, terminations,
    # scene entities, action spaces, AND existing observation terms).
    env_cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs)
    env_cfg.seed = seed

    # --- Scene: add overhead tiled camera ---
    # Camera at (0.5, 0.0, 0.5) in the env frame, angled down at the table.
    # The rotation quaternion matches Isaac Lab's Stack-Cube table camera
    # (ROS convention: x-forward, y-left, z-up at the camera).
    env_cfg.scene.tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=image_height,
        width=image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 0.5),
            rot=(0.35355, -0.61237, -0.61237, 0.35355),
            convention="ros",
        ),
    )

    # --- Observations: add camera to existing group, switch to Dict mode ---
    # The base config already defines state obs terms (joint_pos, joint_vel,
    # object_position, target_object_position, actions) using task-specific
    # mdp functions. We add the camera term and set concatenate_terms=False
    # so SB3 sees a Dict obs and routes the image through NatureCNN.
    obs_policy = env_cfg.observations.policy
    obs_policy.tiled_camera = ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("tiled_camera"),
            "data_type": "rgb",
            "normalize": False,
        },
        clip=(0, 255),
    )
    obs_policy.concatenate_terms = False
    obs_policy.enable_corruption = False

    # --- Disable curriculum (Lesson 14) ---
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        for attr_name in list(vars(env_cfg.curriculum)):
            if not attr_name.startswith("_"):
                setattr(env_cfg.curriculum, attr_name, None)
        print("[appendix-e] Disabled CurriculumManager (off-policy compatibility)")

    print(
        f"[appendix-e] Created visuomotor config: "
        f"tiled_camera {image_height}x{image_width} RGB, "
        f"{num_envs} envs"
    )
    return env_cfg


def _make_isaac_env(
    env_id: str,
    *,
    device: str,
    num_envs: int = 1,
    seed: int | None = None,
    disable_curriculum: bool = True,
    render_mode: str | None = None,
    pixel: bool = False,
):
    """Create an Isaac Lab gym env from registry config.

    Args:
        disable_curriculum: If True, disable the CurriculumManager by clearing
            its term configs. This is critical for off-policy algorithms (SAC/TD3)
            because reward scale changes mid-training cause catastrophic replay
            buffer mismatch: old transitions have rewards at the old scale, new
            transitions at the new scale, and the Q-function cannot reconcile
            them. On-policy algorithms (PPO) don't have this problem because they
            discard old experience after each update.
        render_mode: If set (e.g., "rgb_array"), passed through to gym.make()
            to enable rendering. Requires --enable_cameras in Isaac args.
        pixel: If True, use native TiledCamera sensor for per-env pixel
            observations. Creates a visuomotor config with Dict obs space
            (state vectors + camera image). Multi-env compatible.
    """
    _require_gym()

    if pixel:
        # Native camera sensor: TiledCamera renders all envs in parallel.
        # Curriculum is disabled inside _create_visuomotor_lift_cfg.
        env_cfg = _create_visuomotor_lift_cfg(
            env_id, device, num_envs, seed or 0,
        )
    else:
        try:
            from isaaclab_tasks.utils import parse_env_cfg
        except ModuleNotFoundError:
            from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore

        env_cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs)
        if seed is not None:
            env_cfg.seed = seed

        # Disable curriculum for off-policy SAC compatibility.
        # Isaac Lab's Lift-Cube env ramps action_rate/joint_vel penalties from
        # -0.0001 to -0.1 (1000x) via CurriculumManager. This causes catastrophic
        # reward scale mismatch in SAC's replay buffer (Lesson 14).
        if disable_curriculum and hasattr(env_cfg, "curriculum"):
            try:
                curriculum_cfg = env_cfg.curriculum
                if curriculum_cfg is not None:
                    for attr_name in list(vars(curriculum_cfg)):
                        if not attr_name.startswith("_"):
                            setattr(curriculum_cfg, attr_name, None)
                    print("[appendix-e] Disabled CurriculumManager (off-policy compatibility)")
            except Exception as exc:
                print(f"[appendix-e] Warning: could not disable curriculum: {exc}")

    kwargs: dict[str, Any] = {"cfg": env_cfg}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)


def _wrap_for_sb3(isaac_env):
    """Wrap an Isaac Lab env with the official SB3 VecEnv adapter.

    Uses Isaac Lab's Sb3VecEnvWrapper which:
    - Subclasses SB3 VecEnv directly (bypasses DummyVecEnv)
    - Extracts obs['policy'] and converts to numpy
    - Clips infinite action bounds to [-100, 100]
    - Handles batched tensor <-> numpy conversion
    """
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    return Sb3VecEnvWrapper(isaac_env)


def _maybe_frame_stack(env, n_stack: int):
    """Wrap env with VecFrameStack if n_stack > 1."""
    if n_stack <= 1:
        return env
    from stable_baselines3.common.vec_env import VecFrameStack

    env = VecFrameStack(env, n_stack=n_stack)
    obs_space = env.observation_space
    if hasattr(obs_space, "shape"):
        print(f"[appendix-e] Frame stacking: {n_stack} -> obs_shape={obs_space.shape}")
    else:
        # Dict obs space (e.g., pixel mode): report per-key shapes
        shapes = {k: v.shape for k, v in obs_space.spaces.items()}
        print(f"[appendix-e] Frame stacking: {n_stack} -> obs_shapes={shapes}")
    return env


# ---------------------------------------------------------------------------
# SB3 training/evaluation
# ---------------------------------------------------------------------------


def _parse_ent_coef(val: str) -> str | float:
    if val == "auto":
        return "auto"
    return float(val)


def _is_goal_conditioned_env(env) -> bool:
    """Check if the Isaac env has Gymnasium-Robotics goal-conditioned structure.

    Most Isaac Lab envs expose {'policy': flat_vector} and are NOT goal-conditioned.
    A goal-conditioned env would need observation, achieved_goal, desired_goal keys
    inside the 'policy' observation group, plus a compute_reward method.
    """
    obs_space = env.observation_space
    if not isinstance(obs_space, gym.spaces.Dict):
        return False
    needed_keys = {"observation", "achieved_goal", "desired_goal"}
    has_keys = needed_keys.issubset(set(obs_space.spaces.keys()))
    has_reward_fn = hasattr(env.unwrapped, "compute_reward")
    return bool(has_keys and has_reward_fn)


def _is_image_space(space) -> bool:
    """Check if a Gymnasium space looks like an image (H, W, C) or (C, H, W)."""
    if not isinstance(space, gym.spaces.Box):
        return False
    shape = space.shape
    if len(shape) != 3:
        return False
    # (H, W, C) with C in {1, 3, 4} or (C, H, W) with C in {1, 3, 4}
    return shape[-1] in (1, 3, 4) or shape[0] in (1, 3, 4)


def _policy_name_for_env(env) -> str:
    """Pick SB3 policy class name based on observation space.

    After Sb3VecEnvWrapper extraction, Isaac Lab envs expose the inner
    observation space (typically a Box). Goal-conditioned envs with Dict
    obs get MultiInputPolicy. Image observations get CnnPolicy.
    """
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Dict):
        # Check if any value in the dict is an image
        has_image = any(_is_image_space(v) for v in obs_space.spaces.values())
        return "MultiInputPolicy" if not has_image else "MultiInputPolicy"
    if _is_image_space(obs_space):
        return "CnnPolicy"
    return "MlpPolicy"


def _build_sac_model(cfg: AppendixEConfig, env, is_goal_conditioned: bool):
    from stable_baselines3 import SAC

    use_her = False
    if cfg.her == "on":
        use_her = True
    elif cfg.her == "auto":
        use_her = is_goal_conditioned

    replay_buffer_class = None
    replay_buffer_kwargs = None
    if use_her:
        from stable_baselines3 import HerReplayBuffer

        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = {
            "n_sampled_goal": cfg.her_n_sampled_goal,
            "goal_selection_strategy": cfg.her_goal_selection_strategy,
        }

    net_arch = [int(x) for x in cfg.net_arch.split(",")]
    policy_kwargs = {"net_arch": net_arch}
    print(f"[appendix-e] net_arch: {net_arch}")

    model = SAC(
        _policy_name_for_env(env),
        env,
        verbose=1,
        device="auto",
        seed=cfg.seed,
        tensorboard_log=str(_ensure_dir(cfg.log_dir)),
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        tau=cfg.tau,
        ent_coef=_parse_ent_coef(cfg.ent_coef),
        policy_kwargs=policy_kwargs,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )

    return model, use_her


def _do_train(
    cfg: AppendixEConfig,
    env_id: str,
    total_steps: int,
) -> tuple[Path, dict[str, Any]]:
    """Train SAC on a single env. Assumes Isaac is already booted."""
    print(f"[appendix-e] Training SAC on {env_id}")
    print(f"[appendix-e] seed={cfg.seed}, total_steps={total_steps}, device={cfg.device}, num_envs={cfg.num_envs}")
    print(f"[appendix-e] pixel={cfg.pixel}, frame_stack={cfg.frame_stack}, net_arch={cfg.net_arch}")

    isaac_env = _make_isaac_env(
        env_id, device=cfg.device, num_envs=cfg.num_envs,
        seed=cfg.seed, pixel=cfg.pixel,
    )
    env = _wrap_for_sb3(isaac_env)
    goal_conditioned = _is_goal_conditioned_env(env)
    env = _maybe_frame_stack(env, cfg.frame_stack)

    print(f"[appendix-e] obs_space: {env.observation_space}")
    print(f"[appendix-e] act_space: {env.action_space}")
    if cfg.pixel and isinstance(env.observation_space, gym.spaces.Dict):
        # Estimate replay buffer RAM: obs + next_obs for image key
        for key, space in env.observation_space.spaces.items():
            if _is_image_space(space):
                pixel_shape = space.shape
                pixel_bytes = int(np.prod(pixel_shape))
                buffer_gb = 2 * cfg.buffer_size * pixel_bytes / (1024**3)
                print(f"[appendix-e] Replay buffer estimate: {buffer_gb:.1f} GB "
                      f"(buffer_size={cfg.buffer_size}, {key}={pixel_shape})")
                break
    print(f"[appendix-e] goal_conditioned: {goal_conditioned}")

    steps_done = 0
    if cfg.resume_ckpt:
        from stable_baselines3 import SAC

        print(f"[appendix-e] Resuming from {cfg.resume_ckpt}")
        model = SAC.load(cfg.resume_ckpt, env=env, device="auto")
        used_her = cfg.her == "on" or (cfg.her == "auto" and goal_conditioned)
        # Parse steps already completed from checkpoint filename (e.g., _300000.zip).
        m = re.search(r"_(\d+)\.zip$", cfg.resume_ckpt)
        steps_done = int(m.group(1)) if m else 0
        remaining = max(0, total_steps - steps_done)
        if remaining == 0:
            print(f"[appendix-e] Checkpoint already at {steps_done} steps >= {total_steps}. Nothing to do.")
            try:
                env.close()
            except Exception:
                pass
            return _ckpt_path(cfg, env_id), {}
        print(f"[appendix-e] Steps done: {steps_done}, remaining: {remaining}")
        total_steps = remaining
    else:
        model, used_her = _build_sac_model(cfg, env, goal_conditioned)

    # Build callbacks.
    callbacks: list = []
    if cfg.checkpoint_freq > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback

        ckpt_dir = _ensure_dir(cfg.checkpoints_dir)
        callbacks.append(CheckpointCallback(
            save_freq=max(1, cfg.checkpoint_freq // cfg.num_envs),
            save_path=str(ckpt_dir),
            name_prefix=f"appendix_e_sac_{_safe_env(env_id)}_seed{cfg.seed}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ))

    run_id = f"appendix_e/sac/{_safe_env(env_id)}/seed{cfg.seed}"
    t0 = time.perf_counter()
    model.learn(
        total_timesteps=total_steps,
        reset_num_timesteps=not bool(cfg.resume_ckpt),
        tb_log_name=run_id,
        callback=callbacks if callbacks else None,
    )
    elapsed = time.perf_counter() - t0

    ckpt = _ckpt_path(cfg, env_id)
    model.save(str(ckpt))

    meta = {
        "created_at": _now_iso(),
        "pipeline": "appendix_e_isaac_peg",
        "algo": "sac",
        "env_id": env_id,
        "seed": cfg.seed,
        "num_envs": cfg.num_envs,
        "total_steps": int(total_steps),
        "device": cfg.device,
        "used_her": bool(used_her),
        "obs_space_type": type(env.observation_space).__name__,
        "obs_space_shape": (
            {k: list(v.shape) for k, v in env.observation_space.spaces.items()}
            if hasattr(env.observation_space, "spaces")
            else list(env.observation_space.shape)
        ),
        "action_space_shape": list(env.action_space.shape)
        if hasattr(env.action_space, "shape") else "unknown",
        "is_goal_conditioned": bool(goal_conditioned),
        "wrapper": "Sb3VecEnvWrapper",
        "pixel": bool(cfg.pixel),
        "hyperparameters": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.ent_coef,
            "pixel": cfg.pixel,
            "frame_stack": cfg.frame_stack,
            "net_arch": cfg.net_arch,
            "her_mode": cfg.her,
            "her_n_sampled_goal": cfg.her_n_sampled_goal,
            "her_goal_selection_strategy": cfg.her_goal_selection_strategy,
            "checkpoint_freq": cfg.checkpoint_freq,
        },
        "checkpoint": str(ckpt),
        "training_seconds": elapsed,
        "steps_per_second": float(total_steps / max(elapsed, 1e-6)),
        "versions": _gather_versions(),
    }
    _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"[appendix-e] Saved checkpoint: {ckpt}")
    print(f"[appendix-e] Wrote metadata: {_meta_path(ckpt)}")

    try:
        env.close()
    except Exception:
        pass

    return ckpt, meta


def _do_eval(
    cfg: AppendixEConfig,
    ckpt_path: Path,
    env_id: str,
    *,
    n_episodes: int | None = None,
) -> Path:
    """Evaluate a checkpoint. Assumes Isaac is already booted.

    Uses SB3's VecEnv API (step_async/step_wait) for the eval loop.
    Episodes are collected sequentially from env index 0 when num_envs=1.
    """
    from stable_baselines3 import SAC

    episodes = int(n_episodes if n_episodes is not None else cfg.eval_episodes)
    print(f"[appendix-e] Evaluating {ckpt_path.name} on {env_id} ({episodes} episodes)")

    # Use num_envs=1 for deterministic sequential evaluation.
    isaac_env = _make_isaac_env(
        env_id, device=cfg.device, num_envs=1, seed=cfg.seed,
        pixel=cfg.pixel,
    )
    env = _wrap_for_sb3(isaac_env)
    env = _maybe_frame_stack(env, cfg.frame_stack)
    model = SAC.load(str(ckpt_path), env=env, device="auto")

    ep_returns: list[float] = []
    ep_lengths: list[int] = []

    # VecEnv API: reset() returns obs, step_wait() returns (obs, rewards, dones, infos).
    obs = env.reset()
    current_return = 0.0
    current_len = 0

    while len(ep_returns) < episodes:
        action, _ = model.predict(obs, deterministic=cfg.deterministic_eval)
        obs, rewards, dones, infos = env.step(action)
        current_return += float(rewards[0])
        current_len += 1

        if dones[0]:
            # Sb3VecEnvWrapper logs episodic info in infos[idx]["episode"].
            ep_info = infos[0].get("episode")
            if ep_info is not None:
                ep_returns.append(float(ep_info["r"]))
                ep_lengths.append(int(ep_info["l"]))
            else:
                ep_returns.append(current_return)
                ep_lengths.append(current_len)
            current_return = 0.0
            current_len = 0

    agg: dict[str, Any] = {
        "return_mean": float(np.mean(ep_returns)),
        "return_std": float(np.std(ep_returns, ddof=0)),
        "ep_len_mean": float(np.mean(ep_lengths)),
        "ep_len_std": float(np.std(ep_lengths, ddof=0)),
    }

    report = {
        "created_at": _now_iso(),
        "pipeline": "appendix_e_isaac_peg",
        "env_id": env_id,
        "checkpoint": str(ckpt_path),
        "seed_base": cfg.seed,
        "n_episodes": episodes,
        "deterministic": cfg.deterministic_eval,
        "aggregate": agg,
        "per_episode": [
            {
                "episode": i,
                "return": float(ep_returns[i]),
                "length": int(ep_lengths[i]),
            }
            for i in range(episodes)
        ],
        "versions": _gather_versions(),
    }

    out_path = _eval_path(cfg, env_id)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[appendix-e] Wrote eval report: {out_path}")

    try:
        env.close()
    except Exception:
        pass

    return out_path


def _probe_env_obs_space(env_id: str, device: str) -> dict[str, Any] | None:
    """Probe a single env's observation and action spaces.

    Uses the official Sb3VecEnvWrapper to show the exact spaces SB3 will see.
    For visuomotor envs, Sb3VecEnvWrapper auto-detects image keys and
    handles HWC->CHW transpose + uint8 casting.
    """
    try:
        isaac_env = _make_isaac_env(env_id, device=device, num_envs=1)
        env = _wrap_for_sb3(isaac_env)

        obs_space = env.observation_space
        act_space = env.action_space

        info: dict[str, Any] = {
            "obs_type": type(obs_space).__name__,
            "action_type": type(act_space).__name__,
            "is_goal_conditioned": _is_goal_conditioned_env(env),
            "num_envs": env.num_envs,
        }

        if isinstance(obs_space, gym.spaces.Dict):
            info["obs_keys"] = list(obs_space.spaces.keys())
            info["obs_shapes"] = {k: list(v.shape) for k, v in obs_space.spaces.items()}
            info["obs_dtypes"] = {k: str(v.dtype) for k, v in obs_space.spaces.items()}
            # Note which keys are images
            info["image_keys"] = [
                k for k, v in obs_space.spaces.items() if _is_image_space(v)
            ]
        elif isinstance(obs_space, gym.spaces.Box):
            info["obs_shape"] = list(obs_space.shape)
            info["obs_dtype"] = str(obs_space.dtype)

        if isinstance(act_space, gym.spaces.Box):
            info["action_shape"] = list(act_space.shape)
            info["action_bounds"] = [float(act_space.low[0]), float(act_space.high[0])]

        # Report policy class that would be selected
        info["policy_class"] = _policy_name_for_env(env)

        env.close()
        return info
    except Exception as exc:
        print(f"[appendix-e]   Probe failed for {env_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_discover_envs(cfg: AppendixEConfig, isaac_extra_args: list[str], pattern: str = "") -> int:
    sim_app = _init_isaac(isaac_extra_args)
    _import_isaac_tasks()
    try:
        env_ids = _registered_isaac_env_ids()
        peg = _find_peg_candidates(env_ids)

        if pattern:
            rgx = re.compile(pattern, flags=re.IGNORECASE)
            env_ids = [eid for eid in env_ids if rgx.search(eid)]
            peg = [eid for eid in peg if rgx.search(eid)]

        print(f"[appendix-e] Registered Isaac envs: {len(env_ids)}")
        for env_id in env_ids:
            print(f"  {env_id}")

        print(f"\n[appendix-e] Peg/insertion-like candidates: {len(peg)}")
        for env_id in peg:
            print(f"  {env_id}")

        # Probe observation space for the dense-first env only.
        # Isaac Lab cannot reliably create multiple envs in the same process:
        # SimulationContext is a singleton -- creating a second env raises
        # RuntimeError("Simulation context already exists").
        probed_envs: dict[str, Any] = {}
        probe_env_id = cfg.dense_env_id if cfg.dense_env_id in env_ids else None
        if probe_env_id:
            print(f"\n[appendix-e] Probing observation space for {probe_env_id}...")
            print("[appendix-e]   (Only one env probed per process -- SimulationContext singleton)")
            result = _probe_env_obs_space(probe_env_id, device=cfg.device)
            if result is not None:
                probed_envs[probe_env_id] = result
                gc_str = "goal-conditioned" if result["is_goal_conditioned"] else "flat obs"
                obs_detail = ""
                if "obs_keys" in result:
                    obs_detail = f", keys={result['obs_keys']}"
                elif "obs_shape" in result:
                    obs_detail = f", shape={result['obs_shape']}"
                print(f"[appendix-e]   -> {gc_str}{obs_detail}")
        if peg and not any(eid in probed_envs for eid in peg):
            print(f"[appendix-e]   To probe a peg candidate, re-run with: --dense-env-id {peg[0]}")

        report = {
            "created_at": _now_iso(),
            "total_isaac_envs": len(env_ids),
            "isaac_env_ids": env_ids,
            "peg_candidates": peg,
            "probed_envs": probed_envs,
            "patterns": PEG_ENV_PATTERNS,
            "versions": _gather_versions(),
        }
        out = _ensure_dir(cfg.results_dir) / "appendix_e_isaac_env_catalog.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[appendix-e] Wrote env catalog: {out}")
        return 0
    finally:
        sim_app.close()


def cmd_smoke(cfg: AppendixEConfig, isaac_extra_args: list[str]) -> int:
    """Dense-first smoke run. Boots Isaac, trains briefly, closes."""
    if cfg.pixel:
        isaac_extra_args = _ensure_enable_cameras(isaac_extra_args)
    sim_app = _init_isaac(isaac_extra_args)
    _import_isaac_tasks()
    try:
        print("[appendix-e] Smoke run (dense-first wiring check)")
        _do_train(cfg, cfg.dense_env_id, cfg.smoke_steps)
        return 0
    except Exception as exc:
        # Print before sim_app.close() -- close() may call sys.exit(0)
        # and swallow the traceback (Isaac Lab known behavior).
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        sim_app.close()


def cmd_train(cfg: AppendixEConfig, isaac_extra_args: list[str]) -> int:
    """Train SAC. Boots Isaac once -- resolves env_id if needed, then trains."""
    if cfg.pixel:
        isaac_extra_args = _ensure_enable_cameras(isaac_extra_args)
    sim_app = _init_isaac(isaac_extra_args)
    _import_isaac_tasks()
    try:
        if not cfg.env_id:
            cfg.env_id = _resolve_train_env_id(cfg)
        _do_train(cfg, cfg.env_id, cfg.total_steps)
        return 0
    finally:
        sim_app.close()


def _infer_env_from_ckpt_name(ckpt_name: str) -> str | None:
    # Match final checkpoint: appendix_e_sac_<env>_seed0.zip
    m = re.match(r"appendix_e_sac_(.+)_seed\d+\.zip", ckpt_name)
    if m:
        return m.group(1)
    # Match intermediate checkpoint: appendix_e_sac_<env>_seed0_199680_steps.zip
    m = re.match(r"appendix_e_sac_(.+)_seed\d+_\d+_steps\.zip", ckpt_name)
    if m:
        return m.group(1)
    return None


def _load_env_from_meta(ckpt_path: Path) -> str | None:
    meta = _meta_path(ckpt_path)
    if not meta.exists():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    env_id = payload.get("env_id")
    return str(env_id) if env_id else None


def cmd_eval(cfg: AppendixEConfig, isaac_extra_args: list[str], ckpt: str) -> int:
    """Evaluate a checkpoint. Boots Isaac once."""
    ckpt_path = Path(ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"[appendix-e] Checkpoint not found: {ckpt_path}")

    env_id = cfg.env_id or _load_env_from_meta(ckpt_path)
    if not env_id:
        env_id = _infer_env_from_ckpt_name(ckpt_path.name)
    if not env_id:
        raise SystemExit(
            "[appendix-e] Could not infer env_id. Pass --env-id explicitly or keep .meta.json next to checkpoint."
        )

    if cfg.pixel:
        isaac_extra_args = _ensure_enable_cameras(isaac_extra_args)
    sim_app = _init_isaac(isaac_extra_args)
    _import_isaac_tasks()
    try:
        _do_eval(cfg, ckpt_path, env_id)
        return 0
    finally:
        sim_app.close()


def cmd_record(cfg: AppendixEConfig, isaac_extra_args: list[str], ckpt: str, video_out: str) -> int:
    """Record a video from a checkpoint. Boots Isaac with cameras enabled."""
    ckpt_path = Path(ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"[appendix-e] Checkpoint not found: {ckpt_path}")

    env_id = cfg.env_id or _load_env_from_meta(ckpt_path)
    if not env_id:
        env_id = _infer_env_from_ckpt_name(ckpt_path.name)
    if not env_id:
        raise SystemExit(
            "[appendix-e] Could not infer env_id. Pass --env-id explicitly."
        )

    # Determine output path
    if not video_out:
        videos_dir = _ensure_dir("videos")
        video_out = str(videos_dir / f"appendix_e_{_safe_env(env_id)}_{ckpt_path.stem}.mp4")

    # Cameras required for rendering
    isaac_extra_args = _ensure_enable_cameras(isaac_extra_args)

    sim_app = _init_isaac(isaac_extra_args)
    _import_isaac_tasks()
    try:
        from stable_baselines3 import SAC

        print(f"[appendix-e] Recording video: {ckpt_path.name} on {env_id}")
        isaac_env = _make_isaac_env(
            env_id, device=cfg.device, num_envs=1,
            seed=cfg.seed, render_mode="rgb_array",
        )
        env = _wrap_for_sb3(isaac_env)
        model = SAC.load(str(ckpt_path), env=env, device="auto")

        # Reset + warm up renderer. Isaac Lab's Vulkan renderer requires the
        # simulation to be active (reset + stepped) before render() produces
        # frames. First few frames may also be black due to shader compilation.
        # Use isaac_env.render() directly -- Sb3VecEnvWrapper.render() hangs
        # because SB3's VecEnv.render() routes through get_images()/tile_images()
        # which is incompatible with Isaac Lab's synchronous Vulkan renderer.
        obs = env.reset()
        for _ in range(5):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(action)
            isaac_env.render()

        # Fresh reset for the actual recorded episode
        obs = env.reset()

        # Run one episode and collect frames
        frames: list[np.ndarray] = []
        done = False
        step_count = 0
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_return += float(rewards[0])
            step_count += 1

            frame = isaac_env.render()
            if frame is not None:
                frame = np.asarray(frame)
                # Handle batched frames: (batch, H, W, C) -> (H, W, C)
                if frame.ndim == 4:
                    frame = frame[0]
                # Normalize to uint8
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                frames.append(frame)

            if dones[0]:
                done = True

        print(f"[appendix-e] Episode: {step_count} steps, return={ep_return:.3f}, frames={len(frames)}")

        if frames:
            import imageio.v3 as iio

            out_path = Path(video_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(out_path, frames, fps=30)
            print(f"[appendix-e] Wrote video: {out_path} ({len(frames)} frames)")
        else:
            print("[appendix-e] WARNING: No frames captured")

        try:
            env.close()
        except Exception:
            pass

        return 0
    finally:
        sim_app.close()


def _build_common_cli_args(cfg: AppendixEConfig) -> list[str]:
    """Serialize config back to CLI arguments for subprocess calls."""
    args = [
        "--seed", str(cfg.seed),
        "--device", cfg.device,
        "--num-envs", str(cfg.num_envs),
        "--frame-stack", str(cfg.frame_stack),
        "--net-arch", cfg.net_arch,
        "--batch-size", str(cfg.batch_size),
        "--buffer-size", str(cfg.buffer_size),
        "--learning-starts", str(cfg.learning_starts),
        "--learning-rate", str(cfg.learning_rate),
        "--gamma", str(cfg.gamma),
        "--tau", str(cfg.tau),
        "--ent-coef", cfg.ent_coef,
        "--her", cfg.her,
        "--her-n-sampled-goal", str(cfg.her_n_sampled_goal),
        "--her-goal-selection-strategy", cfg.her_goal_selection_strategy,
        "--checkpoint-freq", str(cfg.checkpoint_freq),
        "--log-dir", str(cfg.log_dir),
        "--checkpoints-dir", str(cfg.checkpoints_dir),
        "--results-dir", str(cfg.results_dir),
        "--dense-env-id", cfg.dense_env_id,
    ]
    if cfg.pixel:
        args.append("--pixel")
    if cfg.env_id:
        args += ["--env-id", cfg.env_id]
    if cfg.resume_ckpt:
        args += ["--resume-ckpt", cfg.resume_ckpt]
    return args


def cmd_all(cfg: AppendixEConfig, isaac_extra_args: list[str]) -> int:
    """Smoke -> train -> eval, each phase as a separate subprocess.

    Isaac Lab's SimulationContext is a process-level singleton: only one
    Isaac env can exist per Python process. Creating a second env after
    closing the first raises RuntimeError. We handle this by running each
    phase in its own subprocess with a fresh SimulationApp.
    """
    import subprocess

    script = str(Path(__file__).resolve())

    # If env_id not set, resolve via discover-envs subprocess.
    if not cfg.env_id:
        print("[appendix-e] Resolving target env ID via discover-envs...")
        disc_args = _build_common_cli_args(cfg)
        rc = subprocess.call(
            [sys.executable, script, "discover-envs"] + disc_args + isaac_extra_args,
        )
        if rc != 0:
            print(f"[appendix-e] discover-envs failed (exit {rc})")
            return rc
        catalog = cfg.results_dir / "appendix_e_isaac_env_catalog.json"
        if not catalog.exists():
            raise SystemExit(
                "[appendix-e] discover-envs did not produce catalog. Cannot resolve env_id."
            )
        data = json.loads(catalog.read_text(encoding="utf-8"))
        candidates = data.get("peg_candidates", [])
        if not candidates:
            raise SystemExit(
                "[appendix-e] No peg/insertion-like env found. Pass --env-id explicitly."
            )
        cfg.env_id = candidates[0]
        print(f"[appendix-e] Auto-selected target env: {cfg.env_id}")

    common = _build_common_cli_args(cfg)

    def _run_phase(subcmd: str, extra: list[str]) -> int:
        cmd = [sys.executable, script, subcmd] + common + extra + isaac_extra_args
        print(f"[appendix-e] Running subprocess: {subcmd}")
        return subprocess.call(cmd)

    if not cfg.resume_ckpt:
        # Phase 1: Smoke (dense-first wiring check).
        print("\n[appendix-e] === Phase 1/3: Smoke (dense-first wiring check) ===")
        rc = _run_phase("smoke", ["--smoke-steps", str(cfg.smoke_steps)])
        if rc != 0:
            print(f"[appendix-e] Smoke phase failed (exit {rc})")
            return rc
    else:
        print("\n[appendix-e] === Skipping smoke (resuming from checkpoint) ===")

    # Phase 2: Train on target env.
    print("\n[appendix-e] === Phase 2/3: Train ===")
    rc = _run_phase("train", ["--total-steps", str(cfg.total_steps)])
    if rc != 0:
        print(f"[appendix-e] Train phase failed (exit {rc})")
        return rc

    # Phase 3: Evaluate the checkpoint from Phase 2.
    ckpt = _ckpt_path(cfg, cfg.env_id)
    if not ckpt.exists():
        print(f"[appendix-e] Expected checkpoint not found: {ckpt}")
        return 1

    print("\n[appendix-e] === Phase 3/3: Eval ===")
    eval_extra = [
        "--ckpt", str(ckpt),
        "--eval-episodes", str(cfg.eval_episodes),
        "--success-threshold", str(cfg.success_threshold),
    ]
    if cfg.deterministic_eval:
        eval_extra.append("--deterministic-eval")
    else:
        eval_extra.append("--no-deterministic-eval")
    rc = _run_phase("eval", eval_extra)
    if rc != 0:
        print(f"[appendix-e] Eval phase failed (exit {rc})")
        return rc

    print("\n[appendix-e] === All phases complete ===")
    return 0


def cmd_compare(cfg: AppendixEConfig, env_id: str, result_paths: list[str]) -> int:
    rows: list[dict[str, Any]] = []
    for path_str in result_paths:
        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"[appendix-e] Missing result file: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        agg = payload.get("aggregate", {})
        rows.append(
            {
                "path": str(p),
                "return_mean": agg.get("return_mean"),
                "ep_len_mean": agg.get("ep_len_mean"),
                "success_rate": agg.get("success_rate"),
                "final_goal_distance_mean": agg.get("final_goal_distance_mean"),
            }
        )

    print("[appendix-e] Comparison")
    print(f"{'Result file':<70} {'Return':>10} {'Succ':>8} {'Len':>8} {'GoalDist':>10}")
    print("-" * 114)
    for r in rows:
        succ = "n/a" if r["success_rate"] is None else f"{float(r['success_rate']):.3f}"
        dist = "n/a" if r["final_goal_distance_mean"] is None else f"{float(r['final_goal_distance_mean']):.4f}"
        print(
            f"{Path(r['path']).name:<70} "
            f"{float(r['return_mean']):>10.3f} {succ:>8} {float(r['ep_len_mean']):>8.1f} {dist:>10}"
        )

    out = {
        "created_at": _now_iso(),
        "pipeline": "appendix_e_isaac_peg",
        "env_id": env_id,
        "rows": rows,
    }
    out_path = _comparison_path(cfg, env_id)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[appendix-e] Wrote comparison: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Appendix E Isaac Lab manipulation pipeline. "
            "Primary target: Isaac-Lift-Cube-Franka-v0 (SAC, 256 envs, ~774 fps). "
            "Subcommands: discover-envs, smoke, train, eval, all, compare, record."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--env-id", default="", help="Target Isaac env id (e.g., Isaac-Lift-Cube-Franka-v0). Empty = auto-select peg/insertion candidate")
        p.add_argument("--dense-env-id", default=DEFAULT_DENSE_ENV_ID, help="Known-easy env for dense-first smoke")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cuda:0", help="Isaac device for parse_env_cfg, e.g., cuda:0")
        p.add_argument("--num-envs", type=int, default=1, help="Number of parallel Isaac envs")
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--buffer-size", type=int, default=500_000)
        p.add_argument("--learning-starts", type=int, default=5_000)
        p.add_argument("--learning-rate", type=float, default=3e-4)
        p.add_argument("--gamma", type=float, default=0.99)
        p.add_argument("--tau", type=float, default=0.005)
        p.add_argument("--ent-coef", default="auto")
        p.add_argument("--her", choices=["auto", "on", "off"], default="auto")
        p.add_argument("--her-n-sampled-goal", type=int, default=4)
        p.add_argument("--her-goal-selection-strategy", choices=["future", "final", "episode"], default="future")
        p.add_argument("--frame-stack", type=int, default=1, help="Number of frames to stack (1 = no stacking)")
        p.add_argument("--net-arch", default="256,256", help="MLP hidden layer sizes (comma-separated)")
        p.add_argument("--pixel", action="store_true", help="Use pixel observations via native TiledCamera sensor")
        p.add_argument("--checkpoint-freq", type=int, default=100_000, help="Save every N steps (0 = end only)")
        p.add_argument("--log-dir", default="runs")
        p.add_argument("--checkpoints-dir", default="checkpoints")
        p.add_argument("--results-dir", default="results")

    p_disc = sub.add_parser("discover-envs", help="List Isaac env IDs and peg/insertion candidates")
    add_common(p_disc)
    p_disc.add_argument("--pattern", default="", help="Optional regex filter applied after discovery")

    p_smoke = sub.add_parser("smoke", help="Dense-first short training run for wiring checks")
    add_common(p_smoke)
    p_smoke.add_argument("--smoke-steps", type=int, default=10_000)

    p_train = sub.add_parser("train", help="Train SAC on target env")
    add_common(p_train)
    p_train.add_argument("--total-steps", type=int, default=500_000)
    p_train.add_argument("--resume-ckpt", default="", help="Checkpoint to resume training from")

    p_eval = sub.add_parser("eval", help="Evaluate checkpoint")
    add_common(p_eval)
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--eval-episodes", type=int, default=100)
    p_eval.add_argument("--deterministic-eval", action=argparse.BooleanOptionalAction, default=True)
    p_eval.add_argument("--success-threshold", type=float, default=0.05)

    p_all = sub.add_parser("all", help="Smoke -> train -> eval")
    add_common(p_all)
    p_all.add_argument("--smoke-steps", type=int, default=10_000)
    p_all.add_argument("--total-steps", type=int, default=500_000)
    p_all.add_argument("--resume-ckpt", default="", help="Checkpoint to resume training from (skips smoke)")
    p_all.add_argument("--eval-episodes", type=int, default=100)
    p_all.add_argument("--deterministic-eval", action=argparse.BooleanOptionalAction, default=True)
    p_all.add_argument("--success-threshold", type=float, default=0.05)

    p_cmp = sub.add_parser("compare", help="Compare multiple eval JSON files")
    add_common(p_cmp)
    p_cmp.add_argument("--result", action="append", required=True, help="Path to eval JSON (repeatable)")

    p_rec = sub.add_parser("record", help="Record a video from a checkpoint")
    add_common(p_rec)
    p_rec.add_argument("--ckpt", required=True, help="Path to SB3 checkpoint")
    p_rec.add_argument("--video-out", default="", help="Output video path (default: videos/appendix_e_<env>_<ckpt>.mp4)")

    return parser


def _cfg_from_args(args: argparse.Namespace) -> AppendixEConfig:
    return AppendixEConfig(
        env_id=args.env_id,
        dense_env_id=args.dense_env_id,
        seed=args.seed,
        device=args.device,
        num_envs=getattr(args, "num_envs", 1),
        pixel=getattr(args, "pixel", False),
        frame_stack=args.frame_stack,
        net_arch=args.net_arch,
        smoke_steps=getattr(args, "smoke_steps", 10_000),
        total_steps=getattr(args, "total_steps", 500_000),
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef=args.ent_coef,
        her=args.her,
        her_n_sampled_goal=args.her_n_sampled_goal,
        her_goal_selection_strategy=args.her_goal_selection_strategy,
        eval_episodes=getattr(args, "eval_episodes", 100),
        deterministic_eval=getattr(args, "deterministic_eval", True),
        success_threshold=getattr(args, "success_threshold", 0.05),
        checkpoint_freq=getattr(args, "checkpoint_freq", 100_000),
        resume_ckpt=getattr(args, "resume_ckpt", ""),
        log_dir=Path(args.log_dir),
        checkpoints_dir=Path(args.checkpoints_dir),
        results_dir=Path(args.results_dir),
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = _build_parser()
    args, isaac_extra_args = parser.parse_known_args(argv)
    cfg = _cfg_from_args(args)

    # Force unbuffered logs for long-running training readability.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    if args.cmd == "discover-envs":
        return cmd_discover_envs(cfg, isaac_extra_args, pattern=args.pattern)
    if args.cmd == "smoke":
        return cmd_smoke(cfg, isaac_extra_args)
    if args.cmd == "train":
        return cmd_train(cfg, isaac_extra_args)
    if args.cmd == "eval":
        return cmd_eval(cfg, isaac_extra_args, ckpt=args.ckpt)
    if args.cmd == "all":
        return cmd_all(cfg, isaac_extra_args)
    if args.cmd == "compare":
        env_id = cfg.env_id or "unknown_env"
        return cmd_compare(cfg, env_id=env_id, result_paths=args.result)
    if args.cmd == "record":
        return cmd_record(cfg, isaac_extra_args, ckpt=args.ckpt, video_out=args.video_out)

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
