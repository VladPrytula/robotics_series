#!/usr/bin/env python3
"""Appendix E: Isaac Lab Peg-In-Hole Pipeline (book-style Run It script).

This script prepares a reproducible SAC (+ optional HER) pipeline for Appendix E
without coupling to MuJoCo/Gymnasium-Robotics assumptions.

Design goals (aligned with the tutorial/book workflow):
1) Dense-first debugging: run a short smoke train on a known-easy Isaac env
   before long insertion runs.
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
    # Discover available Isaac env IDs and peg-like candidates
    python3 scripts/appendix_e_isaac_peg.py discover-envs --headless

    # Dense-first smoke test (short Reach run)
    python3 scripts/appendix_e_isaac_peg.py smoke --headless --seed 0

    # Train on auto-selected peg/insertion env (if available)
    python3 scripts/appendix_e_isaac_peg.py train --headless --seed 0

    # Train on a specific env explicitly
    python3 scripts/appendix_e_isaac_peg.py train --headless --env-id Isaac-Factory-PegInsert-Direct-v0

    # Evaluate a checkpoint
    python3 scripts/appendix_e_isaac_peg.py eval --headless \
        --ckpt checkpoints/appendix_e_sac_Isaac-Factory-PegInsert-Direct-v0_seed0.zip

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


def _make_isaac_env(env_id: str, *, device: str, num_envs: int = 1, seed: int | None = None):
    """Create an Isaac Lab gym env from registry config."""
    _require_gym()
    try:
        from isaaclab_tasks.utils import parse_env_cfg
    except ModuleNotFoundError:
        from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore

    env_cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs)
    if seed is not None:
        env_cfg.seed = seed
    return gym.make(env_id, cfg=env_cfg)


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
    print(f"[appendix-e] Frame stacking: {n_stack} -> obs_shape={env.observation_space.shape}")
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


def _policy_name_for_env(env) -> str:
    """Pick SB3 policy class name based on observation space.

    After Sb3VecEnvWrapper extraction, Isaac Lab envs expose the inner
    observation space (typically a Box). Goal-conditioned envs with Dict
    obs get MultiInputPolicy.
    """
    return "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy"


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
    print(f"[appendix-e] frame_stack={cfg.frame_stack}, net_arch={cfg.net_arch}")

    isaac_env = _make_isaac_env(env_id, device=cfg.device, num_envs=cfg.num_envs, seed=cfg.seed)
    env = _wrap_for_sb3(isaac_env)
    goal_conditioned = _is_goal_conditioned_env(env)
    env = _maybe_frame_stack(env, cfg.frame_stack)

    print(f"[appendix-e] obs_space: {env.observation_space}")
    print(f"[appendix-e] act_space: {env.action_space}")
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
        "obs_space_shape": list(env.observation_space.shape)
        if hasattr(env.observation_space, "shape") else "dict",
        "action_space_shape": list(env.action_space.shape)
        if hasattr(env.action_space, "shape") else "unknown",
        "is_goal_conditioned": bool(goal_conditioned),
        "wrapper": "Sb3VecEnvWrapper",
        "hyperparameters": {
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.buffer_size,
            "learning_starts": cfg.learning_starts,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "ent_coef": cfg.ent_coef,
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
    isaac_env = _make_isaac_env(env_id, device=cfg.device, num_envs=1, seed=cfg.seed)
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
        elif isinstance(obs_space, gym.spaces.Box):
            info["obs_shape"] = list(obs_space.shape)
            info["obs_dtype"] = str(obs_space.dtype)

        if isinstance(act_space, gym.spaces.Box):
            info["action_shape"] = list(act_space.shape)
            info["action_bounds"] = [float(act_space.low[0]), float(act_space.high[0])]

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
    sim_app = _init_isaac(isaac_extra_args)
    _import_isaac_tasks()
    try:
        print("[appendix-e] Smoke run (dense-first wiring check)")
        _do_train(cfg, cfg.dense_env_id, cfg.smoke_steps)
        return 0
    finally:
        sim_app.close()


def cmd_train(cfg: AppendixEConfig, isaac_extra_args: list[str]) -> int:
    """Train SAC. Boots Isaac once -- resolves env_id if needed, then trains."""
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
    m = re.match(r"appendix_e_sac_(.+)_seed\d+\.zip", ckpt_name)
    if not m:
        return None
    return m.group(1)


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

    sim_app = _init_isaac(isaac_extra_args)
    _import_isaac_tasks()
    try:
        _do_eval(cfg, ckpt_path, env_id)
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
        description="Appendix E Isaac Lab pipeline (discover, smoke, train, eval, compare)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--env-id", default="", help="Target Isaac env id. Empty = auto-select peg/insertion candidate")
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

    return parser


def _cfg_from_args(args: argparse.Namespace) -> AppendixEConfig:
    return AppendixEConfig(
        env_id=args.env_id,
        dense_env_id=args.dense_env_id,
        seed=args.seed,
        device=args.device,
        num_envs=getattr(args, "num_envs", 1),
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

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
