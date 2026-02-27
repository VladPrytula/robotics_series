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

    smoke_steps: int = 10_000
    total_steps: int = 500_000

    batch_size: int = 256
    buffer_size: int = 500_000
    learning_starts: int = 5_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = "auto"

    her: str = "auto"  # {auto,on,off}
    her_n_sampled_goal: int = 4
    her_goal_selection_strategy: str = "future"

    eval_episodes: int = 100
    deterministic_eval: bool = True
    success_threshold: float = 0.05

    log_dir: Path = Path("runs")
    checkpoints_dir: Path = CHECKPOINTS_DIR
    results_dir: Path = RESULTS_DIR


if gym is None:
    _GymEnvBase = object
else:
    _GymEnvBase = gym.Env


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


def _make_isaac_env(env_id: str, *, device: str, num_envs: int = 1, render_mode: str | None = None):
    """Create an Isaac Lab gym env from registry config."""
    _require_gym()
    try:
        from isaaclab_tasks.utils import parse_env_cfg
    except ModuleNotFoundError:
        from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore

    env_cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs)
    kwargs: dict[str, Any] = {"cfg": env_cfg}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)


# ---------------------------------------------------------------------------
# Isaac <-> SB3 adapter
# ---------------------------------------------------------------------------


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _strip_batch_np(x: np.ndarray) -> np.ndarray:
    if x.ndim >= 1 and x.shape[0] == 1:
        return x[0]
    return x


def _strip_batch_space(space: gym.Space) -> gym.Space:
    _require_gym()
    if isinstance(space, gym.spaces.Box):
        if len(space.shape) >= 1 and space.shape[0] == 1:
            low = _strip_batch_np(np.asarray(space.low))
            high = _strip_batch_np(np.asarray(space.high))
            return gym.spaces.Box(low=low, high=high, dtype=space.dtype)
        return space

    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: _strip_batch_space(v) for k, v in space.spaces.items()})

    return space


def _convert_obs(obs: Any) -> Any:
    if isinstance(obs, dict):
        converted: dict[str, np.ndarray] = {}
        for key, val in obs.items():
            arr = _to_numpy(val)
            arr = _strip_batch_np(arr)
            converted[key] = arr
        return converted

    arr = _to_numpy(obs)
    return _strip_batch_np(arr)


def _extract_scalar_done(done_like: Any) -> bool:
    arr = _to_numpy(done_like)
    if arr.ndim == 0:
        return bool(arr.item())
    return bool(_strip_batch_np(arr).item())


def _extract_scalar_reward(reward_like: Any) -> float:
    arr = _to_numpy(reward_like)
    if arr.ndim == 0:
        return float(arr.item())
    return float(_strip_batch_np(arr).item())


def _info_to_python(info: Any) -> Any:
    if isinstance(info, dict):
        out: dict[str, Any] = {}
        for k, v in info.items():
            if isinstance(v, dict):
                out[k] = _info_to_python(v)
                continue
            arr = _to_numpy(v)
            if arr.ndim == 0:
                out[k] = arr.item()
            else:
                # Keep arrays for debugging, but strip singleton batch.
                out[k] = _strip_batch_np(arr).tolist()
        return out
    return info


class IsaacSb3Adapter(_GymEnvBase):
    """Gym Env wrapper that adapts Isaac tensor-based API to SB3-friendly numpy API."""

    metadata = {"render_modes": ["rgb_array", None]}

    def __init__(self, isaac_env, device: str = "cuda:0"):
        super().__init__()
        self._env = isaac_env
        self._device = device

        self.observation_space = _strip_batch_space(isaac_env.observation_space)
        self.action_space = _strip_batch_space(isaac_env.action_space)

        sample = _to_numpy(isaac_env.action_space.sample())
        self._expects_batched_action = bool(sample.ndim >= 2 and sample.shape[0] == 1)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = int(seed)
        if options is not None:
            kwargs["options"] = options

        try:
            obs, info = self._env.reset(**kwargs)
        except TypeError:
            obs, info = self._env.reset()

        return _convert_obs(obs), _info_to_python(info)

    def step(self, action):
        import torch

        action_np = np.asarray(action, dtype=np.float32)
        if self._expects_batched_action and action_np.ndim == 1:
            action_np = action_np[None, :]

        action_t = torch.as_tensor(action_np, device=self._device)
        obs, reward, terminated, truncated, info = self._env.step(action_t)

        return (
            _convert_obs(obs),
            _extract_scalar_reward(reward),
            _extract_scalar_done(terminated),
            _extract_scalar_done(truncated),
            _info_to_python(info),
        )

    def render(self):
        frame = self._env.render()
        if frame is None:
            return None
        frame_np = _to_numpy(frame)
        frame_np = _strip_batch_np(frame_np)
        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
        return frame_np

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# SB3 training/evaluation
# ---------------------------------------------------------------------------


def _parse_ent_coef(val: str) -> str | float:
    if val == "auto":
        return "auto"
    return float(val)


def _is_goal_conditioned_env(env: gym.Env) -> bool:
    _require_gym()
    obs_space = env.observation_space
    if not isinstance(obs_space, gym.spaces.Dict):
        return False

    needed_keys = {"observation", "achieved_goal", "desired_goal"}
    has_keys = needed_keys.issubset(set(obs_space.spaces.keys()))
    has_reward_fn = hasattr(env.unwrapped, "compute_reward")
    return bool(has_keys and has_reward_fn)


def _policy_name_for_env(env: gym.Env) -> str:
    return "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy"


def _build_sac_model(cfg: AppendixEConfig, env: gym.Env):
    from stable_baselines3 import SAC

    use_her = False
    if cfg.her == "on":
        use_her = True
    elif cfg.her == "auto":
        use_her = _is_goal_conditioned_env(env)

    replay_buffer_class = None
    replay_buffer_kwargs = None
    if use_her:
        try:
            from stable_baselines3 import HerReplayBuffer
        except Exception:
            from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = {
            "n_sampled_goal": cfg.her_n_sampled_goal,
            "goal_selection_strategy": cfg.her_goal_selection_strategy,
        }

    model = SAC(
        _policy_name_for_env(env),
        env,
        verbose=1,
        device="auto",
        tensorboard_log=str(_ensure_dir(cfg.log_dir)),
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        tau=cfg.tau,
        ent_coef=_parse_ent_coef(cfg.ent_coef),
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )

    return model, use_her


def _extract_success(info: dict[str, Any], obs: Any, threshold: float) -> int | None:
    if "is_success" in info:
        val = info["is_success"]
        try:
            arr = np.asarray(val)
            if arr.size > 0:
                return int(bool(arr.reshape(-1)[0]))
        except Exception:
            pass

    if isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs:
        ag = np.asarray(obs["achieved_goal"], dtype=np.float32)
        dg = np.asarray(obs["desired_goal"], dtype=np.float32)
        dist = float(np.linalg.norm(ag - dg))
        return int(dist <= threshold)

    return None


def _extract_goal_distance(obs: Any) -> float | None:
    if not isinstance(obs, dict):
        return None
    if "achieved_goal" not in obs or "desired_goal" not in obs:
        return None

    ag = np.asarray(obs["achieved_goal"], dtype=np.float32)
    dg = np.asarray(obs["desired_goal"], dtype=np.float32)
    return float(np.linalg.norm(ag - dg))


def _train_on_env(cfg: AppendixEConfig, env_id: str, total_steps: int, isaac_extra_args: list[str]) -> tuple[Path, dict[str, Any]]:
    sim_app = _init_isaac(isaac_extra_args)
    _import_isaac_tasks()

    print(f"[appendix-e] Training SAC on {env_id}")
    print(f"[appendix-e] seed={cfg.seed}, total_steps={total_steps}, device={cfg.device}")

    try:
        isaac_env = _make_isaac_env(env_id, device=cfg.device, num_envs=1)
        env = IsaacSb3Adapter(isaac_env, device=cfg.device)

        try:
            from stable_baselines3.common.utils import set_random_seed

            set_random_seed(cfg.seed)
        except Exception:
            pass

        model, used_her = _build_sac_model(cfg, env)

        run_id = f"appendix_e/sac/{_safe_env(env_id)}/seed{cfg.seed}"
        t0 = time.perf_counter()
        model.learn(total_timesteps=total_steps, tb_log_name=run_id)
        elapsed = time.perf_counter() - t0

        ckpt = _ckpt_path(cfg, env_id)
        model.save(str(ckpt))

        meta = {
            "created_at": _now_iso(),
            "pipeline": "appendix_e_isaac_peg",
            "algo": "sac",
            "env_id": env_id,
            "seed": cfg.seed,
            "total_steps": int(total_steps),
            "device": cfg.device,
            "used_her": bool(used_her),
            "hyperparameters": {
                "batch_size": cfg.batch_size,
                "buffer_size": cfg.buffer_size,
                "learning_starts": cfg.learning_starts,
                "learning_rate": cfg.learning_rate,
                "gamma": cfg.gamma,
                "tau": cfg.tau,
                "ent_coef": cfg.ent_coef,
                "her_mode": cfg.her,
                "her_n_sampled_goal": cfg.her_n_sampled_goal,
                "her_goal_selection_strategy": cfg.her_goal_selection_strategy,
            },
            "checkpoint": str(ckpt),
            "training_seconds": elapsed,
            "steps_per_second": float(total_steps / max(elapsed, 1e-6)),
            "versions": _gather_versions(),
        }
        _meta_path(ckpt).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        print(f"[appendix-e] Saved checkpoint: {ckpt}")
        print(f"[appendix-e] Wrote metadata: {_meta_path(ckpt)}")

        return ckpt, meta
    finally:
        try:
            env.close()
        except Exception:
            pass
        sim_app.close()


def _evaluate_ckpt(
    cfg: AppendixEConfig,
    ckpt_path: Path,
    env_id: str,
    isaac_extra_args: list[str],
    *,
    n_episodes: int | None = None,
) -> Path:
    from stable_baselines3 import SAC

    episodes = int(n_episodes if n_episodes is not None else cfg.eval_episodes)

    sim_app = _init_isaac(isaac_extra_args)
    _import_isaac_tasks()

    print(f"[appendix-e] Evaluating {ckpt_path.name} on {env_id} ({episodes} episodes)")

    try:
        isaac_env = _make_isaac_env(env_id, device=cfg.device, num_envs=1)
        env = IsaacSb3Adapter(isaac_env, device=cfg.device)

        model = SAC.load(str(ckpt_path), env=env, device="auto")

        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        ep_success: list[int | None] = []
        ep_goal_dist: list[float | None] = []

        for i in range(episodes):
            obs, info = env.reset(seed=cfg.seed + i)
            done = False
            ep_return = 0.0
            ep_len = 0
            final_success: int | None = None

            while not done:
                action, _ = model.predict(obs, deterministic=cfg.deterministic_eval)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)
                final_success = _extract_success(info, obs, cfg.success_threshold)

            ep_returns.append(ep_return)
            ep_lengths.append(ep_len)
            ep_success.append(final_success)

            gd = _extract_goal_distance(obs)
            ep_goal_dist.append(gd)

        agg: dict[str, Any] = {
            "return_mean": float(np.mean(ep_returns)),
            "return_std": float(np.std(ep_returns, ddof=0)),
            "ep_len_mean": float(np.mean(ep_lengths)),
            "ep_len_std": float(np.std(ep_lengths, ddof=0)),
        }
        success_values = [v for v in ep_success if v is not None]
        if success_values:
            agg["success_rate"] = float(np.mean(success_values))

        goal_dist_values = [v for v in ep_goal_dist if v is not None]
        if goal_dist_values:
            agg["final_goal_distance_mean"] = float(np.mean(goal_dist_values))
            agg["final_goal_distance_std"] = float(np.std(goal_dist_values, ddof=0))

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
                    "success": None if ep_success[i] is None else int(ep_success[i]),
                    "final_goal_distance": None if ep_goal_dist[i] is None else float(ep_goal_dist[i]),
                }
                for i in range(episodes)
            ],
            "versions": _gather_versions(),
        }

        out_path = _eval_path(cfg, env_id)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[appendix-e] Wrote eval report: {out_path}")
        return out_path
    finally:
        try:
            env.close()
        except Exception:
            pass
        sim_app.close()


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

        report = {
            "created_at": _now_iso(),
            "total_isaac_envs": len(env_ids),
            "isaac_env_ids": env_ids,
            "peg_candidates": peg,
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
    print("[appendix-e] Smoke run (dense-first wiring check)")
    _train_on_env(cfg, cfg.dense_env_id, cfg.smoke_steps, isaac_extra_args)
    return 0


def cmd_train(cfg: AppendixEConfig, isaac_extra_args: list[str]) -> int:
    # Resolve auto env by reading registry in a booted Isaac session.
    if not cfg.env_id:
        sim_app = _init_isaac(isaac_extra_args)
        _import_isaac_tasks()
        try:
            cfg.env_id = _resolve_train_env_id(cfg)
        finally:
            sim_app.close()

    _train_on_env(cfg, cfg.env_id, cfg.total_steps, isaac_extra_args)
    return 0


def _infer_env_from_ckpt_name(ckpt_name: str) -> str | None:
    # Matches checkpoint names produced by _ckpt_path.
    # appendix_e_sac_<env_id>_seedN.zip
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

    _evaluate_ckpt(cfg, ckpt_path, env_id, isaac_extra_args)
    return 0


def cmd_all(cfg: AppendixEConfig, isaac_extra_args: list[str]) -> int:
    cmd_smoke(cfg, isaac_extra_args)
    cmd_train(cfg, isaac_extra_args)
    ckpt = _ckpt_path(cfg, cfg.env_id)
    _evaluate_ckpt(cfg, ckpt, cfg.env_id, isaac_extra_args)
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
