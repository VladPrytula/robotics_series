#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _set_mujoco_gl(backend: str) -> None:
    os.environ["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)


def _parse_seeds(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    if "," in raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    if "-" in raw:
        lo, hi = raw.split("-", 1)
        start, end = int(lo.strip()), int(hi.strip())
        if end < start:
            raise SystemExit(f"Invalid seed range: {raw}")
        return list(range(start, end + 1))
    return [int(raw)]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _gather_versions() -> dict[str, str]:
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


def _load_model(algo: str, ckpt: str, *, device: str):
    from stable_baselines3 import PPO, SAC, TD3

    candidates: list[tuple[str, object]] = [("ppo", PPO), ("sac", SAC), ("td3", TD3)]
    if algo != "auto":
        candidates = [(algo, {"ppo": PPO, "sac": SAC, "td3": TD3}[algo])]

    errors: list[str] = []
    for name, cls in candidates:
        try:
            model = cls.load(ckpt, device=device)
            return name, model
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
    raise SystemExit("Could not load checkpoint with PPO/SAC/TD3.\n" + "\n".join(errors))


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an SB3 checkpoint on Gymnasium-Robotics Fetch tasks and emit a JSON metrics file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", required=True, help="Path to SB3 .zip checkpoint.")
    parser.add_argument("--env", dest="env_id", required=True, help="Gym env id (e.g., FetchReachDense-v4).")
    parser.add_argument("--algo", choices=["auto", "ppo", "sac", "td3"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Base seed used when --seeds is not provided.")
    parser.add_argument("--seeds", default="", help="Comma list (e.g., 0,1,2) or range (e.g., 0-99).")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--json-out", default="results/metrics.json")
    parser.add_argument("--video", action="store_true", help="Record the first episode to an mp4 (requires EGL).")
    parser.add_argument("--video-out", default="videos/eval.mp4")
    args = parser.parse_args()

    if args.n_episodes < 1:
        raise SystemExit("--n-episodes must be >= 1")

    if args.video:
        _set_mujoco_gl("egl")
    else:
        _set_mujoco_gl("disable")

    device = _resolve_device(args.device)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    out_path = Path(args.json_out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed + i for i in range(args.n_episodes)]
    if len(seeds) < args.n_episodes:
        raise SystemExit(f"Not enough seeds ({len(seeds)}) for --n-episodes={args.n_episodes}")
    seeds = seeds[: args.n_episodes]

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401  (registers envs)
    import numpy as np

    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.env_id, render_mode=render_mode)
    try:
        algo_name, model = _load_model(args.algo, str(ckpt_path), device=device)

        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        episode_success: list[int] = []
        episode_final_distance: list[float] = []
        episode_time_to_success: list[int | None] = []
        episode_action_smoothness: list[float] = []
        episode_action_max_abs: list[float] = []

        video_frames: list[object] = []

        for ep, ep_seed in enumerate(seeds):
            obs, info = env.reset(seed=int(ep_seed))
            terminated = truncated = False
            ep_return = 0.0
            ep_len = 0
            first_success_t: int | None = None
            prev_action = None
            smoothness = 0.0
            max_abs = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                action_arr = np.asarray(action, dtype=np.float32)
                max_abs = float(max(max_abs, float(np.max(np.abs(action_arr)))))
                if prev_action is not None:
                    da = action_arr - prev_action
                    smoothness += float(np.sum(da * da))
                prev_action = action_arr

                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                ep_len += 1

                if first_success_t is None and bool(info.get("is_success", False)):
                    first_success_t = ep_len

                if args.video and ep == 0:
                    frame = env.render()
                    if frame is not None:
                        video_frames.append(frame)

            achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
            desired = np.asarray(obs["desired_goal"], dtype=np.float32)
            final_distance = float(np.linalg.norm(achieved - desired))
            success = int(bool(info.get("is_success", False)))

            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)
            episode_success.append(success)
            episode_final_distance.append(final_distance)
            episode_time_to_success.append(first_success_t)
            episode_action_smoothness.append(float(smoothness))
            episode_action_max_abs.append(float(max_abs))

        if args.video and video_frames:
            try:
                import imageio.v3 as iio

                video_out = Path(args.video_out).expanduser().resolve()
                video_out.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(video_out, video_frames, fps=20)
            except Exception as exc:
                print(f"WARNING: video write failed: {exc}", file=sys.stderr)

        metrics = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "algo": algo_name,
            "env_id": args.env_id,
            "checkpoint": str(ckpt_path),
            "device": device,
            "n_episodes": args.n_episodes,
            "seeds": seeds,
            "deterministic": bool(args.deterministic),
            "aggregate": {
                "success_rate": _mean(episode_success),
                "return_mean": _mean(episode_returns),
                "return_std": float(np.std(np.asarray(episode_returns), ddof=0)),
                "final_distance_mean": _mean(episode_final_distance),
                "final_distance_std": float(np.std(np.asarray(episode_final_distance), ddof=0)),
                "ep_len_mean": _mean(episode_lengths),
                "ep_len_std": float(np.std(np.asarray(episode_lengths), ddof=0)),
                "action_smoothness_mean": _mean(episode_action_smoothness),
                "action_max_abs_mean": _mean(episode_action_max_abs),
            },
            "per_episode": [
                {
                    "seed": int(seeds[i]),
                    "return": float(episode_returns[i]),
                    "length": int(episode_lengths[i]),
                    "success": int(episode_success[i]),
                    "final_distance": float(episode_final_distance[i]),
                    "time_to_success": None if episode_time_to_success[i] is None else int(episode_time_to_success[i]),
                    "action_smoothness": float(episode_action_smoothness[i]),
                    "action_max_abs": float(episode_action_max_abs[i]),
                }
                for i in range(args.n_episodes)
            ],
            "versions": _gather_versions(),
        }

        out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        env.close()

    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

