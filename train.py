#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _set_mujoco_gl(backend: str) -> None:
    os.environ["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SB3 agents on Gymnasium-Robotics Fetch tasks (DGX/Docker-first).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo", choices=["ppo", "sac", "td3"], required=True)
    parser.add_argument("--env", dest="env_id", required=True, help="Gym env id (e.g., FetchReachDense-v4).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument(
        "--out",
        default=None,
        help="Output checkpoint path prefix (SB3 appends .zip). Defaults to checkpoints/<algo>_<env>_seed<seed>.",
    )
    parser.add_argument("--track", choices=["none", "tb"], default="tb")
    parser.add_argument("--log-dir", default="runs", help="TensorBoard log root (used when --track=tb).")

    ppo = parser.add_argument_group("PPO")
    ppo.add_argument("--ppo-n-steps", type=int, default=1024)
    ppo.add_argument("--ppo-batch-size", type=int, default=256)

    off = parser.add_argument_group("SAC/TD3")
    off.add_argument("--off-batch-size", type=int, default=256)
    off.add_argument("--off-buffer-size", type=int, default=1_000_000)
    off.add_argument("--off-learning-starts", type=int, default=10_000)

    her = parser.add_argument_group("HER (SAC/TD3 only)")
    her.add_argument("--her", action="store_true")
    her.add_argument("--her-n-sampled-goal", type=int, default=4)
    her.add_argument("--her-goal-selection-strategy", choices=["future", "final", "episode"], default="future")

    return parser.parse_args()


def _default_out(env_id: str, algo: str, seed: int) -> Path:
    safe_env = env_id.replace("/", "_").replace(":", "_")
    return Path("checkpoints") / f"{algo}_{safe_env}_seed{seed}"


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


def main() -> int:
    args = _parse_args()

    _set_mujoco_gl("disable")
    device = _resolve_device(args.device)

    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(args.seed)
    except Exception:
        pass

    import gymnasium_robotics  # noqa: F401  (registers envs)
    from stable_baselines3.common.env_util import make_vec_env

    if args.n_envs < 1:
        raise SystemExit("--n-envs must be >= 1")

    out_path = Path(args.out) if args.out else _default_out(args.env_id, args.algo, args.seed)
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = f"{args.algo}/{args.env_id}/seed{args.seed}"
    tb_log = str(Path(args.log_dir).expanduser().resolve()) if args.track == "tb" else None

    env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.seed)
    try:
        if args.algo == "ppo":
            from stable_baselines3 import PPO

            model = PPO(
                "MultiInputPolicy",
                env,
                n_steps=args.ppo_n_steps,
                batch_size=args.ppo_batch_size,
                verbose=1,
                device=device,
                tensorboard_log=tb_log,
            )
        else:
            if args.her:
                try:
                    from stable_baselines3 import HerReplayBuffer
                except Exception:
                    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

                replay_buffer_class = HerReplayBuffer
                replay_buffer_kwargs = {
                    "n_sampled_goal": args.her_n_sampled_goal,
                    "goal_selection_strategy": args.her_goal_selection_strategy,
                }
            else:
                replay_buffer_class = None
                replay_buffer_kwargs = None

            if args.algo == "sac":
                from stable_baselines3 import SAC

                model = SAC(
                    "MultiInputPolicy",
                    env,
                    verbose=1,
                    device=device,
                    tensorboard_log=tb_log,
                    batch_size=args.off_batch_size,
                    buffer_size=args.off_buffer_size,
                    learning_starts=args.off_learning_starts,
                    replay_buffer_class=replay_buffer_class,
                    replay_buffer_kwargs=replay_buffer_kwargs,
                )
            else:
                from stable_baselines3 import TD3

                model = TD3(
                    "MultiInputPolicy",
                    env,
                    verbose=1,
                    device=device,
                    tensorboard_log=tb_log,
                    batch_size=args.off_batch_size,
                    buffer_size=args.off_buffer_size,
                    learning_starts=args.off_learning_starts,
                    replay_buffer_class=replay_buffer_class,
                    replay_buffer_kwargs=replay_buffer_kwargs,
                )

        model.learn(total_timesteps=args.total_steps, tb_log_name=run_id if args.track == "tb" else None)
        model.save(str(out_path))
    finally:
        env.close()

    suffix = "" if str(out_path).endswith(".zip") else ".zip"
    meta_path = out_path.with_suffix(".meta.json") if suffix else Path(str(out_path) + ".meta.json")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algo": args.algo,
        "env_id": args.env_id,
        "seed": args.seed,
        "device": device,
        "n_envs": args.n_envs,
        "total_steps": args.total_steps,
        "checkpoint": str(out_path) + suffix,
        "versions": _gather_versions(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"OK: saved {out_path}{suffix}")
    print(f"OK: wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

