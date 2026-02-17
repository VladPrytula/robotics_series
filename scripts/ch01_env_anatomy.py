#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _set_mujoco_gl(backend: str) -> None:
    os.environ["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)


def _print_gl_env() -> None:
    for key in ["MUJOCO_GL", "PYOPENGL_PLATFORM"]:
        if os.environ.get(key):
            print(f"{key}={os.environ[key]}")


def _import_fetch() -> None:
    import gymnasium_robotics  # noqa: F401  (registers envs)


def _list_fetch_envs() -> list[str]:
    import gymnasium as gym

    _import_fetch()
    # Normalize to strings for compatibility across Gymnasium versions
    # (some versions return EnvSpec objects from registry.keys())
    return sorted(str(k) for k in gym.registry.keys() if str(k).startswith("Fetch"))


def _pick_env_id(explicit: str | None, preferred: Iterable[str]) -> str:
    import gymnasium as gym

    _import_fetch()
    if explicit and explicit != "auto":
        if explicit not in gym.registry:
            raise SystemExit(f"Env id not found in gym registry: {explicit}")
        return explicit

    for env_id in preferred:
        if env_id in gym.registry:
            return env_id

    def parse_version(env: str) -> int | None:
        if "-v" not in env:
            return None
        try:
            return int(env.rsplit("-v", 1)[1])
        except ValueError:
            return None

    candidates: list[tuple[int, str]] = []
    for env_id in _list_fetch_envs():
        ver = parse_version(env_id)
        if ver is None or ver <= 1:
            continue
        candidates.append((ver, env_id))
    if candidates:
        return max(candidates)[1]

    legacy = _list_fetch_envs()
    if legacy:
        raise SystemExit(
            "Only legacy Fetch envs are available (e.g., -v1), which typically require mujoco-py. "
            "Upgrade gymnasium-robotics to obtain mujoco-based v4 envs."
        )
    raise SystemExit("No Fetch* envs found in gym registry. Is `gymnasium-robotics` installed?")


def _space_summary(space: Any) -> Any:
    try:
        import gymnasium as gym

        spaces = gym.spaces
    except Exception:
        return repr(space)

    if isinstance(space, spaces.Dict):
        return {k: _space_summary(v) for k, v in space.spaces.items()}
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": [float(x) for x in space.low.flat],
            "high": [float(x) for x in space.high.flat],
        }
    return {"type": type(space).__name__, "repr": repr(space)}


def cmd_list_envs(_: argparse.Namespace) -> int:
    _set_mujoco_gl("disable")
    _print_gl_env()
    for env_id in _list_fetch_envs():
        print(env_id)
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    _set_mujoco_gl("disable")
    _print_gl_env()

    import gymnasium as gym

    env_id = _pick_env_id(args.env_id, args.preferred)
    env = gym.make(env_id)
    try:
        obs, info = env.reset(seed=args.seed)
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "env_id": env_id,
            "seed": args.seed,
            "action_space": _space_summary(env.action_space),
            "observation_space": _space_summary(env.observation_space),
            "reset_obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else None,
            "reset_info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
        }
    finally:
        env.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"OK: wrote {out_path}")
    return 0


def cmd_reward_check(args: argparse.Namespace) -> int:
    _set_mujoco_gl("disable")
    _print_gl_env()

    import gymnasium as gym
    import numpy as np

    env_id = _pick_env_id(args.env_id, args.preferred)
    env = gym.make(env_id)
    try:
        if not hasattr(env.unwrapped, "compute_reward"):
            raise SystemExit("env.unwrapped.compute_reward(...) not found; HER-style reward checks are unavailable.")

        obs, info = env.reset(seed=args.seed)
        mismatches: list[dict[str, Any]] = []

        for t in range(args.n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            achieved = np.asarray(obs["achieved_goal"])
            desired = np.asarray(obs["desired_goal"])
            computed = env.unwrapped.compute_reward(achieved, desired, info)
            computed_f = float(np.asarray(computed).item())
            reward_f = float(np.asarray(reward).item())

            if abs(computed_f - reward_f) > args.atol:
                mismatches.append(
                    {
                        "t": t,
                        "reward": reward_f,
                        "computed_reward": computed_f,
                        "abs_err": abs(computed_f - reward_f),
                        "is_success": bool(info.get("is_success", False)),
                    }
                )
                if len(mismatches) >= args.max_mismatches:
                    break

            if terminated or truncated:
                # Use timestep index (not episode count) so each reset gets a unique seed,
                # even if episodes vary in length across runs.
                obs, info = env.reset(seed=args.seed + t + 1)

        if mismatches:
            print(f"FAIL: reward mismatch count={len(mismatches)} (showing up to {args.max_mismatches})", file=sys.stderr)
            print(json.dumps(mismatches, indent=2, sort_keys=True), file=sys.stderr)
            return 2

        print(f"OK: env reward matches compute_reward within atol={args.atol} for n_steps={args.n_steps}")
        return 0
    finally:
        env.close()


def cmd_random_episodes(args: argparse.Namespace) -> int:
    _set_mujoco_gl("disable")
    _print_gl_env()

    import gymnasium as gym
    import numpy as np

    env_id = _pick_env_id(args.env_id, args.preferred)
    env = gym.make(env_id)
    try:
        per_episode: list[dict[str, Any]] = []
        for ep in range(args.n_episodes):
            seed = args.seed + ep
            obs, info = env.reset(seed=seed)
            terminated = truncated = False
            ret = 0.0
            length = 0
            first_success_t: int | None = None
            prev_action = None
            smoothness = 0.0
            max_abs = 0.0

            while not (terminated or truncated):
                action = env.action_space.sample()
                action_arr = np.asarray(action, dtype=np.float32)
                max_abs = float(max(max_abs, float(np.max(np.abs(action_arr)))))
                if prev_action is not None:
                    da = action_arr - prev_action
                    smoothness += float(np.sum(da * da))
                prev_action = action_arr

                obs, reward, terminated, truncated, info = env.step(action)
                ret += float(reward)
                length += 1
                if first_success_t is None and bool(info.get("is_success", False)):
                    first_success_t = length

            achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
            desired = np.asarray(obs["desired_goal"], dtype=np.float32)
            final_distance = float(np.linalg.norm(achieved - desired))
            success = int(bool(info.get("is_success", False)))

            per_episode.append(
                {
                    "episode": ep,
                    "seed": int(seed),
                    "return": float(ret),
                    "length": int(length),
                    "success": int(success),
                    "final_distance": float(final_distance),
                    "time_to_success": None if first_success_t is None else int(first_success_t),
                    "action_smoothness": float(smoothness),
                    "action_max_abs": float(max_abs),
                }
            )

        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "env_id": env_id,
            "n_episodes": args.n_episodes,
            "seed_base": args.seed,
            "aggregate": {
                "success_rate": float(np.mean([x["success"] for x in per_episode])),
                "return_mean": float(np.mean([x["return"] for x in per_episode])),
                "return_std": float(np.std([x["return"] for x in per_episode], ddof=0)),
                "final_distance_mean": float(np.mean([x["final_distance"] for x in per_episode])),
                "final_distance_std": float(np.std([x["final_distance"] for x in per_episode], ddof=0)),
            },
            "per_episode": per_episode,
        }

        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.json_out:
            out_path = Path(args.json_out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"OK: wrote {out_path}")
        return 0
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chapter 1 utilities: inspect Fetch env anatomy (obs/action/reward semantics).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    preferred_default = [
        "FetchReachDense-v4",
        "FetchReachDense-v3",
        "FetchReachDense-v2",
        "FetchReach-v4",
        "FetchReach-v3",
        "FetchReach-v2",
    ]

    p_list = sub.add_parser("list-envs", help="List Fetch* env IDs registered in Gym.")
    p_list.set_defaults(func=cmd_list_envs)

    p_desc = sub.add_parser("describe", help="Print action/observation space schema as JSON.")
    p_desc.add_argument("--env-id", default="auto")
    p_desc.add_argument("--preferred", nargs="*", default=preferred_default)
    p_desc.add_argument("--seed", type=int, default=0)
    p_desc.add_argument("--json-out", default="", help="Write JSON summary to this path (optional).")
    p_desc.set_defaults(func=cmd_describe)

    p_reward = sub.add_parser("reward-check", help="Sanity-check env reward against env.unwrapped.compute_reward.")
    p_reward.add_argument("--env-id", default="auto")
    p_reward.add_argument("--preferred", nargs="*", default=preferred_default)
    p_reward.add_argument("--seed", type=int, default=0)
    p_reward.add_argument("--n-steps", type=int, default=500)
    p_reward.add_argument("--atol", type=float, default=1e-6)
    p_reward.add_argument("--max-mismatches", type=int, default=5)
    p_reward.set_defaults(func=cmd_reward_check)

    p_rand = sub.add_parser("random-episodes", help="Run N random episodes and emit a metrics JSON.")
    p_rand.add_argument("--env-id", default="auto")
    p_rand.add_argument("--preferred", nargs="*", default=preferred_default)
    p_rand.add_argument("--seed", type=int, default=0)
    p_rand.add_argument("--n-episodes", type=int, default=10)
    p_rand.add_argument("--json-out", default="", help="Write JSON summary to this path (optional).")
    p_rand.set_defaults(func=cmd_random_episodes)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

