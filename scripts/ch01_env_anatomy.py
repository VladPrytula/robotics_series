#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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

    def json_scalar(x: Any) -> float | str:
        try:
            xf = float(x)
        except Exception:
            return repr(x)
        if math.isfinite(xf):
            return xf
        return "inf" if xf > 0 else "-inf"

    if isinstance(space, spaces.Dict):
        return {k: _space_summary(v) for k, v in space.spaces.items()}
    if isinstance(space, spaces.Box):
        low = [json_scalar(x) for x in space.low.flat]
        high = [json_scalar(x) for x in space.high.flat]
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": low,
            "high": high,
        }
    return {"type": type(space).__name__, "repr": repr(space)}

def _gather_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.replace("\n", " ")}
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", "unknown")
        versions["torch_cuda"] = str(getattr(torch.version, "cuda", "unknown"))
    except Exception:
        pass
    for module_name in ["gymnasium", "gymnasium_robotics", "mujoco", "stable_baselines3", "numpy"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            continue
    return versions


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
        max_episode_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        distance_threshold = getattr(env.unwrapped, "distance_threshold", None)
        reward_type = getattr(env.unwrapped, "reward_type", None)
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "env_id": env_id,
            "seed": args.seed,
            "max_episode_steps": max_episode_steps,
            "distance_threshold": None if distance_threshold is None else float(distance_threshold),
            "reward_type": None if reward_type is None else str(reward_type),
            "action_space": _space_summary(env.action_space),
            "observation_space": _space_summary(env.observation_space),
            "reset_obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else None,
            "reset_info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
            "versions": _gather_versions(),
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

        reward_type = getattr(env.unwrapped, "reward_type", None)
        if reward_type is None:
            reward_type = "dense" if "Dense" in env_id else "sparse"
        reward_type = str(reward_type).lower()
        if reward_type not in {"dense", "sparse"}:
            raise SystemExit(f"Unexpected reward_type={reward_type!r}; expected 'dense' or 'sparse'.")

        distance_threshold = getattr(env.unwrapped, "distance_threshold", None)
        if distance_threshold is None:
            distance_threshold = 0.05
        distance_threshold = float(distance_threshold)

        def expected_reward(*, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
            dist = float(np.linalg.norm(achieved_goal - desired_goal))
            if reward_type == "dense":
                return -dist
            # Gymnasium-Robotics sparse Fetch rewards are 0 on success, -1 otherwise.
            # Use <= to match the common implementation (-[d > threshold]).
            return 0.0 if dist <= distance_threshold else -1.0

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
                        "kind": "step_vs_compute_reward",
                        "reward": reward_f,
                        "computed_reward": computed_f,
                        "abs_err": abs(computed_f - reward_f),
                        "is_success": bool(info.get("is_success", False)),
                    }
                )
                if len(mismatches) >= args.max_mismatches:
                    break

            expected_f = expected_reward(achieved_goal=achieved, desired_goal=desired)
            if abs(computed_f - expected_f) > args.atol:
                mismatches.append(
                    {
                        "t": t,
                        "kind": "compute_reward_vs_distance_formula",
                        "reward_type": reward_type,
                        "distance_threshold": distance_threshold,
                        "computed_reward": computed_f,
                        "expected_reward": expected_f,
                        "distance": float(np.linalg.norm(achieved - desired)),
                        "abs_err": abs(computed_f - expected_f),
                    }
                )
                if len(mismatches) >= args.max_mismatches:
                    break

            if args.n_random_goals > 0:
                goal_space = None
                if hasattr(env.observation_space, "spaces"):
                    goal_space = env.observation_space.spaces.get("desired_goal")
                for j in range(args.n_random_goals):
                    if goal_space is None:
                        break
                    desired2 = np.asarray(goal_space.sample())
                    computed2 = env.unwrapped.compute_reward(achieved, desired2, info)
                    computed2_f = float(np.asarray(computed2).item())
                    expected2_f = expected_reward(achieved_goal=achieved, desired_goal=desired2)
                    if abs(computed2_f - expected2_f) > args.atol:
                        mismatches.append(
                            {
                                "t": t,
                                "kind": "compute_reward_random_goal_vs_distance_formula",
                                "reward_type": reward_type,
                                "distance_threshold": distance_threshold,
                                "computed_reward": computed2_f,
                                "expected_reward": expected2_f,
                                "distance": float(np.linalg.norm(achieved - desired2)),
                                "abs_err": abs(computed2_f - expected2_f),
                            }
                        )
                        if len(mismatches) >= args.max_mismatches:
                            break
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

        print(
            "OK: reward checks passed "
            f"(reward_type={reward_type}, distance_threshold={distance_threshold}, "
            f"atol={args.atol}, n_steps={args.n_steps}, n_random_goals={args.n_random_goals})"
        )
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
                "ep_len_mean": float(np.mean([x["length"] for x in per_episode])),
                "ep_len_std": float(np.std([x["length"] for x in per_episode], ddof=0)),
                "action_smoothness_mean": float(np.mean([x["action_smoothness"] for x in per_episode])),
                "action_max_abs_mean": float(np.mean([x["action_max_abs"] for x in per_episode])),
            },
            "per_episode": per_episode,
            "versions": _gather_versions(),
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

def cmd_all(args: argparse.Namespace) -> int:
    script = str(Path(__file__).resolve())

    def run_step(step_args: list[str]) -> int:
        env = os.environ.copy()
        env["MUJOCO_GL"] = "disable"
        env.pop("PYOPENGL_PLATFORM", None)
        return subprocess.run([sys.executable, script, *step_args], env=env).returncode

    describe_args = [
        "describe",
        "--env-id",
        args.env_id,
        "--seed",
        str(args.seed),
        "--json-out",
        args.describe_out,
        "--preferred",
        *args.preferred,
    ]

    reward_args = [
        "reward-check",
        "--env-id",
        args.env_id,
        "--seed",
        str(args.seed),
        "--n-steps",
        str(args.n_steps),
        "--atol",
        str(args.atol),
        "--max-mismatches",
        str(args.max_mismatches),
        "--n-random-goals",
        str(args.n_random_goals),
        "--preferred",
        *args.preferred,
    ]

    random_args = [
        "random-episodes",
        "--env-id",
        args.env_id,
        "--seed",
        str(args.seed),
        "--n-episodes",
        str(args.n_episodes),
        "--json-out",
        args.random_out,
        "--preferred",
        *args.preferred,
    ]

    print("== Describe obs/action spaces ==")
    rc = run_step(describe_args)
    if rc != 0:
        return rc

    print("\n== Reward consistency check ==")
    rc = run_step(reward_args)
    if rc != 0:
        return rc

    print("\n== Random policy baseline ==")
    return run_step(random_args)


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
    p_reward.add_argument(
        "--n-random-goals",
        type=int,
        default=3,
        help="Per step, sample this many random desired goals and verify compute_reward matches the distance-based formula.",
    )
    p_reward.set_defaults(func=cmd_reward_check)

    p_rand = sub.add_parser("random-episodes", help="Run N random episodes and emit a metrics JSON.")
    p_rand.add_argument("--env-id", default="auto")
    p_rand.add_argument("--preferred", nargs="*", default=preferred_default)
    p_rand.add_argument("--seed", type=int, default=0)
    p_rand.add_argument("--n-episodes", type=int, default=10)
    p_rand.add_argument("--json-out", default="", help="Write JSON summary to this path (optional).")
    p_rand.set_defaults(func=cmd_random_episodes)

    p_all = sub.add_parser(
        "all",
        help="Run describe + reward-check + random-episodes (writes results JSON artifacts).",
    )
    p_all.add_argument("--env-id", default="auto")
    p_all.add_argument("--preferred", nargs="*", default=preferred_default)
    p_all.add_argument("--seed", type=int, default=0)
    p_all.add_argument("--describe-out", default="results/ch01_env_describe.json")
    p_all.add_argument("--random-out", default="results/ch01_random_metrics.json")
    p_all.add_argument("--n-episodes", type=int, default=10, help="Random-policy episodes for the baseline report.")
    p_all.add_argument("--n-steps", type=int, default=500, help="Steps for reward-check.")
    p_all.add_argument("--atol", type=float, default=1e-6, help="Tolerance for reward-check.")
    p_all.add_argument("--max-mismatches", type=int, default=5, help="Stop after this many reward-check mismatches.")
    p_all.add_argument(
        "--n-random-goals",
        type=int,
        default=3,
        help="Per step, sample this many random desired goals for reward-check.",
    )
    p_all.set_defaults(func=cmd_all)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
