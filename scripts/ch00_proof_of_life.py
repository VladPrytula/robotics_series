#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _set_gl_backend(backend: str, *, force: bool) -> None:
    if force or not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = backend

    mujoco_gl = os.environ.get("MUJOCO_GL", backend)
    if mujoco_gl in {"egl", "osmesa"}:
        if force or not os.environ.get("PYOPENGL_PLATFORM"):
            os.environ["PYOPENGL_PLATFORM"] = mujoco_gl
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)


def _print_gl_env() -> None:
    mujoco_gl = os.environ.get("MUJOCO_GL")
    if mujoco_gl:
        print(f"MUJOCO_GL={mujoco_gl}")
    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM")
    if pyopengl_platform:
        print(f"PYOPENGL_PLATFORM={pyopengl_platform}")


def _should_fallback_from_egl(error: BaseException) -> bool:
    if os.environ.get("MUJOCO_GL") != "egl":
        return False
    msg = str(error)
    if isinstance(error, AttributeError) and "eglQueryString" in msg:
        return True
    lower = msg.lower()
    if isinstance(error, (ImportError, OSError)) and ("libegl" in lower or "egl" in lower):
        return True
    return False


def _should_fallback_from_osmesa(error: BaseException) -> bool:
    if os.environ.get("MUJOCO_GL") != "osmesa":
        return False
    msg = str(error)
    if isinstance(error, AttributeError) and "glGetError" in msg:
        return True
    lower = msg.lower()
    if isinstance(error, (ImportError, OSError)) and ("osmesa" in lower or "libgl" in lower):
        return True
    return False


def _fallback_tried() -> set[str]:
    raw = os.environ.get("ROBOTICS_GL_FALLBACK_TRIED", "")
    return {x for x in raw.split(",") if x}


def _reexec_with_gl_backend(backend: str) -> "None":
    env = os.environ.copy()
    env["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        env["PYOPENGL_PLATFORM"] = backend
    else:
        env.pop("PYOPENGL_PLATFORM", None)
    tried = _fallback_tried()
    tried.add(backend)
    env["ROBOTICS_GL_FALLBACK_TRIED"] = ",".join(sorted(tried))
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _import_fetch():
    try:
        import gymnasium_robotics  # noqa: F401  (registers envs)
    except Exception as exc:
        tried = _fallback_tried()
        if "osmesa" not in tried and _should_fallback_from_egl(exc):
            print("EGL initialization failed; retrying with MUJOCO_GL=osmesa ...", file=sys.stderr)
            _reexec_with_gl_backend("osmesa")
        if "disable" not in tried and _should_fallback_from_osmesa(exc):
            print("OSMesa initialization failed; retrying with MUJOCO_GL=disable (no rendering) ...", file=sys.stderr)
            _reexec_with_gl_backend("disable")
        raise


def list_fetch_envs() -> list[str]:
    import gymnasium as gym

    _import_fetch()
    # Normalize to plain strings for robustness across Gymnasium versions.
    return sorted(str(k) for k in gym.registry.keys() if str(k).startswith("Fetch"))


def pick_env_id(preferred: Iterable[str] | None, explicit: str | None) -> str:
    import gymnasium as gym

    _import_fetch()
    if explicit:
        if explicit not in gym.registry:
            raise SystemExit(f"Env id not found in gym registry: {explicit}")
        return explicit

    preferred_list = list(preferred or [])
    for env_id in preferred_list:
        if env_id in gym.registry:
            return env_id

    def parse_version(env: str) -> int | None:
        # Typical ids look like "FetchReach-v4".
        if "-v" not in env:
            return None
        try:
            return int(env.rsplit("-v", 1)[1])
        except ValueError:
            return None

    def best(prefix: str) -> str | None:
        candidates: list[tuple[int, str]] = []
        for env_id in list_fetch_envs():
            if not env_id.startswith(prefix):
                continue
            ver = parse_version(env_id)
            if ver is None:
                continue
            # v1 Fetch envs typically require mujoco-py; prefer modern mujoco-based versions.
            if ver <= 1:
                continue
            candidates.append((ver, env_id))
        if not candidates:
            return None
        return max(candidates)[1]

    # Prefer ReachDense -> Reach -> anything modern (v>=2).
    env_id = best("FetchReachDense") or best("FetchReach")
    if env_id:
        return env_id

    modern: list[tuple[int, str]] = []
    legacy: list[str] = []
    for env_id in list_fetch_envs():
        ver = parse_version(env_id)
        if ver is None:
            legacy.append(env_id)
        elif ver > 1:
            modern.append((ver, env_id))
        else:
            legacy.append(env_id)

    if modern:
        return max(modern)[1]
    if legacy:
        return sorted(legacy)[0]
    raise SystemExit("No Fetch* envs found in gym registry. Is `gymnasium-robotics` installed?")


def cmd_gpu_check(_: argparse.Namespace) -> int:
    import torch

    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        dev_count = torch.cuda.device_count()
        print(f"OK: CUDA available -- {dev_count} device(s), primary: {dev_name}")
    else:
        print("WARN: CUDA not available; training will use CPU (this is expected on Mac)")
    return 0


def cmd_list_envs(_: argparse.Namespace) -> int:
    _set_gl_backend("disable", force=True)
    _print_gl_env()
    for env_id in list_fetch_envs():
        print(env_id)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import imageio.v3 as iio

    # For rendering we default to EGL, but we must not clobber a backend selected by fallback re-exec.
    if not os.environ.get("ROBOTICS_GL_FALLBACK_TRIED") and os.environ.get("MUJOCO_GL") not in {"egl", "osmesa"}:
        _set_gl_backend("egl", force=True)
    else:
        _set_gl_backend(os.environ.get("MUJOCO_GL", "egl"), force=False)
    _print_gl_env()

    if os.environ.get("MUJOCO_GL") == "disable":
        raise SystemExit(
            "Rendering is unavailable (MUJOCO_GL=disable). "
            "Use an image with libEGL/libGL installed (recommended: rerun `bash docker/dev.sh` so it can build `robotics-rl:latest`)."
        )

    env_id = pick_env_id(
        preferred=args.preferred,
        explicit=args.env_id if args.env_id != "auto" else None,
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    try:
        env.reset(seed=args.seed)
        frame = env.render()
        if frame is None:
            env.step(env.action_space.sample())
            frame = env.render()
        if frame is None:
            raise SystemExit("env.render() returned None (even after a step). Check render_mode and MUJOCO_GL.")
        iio.imwrite(out_path, frame)
    finally:
        env.close()

    print(f"OK: wrote {out_path}")
    return 0


def cmd_ppo_smoke(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    # Training does not require rendering; force-disable OpenGL to avoid DGX driver/GL stack pitfalls.
    _set_gl_backend("disable", force=True)
    _print_gl_env()

    env_id = pick_env_id(
        preferred=args.preferred,
        explicit=args.env_id if args.env_id != "auto" else None,
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.n_envs < 1:
        raise SystemExit("--n-envs must be >= 1")

    _import_fetch()
    env = make_vec_env(env_id, n_envs=args.n_envs, seed=args.seed)
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            verbose=1,
            device=device,
        )
        model.learn(total_timesteps=args.total_steps)
        model.save(str(out_path))
    finally:
        env.close()

    suffix = "" if str(out_path).endswith(".zip") else ".zip"
    print(f"OK: saved {out_path}{suffix} (device={device}, env={env_id})")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    script = str(Path(__file__).resolve())

    def run_step(step_args: list[str], env_overrides: dict[str, str]) -> int:
        env = os.environ.copy()
        env.update(env_overrides)
        env.pop("ROBOTICS_GL_FALLBACK_TRIED", None)
        return subprocess.run([sys.executable, script, *step_args], env=env).returncode

    print("== GPU check ==")
    rc = run_step(["gpu-check"], {})
    if rc != 0:
        return rc

    print("\n== Fetch env registry (Fetch*) ==")
    rc = run_step(["list-envs"], {"MUJOCO_GL": "disable"})
    if rc != 0:
        return rc

    print("\n== Headless render smoke ==")
    render_args = [
        "render",
        "--env-id",
        args.env_id,
        "--seed",
        str(args.seed),
        "--out",
        args.render_out,
        "--preferred",
        *args.preferred,
    ]
    parent_backend = os.environ.get("MUJOCO_GL")
    if parent_backend in {"egl", "osmesa"}:
        backend = parent_backend
    else:
        backend = "egl" if shutil.which("nvidia-smi") else "osmesa"

    render_gl = {"MUJOCO_GL": backend}
    if backend in {"egl", "osmesa"}:
        render_gl["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", backend)
    rc = run_step(render_args, render_gl)
    if rc != 0:
        return rc

    if args.skip_train:
        print("\n== PPO smoke train (skipped) ==")
        return 0

    print("\n== PPO smoke train ==")
    train_args = [
        "ppo-smoke",
        "--env-id",
        args.env_id,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--n-envs",
        str(args.n_envs),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--total-steps",
        str(args.total_steps),
        "--out",
        args.train_out,
        "--preferred",
        *args.preferred,
    ]
    return run_step(train_args, {"MUJOCO_GL": "disable"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spark DGX 'proof of life' utilities for Gymnasium-Robotics Fetch + MuJoCo + SB3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gpu = sub.add_parser("gpu-check", help="Check GPU/CUDA availability via PyTorch.")
    p_gpu.set_defaults(func=cmd_gpu_check)

    p_list = sub.add_parser("list-envs", help="List Fetch* env IDs registered in Gym.")
    p_list.set_defaults(func=cmd_list_envs)

    preferred_default = [
        "FetchReachDense-v4",
        "FetchReachDense-v3",
        "FetchReachDense-v2",
        "FetchReach-v4",
        "FetchReach-v3",
        "FetchReach-v2",
    ]

    p_render = sub.add_parser("render", help="Render one frame headlessly and save it as an image.")
    p_render.add_argument("--env-id", default="auto", help="Gym env id (or 'auto').")
    p_render.add_argument("--preferred", nargs="*", default=preferred_default, help="Preferred env IDs (in order).")
    p_render.add_argument("--seed", type=int, default=0)
    p_render.add_argument("--out", default="smoke_frame.png")
    p_render.set_defaults(func=cmd_render)

    p_train = sub.add_parser("ppo-smoke", help="Short PPO training run to validate end-to-end training.")
    p_train.add_argument("--env-id", default="auto", help="Gym env id (or 'auto').")
    p_train.add_argument("--preferred", nargs="*", default=preferred_default, help="Preferred env IDs (in order).")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p_train.add_argument("--n-envs", type=int, default=8)
    p_train.add_argument("--n-steps", type=int, default=1024, help="SB3 PPO n_steps (rollout length per env).")
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--total-steps", type=int, default=50_000)
    p_train.add_argument("--out", default="ppo_smoke", help="Output path prefix (SB3 appends .zip).")
    p_train.set_defaults(func=cmd_ppo_smoke)

    p_all = sub.add_parser("all", help="Run gpu-check + list-envs + render + PPO smoke train.")
    p_all.add_argument("--env-id", default="auto", help="Gym env id (or 'auto').")
    p_all.add_argument("--preferred", nargs="*", default=preferred_default, help="Preferred env IDs (in order).")
    p_all.add_argument("--seed", type=int, default=0)
    p_all.add_argument("--render-out", default="smoke_frame.png")
    p_all.add_argument("--skip-train", action="store_true")
    p_all.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p_all.add_argument("--n-envs", type=int, default=8)
    p_all.add_argument("--n-steps", type=int, default=1024)
    p_all.add_argument("--batch-size", type=int, default=256)
    p_all.add_argument("--total-steps", type=int, default=50_000)
    p_all.add_argument("--train-out", default="ppo_smoke")
    p_all.set_defaults(func=cmd_all)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
