#!/usr/bin/env python3
"""Isaac Lab proof of life -- validates GPU, environment, and rendering.

Usage (inside Isaac Lab container):
    python scripts/isaac_proof_of_life.py gpu-check
    python scripts/isaac_proof_of_life.py env-step --headless
    python scripts/isaac_proof_of_life.py render --headless
    python scripts/isaac_proof_of_life.py all --headless

Run via Docker:
    bash docker/dev-isaac.sh python3 scripts/isaac_proof_of_life.py all --headless

Note: --enable_cameras is automatically injected for render and all subcommands
(required by Isaac Lab for headless RGB rendering via Vulkan).

Important: Isaac Lab can only create ONE environment per process. The `all`
subcommand creates a single env with render_mode="rgb_array" and uses it
for both env-step and render checks.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ENV_ID = "Isaac-Reach-Franka-v0"
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# gpu-check -- pure PyTorch, no Isaac Lab imports
# ---------------------------------------------------------------------------

def cmd_gpu_check() -> int:
    """Check GPU/CUDA availability using only PyTorch."""
    import torch

    print("== GPU check ==")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        print(f"OK: CUDA available -- {count} device(s), primary: {name}")
        return 0
    else:
        print("FAIL: CUDA not available; Isaac Lab requires an NVIDIA GPU")
        return 1


# ---------------------------------------------------------------------------
# Isaac Lab initialization (must happen before env imports)
# ---------------------------------------------------------------------------

def _init_isaac(extra_args: list[str]) -> tuple:
    """Initialize AppLauncher and return (parsed_args, simulation_app).

    AppLauncher boots Isaac Sim / Omniverse Kit.  This takes 30-90s on
    first run (shader compilation) and ~10s on subsequent runs (cached).
    All Isaac Lab environment imports must happen AFTER this call.
    """
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Isaac Lab proof of life")
    parser.add_argument(
        "--env-id", default=DEFAULT_ENV_ID,
        help="Isaac Lab gymnasium env ID (default: %(default)s)",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(extra_args)

    app_launcher = AppLauncher(args)
    return args, app_launcher.app


def _import_isaac_tasks() -> None:
    """Register Isaac Lab task environments with gymnasium."""
    try:
        import isaaclab_tasks  # noqa: F401 -- registers gymnasium envs
    except ImportError:
        # Older Isaac Lab versions use a different module path
        import omni.isaac.lab_tasks  # noqa: F401


def _make_env(env_id: str, render_mode: str | None = None):
    """Create an Isaac Lab gymnasium environment.

    Uses parse_env_cfg to build the config from the registry entry point,
    then passes it to gym.make() with the cfg kwarg that Isaac Lab requires.
    """
    import gymnasium as gym

    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(env_id, device="cuda:0", num_envs=1)
    kwargs = {"cfg": env_cfg}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)


# ---------------------------------------------------------------------------
# env-step
# ---------------------------------------------------------------------------

def _do_env_step(env) -> dict:
    """Reset + step the given env and return an info dict."""
    import torch

    print("\n== Environment step ==")
    try:
        obs, info = env.reset()

        # Isaac Lab returns numpy from action_space.sample() but expects
        # torch tensors for env.step(). Convert explicitly.
        action_np = env.action_space.sample()
        action = torch.tensor(action_np, device="cuda:0")
        obs, reward, terminated, truncated, info = env.step(action)

        # Report observation structure
        if isinstance(obs, dict):
            obs_shape = {k: tuple(v.shape) for k, v in obs.items()}
        elif hasattr(obs, "shape"):
            obs_shape = tuple(obs.shape)
        else:
            obs_shape = str(type(obs))

        action_shape = tuple(action.shape)

        env_id = env.spec.id if env.spec else "unknown"
        print(f"OK: {env_id} reset + step successful")
        print(f"  obs shape: {obs_shape}")
        print(f"  action shape: {action_shape}")
        return {
            "env_id": env_id,
            "obs_shape": str(obs_shape),
            "action_shape": str(action_shape),
            "status": "ok",
        }
    except Exception as exc:
        print(f"FAIL: env step -- {exc}")
        return {"status": "fail", "error": str(exc)}


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def _do_render(env, out_path: Path) -> dict:
    """Render one frame from the given env and save as PNG."""
    import numpy as np
    import torch

    print("\n== Headless render ==")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Warm up the Vulkan headless renderer: step + render several times.
        # The first few frames are often black while shaders compile.
        for _ in range(5):
            action = torch.tensor(env.action_space.sample(), device="cuda:0")
            env.step(action)
            env.render()

        # Capture the actual frame
        action = torch.tensor(env.action_space.sample(), device="cuda:0")
        env.step(action)
        frame = env.render()

        if frame is None:
            print("FAIL: env.render() returned None")
            return {"status": "fail", "error": "render returned None"}

        # Handle batched frames from vectorized envs
        if isinstance(frame, (list, tuple)):
            frame = frame[0]
        frame = np.asarray(frame)
        if frame.ndim == 4:  # (batch, H, W, C) -> (H, W, C)
            frame = frame[0]

        # Normalize to uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        import imageio.v3 as iio

        iio.imwrite(out_path, frame)
        print(f"OK: wrote {out_path} (shape={frame.shape})")
        return {
            "status": "ok",
            "path": str(out_path),
            "shape": list(frame.shape),
        }
    except Exception as exc:
        print(f"FAIL: render -- {exc}")
        return {"status": "fail", "error": str(exc)}


# ---------------------------------------------------------------------------
# version report
# ---------------------------------------------------------------------------

def _gather_versions() -> dict:
    """Collect version info for Isaac Lab, Isaac Sim, PyTorch, etc."""
    import torch

    versions: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", "N/A"),
    }

    if torch.cuda.is_available():
        versions["gpu"] = torch.cuda.get_device_name(0)
        versions["gpu_count"] = torch.cuda.device_count()

    # Isaac Lab version
    for mod_name in ["isaaclab", "omni.isaac.lab"]:
        try:
            mod = __import__(mod_name)
            versions["isaaclab"] = getattr(mod, "__version__", "unknown")
            break
        except (ImportError, ModuleNotFoundError):
            continue

    # Isaac Sim version
    for mod_name in ["isaacsim", "omni.isaac.version"]:
        try:
            mod = __import__(mod_name)
            versions["isaacsim"] = getattr(mod, "__version__", "unknown")
            break
        except (ImportError, ModuleNotFoundError):
            continue

    # Additional packages
    for pkg in ["numpy", "gymnasium", "stable_baselines3"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except (ImportError, ModuleNotFoundError):
            continue

    return versions


# ---------------------------------------------------------------------------
# subcommand dispatch
# ---------------------------------------------------------------------------

def _ensure_enable_cameras(extra_args: list[str]) -> list[str]:
    """Inject --enable_cameras if not already present (needed for rendering)."""
    if "--enable_cameras" not in extra_args:
        extra_args = ["--enable_cameras"] + extra_args
    return extra_args


def cmd_env_step(extra_args: list[str]) -> int:
    args, sim_app = _init_isaac(extra_args)
    _import_isaac_tasks()
    try:
        env = _make_env(args.env_id)
        try:
            result = _do_env_step(env)
            return 0 if result["status"] == "ok" else 1
        finally:
            env.close()
    finally:
        sim_app.close()


def cmd_render(extra_args: list[str]) -> int:
    extra_args = _ensure_enable_cameras(extra_args)
    args, sim_app = _init_isaac(extra_args)
    _import_isaac_tasks()
    try:
        env = _make_env(args.env_id, render_mode="rgb_array")
        try:
            env.reset()
            out_path = RESULTS_DIR / "isaac_proof_of_life.png"
            result = _do_render(env, out_path)
            return 0 if result["status"] == "ok" else 1
        finally:
            env.close()
    finally:
        sim_app.close()


def cmd_all(extra_args: list[str]) -> int:
    """Run all checks in a single process with a single environment.

    Isaac Lab can only create one simulation scene per process. We create
    one env with render_mode="rgb_array" and use it for both env-step
    and render checks.
    """
    # GPU check first -- no Isaac imports, instant
    rc = cmd_gpu_check()
    if rc != 0:
        return rc

    # Rendering needs --enable_cameras for headless Vulkan rendering
    extra_args = _ensure_enable_cameras(extra_args)

    # Initialize Isaac Lab once
    args, sim_app = _init_isaac(extra_args)
    _import_isaac_tasks()
    try:
        # Create ONE environment for all checks (Isaac Lab singleton constraint)
        env = _make_env(args.env_id, render_mode="rgb_array")
        try:
            # Environment step
            env_result = _do_env_step(env)
            if env_result["status"] != "ok":
                return 1

            # Render (reuses same env).
            # Vulkan headless rendering can fail under memory pressure or if
            # the GPU driver doesn't expose ray-tracing capabilities.  We treat
            # render failure as non-fatal: the env-step already proves the
            # Isaac Lab pipeline works end-to-end.
            png_path = RESULTS_DIR / "isaac_proof_of_life.png"
            render_result = _do_render(env, png_path)
        finally:
            env.close()

        # Version report
        print("\n== Version report ==")
        versions = _gather_versions()
        for k, v in sorted(versions.items()):
            print(f"  {k}: {v}")

        # Save JSON artifact (always, even if render failed)
        json_path = RESULTS_DIR / "isaac_proof_of_life.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "env": env_result,
            "render": render_result,
            "versions": versions,
        }
        json_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"OK: wrote {json_path}")

        # Exit 0 if env worked; render failure is a warning, not a hard error.
        if render_result["status"] != "ok":
            print("\nWARNING: rendering failed (env-step OK). "
                  "This is usually caused by low system RAM or missing "
                  "Vulkan ray-tracing support. Try again when the system "
                  "is less loaded.")
            return 1
        return 0
    finally:
        sim_app.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

COMMANDS = {
    "gpu-check": "Check GPU/CUDA availability (no Isaac Lab startup)",
    "env-step": "Create an Isaac Lab env, reset, and step once",
    "render": "Render one frame and save as PNG",
    "all": "Run gpu-check + env-step + render + version report",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Commands:")
        for name, desc in COMMANDS.items():
            print(f"  {name:12s}  {desc}")
        print(f"\nDefault env: {DEFAULT_ENV_ID}")
        print("For env-step/render/all, pass --headless for headless mode.")
        return 0

    command = sys.argv[1]
    extra = sys.argv[2:]

    if command == "gpu-check":
        return cmd_gpu_check()
    elif command == "env-step":
        return cmd_env_step(extra)
    elif command == "render":
        return cmd_render(extra)
    elif command == "all":
        return cmd_all(extra)
    else:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(COMMANDS)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
