#!/usr/bin/env python3
"""
Generate annotated environment screenshots for the Manning book proposal.

Creates publication-quality PNG images showing Fetch environments with
labeled components (goals, gripper, observation structure) and diagrams
(dense vs sparse reward, obs dict structure).

All figures are self-generated from MuJoCo renders -- no licensing issues.

Usage:
    python scripts/capture_proposal_figures.py env-setup [--envs ...] [--out-dir figures/]
    python scripts/capture_proposal_figures.py reward-diagram [--out-dir figures/]
    python scripts/capture_proposal_figures.py compare --env ... --ckpt ... [--out-dir figures/]
    python scripts/capture_proposal_figures.py all [--out-dir figures/]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# GL backend management (from ch00_proof_of_life.py pattern)
# ---------------------------------------------------------------------------
def _set_gl_backend(backend: str, *, force: bool) -> None:
    if force or not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = backend
    mujoco_gl = os.environ.get("MUJOCO_GL", backend)
    if mujoco_gl in {"egl", "osmesa"}:
        if force or not os.environ.get("PYOPENGL_PLATFORM"):
            os.environ["PYOPENGL_PLATFORM"] = mujoco_gl
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)


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


def _reexec_with_gl_backend(backend: str) -> None:
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
            print("EGL failed; retrying with MUJOCO_GL=osmesa ...", file=sys.stderr)
            _reexec_with_gl_backend("osmesa")
        if "disable" not in tried and _should_fallback_from_osmesa(exc):
            print("OSMesa failed; retrying with MUJOCO_GL=disable ...", file=sys.stderr)
            _reexec_with_gl_backend("disable")
        raise


# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011)
# ---------------------------------------------------------------------------
COLOR_BLUE = "#0072B2"
COLOR_ORANGE = "#E69F00"
COLOR_GREEN = "#009E73"
COLOR_VERMILLION = "#D55E00"
COLOR_GRAY = "#999999"

# RGB tuples for PIL
RGB_BLUE = (0, 114, 178)
RGB_ORANGE = (230, 159, 0)
RGB_GREEN = (0, 158, 115)
RGB_VERMILLION = (213, 94, 0)
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RGB_SHADOW = (30, 30, 30)


# ---------------------------------------------------------------------------
# Font loading (from generate_demo_videos.py pattern)
# ---------------------------------------------------------------------------
def _load_font(size: int = 16):
    from PIL import ImageFont

    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_font_bold(size: int = 16):
    from PIL import ImageFont

    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------
def _annotate_frame(frame, labels: list[dict]) -> "ndarray":
    """Add text labels with shadow to a frame.

    Each label dict: {"text": str, "xy": (x, y), "color": (r, g, b), "size": int}
    """
    from PIL import Image, ImageDraw
    import numpy as np

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    for label in labels:
        text = label["text"]
        x, y = label["xy"]
        color = label.get("color", RGB_WHITE)
        size = label.get("size", 16)
        font = _load_font_bold(size) if label.get("bold") else _load_font(size)

        # Shadow for readability
        draw.text((x + 2, y + 2), text, fill=RGB_SHADOW, font=font)
        draw.text((x, y), text, fill=color, font=font)

    return np.array(img)


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------
ENV_INFO = {
    "FetchReach-v4": {
        "name": "Fetch Reach",
        "file_stem": "fetch_reach_setup",
        "task": "Move gripper to target",
        "object": "None",
        "obs_dim": 10,
        "goal_dim": 3,
        "achieved_label": "achieved_goal (gripper tip)",
    },
    "FetchPush-v4": {
        "name": "Fetch Push",
        "file_stem": "fetch_push_setup",
        "task": "Push block to target",
        "object": "Block",
        "obs_dim": 25,
        "goal_dim": 3,
        "achieved_label": "achieved_goal (block)",
    },
    "FetchPickAndPlace-v4": {
        "name": "Fetch Pick and Place",
        "file_stem": "fetch_pick_and_place_setup",
        "task": "Pick up block and place at target",
        "object": "Block",
        "obs_dim": 25,
        "goal_dim": 3,
        "achieved_label": "achieved_goal (block)",
    },
}

DEFAULT_ENVS = list(ENV_INFO.keys())


# ---------------------------------------------------------------------------
# Core rendering functions
# ---------------------------------------------------------------------------
def _render_env_setup(
    env_id: str,
    seed: int,
    width: int,
    height: int,
    out_path: Path,
) -> None:
    """Render one annotated reset screenshot for an environment."""
    import gymnasium as gym
    import numpy as np

    info = ENV_INFO[env_id]
    print(f"  Rendering {info['name']} -> {out_path}")

    env = gym.make(env_id, render_mode="rgb_array", width=width, height=height)
    obs, _ = env.reset(seed=seed)
    frame = env.render()

    # Compute distance
    goal = obs["desired_goal"]
    achieved = obs["achieved_goal"]
    dist = np.linalg.norm(goal - achieved)

    # Determine success threshold (Fetch default epsilon = 0.05)
    reward = env.unwrapped.compute_reward(achieved, goal, {})
    is_sparse = env_id.endswith("-v4") and "Dense" not in env_id

    # Build annotation labels
    labels = [
        # Title
        {
            "text": info["name"],
            "xy": (10, 8),
            "color": RGB_WHITE,
            "size": 22,
            "bold": True,
        },
        # Task description
        {
            "text": f"Task: {info['task']}",
            "xy": (10, 36),
            "color": RGB_ORANGE,
            "size": 14,
        },
        # Goal label
        {
            "text": "desired_goal (target)",
            "xy": (10, height - 90),
            "color": RGB_VERMILLION,
            "size": 14,
            "bold": True,
        },
        # Achieved label
        {
            "text": info["achieved_label"],
            "xy": (10, height - 70),
            "color": RGB_GREEN,
            "size": 14,
            "bold": True,
        },
        # Action space
        {
            "text": "Action: [dx, dy, dz, grip]",
            "xy": (10, height - 48),
            "color": RGB_BLUE,
            "size": 13,
        },
        # Distance and reward info
        {
            "text": f"Distance: {dist:.3f}m | Reward: {reward:.2f} ({'sparse' if is_sparse else 'dense'})",
            "xy": (10, height - 26),
            "color": RGB_WHITE,
            "size": 13,
        },
    ]

    annotated = _annotate_frame(frame, labels)

    import imageio.v3 as iio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, annotated)
    env.close()
    print(f"    Saved {out_path} ({annotated.shape[1]}x{annotated.shape[0]})")


def _render_reward_diagram(out_path: Path) -> None:
    """Generate a matplotlib plot comparing dense vs sparse reward curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"  Rendering reward diagram -> {out_path}")

    distances = np.linspace(0, 0.3, 200)
    epsilon = 0.05

    # Dense reward: -distance
    dense_reward = -distances

    # Sparse reward: 0 if distance < epsilon, else -1
    sparse_reward = np.where(distances < epsilon, 0.0, -1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # Dense reward plot
    ax1.plot(distances, dense_reward, color=COLOR_BLUE, linewidth=2.5, label="Dense: $r = -\\|g - g'\\|$")
    ax1.set_xlabel("Distance to goal (m)", fontsize=11)
    ax1.set_ylabel("Reward", fontsize=11)
    ax1.set_title("Dense Reward", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.3)
    ax1.set_ylim(-0.35, 0.05)

    # Sparse reward plot
    ax2.plot(distances, sparse_reward, color=COLOR_VERMILLION, linewidth=2.5,
             label="Sparse: $r = -\\mathbb{1}[\\|g - g'\\| \\geq \\epsilon]$")
    ax2.axvline(x=epsilon, color=COLOR_GRAY, linestyle="--", alpha=0.7,
                label=f"$\\epsilon$ = {epsilon}")
    ax2.set_xlabel("Distance to goal (m)", fontsize=11)
    ax2.set_ylabel("Reward", fontsize=11)
    ax2.set_title("Sparse Reward", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="center right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.3)
    ax2.set_ylim(-1.2, 0.2)

    fig.suptitle("Dense vs Sparse Rewards in Fetch Environments", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved {out_path}")


def _render_obs_structure(out_path: Path) -> None:
    """Generate a diagram showing the observation dictionary structure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print(f"  Rendering obs structure diagram -> {out_path}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(5, 5.5, "Goal-Conditioned Observation Dictionary", fontsize=15,
            fontweight="bold", ha="center", va="center")

    # Main obs dict box
    main_box = mpatches.FancyBboxPatch(
        (0.5, 0.3), 9, 4.8, boxstyle="round,pad=0.15",
        facecolor="#f0f0f0", edgecolor="#333333", linewidth=2,
    )
    ax.add_patch(main_box)
    ax.text(1.0, 4.7, "obs = env.reset()", fontsize=11,
            fontfamily="monospace", color="#333333")

    # observation box
    obs_box = mpatches.FancyBboxPatch(
        (0.8, 2.8), 2.5, 1.5, boxstyle="round,pad=0.1",
        facecolor=COLOR_BLUE, edgecolor=COLOR_BLUE, alpha=0.15, linewidth=1.5,
    )
    ax.add_patch(obs_box)
    ax.text(2.05, 4.0, '"observation"', fontsize=10, fontweight="bold",
            ha="center", fontfamily="monospace", color=COLOR_BLUE)
    ax.text(2.05, 3.55, "Robot state", fontsize=9, ha="center", color="#333333")
    ax.text(2.05, 3.2, "gripper pos, vel,\nobject pos, rel pos", fontsize=8,
            ha="center", color="#666666")

    # desired_goal box
    goal_box = mpatches.FancyBboxPatch(
        (3.7, 2.8), 2.5, 1.5, boxstyle="round,pad=0.1",
        facecolor=COLOR_VERMILLION, edgecolor=COLOR_VERMILLION, alpha=0.15, linewidth=1.5,
    )
    ax.add_patch(goal_box)
    ax.text(4.95, 4.0, '"desired_goal"', fontsize=10, fontweight="bold",
            ha="center", fontfamily="monospace", color=COLOR_VERMILLION)
    ax.text(4.95, 3.55, "Target position", fontsize=9, ha="center", color="#333333")
    ax.text(4.95, 3.2, "[x, y, z]\nshape: (3,)", fontsize=8,
            ha="center", fontfamily="monospace", color="#666666")

    # achieved_goal box
    ach_box = mpatches.FancyBboxPatch(
        (6.6, 2.8), 2.5, 1.5, boxstyle="round,pad=0.1",
        facecolor=COLOR_GREEN, edgecolor=COLOR_GREEN, alpha=0.15, linewidth=1.5,
    )
    ax.add_patch(ach_box)
    ax.text(7.85, 4.0, '"achieved_goal"', fontsize=10, fontweight="bold",
            ha="center", fontfamily="monospace", color=COLOR_GREEN)
    ax.text(7.85, 3.55, "Current position", fontsize=9, ha="center", color="#333333")
    ax.text(7.85, 3.2, "[x, y, z]\nshape: (3,)", fontsize=8,
            ha="center", fontfamily="monospace", color="#666666")

    # Reward computation arrow and box
    ax.annotate(
        "", xy=(5.0, 2.1), xytext=(5.0, 2.7),
        arrowprops=dict(arrowstyle="->", color=COLOR_ORANGE, lw=2),
    )
    ax.annotate(
        "", xy=(7.85, 2.1), xytext=(7.85, 2.7),
        arrowprops=dict(arrowstyle="->", color=COLOR_ORANGE, lw=2),
    )

    reward_box = mpatches.FancyBboxPatch(
        (3.5, 0.6), 5.5, 1.3, boxstyle="round,pad=0.1",
        facecolor=COLOR_ORANGE, edgecolor=COLOR_ORANGE, alpha=0.15, linewidth=1.5,
    )
    ax.add_patch(reward_box)
    ax.text(6.25, 1.6, "compute_reward(achieved_goal, desired_goal, info)",
            fontsize=9, fontweight="bold", ha="center", fontfamily="monospace",
            color=COLOR_ORANGE)
    ax.text(6.25, 1.1, "Dense: $r = -\\|g - g'\\|$    Sparse: $r = -\\mathbb{1}[\\|g - g'\\| \\geq \\epsilon]$",
            fontsize=9, ha="center", color="#333333")

    # Dimension annotations
    ax.text(2.05, 2.9, "Reach: (10,)  Push: (25,)", fontsize=7,
            ha="center", fontfamily="monospace", color=COLOR_GRAY)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved {out_path}")


def _render_comparison(
    env_id: str,
    ckpt_path: Path,
    seed: int,
    width: int,
    height: int,
    out_path: Path,
) -> None:
    """Render side-by-side: random policy vs trained policy."""
    import gymnasium as gym
    import numpy as np

    info = ENV_INFO.get(env_id, {"name": env_id, "file_stem": env_id.lower().replace("-", "_")})
    print(f"  Rendering comparison for {info['name']} -> {out_path}")

    # Random policy frame
    env = gym.make(env_id, render_mode="rgb_array", width=width, height=height)
    obs, _ = env.reset(seed=seed)
    # Run 25 random steps to show typical random behavior
    for _ in range(25):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    random_frame = env.render()
    random_dist = np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])
    env.close()

    # Trained policy frame
    from stable_baselines3 import PPO, SAC, TD3

    env = gym.make(env_id, render_mode="rgb_array", width=width, height=height)
    model = None
    for cls in [SAC, PPO, TD3]:
        try:
            model = cls.load(str(ckpt_path), device="cpu")
            break
        except Exception:
            continue
    if model is None:
        print(f"    WARNING: Could not load checkpoint {ckpt_path}, skipping comparison")
        env.close()
        return

    obs, _ = env.reset(seed=seed)
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    trained_frame = env.render()
    trained_dist = np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])
    env.close()

    # Annotate both frames
    random_labels = [
        {"text": "Random Policy", "xy": (10, 8), "color": RGB_VERMILLION, "size": 18, "bold": True},
        {"text": f"Distance: {random_dist:.3f}m", "xy": (10, height - 26), "color": RGB_WHITE, "size": 13},
    ]
    trained_labels = [
        {"text": "Trained Policy", "xy": (10, 8), "color": RGB_GREEN, "size": 18, "bold": True},
        {"text": f"Distance: {trained_dist:.3f}m", "xy": (10, height - 26), "color": RGB_WHITE, "size": 13},
    ]

    random_annotated = _annotate_frame(random_frame, random_labels)
    trained_annotated = _annotate_frame(trained_frame, trained_labels)

    # Stitch side by side
    combined = np.concatenate([random_annotated, trained_annotated], axis=1)

    import imageio.v3 as iio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, combined)
    print(f"    Saved {out_path} ({combined.shape[1]}x{combined.shape[0]})")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_env_setup(args: argparse.Namespace) -> None:
    """Generate annotated reset screenshots for each environment."""
    if not os.environ.get("ROBOTICS_GL_FALLBACK_TRIED"):
        _set_gl_backend("egl", force=True)
    _import_fetch()

    out_dir = Path(args.out_dir)
    envs = args.envs or DEFAULT_ENVS

    print(f"Generating environment setup figures in {out_dir}/")
    for env_id in envs:
        if env_id not in ENV_INFO:
            print(f"  WARNING: Unknown env {env_id}, skipping")
            continue
        info = ENV_INFO[env_id]
        out_path = out_dir / f"{info['file_stem']}.png"
        _render_env_setup(env_id, args.seed, args.width, args.height, out_path)

    print("Done.")


def cmd_reward_diagram(args: argparse.Namespace) -> None:
    """Generate dense vs sparse reward comparison diagram."""
    out_dir = Path(args.out_dir)
    print(f"Generating reward diagram in {out_dir}/")
    _render_reward_diagram(out_dir / "dense_vs_sparse_reward.png")
    _render_obs_structure(out_dir / "obs_dict_structure.png")
    print("Done.")


def cmd_compare(args: argparse.Namespace) -> None:
    """Generate side-by-side random vs trained comparison."""
    if not os.environ.get("ROBOTICS_GL_FALLBACK_TRIED"):
        _set_gl_backend("egl", force=True)
    _import_fetch()

    out_dir = Path(args.out_dir)
    env_id = args.env
    ckpt = Path(args.ckpt)

    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    info = ENV_INFO.get(env_id, {"file_stem": env_id.lower().replace("-", "_")})
    out_path = out_dir / f"{info['file_stem']}_comparison.png"

    print(f"Generating comparison figure in {out_dir}/")
    _render_comparison(env_id, ckpt, args.seed, args.width, args.height, out_path)
    print("Done.")


def cmd_all(args: argparse.Namespace) -> None:
    """Generate all figures (env-setup + reward-diagram)."""
    cmd_env_setup(args)
    cmd_reward_diagram(args)
    print("\nAll proposal figures generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate annotated figures for the Manning book proposal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--out-dir", default="figures", help="Output directory")
    shared.add_argument("--seed", type=int, default=42, help="Random seed for env reset")
    shared.add_argument("--width", type=int, default=640, help="Image width in pixels")
    shared.add_argument("--height", type=int, default=480, help="Image height in pixels")

    # env-setup
    p_env = sub.add_parser("env-setup", parents=[shared],
                           help="Annotated reset screenshots for each env")
    p_env.add_argument("--envs", nargs="*", default=None,
                       help="Environment IDs (default: all Fetch envs)")
    p_env.set_defaults(func=cmd_env_setup)

    # reward-diagram
    p_rew = sub.add_parser("reward-diagram", parents=[shared],
                           help="Dense vs sparse reward diagram + obs structure")
    p_rew.set_defaults(func=cmd_reward_diagram)

    # compare
    p_cmp = sub.add_parser("compare", parents=[shared],
                           help="Side-by-side random vs trained comparison")
    p_cmp.add_argument("--env", required=True, help="Environment ID")
    p_cmp.add_argument("--ckpt", required=True, help="Trained checkpoint path")
    p_cmp.set_defaults(func=cmd_compare)

    # all
    p_all = sub.add_parser("all", parents=[shared],
                           help="Generate all figures (env-setup + diagrams)")
    p_all.add_argument("--envs", nargs="*", default=None,
                       help="Environment IDs (default: all Fetch envs)")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
