#!/usr/bin/env python3
"""
Generate demo videos of trained policies for social media.

Creates eye-catching animations suitable for LinkedIn, Substack, and Twitter.

Usage:
    # Generate videos from trained checkpoint
    python scripts/generate_demo_videos.py --ckpt checkpoints/ppo_FetchReachDense-v4_seed0.zip

    # Quick test (fewer episodes)
    python scripts/generate_demo_videos.py --ckpt checkpoints/ppo_FetchReachDense-v4_seed0.zip --n-episodes 3

    # Custom output
    python scripts/generate_demo_videos.py --ckpt checkpoints/ppo_FetchReachDense-v4_seed0.zip --out videos/my_demo
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Set rendering backend before importing gymnasium
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate demo videos for social media",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", required=True,
        help="Path to trained checkpoint (.zip)"
    )
    parser.add_argument(
        "--env", default="FetchReachDense-v4",
        help="Environment ID"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=5,
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--out", default="videos/demo",
        help="Output path prefix (will create .mp4 and .gif)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Video width"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Video height"
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic policy"
    )
    parser.add_argument(
        "--gif", action="store_true",
        help="Also generate GIF (slower, larger file)"
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Create a 2x2 grid of episodes (requires 4+ episodes)"
    )
    return parser.parse_args()


def load_model(ckpt_path: str, device: str = "auto"):
    """Load a trained SB3 model."""
    from stable_baselines3 import PPO, SAC, TD3

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # Try each algorithm
    for cls in [PPO, SAC, TD3]:
        try:
            return cls.load(str(ckpt), device=device)
        except Exception:
            continue

    raise RuntimeError(f"Could not load checkpoint with PPO/SAC/TD3: {ckpt}")


def record_episode(env, model, deterministic: bool = True) -> list:
    """Record a single episode and return frames."""
    frames = []
    obs, _ = env.reset()
    done = False

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # Add a few frames at the end to show success
    for _ in range(10):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    return frames


def add_text_overlay(frames: list, text: str, position: str = "top") -> list:
    """Add text overlay to frames (requires PIL)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except ImportError:
        print("PIL not available, skipping text overlay")
        return frames

    result = []
    for frame in frames:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        # Position text
        if position == "top":
            xy = (10, 10)
        else:
            xy = (10, frame.shape[0] - 40)

        # Draw text with shadow for visibility
        draw.text((xy[0]+2, xy[1]+2), text, fill=(0, 0, 0), font=font)
        draw.text(xy, text, fill=(255, 255, 255), font=font)

        result.append(np.array(img))

    return result


def create_grid_video(all_episodes: list[list], grid_size: tuple = (2, 2)) -> list:
    """Create a grid of episodes playing simultaneously."""
    import numpy as np

    rows, cols = grid_size
    n_needed = rows * cols

    if len(all_episodes) < n_needed:
        raise ValueError(f"Need {n_needed} episodes for {rows}x{cols} grid, got {len(all_episodes)}")

    # Use first n_needed episodes
    episodes = all_episodes[:n_needed]

    # Find max length
    max_len = max(len(ep) for ep in episodes)

    # Pad shorter episodes by repeating last frame
    for ep in episodes:
        while len(ep) < max_len:
            ep.append(ep[-1])

    # Get frame dimensions from first frame
    h, w = episodes[0][0].shape[:2]

    # Create grid frames
    grid_frames = []
    for i in range(max_len):
        # Create grid for this timestep
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        for idx, ep in enumerate(episodes):
            r = idx // cols
            c = idx % cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = ep[i]

        grid_frames.append(grid)

    return grid_frames


def save_video(frames: list, path: Path, fps: int = 30):
    """Save frames as MP4 video."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Try imageio-ffmpeg first, fall back to raw imageio
    try:
        import imageio_ffmpeg
        import imageio.v3 as iio
        iio.imwrite(
            path,
            frames,
            fps=fps,
            codec="libx264",
        )
        print(f"Saved video: {path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
        return
    except (ImportError, OSError):
        pass

    # Fall back to saving as GIF (works everywhere)
    print(f"ffmpeg not available, saving as GIF instead...")
    gif_path = path.with_suffix(".gif")
    save_gif(frames, gif_path, fps=fps, optimize=False)
    print(f"Saved as GIF: {gif_path}")


def save_gif(frames: list, path: Path, fps: int = 15, optimize: bool = True):
    """Save frames as GIF (downsampled for smaller file size)."""
    import imageio.v3 as iio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    # Downsample frames for smaller GIF
    step = max(1, fps // 15)  # Target ~15fps for GIF
    downsampled = frames[::step]

    # Try to reduce resolution with PIL, fall back to using original
    try:
        from PIL import Image
        resized = []
        for frame in downsampled:
            img = Image.fromarray(frame)
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
            resized.append(np.array(img))
        frames_to_save = resized
    except ImportError:
        print("PIL not available, using full resolution")
        frames_to_save = downsampled

    # Calculate duration in ms per frame
    duration = int(1000 / 15)  # ~15fps

    iio.imwrite(path, frames_to_save, duration=duration, loop=0)
    print(f"Saved GIF: {path} ({len(frames_to_save)} frames)")


def main() -> int:
    args = parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt)

    print(f"Creating environment: {args.env}")
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    env = gym.make(
        args.env,
        render_mode="rgb_array",
        width=args.width,
        height=args.height,
    )

    try:
        all_episodes = []

        for ep in range(args.n_episodes):
            print(f"Recording episode {ep + 1}/{args.n_episodes}...")
            frames = record_episode(env, model, deterministic=args.deterministic)

            # Add episode number overlay
            frames = add_text_overlay(frames, f"Episode {ep + 1}")
            all_episodes.append(frames)
            print(f"  Recorded {len(frames)} frames")

        # Create output directory
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save individual episode compilation
        print("\nCreating compilation video...")
        compilation = []
        for i, ep in enumerate(all_episodes):
            compilation.extend(ep)
            # Add brief pause between episodes
            if i < len(all_episodes) - 1:
                compilation.extend([ep[-1]] * 15)  # 0.5s pause

        save_video(compilation, out_path.with_suffix(".mp4"), fps=args.fps)

        # Create grid video if requested
        if args.grid and args.n_episodes >= 4:
            print("\nCreating 2x2 grid video...")
            grid_frames = create_grid_video(all_episodes, (2, 2))
            grid_frames = add_text_overlay(grid_frames, "Fetch Robot - Goal-Conditioned RL", position="bottom")
            save_video(grid_frames, out_path.with_name(out_path.stem + "_grid.mp4"), fps=args.fps)

        # Create GIF if requested
        if args.gif:
            print("\nCreating GIF...")
            # Use grid if available, otherwise compilation
            if args.grid and args.n_episodes >= 4:
                save_gif(grid_frames, out_path.with_name(out_path.stem + "_grid.gif"), fps=args.fps)
            else:
                # Just use first episode for GIF (smaller)
                save_gif(all_episodes[0], out_path.with_suffix(".gif"), fps=args.fps)

        print("\n" + "="*50)
        print("Demo videos generated successfully!")
        print("="*50)
        print(f"\nFiles created in: {out_path.parent}/")
        print("\nFor social media:")
        print("- LinkedIn: Use .mp4 (max 10 min, <5GB)")
        print("- Twitter: Use .mp4 or .gif (max 2:20, <512MB)")
        print("- Substack: Embed .mp4 or use .gif inline")

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
