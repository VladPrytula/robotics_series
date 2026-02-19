#!/usr/bin/env python3
"""
Pixel Observation Wrapper -- Pedagogical Implementation

This module implements a pixel observation wrapper for Fetch environments,
replacing the flat state vector with rendered camera images.

Observation design variants (controlled by `goal_mode`):
- "none": pixels only (no privileged coordinates exposed in the observation)
- "desired": pixels + desired_goal (goal-conditioned, but still privileged)
- "both": pixels + achieved_goal + desired_goal (fully goal-conditioned, most privileged)

Usage:
    # Run verification (sanity checks, ~30 seconds)
    python scripts/labs/pixel_wrapper.py --verify

    # Demonstrate rendered frames at different reset positions
    python scripts/labs/pixel_wrapper.py --demo

Key regions exported for tutorials:
    - pixel_obs_wrapper: PixelObservationWrapper class
    - render_and_resize: Core rendering logic
    - pixel_replay_buffer: PixelReplayBuffer with uint8 storage
"""
from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym
import numpy as np


# =============================================================================
# Pixel Observation Wrapper
# =============================================================================

# --8<-- [start:render_and_resize]
def render_and_resize(
    env: gym.Env,
    image_size: tuple[int, int] = (84, 84),
) -> np.ndarray:
    """Render the MuJoCo scene and resize to the target resolution.

    The rendering pipeline:
    1. env.render() returns an HWC uint8 array at MuJoCo's default resolution
       (typically 480x480 for Fetch)
    2. PIL resizes to image_size (bilinear interpolation)
    3. We transpose HWC -> CHW for PyTorch/SB3 compatibility

    When ``gym.make(..., width=W, height=H)`` renders natively at the target
    size, step 2 is skipped entirely (no PIL import, no resize).

    Args:
        env: A Gymnasium environment with render_mode="rgb_array".
        image_size: Target (height, width) for the output image.

    Returns:
        CHW uint8 array of shape (3, H, W) with values in [0, 255].
    """
    # MuJoCo render -> HWC uint8 (e.g., 480x480x3)
    frame = env.render()
    assert frame is not None, "env.render() returned None -- is render_mode='rgb_array'?"
    assert frame.ndim == 3 and frame.shape[2] == 3, (
        f"Expected HWC image, got shape {frame.shape}"
    )

    # Fast path: frame already at target size (native rendering).
    # .copy() ensures the array owns its data -- MuJoCo may reuse the
    # internal render buffer on the next call.
    if frame.shape[:2] == image_size:
        return frame.transpose(2, 0, 1).copy()

    # Resize via PIL (bilinear is the default and a good trade-off)
    from PIL import Image

    pil_img = Image.fromarray(frame)
    pil_img = pil_img.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)

    # HWC -> CHW for PyTorch/SB3
    pixels = np.array(pil_img, dtype=np.uint8).transpose(2, 0, 1)
    return pixels
# --8<-- [end:render_and_resize]


# --8<-- [start:pixel_obs_wrapper]
class PixelObservationWrapper(gym.ObservationWrapper):
    """Replace flat state observations with rendered pixel images.

    Fetch environments return dict observations:
        {"observation": float64[10], "achieved_goal": float64[3], "desired_goal": float64[3]}

    This wrapper replaces the "observation" key with a "pixels" key containing
    a rendered camera image. Goal keys are optional (see `goal_mode`).

    The output observation space depends on `goal_mode`:
        goal_mode="none":
            {"pixels": Box(0, 255, (3, H, W), uint8)}
        goal_mode="desired":
            {"pixels": Box(...), "desired_goal": Box(3,)}
        goal_mode="both":
            {"pixels": Box(...), "achieved_goal": Box(3,), "desired_goal": Box(3,)}

    SB3's CombinedExtractor automatically detects the image space via
    is_image_space() and routes it through NatureCNN. Non-image keys
    (for example, goal vectors) are flattened and concatenated; the
    policy/value MLP comes after concatenation. No custom policy_kwargs needed.

    Args:
        env: A goal-conditioned Gymnasium environment with render_mode="rgb_array".
        image_size: Target (height, width) for rendered images. Default (84, 84)
            matches the NatureCNN input size from Mnih et al. (2015).
        goal_mode: One of {"none", "desired", "both"} controlling which (if any)
            goal vectors are exposed in the observation dict. Even when goals
            are not exposed, the wrapper stores the last raw goal dict on
            `last_raw_obs` for evaluation/debugging (not used by the policy).
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] = (84, 84),
        goal_mode: str = "both",
    ):
        if goal_mode not in {"none", "desired", "both"}:
            raise ValueError(f"Invalid goal_mode={goal_mode!r}, expected one of: none, desired, both")

        # Validate render mode before wrapping
        assert env.render_mode == "rgb_array", (
            f"PixelObservationWrapper requires render_mode='rgb_array', "
            f"got '{env.render_mode}'"
        )

        super().__init__(env)
        self._image_size = image_size
        self._goal_mode = goal_mode
        self.last_raw_obs: dict[str, np.ndarray] | None = None

        # Build new observation space: pixels + goal vectors
        # The original dict space has observation, achieved_goal, desired_goal
        old_spaces = env.observation_space.spaces

        new_spaces: dict[str, gym.Space] = {
            "pixels": gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, image_size[0], image_size[1]),
                dtype=np.uint8,
            ),
        }
        if self._goal_mode in {"desired", "both"}:
            new_spaces["desired_goal"] = old_spaces["desired_goal"]
        if self._goal_mode == "both":
            new_spaces["achieved_goal"] = old_spaces["achieved_goal"]

        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Transform observation: replace flat state with pixels.

        This is called after every reset() and step(). The MuJoCo scene
        is already in the correct state, so we just need to render.
        """
        # Keep the raw (goal-conditioned) observation for evaluation/debugging.
        # This is not part of the observation passed to the policy when goal_mode="none".
        self.last_raw_obs = observation
        pixels = render_and_resize(self.env, self._image_size)

        out: dict[str, np.ndarray] = {"pixels": pixels}
        if self._goal_mode in {"desired", "both"}:
            out["desired_goal"] = observation["desired_goal"]
        if self._goal_mode == "both":
            out["achieved_goal"] = observation["achieved_goal"]
        return out
# --8<-- [end:pixel_obs_wrapper]


# =============================================================================
# Pixel Replay Buffer
# =============================================================================

# --8<-- [start:pixel_replay_buffer]
class PixelReplayBuffer:
    """Replay buffer that stores images as uint8 for memory efficiency.

    Standard replay buffers store observations as float32. For 84x84x3 images,
    that is 84,672 bytes per image. Storing as uint8 costs only 21,168 bytes --
    a 4x savings. With 200K transitions (obs + next_obs), this saves ~25GB.

    The buffer converts to float32 [0, 1] only at sample time, when the batch
    is small (typically 256 transitions).

    Args:
        img_shape: Image shape as (C, H, W). E.g., (3, 84, 84).
        goal_dim: Dimensionality of goal vectors. E.g., 3 for Fetch.
        act_dim: Dimensionality of actions. E.g., 4 for Fetch.
        capacity: Maximum number of transitions to store.
    """

    def __init__(
        self,
        img_shape: tuple[int, ...],
        goal_dim: int,
        act_dim: int,
        capacity: int = 200_000,
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Images stored as uint8 for 4x memory savings
        self.pixels = np.zeros((capacity, *img_shape), dtype=np.uint8)
        self.next_pixels = np.zeros((capacity, *img_shape), dtype=np.uint8)

        # Goals stored as float32 (small, 3D vectors)
        self.achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.desired_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.next_achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.next_desired_goals = np.zeros((capacity, goal_dim), dtype=np.float32)

        # Actions and rewards
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: float,
    ) -> None:
        """Store a transition.

        Args:
            obs: Dict with "pixels" (uint8 CHW), "achieved_goal", "desired_goal".
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation dict.
            done: Whether episode terminated.
        """
        assert "achieved_goal" in obs and "desired_goal" in obs, (
            "PixelReplayBuffer requires 'achieved_goal' and 'desired_goal' "
            "in obs. Use PixelObservationWrapper with goal_mode='both'."
        )
        self.pixels[self.ptr] = obs["pixels"]
        self.achieved_goals[self.ptr] = obs["achieved_goal"]
        self.desired_goals[self.ptr] = obs["desired_goal"]

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward

        self.next_pixels[self.ptr] = next_obs["pixels"]
        self.next_achieved_goals[self.ptr] = next_obs["achieved_goal"]
        self.next_desired_goals[self.ptr] = next_obs["desired_goal"]

        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch, converting images to float32 [0, 1].

        The uint8 -> float32 conversion happens here, not in add(), so we
        pay the memory cost only for the batch (256 images) not the full
        buffer (200K images).

        Returns:
            Dict with:
            - "pixels": float32 (B, C, H, W) in [0, 1]
            - "next_pixels": float32 (B, C, H, W) in [0, 1]
            - "achieved_goal", "desired_goal": float32 (B, goal_dim)
            - "actions": float32 (B, act_dim)
            - "rewards": float32 (B,)
            - "dones": float32 (B,)
        """
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "pixels": self.pixels[idx].astype(np.float32) / 255.0,
            "next_pixels": self.next_pixels[idx].astype(np.float32) / 255.0,
            "achieved_goal": self.achieved_goals[idx],
            "desired_goal": self.desired_goals[idx],
            "next_achieved_goal": self.next_achieved_goals[idx],
            "next_desired_goal": self.next_desired_goals[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
        }
# --8<-- [end:pixel_replay_buffer]


# =============================================================================
# Verification
# =============================================================================

def verify_observation_space():
    """Verify observation space structure and types."""
    import gymnasium_robotics  # noqa: F401

    print("Verifying observation space...")

    for goal_mode in ["none", "desired", "both"]:
        env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
        wrapped = PixelObservationWrapper(env, image_size=(84, 84), goal_mode=goal_mode)
        space = wrapped.observation_space

        # Check keys
        assert "pixels" in space.spaces, "Missing 'pixels' key"
        assert "observation" not in space.spaces, "Flat 'observation' should be removed"
        if goal_mode in {"desired", "both"}:
            assert "desired_goal" in space.spaces, "Missing 'desired_goal' key"
        else:
            assert "desired_goal" not in space.spaces, "desired_goal should not be exposed in goal_mode='none'"
        if goal_mode == "both":
            assert "achieved_goal" in space.spaces, "Missing 'achieved_goal' key"
        else:
            assert "achieved_goal" not in space.spaces, "achieved_goal should not be exposed unless goal_mode='both'"

        # Check pixel space
        px_space = space["pixels"]
        assert px_space.shape == (3, 84, 84), f"Expected (3,84,84), got {px_space.shape}"
        assert px_space.dtype == np.uint8, f"Expected uint8, got {px_space.dtype}"
        assert px_space.low.min() == 0 and px_space.high.max() == 255

        # Check goal space shapes when present
        if "desired_goal" in space.spaces:
            assert space["desired_goal"].shape == (3,), f"desired_goal shape: {space['desired_goal'].shape}"
        if "achieved_goal" in space.spaces:
            assert space["achieved_goal"].shape == (3,), f"achieved_goal shape: {space['achieved_goal'].shape}"

        wrapped.close()
        print(f"  goal_mode={goal_mode}: keys={list(space.spaces.keys())}")
    print("  [PASS] Observation space OK")


def verify_pixel_values():
    """Verify pixel values are in expected range."""
    import gymnasium_robotics  # noqa: F401

    print("Verifying pixel values...")

    env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
    wrapped = PixelObservationWrapper(env, image_size=(84, 84))

    obs, _ = wrapped.reset()
    pixels = obs["pixels"]

    assert pixels.dtype == np.uint8, f"Expected uint8, got {pixels.dtype}"
    assert pixels.shape == (3, 84, 84), f"Expected (3,84,84), got {pixels.shape}"
    assert pixels.min() >= 0, f"Min pixel: {pixels.min()}"
    assert pixels.max() <= 255, f"Max pixel: {pixels.max()}"
    # The scene should not be all black or all white
    assert pixels.max() > 10, f"Image looks too dark (max={pixels.max()})"
    assert pixels.min() < 245, f"Image looks too bright (min={pixels.min()})"

    # Step and check again
    action = wrapped.action_space.sample()
    obs2, _, _, _, _ = wrapped.step(action)
    pixels2 = obs2["pixels"]
    assert pixels2.dtype == np.uint8
    assert pixels2.shape == (3, 84, 84)

    wrapped.close()
    print(f"  Reset pixels: min={pixels.min()}, max={pixels.max()}, mean={pixels.mean():.1f}")
    print(f"  Step pixels:  min={pixels2.min()}, max={pixels2.max()}, mean={pixels2.mean():.1f}")
    print("  [PASS] Pixel values OK")


def verify_goal_preservation():
    """Verify that goal vectors are preserved through the wrapper."""
    import gymnasium_robotics  # noqa: F401

    print("Verifying goal preservation...")

    env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
    wrapped = PixelObservationWrapper(env, image_size=(84, 84), goal_mode="both")

    obs, _ = wrapped.reset()
    ag = obs["achieved_goal"]
    dg = obs["desired_goal"]

    # Goals should be 3D float vectors with reasonable values
    assert ag.shape == (3,), f"achieved_goal shape: {ag.shape}"
    assert dg.shape == (3,), f"desired_goal shape: {dg.shape}"
    assert np.isfinite(ag).all(), "achieved_goal contains NaN/Inf"
    assert np.isfinite(dg).all(), "desired_goal contains NaN/Inf"

    # Goals should be in workspace range (Fetch reach: roughly [1.0, 1.6] x [0.4, 1.1] x [0.4, 0.6])
    assert 0.5 < ag[0] < 2.0, f"achieved_goal x={ag[0]} out of workspace"
    assert 0.0 < ag[1] < 1.5, f"achieved_goal y={ag[1]} out of workspace"
    assert 0.2 < ag[2] < 1.0, f"achieved_goal z={ag[2]} out of workspace"

    wrapped.close()
    print(f"  achieved_goal: {ag}")
    print(f"  desired_goal:  {dg}")
    print("  [PASS] Goal preservation OK")


def verify_sb3_compatibility():
    """Verify that SB3 recognizes our pixel space as an image."""
    import gymnasium_robotics  # noqa: F401

    print("Verifying SB3 compatibility...")

    try:
        from stable_baselines3.common.preprocessing import is_image_space
    except ImportError:
        print("  [SKIP] SB3 not installed")
        return

    for goal_mode in ["none", "desired", "both"]:
        env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
        wrapped = PixelObservationWrapper(env, image_size=(84, 84), goal_mode=goal_mode)
        px_space = wrapped.observation_space["pixels"]
        detected = is_image_space(px_space)
        wrapped.close()

        assert detected, (
            "SB3 is_image_space() did not recognize our pixel space. "
            f"Space: shape={px_space.shape}, dtype={px_space.dtype}, "
            f"low={px_space.low.min()}, high={px_space.high.max()}"
        )
        print(f"  goal_mode={goal_mode}: is_image_space(pixels) = {detected}")
    print("  [PASS] SB3 compatibility OK")


def verify_replay_buffer():
    """Verify PixelReplayBuffer storage and sampling."""
    print("Verifying PixelReplayBuffer...")

    img_shape = (3, 84, 84)
    goal_dim = 3
    act_dim = 4
    capacity = 100
    batch_size = 16

    buf = PixelReplayBuffer(img_shape, goal_dim, act_dim, capacity)

    # Add some transitions
    rng = np.random.default_rng(42)
    for i in range(50):
        obs = {
            "pixels": rng.integers(0, 256, size=img_shape, dtype=np.uint8),
            "achieved_goal": rng.standard_normal(goal_dim).astype(np.float32),
            "desired_goal": rng.standard_normal(goal_dim).astype(np.float32),
        }
        action = rng.standard_normal(act_dim).astype(np.float32)
        reward = float(rng.standard_normal())
        next_obs = {
            "pixels": rng.integers(0, 256, size=img_shape, dtype=np.uint8),
            "achieved_goal": rng.standard_normal(goal_dim).astype(np.float32),
            "desired_goal": rng.standard_normal(goal_dim).astype(np.float32),
        }
        done = 0.0
        buf.add(obs, action, reward, next_obs, done)

    assert buf.size == 50, f"Expected size 50, got {buf.size}"

    # Sample and check types/shapes
    batch = buf.sample(batch_size)

    assert batch["pixels"].shape == (batch_size, 3, 84, 84), f"pixels shape: {batch['pixels'].shape}"
    assert batch["pixels"].dtype == np.float32, f"pixels dtype: {batch['pixels'].dtype}"
    assert batch["pixels"].min() >= 0.0 and batch["pixels"].max() <= 1.0, (
        f"pixels range: [{batch['pixels'].min()}, {batch['pixels'].max()}]"
    )
    assert batch["next_pixels"].shape == (batch_size, 3, 84, 84)
    assert batch["achieved_goal"].shape == (batch_size, goal_dim)
    assert batch["actions"].shape == (batch_size, act_dim)
    assert batch["rewards"].shape == (batch_size,)
    assert batch["dones"].shape == (batch_size,)

    # Verify uint8 storage (internal)
    assert buf.pixels.dtype == np.uint8, f"Internal storage should be uint8, got {buf.pixels.dtype}"

    # Memory calculation
    img_bytes = capacity * np.prod(img_shape) * 2  # obs + next_obs, uint8
    img_bytes_float = capacity * np.prod(img_shape) * 2 * 4  # if float32
    savings = img_bytes_float / img_bytes

    print(f"  Buffer size: {buf.size}/{capacity}")
    print(f"  Sampled pixels: shape={batch['pixels'].shape}, dtype={batch['pixels'].dtype}")
    print(f"  Pixel range: [{batch['pixels'].min():.3f}, {batch['pixels'].max():.3f}]")
    print(f"  Internal storage: uint8 ({img_bytes / 1024:.0f} KB vs {img_bytes_float / 1024:.0f} KB float32, {savings:.0f}x savings)")
    print("  [PASS] Replay buffer OK")


def verify_wrap_around():
    """Verify that the buffer overwrites old data correctly."""
    print("Verifying buffer wrap-around...")

    buf = PixelReplayBuffer((3, 8, 8), goal_dim=3, act_dim=4, capacity=10)
    rng = np.random.default_rng(0)

    # Fill past capacity
    for i in range(25):
        obs = {
            "pixels": np.full((3, 8, 8), i % 256, dtype=np.uint8),
            "achieved_goal": np.zeros(3, dtype=np.float32),
            "desired_goal": np.zeros(3, dtype=np.float32),
        }
        next_obs = {
            "pixels": np.full((3, 8, 8), (i + 1) % 256, dtype=np.uint8),
            "achieved_goal": np.zeros(3, dtype=np.float32),
            "desired_goal": np.zeros(3, dtype=np.float32),
        }
        buf.add(obs, np.zeros(4, dtype=np.float32), 0.0, next_obs, 0.0)

    assert buf.size == 10, f"Expected size 10 (capped at capacity), got {buf.size}"
    assert buf.ptr == 5, f"Expected ptr 5 (25 % 10), got {buf.ptr}"

    # The most recent transition (i=24) should have pixels filled with 24
    # It's at index (24 % 10) = 4
    assert buf.pixels[4, 0, 0, 0] == 24, f"Expected 24 at idx 4, got {buf.pixels[4, 0, 0, 0]}"

    print(f"  Buffer after 25 inserts: size={buf.size}, ptr={buf.ptr}")
    print("  [PASS] Wrap-around OK")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("Pixel Observation Wrapper -- Verification")
    print("=" * 60)

    verify_observation_space()
    print()
    verify_pixel_values()
    print()
    verify_goal_preservation()
    print()
    verify_sb3_compatibility()
    print()
    verify_replay_buffer()
    print()
    verify_wrap_around()

    print()
    print("=" * 60)
    print("[ALL PASS] Pixel wrapper verified")
    print("=" * 60)


# =============================================================================
# Demo
# =============================================================================

def run_demo():
    """Render frames at 3 reset positions, print pixel statistics."""
    import gymnasium_robotics  # noqa: F401

    print("=" * 60)
    print("Pixel Observation Wrapper -- Demo")
    print("=" * 60)

    env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
    wrapped = PixelObservationWrapper(env, image_size=(84, 84))

    print(f"\nObservation space: {wrapped.observation_space}")
    print(f"Action space:     {wrapped.action_space}")

    print(f"\n{'Reset':>8} | {'Shape':>14} | {'Min':>5} | {'Max':>5} | {'Mean':>7} | {'Std':>7} | {'Goal distance':>15}")
    print("-" * 75)

    for i in range(3):
        obs, _ = wrapped.reset()
        px = obs["pixels"]
        ag = obs["achieved_goal"]
        dg = obs["desired_goal"]
        dist = np.linalg.norm(ag - dg)

        print(
            f"{i+1:>8} | {str(px.shape):>14} | {px.min():>5} | {px.max():>5} | "
            f"{px.mean():>7.1f} | {px.std():>7.1f} | {dist:>15.4f}m"
        )

    # Also demonstrate step
    print("\nStepping 5 times with random actions:")
    obs, _ = wrapped.reset()
    for step in range(5):
        action = wrapped.action_space.sample()
        obs, reward, term, trunc, info = wrapped.step(action)
        px = obs["pixels"]
        print(f"  Step {step+1}: pixels mean={px.mean():.1f}, reward={reward:.4f}")

    wrapped.close()
    print()
    print("=" * 60)
    print("Demo complete. The wrapper renders MuJoCo scenes as 84x84 pixel images.")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pixel Observation Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --verify    Run sanity checks (~30 seconds)
  --demo      Render frames at 3 reset positions
        """
    )
    parser.add_argument("--verify", action="store_true", help="Run verification checks")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    args = parser.parse_args()

    if args.verify:
        run_verification()
    elif args.demo:
        run_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
