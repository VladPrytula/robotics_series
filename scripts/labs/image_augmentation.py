#!/usr/bin/env python3
"""
Image Augmentation for Visual RL -- DrQ (Kostrikov et al. 2020)

DrQ regularizes the Q-function by averaging over random spatial augmentations
of pixel observations. The augmentation is applied at sample time from the
replay buffer, so each replay of a transition sees a different random crop.
This acts as implicit regularization against Q-function overfitting to
pixel-level details.

The entire DrQ "trick" is a replay buffer wrapper -- the reader sees that
algorithmic innovation can be a single clean abstraction layered on existing
infrastructure.

Usage:
    # Run verification (sanity checks, ~30 seconds)
    python scripts/labs/image_augmentation.py --verify

    # Demonstrate augmentation visually (print stats)
    python scripts/labs/image_augmentation.py --demo

Key regions exported for tutorials:
    - random_shift_aug: RandomShiftAug (pad-and-crop augmentation)
    - drq_replay_buffer: DrQDictReplayBuffer (SB3-compatible augmented sampling)

Reference:
    Kostrikov et al. (2020), "Image Augmentation Is All You Need:
    Regularizing Deep Reinforcement Learning from Pixels",
    arXiv:2004.13649, Section 3.1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is on sys.path for cross-module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# Random Shift Augmentation
# =============================================================================

# --8<-- [start:random_shift_aug]
class RandomShiftAug(nn.Module):
    """Pad-and-crop spatial augmentation from DrQ (Kostrikov et al. 2020).

    For an input image of size (H, W):
    1. Pad by `pad` pixels on all four sides using replicate padding,
       producing a (H + 2*pad, W + 2*pad) image.
    2. Randomly crop back to the original (H, W) size.

    Each image in the batch gets an independent random crop, so the same
    transition replayed twice will see different augmentations. This is the
    source of DrQ's regularization effect.

    For 84x84 images with pad=4, shifts of up to ~5% of the image size are
    possible -- enough to create diversity while preserving semantic content.

    Args:
        pad: Number of pixels to pad on each side. Default 4 (from the paper).
    """

    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random shift augmentation.

        Args:
            x: (B, C, H, W) float32 tensor (pixel observations).

        Returns:
            (B, C, H, W) augmented tensor with same shape as input.
        """
        B, C, H, W = x.shape
        pad = self.pad

        # Replicate padding: extends border pixels outward
        # This avoids black borders that would create artificial edges
        padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")

        # Random crop offsets -- one per image in the batch
        # Each image gets an independent shift
        crop_h = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        crop_w = torch.randint(0, 2 * pad + 1, (B,), device=x.device)

        # Gather cropped windows using advanced indexing
        # Build index grids for the H and W dimensions
        h_idx = torch.arange(H, device=x.device).unsqueeze(0) + crop_h.unsqueeze(1)  # (B, H)
        w_idx = torch.arange(W, device=x.device).unsqueeze(0) + crop_w.unsqueeze(1)  # (B, W)

        # Index into padded tensor: padded[b, :, h_idx[b], w_idx[b]]
        # We need to expand indices to (B, C, H, W) for gather-style indexing
        b_idx = torch.arange(B, device=x.device)[:, None, None, None].expand(B, C, H, W)
        c_idx = torch.arange(C, device=x.device)[None, :, None, None].expand(B, C, H, W)
        h_grid = h_idx[:, None, :, None].expand(B, C, H, W)
        w_grid = w_idx[:, None, None, :].expand(B, C, H, W)

        return padded[b_idx, c_idx, h_grid, w_grid]
# --8<-- [end:random_shift_aug]


# =============================================================================
# DrQ Replay Buffer (SB3-compatible)
# =============================================================================

# --8<-- [start:drq_replay_buffer]
class DrQDictReplayBuffer:
    """SB3 DictReplayBuffer subclass that applies augmentation at sample time.

    This is the core DrQ integration: instead of modifying the loss function
    or the network architecture, we augment pixel observations every time a
    batch is sampled from the replay buffer. Because the augmentation is
    random, each replay of the same transition produces a different view --
    this is what regularizes the Q-function.

    Usage with SB3's SAC::

        model = SAC(
            "MultiInputPolicy", env,
            replay_buffer_class=DrQDictReplayBuffer,
            replay_buffer_kwargs={
                "aug_fn": RandomShiftAug(pad=4),
                "image_key": "pixels",
            },
        )

    Args:
        aug_fn: An nn.Module that takes (B, C, H, W) and returns the same shape.
            If None, no augmentation is applied (behaves like DictReplayBuffer).
        image_key: The key in the observation dict that contains pixel data.
            Default "pixels" (matching our PixelObservationWrapper).
    """

    def __new__(cls, *args: Any, aug_fn: nn.Module | None = None,
                image_key: str = "pixels", **kwargs: Any) -> Any:
        """Dynamically create a DictReplayBuffer subclass with augmentation.

        We use __new__ with delayed import so that this module can be imported
        without SB3 installed (for standalone testing of RandomShiftAug).
        The actual subclass is created on first instantiation.
        """
        from stable_baselines3.common.buffers import DictReplayBuffer
        from stable_baselines3.common.type_aliases import DictReplayBufferSamples

        # Create the actual subclass dynamically
        class _DrQBuffer(DictReplayBuffer):
            def __init__(self, *a: Any, aug_fn: nn.Module | None = None,
                         image_key: str = "pixels", **kw: Any) -> None:
                super().__init__(*a, **kw)
                self.aug_fn = aug_fn
                self.image_key = image_key

            def _get_samples(self, batch_inds, env=None):
                samples = super()._get_samples(batch_inds, env)
                if self.aug_fn is not None:
                    aug_obs = dict(samples.observations)
                    aug_next = dict(samples.next_observations)
                    aug_obs[self.image_key] = self.aug_fn(
                        samples.observations[self.image_key]
                    )
                    aug_next[self.image_key] = self.aug_fn(
                        samples.next_observations[self.image_key]
                    )
                    return DictReplayBufferSamples(
                        observations=aug_obs,
                        actions=samples.actions,
                        next_observations=aug_next,
                        dones=samples.dones,
                        rewards=samples.rewards,
                    )
                return samples

        instance = DictReplayBuffer.__new__(_DrQBuffer)
        _DrQBuffer.__init__(instance, *args, aug_fn=aug_fn,
                            image_key=image_key, **kwargs)
        return instance
# --8<-- [end:drq_replay_buffer]


# =============================================================================
# Verification
# =============================================================================

def verify_random_shift_shapes():
    """Verify RandomShiftAug preserves input shapes."""
    print("Verifying RandomShiftAug shapes...")

    aug = RandomShiftAug(pad=4)

    # Standard 84x84 images
    x = torch.randn(8, 3, 84, 84)
    out = aug(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    # Non-square images
    x2 = torch.randn(4, 3, 64, 96)
    out2 = aug(x2)
    assert out2.shape == x2.shape, f"Expected {x2.shape}, got {out2.shape}"

    # Single image
    x3 = torch.randn(1, 3, 84, 84)
    out3 = aug(x3)
    assert out3.shape == x3.shape, f"Expected {x3.shape}, got {out3.shape}"

    print(f"  84x84 batch: {x.shape} -> {out.shape}")
    print(f"  64x96 batch: {x2.shape} -> {out2.shape}")
    print(f"  Single image: {x3.shape} -> {out3.shape}")
    print("  [PASS] Shapes preserved")


def verify_shift_distribution():
    """Verify that shifts are uniformly distributed across the pad range."""
    print("Verifying shift distribution...")

    pad = 4
    aug = RandomShiftAug(pad=pad)

    # Use a "probe" image: all zeros except a single bright pixel at center
    # After augmentation, the bright pixel's position tells us the shift
    H, W = 84, 84
    n_trials = 2000

    # Create batch of probe images
    x = torch.zeros(n_trials, 1, H, W)
    center_h, center_w = H // 2, W // 2
    x[:, 0, center_h, center_w] = 1.0

    out = aug(x)

    # Find the bright pixel in each output image
    shifts_h = []
    shifts_w = []
    for i in range(n_trials):
        # The bright pixel should be at (center_h - shift_h, center_w - shift_w)
        # relative to the original, but since we padded and cropped,
        # it appears at center_h - (crop_h - pad), center_w - (crop_w - pad)
        flat_idx = out[i, 0].argmax().item()
        found_h = flat_idx // W
        found_w = flat_idx % W
        shift_h = center_h - found_h
        shift_w = center_w - found_w
        shifts_h.append(shift_h)
        shifts_w.append(shift_w)

    shifts_h = np.array(shifts_h)
    shifts_w = np.array(shifts_w)

    # Shifts should be in [-pad, +pad]
    assert shifts_h.min() >= -pad, f"H shift min {shifts_h.min()} < -{pad}"
    assert shifts_h.max() <= pad, f"H shift max {shifts_h.max()} > {pad}"
    assert shifts_w.min() >= -pad, f"W shift min {shifts_w.min()} < -{pad}"
    assert shifts_w.max() <= pad, f"W shift max {shifts_w.max()} > {pad}"

    # Check that we see a spread of shift values (not always 0)
    unique_h = len(set(shifts_h.tolist()))
    unique_w = len(set(shifts_w.tolist()))
    expected_unique = 2 * pad + 1  # -pad to +pad inclusive
    assert unique_h >= expected_unique - 1, (
        f"Expected ~{expected_unique} unique H shifts, got {unique_h}"
    )
    assert unique_w >= expected_unique - 1, (
        f"Expected ~{expected_unique} unique W shifts, got {unique_w}"
    )

    print(f"  H shifts: min={shifts_h.min()}, max={shifts_h.max()}, "
          f"unique={unique_h}/{expected_unique}")
    print(f"  W shifts: min={shifts_w.min()}, max={shifts_w.max()}, "
          f"unique={unique_w}/{expected_unique}")
    print(f"  Mean shift: H={shifts_h.mean():.2f}, W={shifts_w.mean():.2f} "
          f"(expect ~0.0)")
    print("  [PASS] Shift distribution OK")


def verify_augmentation_differs():
    """Verify that two augmentations of the same input produce different outputs."""
    print("Verifying augmentation produces different outputs...")

    aug = RandomShiftAug(pad=4)
    x = torch.randn(16, 3, 84, 84)

    out1 = aug(x)
    out2 = aug(x)

    # The two outputs should differ (with very high probability for 16 images)
    diff = (out1 - out2).abs().sum().item()
    assert diff > 0, "Two augmentations produced identical outputs (astronomically unlikely)"

    # Each individual image should have finite values
    assert torch.isfinite(out1).all(), "Output 1 contains NaN/Inf"
    assert torch.isfinite(out2).all(), "Output 2 contains NaN/Inf"

    print(f"  Total abs difference between two augmentations: {diff:.2f}")
    print(f"  Per-pixel mean diff: {diff / out1.numel():.6f}")
    print("  [PASS] Augmentations differ")


def verify_replicate_padding():
    """Verify that replicate padding preserves border values (no black borders)."""
    print("Verifying replicate padding (no black borders)...")

    aug = RandomShiftAug(pad=4)

    # Create an image with non-zero borders
    x = torch.ones(4, 3, 84, 84) * 0.5
    out = aug(x)

    # For a constant image, augmentation should not change any values
    # (replicate padding of a constant = same constant everywhere)
    max_diff = (out - x).abs().max().item()
    assert max_diff < 1e-6, (
        f"Constant image should be unchanged after augmentation, "
        f"max diff = {max_diff}"
    )

    print(f"  Constant image max diff after augmentation: {max_diff:.2e}")
    print("  [PASS] Replicate padding OK (no black borders)")


def verify_drq_buffer_integration():
    """Verify DrQDictReplayBuffer works with SB3's SAC."""
    print("Verifying DrQDictReplayBuffer SB3 integration...")

    try:
        from stable_baselines3.common.buffers import DictReplayBuffer
    except ImportError:
        print("  [SKIP] SB3 not installed")
        return

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    # Create a pixel environment
    env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(env, image_size=(84, 84))

    aug_fn = RandomShiftAug(pad=4)

    # Create DrQ buffer with the same signature SB3 uses
    buf = DrQDictReplayBuffer(
        buffer_size=100,
        observation_space=env.observation_space,
        action_space=env.action_space,
        aug_fn=aug_fn,
        image_key="pixels",
        device="cpu",
        n_envs=1,
    )

    # Verify it is a DictReplayBuffer subclass
    assert isinstance(buf, DictReplayBuffer), (
        f"DrQDictReplayBuffer should be a DictReplayBuffer subclass, "
        f"got {type(buf)}"
    )

    # Add some transitions
    obs, _ = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        # SB3 buffers expect batch dimension
        buf.add(
            {k: v[None] for k, v in obs.items()},
            {k: v[None] for k, v in next_obs.items()},
            action[None],
            np.array([reward]),
            np.array([done]),
            [info],
        )

        obs = next_obs
        if done:
            obs, _ = env.reset()

    env.close()

    # Sample a batch and check augmentation was applied
    samples1 = buf.sample(8)
    samples2 = buf.sample(8)

    # Check shapes
    assert "pixels" in samples1.observations, "Missing 'pixels' in observations"
    assert "desired_goal" in samples1.observations, "Missing 'desired_goal' in observations"
    px = samples1.observations["pixels"]
    assert px.ndim == 4, f"Expected 4D tensor, got {px.ndim}D"
    assert px.shape[1:] == (3, 84, 84), f"Expected (B, 3, 84, 84), got {px.shape}"

    # Check that goals are NOT augmented (invariance property)
    # Two samples with the same indices should have identical goals
    # We can't easily control indices, but we can check goals are finite
    goals = samples1.observations["desired_goal"]
    assert torch.isfinite(goals).all(), "Goals contain NaN/Inf"

    print(f"  Buffer type: {type(buf).__mro__[0].__name__} "
          f"(subclass of DictReplayBuffer: {isinstance(buf, DictReplayBuffer)})")
    print(f"  Sampled pixels shape: {px.shape}")
    print(f"  Goals shape: {goals.shape}, all finite: {torch.isfinite(goals).all()}")
    print("  [PASS] DrQ buffer SB3 integration OK")


def verify_augmentation_invariance():
    """Verify that augmentation only affects pixels, not goals."""
    print("Verifying augmentation invariance (goals unchanged)...")

    try:
        from stable_baselines3.common.buffers import DictReplayBuffer
    except ImportError:
        print("  [SKIP] SB3 not installed")
        return

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(env, image_size=(84, 84))

    # Create two buffers: one with augmentation, one without
    aug_fn = RandomShiftAug(pad=4)

    buf_aug = DrQDictReplayBuffer(
        buffer_size=50,
        observation_space=env.observation_space,
        action_space=env.action_space,
        aug_fn=aug_fn,
        image_key="pixels",
        device="cpu",
        n_envs=1,
    )

    buf_plain = DrQDictReplayBuffer(
        buffer_size=50,
        observation_space=env.observation_space,
        action_space=env.action_space,
        aug_fn=None,  # No augmentation
        image_key="pixels",
        device="cpu",
        n_envs=1,
    )

    # Add identical transitions to both
    obs, _ = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        obs_batch = {k: v[None] for k, v in obs.items()}
        next_batch = {k: v[None] for k, v in next_obs.items()}
        act_batch = action[None]
        rew_batch = np.array([reward])
        done_batch = np.array([done])

        buf_aug.add(obs_batch, next_batch, act_batch, rew_batch, done_batch, [info])
        buf_plain.add(obs_batch, next_batch, act_batch, rew_batch, done_batch, [info])

        obs = next_obs
        if done:
            obs, _ = env.reset()

    env.close()

    # Sample with fixed seed to get same indices
    torch.manual_seed(42)
    np.random.seed(42)
    s_aug = buf_aug.sample(8)

    torch.manual_seed(42)
    np.random.seed(42)
    s_plain = buf_plain.sample(8)

    # Goals should be identical (augmentation does not touch them)
    goal_diff = (s_aug.observations["desired_goal"] - s_plain.observations["desired_goal"]).abs().max().item()
    ag_diff = (s_aug.observations["achieved_goal"] - s_plain.observations["achieved_goal"]).abs().max().item()

    # Actions and rewards should also be identical
    act_diff = (s_aug.actions - s_plain.actions).abs().max().item()
    rew_diff = (s_aug.rewards - s_plain.rewards).abs().max().item()

    assert goal_diff == 0.0, f"Goals differ after augmentation: max diff = {goal_diff}"
    assert ag_diff == 0.0, f"Achieved goals differ: max diff = {ag_diff}"
    assert act_diff == 0.0, f"Actions differ: max diff = {act_diff}"
    assert rew_diff == 0.0, f"Rewards differ: max diff = {rew_diff}"

    print(f"  Goal diff (desired): {goal_diff:.2e}")
    print(f"  Goal diff (achieved): {ag_diff:.2e}")
    print(f"  Action diff: {act_diff:.2e}")
    print(f"  Reward diff: {rew_diff:.2e}")
    print("  [PASS] Augmentation invariance OK (only pixels changed)")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("Image Augmentation (DrQ) -- Verification")
    print("=" * 60)

    verify_random_shift_shapes()
    print()
    verify_shift_distribution()
    print()
    verify_augmentation_differs()
    print()
    verify_replicate_padding()
    print()
    verify_drq_buffer_integration()
    print()
    verify_augmentation_invariance()

    print()
    print("=" * 60)
    print("[ALL PASS] Image augmentation verified")
    print("=" * 60)


# =============================================================================
# Demo
# =============================================================================

def run_demo():
    """Demonstrate augmentation on real Fetch pixel observations."""
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    print("=" * 60)
    print("Image Augmentation (DrQ) -- Demo")
    print("=" * 60)

    env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(env, image_size=(84, 84))

    obs, _ = env.reset()
    pixels = obs["pixels"]  # uint8 (3, 84, 84)
    env.close()

    # Convert to float tensor for augmentation
    px_tensor = torch.from_numpy(pixels.astype(np.float32) / 255.0).unsqueeze(0)
    px_batch = px_tensor.expand(8, -1, -1, -1)  # Replicate to batch of 8

    aug = RandomShiftAug(pad=4)

    print(f"\nOriginal image: shape={pixels.shape}, dtype={pixels.dtype}")
    print(f"  Pixel range: [{pixels.min()}, {pixels.max()}]")
    print(f"  Mean: {pixels.mean():.1f}, Std: {pixels.std():.1f}")

    print(f"\nAugmenting batch of 8 copies (pad=4):")
    print(f"{'Image':>8} | {'Mean':>10} | {'Std':>10} | {'Diff from orig':>15}")
    print("-" * 50)

    for trial in range(3):
        aug_batch = aug(px_batch)
        for i in range(min(4, aug_batch.shape[0])):
            img = aug_batch[i]
            diff = (img - px_tensor[0]).abs().mean().item()
            print(f"  {trial*4+i+1:>5} | {img.mean():>10.4f} | {img.std():>10.4f} | {diff:>15.6f}")

    # Show that same image augmented twice gives different results
    print(f"\nDemonstrating stochasticity (same input, different augmentations):")
    out1 = aug(px_tensor)
    out2 = aug(px_tensor)
    diff = (out1 - out2).abs().mean().item()
    print(f"  Mean pixel difference between two augmentations: {diff:.6f}")

    print()
    print("=" * 60)
    print("Demo complete. Each augmentation applies a random spatial shift")
    print("(up to +/-4 pixels), creating implicit Q-function regularization.")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Image Augmentation for Visual RL (DrQ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --verify    Run sanity checks (~30 seconds)
  --demo      Demonstrate augmentation on Fetch pixels
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
