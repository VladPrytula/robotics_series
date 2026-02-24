#!/usr/bin/env python3
"""
Manipulation Encoder -- DrQ-v2 CNN + Spatial Softmax for Fetch Push

This module implements a manipulation-appropriate visual encoder that replaces
NatureCNN for tasks where spatial precision matters. The key insight: NatureCNN's
8x8 stride-4 first layer was designed for Atari (large sprites, coarse control).
For manipulation, objects are small (3-5 pixels) and spatial relationships are
the primary signal -- stride-4 destroys exactly the information the policy needs.

Three interventions addressing the root cause:
1. ManipulationCNN: 3x3 kernels, stride 2 only in layers 1 and 4
   (84->42->42->42->21 vs NatureCNN's 84->20->9->7)
2. SpatialSoftmax: outputs WHERE things are (per-channel expected x,y)
   instead of WHAT they look like (flattened feature maps)
3. Proprioception passthrough: CNN only learns world-state (object);
   robot knows itself via joint encoders

Usage:
    # Run verification (sanity checks, ~30 seconds)
    python scripts/labs/manipulation_encoder.py --verify

    # Compare against NatureCNN architecture
    python scripts/labs/manipulation_encoder.py --compare-nature

Key regions exported for tutorials:
    - spatial_softmax: SpatialSoftmax class
    - manipulation_cnn: ManipulationCNN class
    - manipulation_extractor: ManipulationExtractor (SB3 BaseFeaturesExtractor)

References:
    - Levine et al. (2016), arXiv:1504.00702 -- Spatial softmax for visuomotor policies
    - Yarats et al. (2021), arXiv:2107.09645 -- DrQ-v2 encoder architecture
    - Finn et al. (2016), arXiv:1509.06113 -- Spatial autoencoders for manipulation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space

# Ensure project root is on sys.path for cross-module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# Spatial Softmax
# =============================================================================

# --8<-- [start:spatial_softmax]
class SpatialSoftmax(nn.Module):
    """Per-channel spatial softmax with expected coordinate output.

    Instead of flattening a (B, C, H, W) feature map into a vector (which
    destroys spatial structure), spatial softmax computes the expected (x, y)
    position of each channel's activation peak. The output is (B, 2C) --
    two spatial coordinates per channel.

    This answers "WHERE does each feature detector fire?" rather than
    "WHAT does the whole image look like?" -- exactly what manipulation
    tasks need.

    The operation per channel:
        1. Softmax over (H * W) spatial positions: attention = softmax(features / temperature)
        2. Expected x = sum(pos_x * attention), where pos_x in [-1, 1]
        3. Expected y = sum(pos_y * attention), where pos_y in [-1, 1]

    The temperature parameter is learnable: high temperature -> uniform
    attention (early training), low temperature -> peaked attention
    (converged policy focusing on precise locations).

    Reference: Levine et al. (2016), arXiv:1504.00702, Section 4.

    Args:
        height: Feature map height (e.g., 21 after ManipulationCNN).
        width: Feature map width (e.g., 21).
        num_channels: Number of feature map channels (e.g., 32).
        temperature: Initial softmax temperature. Higher -> softer attention.
    """

    def __init__(self, height: int, width: int, num_channels: int,
                 temperature: float = 1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels

        # Learnable temperature -- network can sharpen or soften attention
        self.temperature = nn.Parameter(
            torch.ones(1) * temperature, requires_grad=True,
        )

        # Coordinate grids normalized to [-1, 1]
        # After marginalizing over spatial dims, we get (B, C, W) and (B, C, H).
        # pos_x: (1, 1, W) -- broadcasts against (B, C, W)
        # pos_y: (1, 1, H) -- broadcasts against (B, C, H)
        pos_x = torch.linspace(-1.0, 1.0, width)
        pos_y = torch.linspace(-1.0, 1.0, height)
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute expected (x, y) coordinates per channel.

        Args:
            features: (B, C, H, W) feature map from CNN.

        Returns:
            (B, 2C) tensor of expected spatial coordinates in [-1, 1].
        """
        B, C, H, W = features.shape

        # Softmax over spatial dimensions: (B, C, H*W) -> attention weights
        softmax_attention = F.softmax(
            features.reshape(B, C, -1) / self.temperature, dim=-1,
        ).reshape(B, C, H, W)

        # Expected x: marginalize over height, then weight by x-coordinates
        # softmax_attention.sum(dim=2) -> (B, C, W) -- marginal over rows
        expected_x = (softmax_attention.sum(dim=2) * self.pos_x).sum(dim=-1)

        # Expected y: marginalize over width, then weight by y-coordinates
        # softmax_attention.sum(dim=3) -> (B, C, H) -- marginal over columns
        expected_y = (softmax_attention.sum(dim=3) * self.pos_y).sum(dim=-1)

        # Concatenate: (B, C) + (B, C) -> (B, 2C)
        return torch.cat([expected_x, expected_y], dim=-1)
# --8<-- [end:spatial_softmax]


# =============================================================================
# Manipulation CNN
# =============================================================================

# --8<-- [start:manipulation_cnn]
class ManipulationCNN(nn.Module):
    """CNN encoder with gentle spatial downsampling for manipulation tasks.

    Based on the DrQ-v2 encoder (Yarats et al. 2021) with modifications:
    - 3x3 kernels throughout (vs NatureCNN's 8/4/3)
    - Stride 2 only in layers 1 and 4 (vs stride 4 in layer 1)
    - 32 channels everywhere (vs escalating 32/64/64)
    - Returns feature MAP (not flattened) for SpatialSoftmax

    Spatial progression for 84x84 input:
        Layer 1: Conv2d(C_in, 32, 3, stride=2, pad=1)  84 -> 42
        Layer 2: Conv2d(32,   32, 3, stride=1, pad=1)  42 -> 42
        Layer 3: Conv2d(32,   32, 3, stride=1, pad=1)  42 -> 42
        Layer 4: Conv2d(32,   32, 3, stride=2, pad=1)  42 -> 21

    A 4-pixel object in the input becomes ~2 pixels after layer 1
    (vs ~1 pixel with NatureCNN) -- the spatial relationship between
    gripper and puck is preserved.

    Args:
        in_channels: Number of input channels (3 for RGB, 12 for 4-frame stack).
        num_filters: Number of channels per conv layer. Default 32.
    """

    def __init__(self, in_channels: int = 3, num_filters: int = 32):
        super().__init__()
        self.num_filters = num_filters

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels to a feature map (not flattened).

        Args:
            pixels: (B, C, H, W) float32 in [0, 1].

        Returns:
            (B, num_filters, H', W') feature map. For 84x84 input: (B, 32, 21, 21).
        """
        return self.conv(pixels)
# --8<-- [end:manipulation_cnn]


# =============================================================================
# Manipulation Features Extractor (SB3 compatible)
# =============================================================================

# --8<-- [start:manipulation_extractor]
class ManipulationExtractor(BaseFeaturesExtractor):
    """SB3 features extractor with ManipulationCNN + optional SpatialSoftmax.

    Routes observation dict keys to appropriate sub-encoders:
    - Image keys (detected by is_image_space) -> ManipulationCNN + SpatialSoftmax
      + LayerNorm + Tanh
    - Vector keys (proprioception, goals) -> Flatten (identity)

    All sub-encoder outputs are concatenated into the final feature vector.

    For FetchPush with spatial_softmax=True, proprio, and goal_mode="both":
        pixels (84x84x12)  -> ManipulationCNN -> SpatialSoftmax -> LN -> Tanh -> 64D
        proprioception (10D) -> Flatten -> 10D
        achieved_goal (3D)  -> Flatten -> 3D
        desired_goal (3D)   -> Flatten -> 3D
        Total: 80D

    Without spatial_softmax (flatten mode):
        pixels -> ManipulationCNN -> Flatten -> Linear(flat_dim, flat_features_dim)
        -> LayerNorm -> Tanh -> flat_features_dim

    Args:
        observation_space: Dict space with image and vector keys.
        spatial_softmax: If True, use SpatialSoftmax on CNN output. If False,
            flatten + linear projection (like DrQ-v2 trunk).
        num_filters: Number of CNN channels. Default 32.
        flat_features_dim: Output dim when spatial_softmax=False. Default 50
            (matches DrQ-v2). Ignored when spatial_softmax=True.
        normalized_image: Whether images are already normalized to [0, 1].
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        spatial_softmax: bool = True,
        num_filters: int = 32,
        flat_features_dim: int = 50,
        include_images: bool = True,
        normalized_image: bool = False,
    ):
        # Placeholder features_dim -- updated after computing total size
        super().__init__(observation_space, features_dim=1)

        extractors: dict[str, nn.Module] = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                if not include_images:
                    continue
                in_channels = subspace.shape[0]
                cnn = ManipulationCNN(in_channels, num_filters)

                # Compute feature map size with a dummy forward pass
                with torch.no_grad():
                    dummy = torch.zeros(1, *subspace.shape)
                    if not normalized_image:
                        dummy = dummy / 255.0  # match runtime normalization
                    feat_map = cnn(dummy)
                    _, C, H, W = feat_map.shape

                if spatial_softmax:
                    # SpatialSoftmax -> 2*C features + LayerNorm + Tanh
                    spatial_dim = 2 * C
                    extractors[key] = nn.Sequential(
                        cnn,
                        SpatialSoftmax(H, W, C),
                        nn.LayerNorm(spatial_dim),
                        nn.Tanh(),
                    )
                    total_concat_size += spatial_dim
                else:
                    # Flatten + Linear + LayerNorm + Tanh (DrQ-v2 trunk pattern)
                    flat_size = C * H * W
                    extractors[key] = nn.Sequential(
                        cnn,
                        nn.Flatten(),
                        nn.Linear(flat_size, flat_features_dim),
                        nn.LayerNorm(flat_features_dim),
                        nn.Tanh(),
                    )
                    total_concat_size += flat_features_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += gym.spaces.utils.flatdim(subspace)

        if total_concat_size == 0:
            raise ValueError(
                "ManipulationExtractor would produce features_dim=0. "
                "Check your observation_space and include_images setting."
            )

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = []
        for key, ext in self.extractors.items():
            obs = observations[key]
            # Normalize uint8 images to [0, 1] at runtime
            if obs.dtype == torch.uint8:
                obs = obs.float() / 255.0
            encoded.append(ext(obs))
        return torch.cat(encoded, dim=1)
# --8<-- [end:manipulation_extractor]


# =============================================================================
# Verification
# =============================================================================

def verify_spatial_softmax():
    """Verify SpatialSoftmax shapes, coordinate output, and temperature gradient."""
    print("Verifying SpatialSoftmax...")

    H, W, C = 21, 21, 32
    ssm = SpatialSoftmax(H, W, C, temperature=1.0)

    # Shape check
    features = torch.randn(4, C, H, W)
    out = ssm(features)
    assert out.shape == (4, 2 * C), f"Expected (4, {2*C}), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"

    # Coordinate range: should be in [-1, 1] since we use softmax weights
    assert out.min() >= -1.0 - 1e-5, f"Min coordinate {out.min():.4f} < -1"
    assert out.max() <= 1.0 + 1e-5, f"Max coordinate {out.max():.4f} > 1"

    # Known-peak test: if one spatial position is much larger, the expected
    # coordinate should be close to that position
    features_peak = torch.zeros(1, 1, H, W)
    features_peak[0, 0, 0, 0] = 100.0  # Top-left corner: y=-1, x=-1
    ssm_1ch = SpatialSoftmax(H, W, 1, temperature=1.0)
    out_peak = ssm_1ch(features_peak)
    # Expected: x ~ -1.0, y ~ -1.0
    assert out_peak[0, 0].item() < -0.9, f"Expected x ~ -1.0, got {out_peak[0, 0]:.3f}"
    assert out_peak[0, 1].item() < -0.9, f"Expected y ~ -1.0, got {out_peak[0, 1]:.3f}"

    # Center peak test
    features_center = torch.zeros(1, 1, H, W)
    features_center[0, 0, H // 2, W // 2] = 100.0
    out_center = ssm_1ch(features_center)
    assert abs(out_center[0, 0].item()) < 0.1, f"Expected x ~ 0.0, got {out_center[0, 0]:.3f}"
    assert abs(out_center[0, 1].item()) < 0.1, f"Expected y ~ 0.0, got {out_center[0, 1]:.3f}"

    # Temperature gradient: should be learnable
    loss = out.sum()
    loss.backward()
    assert ssm.temperature.grad is not None, "No gradient on temperature"
    assert torch.isfinite(ssm.temperature.grad).all(), "NaN/Inf in temperature gradient"

    n_params = sum(p.numel() for p in ssm.parameters())
    print(f"  Output shape: {out.shape} (2 * {C} channels)")
    print(f"  Coordinate range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"  Top-left peak: x={out_peak[0,0]:.3f}, y={out_peak[0,1]:.3f}")
    print(f"  Center peak: x={out_center[0,0]:.3f}, y={out_center[0,1]:.3f}")
    print(f"  Temperature grad: {ssm.temperature.grad.item():.6f}")
    print(f"  Parameters: {n_params} (temperature only)")
    print("  [PASS] SpatialSoftmax OK")


def verify_manipulation_cnn():
    """Verify ManipulationCNN output shape and parameter count."""
    print("Verifying ManipulationCNN...")

    # Standard: 12 channels (4-frame stack), 84x84
    cnn = ManipulationCNN(in_channels=12, num_filters=32)
    batch = torch.randn(4, 12, 84, 84)
    out = cnn(batch)

    assert out.shape == (4, 32, 21, 21), f"Expected (4, 32, 21, 21), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"

    n_params = sum(p.numel() for p in cnn.parameters())

    # 3-channel input (no frame stacking)
    cnn3 = ManipulationCNN(in_channels=3, num_filters=32)
    out3 = cnn3(torch.randn(2, 3, 84, 84))
    assert out3.shape == (2, 32, 21, 21), f"Expected (2, 32, 21, 21), got {out3.shape}"

    n_params3 = sum(p.numel() for p in cnn3.parameters())

    print(f"  12ch input -> output shape: {out.shape}")
    print(f"   3ch input -> output shape: {out3.shape}")
    print(f"  Parameters (12ch): {n_params:,}")
    print(f"  Parameters (3ch):  {n_params3:,}")
    print("  [PASS] ManipulationCNN OK")


def verify_manipulation_extractor():
    """Verify ManipulationExtractor with mock FetchPush observation space."""
    print("Verifying ManipulationExtractor...")

    # Mock FetchPush pixel obs space: pixels + proprio + goals
    obs_space = spaces.Dict({
        "pixels": spaces.Box(0, 255, (12, 84, 84), dtype=np.uint8),
        "proprioception": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float64),
        "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
        "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
    })

    # With spatial softmax (default)
    ext_spatial = ManipulationExtractor(obs_space, spatial_softmax=True)
    expected_dim = 64 + 10 + 3 + 3  # spatial + proprio + ag + dg
    assert ext_spatial.features_dim == expected_dim, (
        f"Expected features_dim={expected_dim}, got {ext_spatial.features_dim}"
    )

    # Forward pass with mock data
    mock_obs = {
        "pixels": torch.randint(0, 256, (4, 12, 84, 84), dtype=torch.uint8),
        "proprioception": torch.randn(4, 10),
        "achieved_goal": torch.randn(4, 3),
        "desired_goal": torch.randn(4, 3),
    }
    out_spatial = ext_spatial(mock_obs)
    assert out_spatial.shape == (4, expected_dim), (
        f"Expected (4, {expected_dim}), got {out_spatial.shape}"
    )
    assert torch.isfinite(out_spatial).all(), "Output contains NaN/Inf"

    # Without spatial softmax (flatten mode)
    ext_flat = ManipulationExtractor(obs_space, spatial_softmax=False, flat_features_dim=50)
    expected_flat_dim = 50 + 10 + 3 + 3
    assert ext_flat.features_dim == expected_flat_dim, (
        f"Expected features_dim={expected_flat_dim}, got {ext_flat.features_dim}"
    )
    out_flat = ext_flat(mock_obs)
    assert out_flat.shape == (4, expected_flat_dim), (
        f"Expected (4, {expected_flat_dim}), got {out_flat.shape}"
    )

    # Vector-only mode (ignore image keys entirely)
    ext_vec_only = ManipulationExtractor(obs_space, include_images=False)
    expected_vec_only = 10 + 3 + 3
    assert ext_vec_only.features_dim == expected_vec_only, (
        f"Expected features_dim={expected_vec_only}, got {ext_vec_only.features_dim}"
    )
    out_vec_only = ext_vec_only(mock_obs)
    assert out_vec_only.shape == (4, expected_vec_only), (
        f"Expected (4, {expected_vec_only}), got {out_vec_only.shape}"
    )

    # Without proprioception (pixels + goals only)
    obs_space_no_proprio = spaces.Dict({
        "pixels": spaces.Box(0, 255, (12, 84, 84), dtype=np.uint8),
        "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
        "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
    })
    ext_no_proprio = ManipulationExtractor(obs_space_no_proprio, spatial_softmax=True)
    expected_no_proprio = 64 + 3 + 3
    assert ext_no_proprio.features_dim == expected_no_proprio

    n_params = sum(p.numel() for p in ext_spatial.parameters())

    print(f"  Spatial softmax: features_dim={ext_spatial.features_dim} "
          f"(64 spatial + 10 proprio + 3 ag + 3 dg)")
    print(f"  Flatten mode:    features_dim={ext_flat.features_dim} "
          f"(50 flat + 10 proprio + 3 ag + 3 dg)")
    print(f"  No proprio:      features_dim={ext_no_proprio.features_dim} "
          f"(64 spatial + 3 ag + 3 dg)")
    print(f"  Output shape:    {out_spatial.shape}")
    print(f"  Parameters:      {n_params:,}")
    print("  [PASS] ManipulationExtractor OK")


def verify_sb3_integration():
    """Verify ManipulationExtractor works with SB3 SAC."""
    print("Verifying SB3 integration (SAC + ManipulationExtractor)...")

    try:
        from stable_baselines3 import SAC
    except ImportError:
        print("  [SKIP] SB3 not installed")
        return

    import gymnasium_robotics  # noqa: F401
    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    push_proprio = [0, 1, 2, 9, 10, 20, 21, 22, 23, 24]

    env = gym.make("FetchPush-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(
        env, image_size=(84, 84), goal_mode="both",
        frame_stack=4, proprio_indices=push_proprio,
    )

    # Verify obs space has the right keys
    obs_keys = sorted(env.observation_space.spaces.keys())
    assert obs_keys == ["achieved_goal", "desired_goal", "pixels", "proprioception"], (
        f"Unexpected obs keys: {obs_keys}"
    )

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=0,
        device="cpu",
        buffer_size=200,
        learning_starts=50,
        batch_size=16,
        policy_kwargs={
            "features_extractor_class": ManipulationExtractor,
            "features_extractor_kwargs": {"spatial_softmax": True},
        },
    )

    # Verify extractor type -- SAC uses separate actor/critic extractors
    # (share_features_extractor=False by default), so policy.features_extractor
    # is None. Check the actor's extractor instead.
    ext = model.policy.actor.features_extractor
    assert isinstance(ext, ManipulationExtractor), (
        f"Expected ManipulationExtractor, got {type(ext).__name__}"
    )
    assert ext.features_dim == 80, (
        f"Expected features_dim=80, got {ext.features_dim}"
    )

    # Train 50 steps -- just verify no crash
    model.learn(total_timesteps=50)

    env.close()
    print(f"  Obs keys: {obs_keys}")
    print(f"  Extractor: {type(ext).__name__}, features_dim={ext.features_dim}")
    print("  50-step training completed without crashes")
    print("  [PASS] SB3 integration OK")


def verify_comparison_table():
    """Print comparison between NatureCNN and ManipulationCNN architectures."""
    print("Architecture comparison: NatureCNN vs ManipulationCNN")
    print()

    # NatureCNN
    nature_layers = [
        ("Conv2d(C, 32, 8x8, stride=4)", "84x84 -> 20x20"),
        ("Conv2d(32, 64, 4x4, stride=2)", "20x20 -> 9x9"),
        ("Conv2d(64, 64, 3x3, stride=1)", "9x9 -> 7x7"),
        ("Flatten + Linear(3136, 256)", "7x7x64 -> 256D"),
    ]

    manip_layers = [
        ("Conv2d(C, 32, 3x3, stride=2, pad=1)", "84x84 -> 42x42"),
        ("Conv2d(32, 32, 3x3, stride=1, pad=1)", "42x42 -> 42x42"),
        ("Conv2d(32, 32, 3x3, stride=1, pad=1)", "42x42 -> 42x42"),
        ("Conv2d(32, 32, 3x3, stride=2, pad=1)", "42x42 -> 21x21"),
        ("SpatialSoftmax(32, 21, 21)", "21x21x32 -> 64D"),
    ]

    print(f"  {'NatureCNN (Mnih 2015)':<45} {'ManipulationCNN (DrQ-v2 + spatial softmax)'}")
    print(f"  {'=' * 45} {'=' * 45}")

    max_rows = max(len(nature_layers), len(manip_layers))
    for i in range(max_rows):
        left = f"{nature_layers[i][0]:<30} {nature_layers[i][1]}" if i < len(nature_layers) else ""
        right = f"{manip_layers[i][0]:<35} {manip_layers[i][1]}" if i < len(manip_layers) else ""
        print(f"  {left:<45} {right}")

    print()

    # Parameter counts
    from scripts.labs.visual_encoder import NatureCNN
    nature = NatureCNN(in_channels=12, features_dim=256)
    manip = ManipulationCNN(in_channels=12, num_filters=32)
    ssm = SpatialSoftmax(21, 21, 32)

    nature_params = sum(p.numel() for p in nature.parameters())
    manip_params = sum(p.numel() for p in manip.parameters()) + sum(p.numel() for p in ssm.parameters())

    print(f"  {'Metric':<30} {'NatureCNN':<20} {'ManipulationCNN'}")
    print(f"  {'-' * 30} {'-' * 20} {'-' * 20}")
    print(f"  {'Parameters':<30} {nature_params:<20,} {manip_params:,}")
    print(f"  {'Output dim':<30} {'256':<20} {'64 (spatial coords)'}")
    print(f"  {'First layer reduction':<30} {'4x (84->20)':<20} {'2x (84->42)'}")
    print(f"  {'4-pixel object after L1':<30} {'~1 pixel':<20} {'~2 pixels'}")
    print(f"  {'Spatial structure':<30} {'Destroyed (flatten)':<20} {'Preserved (x,y coords)'}")

    print()
    print("  [INFO] Comparison table printed")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("Manipulation Encoder -- Verification")
    print("=" * 60)

    verify_spatial_softmax()
    print()
    verify_manipulation_cnn()
    print()
    verify_manipulation_extractor()
    print()
    verify_sb3_integration()

    print()
    print("=" * 60)
    print("[ALL PASS] Manipulation encoder verified")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Manipulation Encoder (DrQ-v2 CNN + Spatial Softmax)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --verify          Run sanity checks (~30 seconds)
  --compare-nature  Print NatureCNN vs ManipulationCNN comparison table
        """
    )
    parser.add_argument("--verify", action="store_true",
                        help="Run verification checks")
    parser.add_argument("--compare-nature", action="store_true",
                        help="Print architecture comparison table")
    args = parser.parse_args()

    if args.verify:
        run_verification()
    elif args.compare_nature:
        verify_comparison_table()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
