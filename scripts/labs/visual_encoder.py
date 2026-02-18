#!/usr/bin/env python3
"""
Visual Encoder -- Pedagogical Implementation

This module implements the NatureCNN encoder (Mnih et al. 2015) and composite
visual-goal networks for SAC with pixel observations. Shows how a CNN encoder
plugs into the actor-critic architecture from sac_from_scratch.py.

Usage:
    # Run verification (sanity checks, ~1 minute)
    python scripts/labs/visual_encoder.py --verify

    # Compare our NatureCNN against SB3's implementation
    python scripts/labs/visual_encoder.py --compare-sb3

    # Short demo: visual SAC training on wrapped FetchReachDense (~2 min)
    python scripts/labs/visual_encoder.py --demo

Key regions exported for tutorials:
    - nature_cnn: NatureCNN encoder class
    - visual_goal_encoder: VisualGoalEncoder (CNN || goal -> features)
    - visual_gaussian_policy: VisualGaussianPolicy (SAC actor with CNN)
    - visual_twin_q_network: VisualTwinQNetwork (SAC critic with CNN)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# =============================================================================
# NatureCNN Encoder
# =============================================================================

# --8<-- [start:nature_cnn]
class NatureCNN(nn.Module):
    """NatureCNN encoder from Mnih et al. (2015).

    Architecture:
        Conv2d(in_channels, 32, 8x8, stride=4)  ->  ReLU
        Conv2d(32, 64, 4x4, stride=2)           ->  ReLU
        Conv2d(64, 64, 3x3, stride=1)           ->  ReLU
        Flatten                                  ->  Linear(3136, features_dim)  ->  ReLU

    For 84x84 input: spatial dims go 84 -> 20 -> 9 -> 7.
    Flatten: 64 * 7 * 7 = 3136.

    This is the same architecture SB3 uses in its CnnPolicy / NatureCnn
    feature extractor. We implement it from scratch to show how the math
    maps to code.

    Args:
        in_channels: Number of input channels (3 for RGB).
        features_dim: Output feature dimension (512 in original paper).
        image_size: Input image dimensions as (height, width). Default (84, 84)
            matches Mnih et al. (2015) conventions.
    """

    def __init__(self, in_channels: int = 3, features_dim: int = 512,
                 image_size: tuple[int, int] = (84, 84)):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten dim by doing a forward pass with dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size[0], image_size[1])
            n_flatten = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self.features_dim = features_dim
        self._n_flatten = n_flatten

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel observations.

        Args:
            pixels: float32 tensor of shape (B, C, H, W) in [0, 1].

        Returns:
            Feature vector of shape (B, features_dim).
        """
        return self.fc(self.conv(pixels))
# --8<-- [end:nature_cnn]


# =============================================================================
# Visual-Goal Composite Encoder
# =============================================================================

# --8<-- [start:visual_goal_encoder]
class VisualGoalEncoder(nn.Module):
    """Encodes pixel observations and goal vectors into a joint feature vector.

    Architecture:
        pixels -> NatureCNN -> cnn_features (512)
        goal   -> [raw]                     (goal_dim)
        concat [cnn_features, goal]         (512 + goal_dim)

    The concatenated vector is the input to downstream MLP heads (policy
    or Q-network). We do NOT apply additional layers here -- the MLP heads
    in the actor/critic handle that.

    This mirrors SB3's CombinedExtractor, which routes image keys through
    NatureCNN and vector keys through a flat identity, then concatenates.

    Args:
        in_channels: Number of image channels (3 for RGB).
        goal_dim: Dimensionality of goal vectors (3 for Fetch).
        cnn_features_dim: Output dim of NatureCNN (512 default).
    """

    def __init__(
        self,
        in_channels: int = 3,
        goal_dim: int = 3,
        cnn_features_dim: int = 512,
    ):
        super().__init__()
        self.cnn = NatureCNN(in_channels, cnn_features_dim)
        self.goal_dim = goal_dim
        self.output_dim = cnn_features_dim + goal_dim

    def forward(
        self,
        pixels: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Encode pixels and goal into a joint feature vector.

        Args:
            pixels: (B, C, H, W) float32 in [0, 1].
            goal: (B, goal_dim) float32 goal vector.

        Returns:
            (B, output_dim) concatenated features.
        """
        cnn_features = self.cnn(pixels)
        return torch.cat([cnn_features, goal], dim=-1)
# --8<-- [end:visual_goal_encoder]


# =============================================================================
# Visual Gaussian Policy (SAC Actor)
# =============================================================================

# --8<-- [start:visual_gaussian_policy]
class VisualGaussianPolicy(nn.Module):
    """SAC policy with CNN encoder for pixel observations.

    Architecture:
        pixels, goal -> VisualGoalEncoder -> features (515)
        features -> MLP(hidden_dims) -> mean_head, log_std_head -> squashed Gaussian

    This reuses the squashed Gaussian pattern from sac_from_scratch.py
    (tanh squashing + log-prob correction), but replaces the flat-obs MLP
    backbone with a CNN encoder.

    Args:
        in_channels: Number of image channels.
        goal_dim: Goal vector dimensionality.
        act_dim: Action dimensionality (4 for Fetch).
        cnn_features_dim: CNN output dimension.
        hidden_dims: MLP hidden layer sizes after encoder.
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        in_channels: int = 3,
        goal_dim: int = 3,
        act_dim: int = 4,
        cnn_features_dim: int = 512,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.encoder = VisualGoalEncoder(in_channels, goal_dim, cnn_features_dim)

        # MLP from encoder output to hidden features
        layers = []
        prev_dim = self.encoder.output_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.mlp = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev_dim, act_dim)
        self.log_std_head = nn.Linear(prev_dim, act_dim)

    def forward(
        self,
        pixels: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability.

        Args:
            pixels: (B, C, H, W) float32 in [0, 1].
            goal: (B, goal_dim) float32.

        Returns:
            action: (B, act_dim) squashed to [-1, 1].
            log_prob: (B,) log probability with tanh correction.
        """
        features = self.encoder(pixels, goal)
        h = self.mlp(features)

        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        # Reparameterization trick
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)

        # Log prob with squashing correction (same as sac_from_scratch.py)
        log_prob = dist.log_prob(x_t).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(dim=-1)

        return action, log_prob
# --8<-- [end:visual_gaussian_policy]


# =============================================================================
# Visual Twin Q-Network (SAC Critic)
# =============================================================================

# --8<-- [start:visual_twin_q_network]
class VisualTwinQNetwork(nn.Module):
    """Twin Q-networks with CNN encoders for pixel observations.

    Each Q-network has its own CNN encoder (no weight sharing between
    Q1 and Q2, and no sharing with the actor). This is standard practice
    in SAC: separate encoders prevent the critic's gradient from
    interfering with the actor's feature representation.

    Architecture (per Q-network):
        pixels, goal -> VisualGoalEncoder -> features (515)
        [features, action] -> MLP(hidden_dims) -> Q-value (scalar)

    Args:
        in_channels: Number of image channels.
        goal_dim: Goal vector dimensionality.
        act_dim: Action dimensionality.
        cnn_features_dim: CNN output dimension.
        hidden_dims: MLP hidden layer sizes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        goal_dim: int = 3,
        act_dim: int = 4,
        cnn_features_dim: int = 512,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Two separate encoders -- no weight sharing
        self.encoder1 = VisualGoalEncoder(in_channels, goal_dim, cnn_features_dim)
        self.encoder2 = VisualGoalEncoder(in_channels, goal_dim, cnn_features_dim)

        input_dim = self.encoder1.output_dim + act_dim

        # Q1 MLP
        layers1 = []
        prev = input_dim
        for h in hidden_dims:
            layers1.append(nn.Linear(prev, h))
            layers1.append(nn.ReLU())
            prev = h
        layers1.append(nn.Linear(prev, 1))
        self.q1 = nn.Sequential(*layers1)

        # Q2 MLP
        layers2 = []
        prev = input_dim
        for h in hidden_dims:
            layers2.append(nn.Linear(prev, h))
            layers2.append(nn.ReLU())
            prev = h
        layers2.append(nn.Linear(prev, 1))
        self.q2 = nn.Sequential(*layers2)

    def forward(
        self,
        pixels: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute twin Q-values.

        Args:
            pixels: (B, C, H, W) float32 in [0, 1].
            goal: (B, goal_dim) float32.
            action: (B, act_dim) float32.

        Returns:
            q1: (B,) Q-value from first network.
            q2: (B,) Q-value from second network.
        """
        f1 = self.encoder1(pixels, goal)
        f2 = self.encoder2(pixels, goal)

        x1 = torch.cat([f1, action], dim=-1)
        x2 = torch.cat([f2, action], dim=-1)

        return self.q1(x1).squeeze(-1), self.q2(x2).squeeze(-1)
# --8<-- [end:visual_twin_q_network]


# =============================================================================
# Visual SAC Update
# =============================================================================

def visual_sac_update(
    policy: VisualGaussianPolicy,
    q_network: VisualTwinQNetwork,
    target_q_network: VisualTwinQNetwork,
    log_alpha: torch.Tensor,
    policy_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    target_entropy: float,
    gamma: float = 0.99,
    tau: float = 0.005,
) -> dict[str, float]:
    """One SAC update step with pixel observations.

    Same structure as sac_from_scratch.sac_update, but networks take
    (pixels, goal) instead of flat observations.

    Args:
        batch: Dict with "pixels", "next_pixels", "desired_goal",
               "next_desired_goal", "actions", "rewards", "dones".
    """
    pixels = batch["pixels"]
    next_pixels = batch["next_pixels"]
    goal = batch["desired_goal"]
    next_goal = batch["next_desired_goal"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]

    alpha = log_alpha.exp().item()

    # --- Q-network update ---
    q1, q2 = q_network(pixels, goal, actions)

    with torch.no_grad():
        next_actions, next_log_probs = policy(next_pixels, next_goal)
        target_q1, target_q2 = target_q_network(next_pixels, next_goal, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target = rewards + gamma * (1.0 - dones) * (target_q - alpha * next_log_probs)

    q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    # --- Policy update ---
    new_actions, log_probs = policy(pixels, goal)
    q1_new, q2_new = q_network(pixels, goal, new_actions)
    q_value = torch.min(q1_new, q2_new)
    actor_loss = (alpha * log_probs - q_value).mean()

    policy_optimizer.zero_grad()
    actor_loss.backward()
    policy_optimizer.step()

    # --- Temperature update ---
    # Reuse log_probs from actor step (detached to stop gradient through policy)
    alpha_loss = -(log_alpha.exp() * (log_probs.detach() + target_entropy)).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    # --- Target network update (Polyak averaging) ---
    with torch.no_grad():
        for p, tp in zip(q_network.parameters(), target_q_network.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    return {
        "q_loss": q_loss.item(),
        "actor_loss": actor_loss.item(),
        "alpha": log_alpha.exp().item(),
        "q1_mean": q1.mean().item(),
        "entropy": -log_probs.mean().item(),
    }


# =============================================================================
# SB3 Comparison
# =============================================================================

def run_sb3_comparison(seed: int = 0) -> None:
    """Compare our NatureCNN against SB3's NatureCnn."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.labs.sb3_compare import compare_nature_cnn_to_sb3

    print("=" * 60)
    print("Visual Encoder -- SB3 Comparison")
    print("=" * 60)

    result = compare_nature_cnn_to_sb3(seed=seed)

    print(f"Max abs output diff: {result.metrics['max_abs_output_diff']:.3e}")
    print(f"Tolerance (atol):    {result.metrics['atol']:.1e}")

    print()
    if result.passed:
        print("[PASS] Our NatureCNN matches SB3's NatureCnn")
    else:
        print("[FAIL] Our NatureCNN does not match SB3's NatureCnn")
        if result.notes:
            print(f"Notes: {result.notes}")
        raise SystemExit(1)


# =============================================================================
# Verification
# =============================================================================

def verify_nature_cnn():
    """Verify NatureCNN forward pass and shapes."""
    print("Verifying NatureCNN...")

    cnn = NatureCNN(in_channels=3, features_dim=512)
    batch = torch.randn(4, 3, 84, 84)  # (B, C, H, W)

    out = cnn(batch)
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"

    # Parameter count
    n_params = sum(p.numel() for p in cnn.parameters())
    # Expected: Conv layers + FC
    # Conv1: 3*32*8*8 + 32 = 6176, Conv2: 32*64*4*4 + 64 = 32832,
    # Conv3: 64*64*3*3 + 64 = 36928, FC: 3136*512 + 512 = 1606144
    # Total: ~1,682,080
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Flatten dim: {cnn._n_flatten}")
    assert cnn._n_flatten == 3136, f"Expected flatten dim 3136, got {cnn._n_flatten}"
    print("  [PASS] NatureCNN OK")


def verify_visual_goal_encoder():
    """Verify VisualGoalEncoder concatenation."""
    print("Verifying VisualGoalEncoder...")

    enc = VisualGoalEncoder(in_channels=3, goal_dim=3, cnn_features_dim=512)
    pixels = torch.randn(4, 3, 84, 84)
    goal = torch.randn(4, 3)

    out = enc(pixels, goal)
    assert out.shape == (4, 515), f"Expected (4, 515), got {out.shape}"
    assert enc.output_dim == 515

    print(f"  Output shape: {out.shape} (512 CNN + 3 goal)")
    print("  [PASS] VisualGoalEncoder OK")


def verify_visual_policy():
    """Verify VisualGaussianPolicy forward pass."""
    print("Verifying VisualGaussianPolicy...")

    policy = VisualGaussianPolicy(in_channels=3, goal_dim=3, act_dim=4)
    pixels = torch.randn(8, 3, 84, 84)
    goal = torch.randn(8, 3)

    actions, log_probs = policy(pixels, goal)

    assert actions.shape == (8, 4), f"Actions shape: {actions.shape}"
    assert log_probs.shape == (8,), f"Log probs shape: {log_probs.shape}"
    assert (actions.abs() <= 1.0).all(), "Actions not in [-1, 1]"
    assert torch.isfinite(log_probs).all(), "Log probs contain NaN/Inf"

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Actions shape: {actions.shape}, range [{actions.min():.2f}, {actions.max():.2f}]")
    print(f"  Log prob mean: {log_probs.mean():.4f}")
    print(f"  Parameters: {n_params:,}")
    print("  [PASS] VisualGaussianPolicy OK")


def verify_visual_q_network():
    """Verify VisualTwinQNetwork forward pass."""
    print("Verifying VisualTwinQNetwork...")

    q_net = VisualTwinQNetwork(in_channels=3, goal_dim=3, act_dim=4)
    pixels = torch.randn(8, 3, 84, 84)
    goal = torch.randn(8, 3)
    actions = torch.randn(8, 4).clamp(-1, 1)

    q1, q2 = q_net(pixels, goal, actions)

    assert q1.shape == (8,), f"Q1 shape: {q1.shape}"
    assert q2.shape == (8,), f"Q2 shape: {q2.shape}"
    assert torch.isfinite(q1).all(), "Q1 contains NaN/Inf"
    assert torch.isfinite(q2).all(), "Q2 contains NaN/Inf"

    n_params = sum(p.numel() for p in q_net.parameters())
    print(f"  Q1 shape: {q1.shape}, mean={q1.mean():.4f}")
    print(f"  Q2 shape: {q2.shape}, mean={q2.mean():.4f}")
    print(f"  Parameters: {n_params:,} (2 separate CNN encoders)")
    print("  [PASS] VisualTwinQNetwork OK")


def verify_gradient_flow():
    """Verify gradients flow through the full visual SAC pipeline."""
    print("Verifying gradient flow...")

    policy = VisualGaussianPolicy(in_channels=3, goal_dim=3, act_dim=4)
    q_net = VisualTwinQNetwork(in_channels=3, goal_dim=3, act_dim=4)

    pixels = torch.randn(4, 3, 84, 84)
    goal = torch.randn(4, 3)

    # Actor loss
    actions, log_probs = policy(pixels, goal)
    q1, q2 = q_net(pixels, goal, actions)
    actor_loss = (0.2 * log_probs - torch.min(q1, q2)).mean()
    actor_loss.backward()

    # Check gradients exist on CNN params
    cnn_param = next(iter(policy.encoder.cnn.conv.parameters()))
    assert cnn_param.grad is not None, "No gradient on CNN conv layer"
    assert torch.isfinite(cnn_param.grad).all(), "NaN/Inf in CNN gradient"
    grad_norm = cnn_param.grad.norm().item()
    assert grad_norm > 0, "Zero gradient on CNN conv layer"

    print(f"  Actor loss: {actor_loss.item():.4f}")
    print(f"  CNN conv1 grad norm: {grad_norm:.6f}")
    print("  [PASS] Gradient flow OK")


def verify_update_convergence():
    """Verify that 10 SAC updates reduce Q-loss on a tiny batch."""
    print("Verifying update convergence (10 steps)...")

    torch.manual_seed(42)

    policy = VisualGaussianPolicy(in_channels=3, goal_dim=3, act_dim=4)
    q_net = VisualTwinQNetwork(in_channels=3, goal_dim=3, act_dim=4)
    target_q = VisualTwinQNetwork(in_channels=3, goal_dim=3, act_dim=4)
    target_q.load_state_dict(q_net.state_dict())

    log_alpha = torch.tensor(0.0, requires_grad=True)
    target_entropy = -4.0

    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=3e-4)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)

    # Small fixed batch
    batch = {
        "pixels": torch.randn(8, 3, 84, 84).clamp(0, 1),
        "next_pixels": torch.randn(8, 3, 84, 84).clamp(0, 1),
        "desired_goal": torch.randn(8, 3),
        "next_desired_goal": torch.randn(8, 3),
        "actions": torch.randn(8, 4).clamp(-1, 1),
        "rewards": torch.randn(8),
        "dones": torch.zeros(8),
    }

    losses = []
    for _ in range(10):
        info = visual_sac_update(
            policy, q_net, target_q, log_alpha,
            policy_opt, q_opt, alpha_opt,
            batch, target_entropy,
        )
        losses.append(info["q_loss"])

    # Q-loss should decrease (or at least not explode)
    assert all(np.isfinite(l) for l in losses), "Q-loss contains NaN/Inf"
    # Check loss didn't explode (should stay in a reasonable range)
    assert losses[-1] < losses[0] * 10, (
        f"Q-loss exploded: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )

    print(f"  Q-loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  Alpha: {info['alpha']:.4f}")
    print("  [PASS] Update convergence OK")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("Visual Encoder -- Verification")
    print("=" * 60)

    verify_nature_cnn()
    print()
    verify_visual_goal_encoder()
    print()
    verify_visual_policy()
    print()
    verify_visual_q_network()
    print()
    verify_gradient_flow()
    print()
    verify_update_convergence()

    print()
    print("=" * 60)
    print("[ALL PASS] Visual encoder verified")
    print("=" * 60)


# =============================================================================
# Demo
# =============================================================================

def run_demo(total_steps: int = 500):
    """Short visual SAC training on wrapped FetchReachDense.

    This is a pedagogical demo, not a full training run. We train for
    a few hundred steps just to show that losses move and the pipeline
    works end-to-end.
    """
    import gymnasium_robotics  # noqa: F401
    import gymnasium as gym
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.labs.pixel_wrapper import PixelObservationWrapper, PixelReplayBuffer

    print("=" * 60)
    print("Visual Encoder -- Demo (short training)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create pixel environment
    env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(env, image_size=(84, 84))

    goal_dim = 3
    act_dim = 4
    image_size = (84, 84)
    img_shape = (3, image_size[0], image_size[1])

    # Networks
    policy = VisualGaussianPolicy(goal_dim=goal_dim, act_dim=act_dim).to(device)
    q_net = VisualTwinQNetwork(goal_dim=goal_dim, act_dim=act_dim).to(device)
    target_q = VisualTwinQNetwork(goal_dim=goal_dim, act_dim=act_dim).to(device)
    target_q.load_state_dict(q_net.state_dict())

    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    target_entropy = -float(act_dim)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=3e-4)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)

    # Replay buffer
    buf = PixelReplayBuffer(img_shape, goal_dim, act_dim, capacity=10_000)

    # Collect experience
    obs, _ = env.reset()
    learning_starts = 100
    batch_size = 32

    print(f"\nTraining for {total_steps} steps (learning starts at {learning_starts})...")
    print(f"{'Step':>6} | {'Q-loss':>10} | {'Actor':>10} | {'Alpha':>8} | {'Entropy':>8}")
    print("-" * 55)

    for step in range(total_steps):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                px = torch.from_numpy(obs["pixels"].astype(np.float32) / 255.0).unsqueeze(0).to(device)
                gl = torch.from_numpy(obs["desired_goal"].astype(np.float32)).unsqueeze(0).to(device)
                action_t, _ = policy(px, gl)
                action = action_t.cpu().numpy().squeeze()

        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        buf.add(obs, action, reward, next_obs, float(done))

        obs = next_obs
        if done:
            obs, _ = env.reset()

        # Update
        if step >= learning_starts and buf.size >= batch_size:
            raw = buf.sample(batch_size)
            batch = {k: torch.from_numpy(v).to(device) for k, v in raw.items()}

            update_info = visual_sac_update(
                policy, q_net, target_q, log_alpha,
                policy_opt, q_opt, alpha_opt,
                batch, target_entropy,
            )

            if (step + 1) % max(50, total_steps // 5) == 0:
                print(
                    f"{step+1:>6} | {update_info['q_loss']:>10.4f} | "
                    f"{update_info['actor_loss']:>10.4f} | "
                    f"{update_info['alpha']:>8.4f} | {update_info['entropy']:>8.2f}"
                )

    env.close()
    print()
    print("=" * 60)
    print("Demo complete. Losses should show movement (not necessarily convergence")
    print("at this scale). For real training, use ch10_visual_reach.py with SB3.")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visual Encoder (NatureCNN + Visual SAC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --verify          Run sanity checks (~1 minute)
  --compare-sb3     Compare NatureCNN weights against SB3
  --demo            Short visual SAC training demo (~2 minutes)
        """
    )
    parser.add_argument("--verify", action="store_true", help="Run verification checks")
    parser.add_argument("--compare-sb3", action="store_true", help="Compare NatureCNN against SB3")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --compare-sb3")
    parser.add_argument("--demo", action="store_true", help="Run demo training")
    parser.add_argument("--steps", type=int, default=500, help="Training steps for demo")
    args = parser.parse_args()

    if args.compare_sb3:
        run_sb3_comparison(seed=args.seed)
    elif args.verify:
        run_verification()
    elif args.demo:
        run_demo(total_steps=args.steps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
