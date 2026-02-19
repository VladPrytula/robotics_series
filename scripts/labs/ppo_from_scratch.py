#!/usr/bin/env python3
"""
PPO From Scratch -- Pedagogical Implementation

This module implements Proximal Policy Optimization (PPO) with explicit tensor
operations to show how the math maps to code. Not for production use.

Usage:
    # Run verification (sanity checks, ~2 minutes)
    python scripts/labs/ppo_from_scratch.py --verify

    # Compare core invariants against SB3 (requires stable-baselines3)
    python scripts/labs/ppo_from_scratch.py --compare-sb3

    # Train on a simple environment (demonstration)
    python scripts/labs/ppo_from_scratch.py --demo

Key regions exported for book chapters:
    - actor_critic_network: Shared-backbone actor-critic with Gaussian policy
    - gae_computation: Generalized Advantage Estimation
    - ppo_loss: Clipped surrogate objective
    - value_loss: Critic MSE loss
    - ppo_update: Full update step combining all losses
    - ppo_training_loop: Rollout collection and batch assembly
"""
from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# =============================================================================
# Optional: Compare Against SB3
# =============================================================================

def run_sb3_comparison(seed: int = 0) -> None:
    """Compare our GAE against SB3 RolloutBuffer (quick invariant check)."""
    from scripts.labs.sb3_compare import compare_ppo_gae_to_sb3

    print("=" * 60)
    print("PPO From Scratch -- SB3 Comparison")
    print("=" * 60)

    result = compare_ppo_gae_to_sb3(seed=seed)

    print(f"Max abs advantage diff: {result.metrics['max_abs_adv_diff']:.3e}")
    print(f"Max abs returns diff:   {result.metrics['max_abs_returns_diff']:.3e}")
    print(f"Tolerance (atol):       {result.metrics['atol']:.1e}")

    print()
    if result.passed:
        print("[PASS] Our GAE matches SB3 RolloutBuffer")
    else:
        print("[FAIL] Our GAE does not match SB3 RolloutBuffer")
        if result.notes:
            print(f"Notes: {result.notes}")
        raise SystemExit(1)


# =============================================================================
# Network Architecture
# =============================================================================

# --8<-- [start:actor_critic_network]
class ActorCritic(nn.Module):
    """Simple actor-critic network with separate heads.

    Architecture:
        observation -> shared_backbone -> actor_head -> (mean, log_std)
                                       -> critic_head -> value
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head: outputs mean of Gaussian policy
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        # Learnable log standard deviation (state-independent)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head: outputs state value estimate
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """Forward pass returning action distribution and value estimate."""
        features = self.backbone(obs)

        # Actor: Gaussian with learned mean and std
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)

        # Critic: scalar value
        value = self.critic(features).squeeze(-1)

        return dist, value
# --8<-- [end:actor_critic_network]


# =============================================================================
# Rollout Storage
# =============================================================================

class RolloutBatch(NamedTuple):
    """Batch of experience for PPO update."""
    observations: torch.Tensor    # (batch_size, obs_dim)
    actions: torch.Tensor         # (batch_size, act_dim)
    old_log_probs: torch.Tensor   # (batch_size,)
    advantages: torch.Tensor      # (batch_size,)
    returns: torch.Tensor         # (batch_size,)


# =============================================================================
# GAE Computation
# =============================================================================

# --8<-- [start:gae_computation]
def compute_gae(
    rewards: torch.Tensor,      # (T,) rewards at each timestep
    values: torch.Tensor,       # (T,) value estimates V(s_t)
    next_value: torch.Tensor,   # scalar: V(s_T) for bootstrap
    dones: torch.Tensor,        # (T,) episode termination flags
    gamma: float = 0.99,        # discount factor
    gae_lambda: float = 0.95,   # GAE lambda for bias-variance tradeoff
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE balances bias and variance in advantage estimation:
        A_t = sum_{k=0}^{inf} (gamma * lambda)^k * delta_{t+k}

    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) is the TD residual.

    Args:
        rewards: Rewards received at each timestep
        values: Value estimates from the critic
        next_value: Bootstrap value for the final state
        dones: Whether episode terminated at each step
        gamma: Discount factor (how much to value future rewards)
        gae_lambda: GAE parameter (0=high bias, 1=high variance)

    Returns:
        advantages: GAE advantages (T,)
        returns: Target values for critic (T,), computed as advantages + values
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)

    # Bootstrap: last advantage uses next_value
    # Work backwards: A_{t} = delta_t + gamma * lambda * A_{t+1}
    last_gae = 0.0

    for t in reversed(range(T)):
        # If episode ended, next value is 0 (no future rewards)
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # Mask out next value if episode terminated
        next_val = next_val * (1.0 - dones[t])

        # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        # This measures: "was this transition better or worse than expected?"
        delta = rewards[t] + gamma * next_val - values[t]

        # GAE recursion: A_t = delta_t + gamma * lambda * A_{t+1}
        # The (1 - done) masks out future advantages at episode boundaries
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae

    # Returns = advantages + values (target for value function)
    returns = advantages + values

    return advantages, returns
# --8<-- [end:gae_computation]


# =============================================================================
# PPO Loss Functions
# =============================================================================

# --8<-- [start:ppo_loss]
def compute_ppo_loss(
    dist: Normal,                # current policy distribution
    old_log_probs: torch.Tensor, # log pi_old(a|s) from rollout
    actions: torch.Tensor,       # actions taken during rollout
    advantages: torch.Tensor,    # GAE advantages
    clip_range: float = 0.2,     # PPO clipping parameter epsilon
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO clipped surrogate objective.

    The key insight: we want to improve the policy, but not too much at once.
    Large policy changes can be catastrophic because our advantage estimates
    become invalid when the policy changes significantly.

    PPO clips the probability ratio r_t = pi(a|s) / pi_old(a|s) to stay
    within [1-epsilon, 1+epsilon], preventing destructively large updates.

    L^CLIP = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]

    Args:
        dist: Current policy distribution pi_theta
        old_log_probs: Log probabilities from the policy that collected data
        actions: Actions taken during rollout
        advantages: GAE advantage estimates
        clip_range: PPO clipping parameter (typically 0.2)

    Returns:
        loss: Negative of clipped objective (we minimize, so negate)
        info: Dictionary with diagnostic metrics
    """
    # Compute log probability of actions under CURRENT policy
    new_log_probs = dist.log_prob(actions).sum(dim=-1)

    # Probability ratio: r_t = pi_new(a|s) / pi_old(a|s)
    # Using log: r_t = exp(log_pi_new - log_pi_old)
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()

    # Clipped ratio: keep within [1-eps, 1+eps]
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    # Two surrogate objectives:
    # 1. Unclipped: r_t * A_t (naive policy gradient)
    # 2. Clipped: clip(r_t) * A_t (conservative update)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages

    # Take the minimum (pessimistic bound)
    # This prevents both:
    # - Making good actions TOO likely (when A > 0, caps at 1+eps)
    # - Making bad actions TOO unlikely (when A < 0, caps at 1-eps)
    policy_loss = -torch.min(surr1, surr2).mean()

    # Diagnostics for monitoring training health
    with torch.no_grad():
        # Approximate KL divergence: KL(pi_old || pi_new)
        # Using the approximation: KL ≈ (ratio - 1) - log(ratio)
        approx_kl = ((ratio - 1) - log_ratio).mean()

        # Fraction of updates that were clipped
        clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean()

    info = {
        "policy_loss": policy_loss.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_fraction.item(),
        "ratio_mean": ratio.mean().item(),
    }

    return policy_loss, info
# --8<-- [end:ppo_loss]


# --8<-- [start:value_loss]
def compute_value_loss(
    values: torch.Tensor,        # V(s) predictions from critic
    returns: torch.Tensor,       # target values (advantages + old_values)
    old_values: torch.Tensor | None = None,  # for value clipping (optional)
    clip_range: float | None = None,         # value clip range (optional)
) -> tuple[torch.Tensor, dict]:
    """
    Compute value function loss (critic update).

    The critic learns to predict expected returns: V(s) ≈ E[sum of future rewards].
    We train with MSE loss against the computed returns.

    Optional: clip value updates similar to policy (can help stability).

    Args:
        values: Current value predictions V_theta(s)
        returns: Target values (computed as advantages + values from rollout)
        old_values: Value predictions from rollout (for clipping)
        clip_range: If provided, clip value updates

    Returns:
        loss: Value function MSE loss
        info: Dictionary with diagnostic metrics
    """
    if clip_range is not None and old_values is not None:
        # Clipped value loss (optional, from PPO paper)
        # Prevents value function from changing too rapidly
        values_clipped = old_values + torch.clamp(
            values - old_values, -clip_range, clip_range
        )
        value_loss_unclipped = (values - returns).pow(2)
        value_loss_clipped = (values_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
    else:
        # Simple MSE loss
        value_loss = 0.5 * (values - returns).pow(2).mean()

    info = {
        "value_loss": value_loss.item(),
        "explained_variance": explained_variance(values, returns),
    }

    return value_loss, info


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute explained variance: 1 - Var(target - pred) / Var(target).

    - EV = 1: perfect predictions
    - EV = 0: predictions are as good as predicting the mean
    - EV < 0: predictions are worse than predicting the mean
    """
    with torch.no_grad():
        var_target = target.var()
        if var_target < 1e-8:
            return 0.0
        return (1.0 - (target - pred).var() / var_target).item()
# --8<-- [end:value_loss]


# --8<-- [start:ppo_update]
def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    max_grad_norm: float = 0.5,
) -> dict:
    """
    Perform one PPO update step.

    Combines three losses:
        L = L_policy + c1 * L_value - c2 * H[pi]

    where:
        - L_policy: clipped surrogate objective (makes good actions more likely)
        - L_value: MSE on value predictions (improves advantage estimates)
        - H[pi]: entropy bonus (encourages exploration)

    Args:
        model: Actor-critic network
        optimizer: Optimizer (typically Adam)
        batch: Rollout batch with obs, actions, log_probs, advantages, returns
        clip_range: PPO clipping parameter epsilon
        value_coef: Weight for value loss (c1)
        entropy_coef: Weight for entropy bonus (c2)
        max_grad_norm: Gradient clipping threshold

    Returns:
        info: Dictionary with all loss components and diagnostics
    """
    # Forward pass: get current policy and value estimates
    dist, values = model(batch.observations)

    # Normalize advantages (common practice, reduces variance)
    advantages = batch.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute individual losses
    policy_loss, policy_info = compute_ppo_loss(
        dist, batch.old_log_probs, batch.actions, advantages, clip_range
    )

    value_loss, value_info = compute_value_loss(values, batch.returns)

    # Entropy bonus: encourages exploration by penalizing overly deterministic policies
    # H[pi] = -E[log pi(a|s)], higher entropy = more random
    entropy = dist.entropy().mean()
    entropy_loss = -entropy  # negative because we minimize total loss

    # Combined loss
    # Note: we MINIMIZE this, so:
    # - policy_loss is already negated (we want to maximize the objective)
    # - value_loss is positive (we want to minimize prediction error)
    # - entropy_loss is negated (we want to maximize entropy)
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    # Gradient update
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping (prevents exploding gradients)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    # Combine all info
    info = {
        **policy_info,
        **value_info,
        "entropy": entropy.item(),
        "total_loss": total_loss.item(),
        "grad_norm": grad_norm.item(),
    }

    return info
# --8<-- [end:ppo_update]


# =============================================================================
# Verification (Sanity Checks)
# =============================================================================

def verify_network():
    """Verify actor-critic network shapes and parameter count."""
    print("Verifying actor-critic network...")

    obs_dim, act_dim = 16, 4
    model = ActorCritic(obs_dim, act_dim)
    obs = torch.randn(1, obs_dim)
    dist, value = model(obs)

    assert dist.mean.shape == (1, 4), f"Action mean shape: {dist.mean.shape}, expected (1, 4)"
    assert value.shape == (1,), f"Value shape: {value.shape}, expected (1,)"
    assert torch.isfinite(dist.mean).all(), "Non-finite action mean"
    assert torch.isfinite(value).all(), "Non-finite value"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  dist.mean shape: {dist.mean.shape}")
    print(f"  value shape:     {value.shape}")
    print(f"  total params:    {total_params:,}")
    print("  [PASS] Actor-critic network OK")


def verify_gae():
    """Verify GAE computation produces sensible values."""
    print("Verifying GAE computation...")

    # Create simple test case
    T = 10
    rewards = torch.zeros(T)
    rewards[-1] = 1.0  # reward only at end
    values = torch.linspace(0, 0.5, T)  # increasing value estimates
    next_value = torch.tensor(0.0)
    dones = torch.zeros(T)
    dones[-1] = 1.0  # episode ends

    advantages, returns = compute_gae(rewards, values, next_value, dones)

    # Checks
    assert torch.isfinite(advantages).all(), "Advantages contain NaN/Inf"
    assert torch.isfinite(returns).all(), "Returns contain NaN/Inf"
    assert advantages[-1] > 0, "Final advantage should be positive (got reward)"
    assert torch.allclose(returns, advantages + values, atol=1e-6), \
        "Returns should equal advantages + values"
    print(f"  Advantages: {advantages.tolist()}")
    print(f"  Returns: {returns.tolist()}")
    print("  [PASS] GAE computation OK")


def verify_ppo_loss():
    """Verify PPO loss computation."""
    print("Verifying PPO loss...")

    # Create mock policy and data
    obs_dim, act_dim = 4, 2
    batch_size = 32

    model = ActorCritic(obs_dim, act_dim)
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)
    advantages = torch.randn(batch_size)

    with torch.no_grad():
        dist, _ = model(obs)
        old_log_probs = dist.log_prob(actions).sum(dim=-1)

    # Compute loss
    dist, _ = model(obs)
    loss, info = compute_ppo_loss(dist, old_log_probs, actions, advantages)

    # Checks: same model, same params -> no clipping, ratio=1, kl~0
    assert torch.isfinite(loss), "Loss is NaN/Inf"
    assert abs(info["clip_fraction"] - 0.0) < 1e-6, \
        f"clip_fraction should be 0.0 for same policy, got {info['clip_fraction']}"
    assert abs(info["ratio_mean"] - 1.0) < 1e-4, \
        f"ratio_mean should be 1.0 for same policy, got {info['ratio_mean']}"
    assert info["approx_kl"] < 1e-4, \
        f"approx_kl should be ~0.0 for same policy, got {info['approx_kl']}"
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Approx KL: {info['approx_kl']:.6f}")
    print(f"  Clip fraction: {info['clip_fraction']:.4f}")
    print(f"  Ratio mean: {info['ratio_mean']:.4f}")
    print("  [PASS] PPO loss OK")


def verify_value_loss():
    """Verify value loss computation."""
    print("Verifying value loss...")

    batch_size = 64
    values = torch.randn(batch_size) * 0.01  # near-zero predictions at init
    returns = torch.randn(batch_size)          # target values with variance

    loss, info = compute_value_loss(values, returns)

    assert torch.isfinite(loss), "Value loss is NaN/Inf"
    assert 0.1 < info["value_loss"] < 2.0, \
        f"Unexpected value_loss: {info['value_loss']:.4f} (expected ~0.5)"
    assert abs(info["explained_variance"]) < 0.2, \
        f"Unexpected explained_variance: {info['explained_variance']:.4f} (expected ~0.0)"
    print(f"  value_loss:         {info['value_loss']:.4f} (expected ~0.5)")
    print(f"  explained_variance: {info['explained_variance']:.4f} (expected ~0.0)")
    print("  [PASS] Value loss OK")


def verify_update():
    """Verify full PPO update step."""
    print("Verifying PPO update...")

    obs_dim, act_dim = 4, 2
    batch_size = 64

    model = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Create mock batch
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)

    with torch.no_grad():
        dist, values = model(obs)
        old_log_probs = dist.log_prob(actions).sum(dim=-1)

    advantages = torch.randn(batch_size)
    returns = values.detach() + advantages

    batch = RolloutBatch(
        observations=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
    )

    # Run multiple updates, check value loss decreases
    initial_value_loss = None
    for i in range(10):
        info = ppo_update(model, optimizer, batch)
        if initial_value_loss is None:
            initial_value_loss = info["value_loss"]

    # Checks
    assert info["value_loss"] < initial_value_loss, "Value loss should decrease"
    assert torch.isfinite(torch.tensor(info["total_loss"])), "Total loss is NaN/Inf"
    assert info["approx_kl"] < 0.05, \
        f"approx_kl should be < 0.05, got {info['approx_kl']}"
    assert torch.isfinite(torch.tensor(info["grad_norm"])), "Grad norm is NaN/Inf"
    print(f"  Initial value loss: {initial_value_loss:.4f}")
    print(f"  Final value loss: {info['value_loss']:.4f}")
    print(f"  Approx KL: {info['approx_kl']:.4f}")
    print(f"  Grad norm: {info['grad_norm']:.4f}")
    print("  [PASS] PPO update OK")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("PPO From Scratch -- Verification")
    print("=" * 60)

    verify_network()
    print()
    verify_gae()
    print()
    verify_ppo_loss()
    print()
    verify_value_loss()
    print()
    verify_update()

    print()
    print("=" * 60)
    print("[ALL PASS] PPO implementation verified")
    print("=" * 60)


# =============================================================================
# Demo: Train on CartPole
# =============================================================================

# --8<-- [start:ppo_training_loop]
def collect_rollout(
    env,
    model: ActorCritic,
    n_steps: int,
    device: torch.device,
) -> tuple[list[dict], float]:
    """
    Collect a rollout of experience from the environment.

    This is the data collection phase of PPO:
    1. Run the policy for n_steps
    2. Store (obs, action, reward, done, log_prob, value) for each step
    3. Return the data for GAE computation and updates

    Args:
        env: Gymnasium environment
        model: Actor-critic network
        n_steps: Number of steps to collect
        device: Torch device

    Returns:
        transitions: List of transition dicts
        episode_return: Total return from completed episodes
    """
    transitions = []
    episode_return = 0.0
    episode_returns = []

    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    for _ in range(n_steps):
        # Get action from policy
        with torch.no_grad():
            dist, value = model(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Step environment
        action_np = action.cpu().numpy().squeeze()
        # CartPole needs discrete action
        action_discrete = int(action_np > 0)  # Simple threshold for demo
        next_obs, reward, terminated, truncated, _ = env.step(action_discrete)
        done = terminated or truncated

        # Store transition
        transitions.append({
            "obs": obs_tensor.squeeze(0),
            "action": action.squeeze(0),
            "reward": reward,
            "done": float(done),
            "log_prob": log_prob.squeeze(0),
            "value": value.squeeze(0),
        })

        episode_return += reward

        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    # Get bootstrap value for final state
    with torch.no_grad():
        _, next_value = model(obs_tensor)

    return transitions, next_value.squeeze(), episode_returns


def transitions_to_batch(
    transitions: list[dict],
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> RolloutBatch:
    """Convert collected transitions to a training batch with GAE."""
    # Stack tensors - ensure all are on the same device
    obs = torch.stack([t["obs"] for t in transitions])
    actions = torch.stack([t["action"] for t in transitions])
    old_log_probs = torch.stack([t["log_prob"] for t in transitions])
    values = torch.stack([t["value"] for t in transitions])
    device = values.device
    rewards = torch.tensor([t["reward"] for t in transitions], device=device)
    dones = torch.tensor([t["done"] for t in transitions], device=device)

    # Compute advantages with GAE
    advantages, returns = compute_gae(rewards, values, next_value, dones, gamma, gae_lambda)

    return RolloutBatch(
        observations=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
    )
# --8<-- [end:ppo_training_loop]


def run_demo():
    """
    Train PPO from scratch on CartPole-v1.

    This demonstrates the full training loop:
    1. Collect rollout with current policy
    2. Compute advantages via GAE
    3. Update policy with multiple epochs
    4. Repeat

    Expected: CartPole should reach ~200 return (solved) within 50k steps.
    """
    import gymnasium as gym

    print("=" * 60)
    print("PPO From Scratch -- CartPole Demo")
    print("=" * 60)
    print()

    # Hyperparameters
    n_steps = 2048       # Steps per rollout
    n_epochs = 10        # Update epochs per rollout
    batch_size = 64      # Minibatch size
    total_steps = 50000  # Total training steps
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Environment
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = 1  # We output a single continuous value, threshold to discrete

    # Model
    model = ActorCritic(obs_dim, act_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    steps_done = 0
    iteration = 0

    while steps_done < total_steps:
        iteration += 1

        # Collect rollout
        transitions, next_value, ep_returns = collect_rollout(env, model, n_steps, device)
        steps_done += len(transitions)

        # Convert to batch
        batch = transitions_to_batch(transitions, next_value, gamma, gae_lambda)

        # Multiple update epochs
        indices = np.arange(len(transitions))
        for epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, len(transitions), batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]

                mb_batch = RolloutBatch(
                    observations=batch.observations[mb_indices],
                    actions=batch.actions[mb_indices],
                    old_log_probs=batch.old_log_probs[mb_indices],
                    advantages=batch.advantages[mb_indices],
                    returns=batch.returns[mb_indices],
                )

                info = ppo_update(model, optimizer, mb_batch, clip_range)

        # Logging
        avg_return = np.mean(ep_returns) if ep_returns else 0.0
        print(f"Iter {iteration:3d} | Steps: {steps_done:6d} | "
              f"Episodes: {len(ep_returns):2d} | "
              f"Avg Return: {avg_return:6.1f} | "
              f"Value Loss: {info['value_loss']:.4f} | "
              f"KL: {info['approx_kl']:.4f}")

        # Early stopping if solved
        if avg_return >= 195.0 and len(ep_returns) >= 5:
            print()
            print(f"[SOLVED] CartPole solved in {steps_done} steps!")
            break

    env.close()

    print()
    print("=" * 60)
    print("Demo complete. This is the same algorithm SB3 uses, just simpler.")
    print("=" * 60)

    return model  # Return trained model for recording


def record_episode(model: ActorCritic, output_path: str = "videos/ppo_cartpole_demo.gif"):
    """
    Record a GIF of the trained policy solving CartPole.

    Args:
        model: Trained actor-critic model
        output_path: Where to save the GIF
    """
    import gymnasium as gym
    from pathlib import Path
    from PIL import Image

    print()
    print("Recording episode...")

    # Create environment with rendering
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    device = next(model.parameters()).device

    frames = []
    obs, _ = env.reset()
    total_reward = 0

    for step in range(500):  # Max steps
        # Render frame
        frame = env.render()
        frames.append(Image.fromarray(frame))

        # Get action from policy (deterministic for recording)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            dist, _ = model(obs_tensor)
            action = dist.mean  # Use mean for deterministic action
            action_discrete = int(action.cpu().numpy().squeeze() > 0)

        obs, reward, terminated, truncated, _ = env.step(action_discrete)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()

    # Save as GIF
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,  # 20 FPS
        loop=0
    )

    print(f"Recorded {len(frames)} frames, total reward: {total_reward}")
    print(f"Saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PPO From Scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --verify           Run sanity checks (all 6 components)
  --compare-sb3      Compare GAE against SB3 RolloutBuffer
  --demo             Train on CartPole and solve it
  --demo --record    Train and save a GIF of the solved policy
        """
    )
    parser.add_argument("--verify", action="store_true", help="Run verification checks")
    parser.add_argument("--compare-sb3", action="store_true", help="Compare core invariants against SB3")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --compare-sb3 (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Run demo training on CartPole")
    parser.add_argument("--record", action="store_true", help="Record a GIF after training")
    args = parser.parse_args()

    if args.compare_sb3:
        run_sb3_comparison(seed=args.seed)
    elif args.verify:
        run_verification()
    elif args.demo:
        model = run_demo()
        if args.record and model is not None:
            record_episode(model, "videos/ppo_cartpole_demo.gif")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
