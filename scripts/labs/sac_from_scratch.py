#!/usr/bin/env python3
"""
SAC From Scratch -- Pedagogical Implementation

This module implements Soft Actor-Critic (SAC) with explicit tensor operations
to show how the math maps to code. Not for production use.

Usage:
    # Run verification (sanity checks, ~2 minutes)
    python scripts/labs/sac_from_scratch.py --verify

    # Train on a simple environment (demonstration)
    python scripts/labs/sac_from_scratch.py --demo

Key regions exported for tutorials:
    - twin_q_loss: Twin Q-network loss with target
    - actor_loss: Policy loss with entropy term
    - temperature_loss: Automatic entropy coefficient tuning
    - sac_update: Full update step combining all components
"""
from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# =============================================================================
# Network Architecture
# =============================================================================

class MLP(nn.Module):
    """Simple MLP with configurable hidden layers."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy with state-dependent mean and log_std.

    Outputs a squashed Gaussian (tanh applied to samples) for bounded actions.
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1])
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action and log probability."""
        features = F.relu(self.backbone.net[:-1](obs))  # All but last layer
        features = self.backbone.net[-1](features)  # Last layer without ReLU

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        # Reparameterization trick: sample = mean + std * noise
        dist = Normal(mean, std)
        x_t = dist.rsample()  # rsample for gradient flow
        action = torch.tanh(x_t)  # squash to [-1, 1]

        # Log prob with squashing correction
        # log pi(a|s) = log mu(u|s) - sum(log(1 - tanh^2(u)))
        log_prob = dist.log_prob(x_t).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(dim=-1)

        return action, log_prob


class TwinQNetwork(nn.Module):
    """Twin Q-networks for reducing overestimation bias."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dims)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# =============================================================================
# Replay Buffer (Minimal)
# =============================================================================

class ReplayBuffer:
    """Simple replay buffer for off-policy learning."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 100000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]).to(device),
            "actions": torch.from_numpy(self.actions[idx]).to(device),
            "rewards": torch.from_numpy(self.rewards[idx]).to(device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(device),
            "dones": torch.from_numpy(self.dones[idx]).to(device),
        }


# =============================================================================
# SAC Loss Functions
# =============================================================================

# --8<-- [start:twin_q_loss]
def compute_q_loss(
    q_network: TwinQNetwork,
    target_q_network: TwinQNetwork,
    policy: GaussianPolicy,
    batch: dict,
    gamma: float = 0.99,
    alpha: float = 0.2,  # entropy coefficient
) -> tuple[torch.Tensor, dict]:
    """
    Compute twin Q-network loss with soft Bellman backup.

    SAC uses two Q-networks and takes the minimum to reduce overestimation:
        y = r + gamma * (1 - d) * (min(Q1_target, Q2_target) - alpha * log_pi)

    The entropy term (-alpha * log_pi) encourages exploration by making
    high-entropy (uncertain) states more valuable.

    Args:
        q_network: Twin Q-networks being trained
        target_q_network: Target networks (slowly updated copies)
        policy: Current policy (for computing next actions)
        batch: Dictionary with obs, actions, rewards, next_obs, dones
        gamma: Discount factor
        alpha: Entropy coefficient (temperature)

    Returns:
        loss: Combined Q1 + Q2 MSE loss
        info: Diagnostic metrics
    """
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_obs = batch["next_obs"]
    dones = batch["dones"]

    # Current Q-values
    q1, q2 = q_network(obs, actions)

    # Compute target Q-values (no gradient through target)
    with torch.no_grad():
        # Sample next actions from current policy
        next_actions, next_log_probs = policy(next_obs)

        # Target Q-values: use minimum of two targets
        target_q1, target_q2 = target_q_network(next_obs, next_actions)
        target_q = torch.min(target_q1, target_q2)

        # Soft Bellman backup with entropy bonus
        # The -alpha * log_pi term makes uncertain states more valuable
        target = rewards + gamma * (1.0 - dones) * (target_q - alpha * next_log_probs)

    # MSE loss for both Q-networks
    q1_loss = F.mse_loss(q1, target)
    q2_loss = F.mse_loss(q2, target)
    q_loss = q1_loss + q2_loss

    info = {
        "q1_loss": q1_loss.item(),
        "q2_loss": q2_loss.item(),
        "q1_mean": q1.mean().item(),
        "q2_mean": q2.mean().item(),
        "target_q_mean": target.mean().item(),
    }

    return q_loss, info
# --8<-- [end:twin_q_loss]


# --8<-- [start:actor_loss]
def compute_actor_loss(
    policy: GaussianPolicy,
    q_network: TwinQNetwork,
    obs: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    Compute actor (policy) loss with entropy regularization.

    SAC maximizes: E[Q(s,a) - alpha * log pi(a|s)]

    This objective has two parts:
    1. Q(s,a): prefer actions that lead to high value
    2. -alpha * log pi: prefer high-entropy (exploratory) policies

    The entropy term prevents the policy from collapsing to a deterministic
    solution too early, which helps exploration and robustness.

    Args:
        policy: Gaussian policy network
        q_network: Twin Q-networks (for evaluating action quality)
        obs: Batch of observations
        alpha: Entropy coefficient (temperature)

    Returns:
        loss: Negative of objective (we minimize)
        info: Diagnostic metrics including entropy
    """
    # Sample actions from current policy
    actions, log_probs = policy(obs)

    # Evaluate actions with Q-networks (use minimum for pessimism)
    q1, q2 = q_network(obs, actions)
    q_value = torch.min(q1, q2)

    # Actor loss: maximize Q-value while maintaining entropy
    # Loss = E[alpha * log_pi - Q]  (negative because we minimize)
    actor_loss = (alpha * log_probs - q_value).mean()

    info = {
        "actor_loss": actor_loss.item(),
        "entropy": -log_probs.mean().item(),  # H = -E[log pi]
        "log_prob_mean": log_probs.mean().item(),
    }

    return actor_loss, info
# --8<-- [end:actor_loss]


# --8<-- [start:temperature_loss]
def compute_temperature_loss(
    log_alpha: torch.Tensor,      # learnable log(alpha)
    log_probs: torch.Tensor,      # log pi(a|s) from policy
    target_entropy: float,        # target entropy (usually -dim(A))
) -> tuple[torch.Tensor, dict]:
    """
    Compute entropy coefficient (temperature) loss for automatic tuning.

    Instead of hand-tuning alpha, SAC can learn it automatically:
        L(alpha) = E[-alpha * (log_pi + H_target)]

    This adjusts alpha to maintain a target entropy level:
    - If entropy is too low (policy too deterministic), increase alpha
    - If entropy is too high (policy too random), decrease alpha

    Args:
        log_alpha: Learnable parameter log(alpha) (ensures alpha > 0)
        log_probs: Log probabilities from the policy
        target_entropy: Desired entropy level (typically -act_dim)

    Returns:
        loss: Temperature loss
        info: Current alpha value
    """
    alpha = log_alpha.exp()

    # Loss: adjust alpha to push entropy toward target
    # If log_pi > -target (entropy too low), loss is positive, alpha increases
    # If log_pi < -target (entropy too high), loss is negative, alpha decreases
    alpha_loss = -(alpha * (log_probs.detach() + target_entropy)).mean()

    info = {
        "alpha_loss": alpha_loss.item(),
        "alpha": alpha.item(),
    }

    return alpha_loss, info
# --8<-- [end:temperature_loss]


# --8<-- [start:sac_update]
def sac_update(
    policy: GaussianPolicy,
    q_network: TwinQNetwork,
    target_q_network: TwinQNetwork,
    log_alpha: torch.Tensor,
    policy_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    batch: dict,
    target_entropy: float,
    gamma: float = 0.99,
    tau: float = 0.005,  # target network update rate
) -> dict:
    """
    Perform one SAC update step.

    SAC updates three components:
    1. Q-networks: minimize Bellman error
    2. Policy: maximize Q-value + entropy
    3. Temperature: maintain target entropy

    Target networks are updated with Polyak averaging: theta_target = tau * theta + (1-tau) * theta_target

    Args:
        policy: Gaussian policy network
        q_network: Twin Q-networks
        target_q_network: Target Q-networks (slowly updated)
        log_alpha: Learnable log(entropy coefficient)
        policy_optimizer: Optimizer for policy
        q_optimizer: Optimizer for Q-networks
        alpha_optimizer: Optimizer for temperature
        batch: Replay buffer sample
        target_entropy: Target entropy for auto-tuning
        gamma: Discount factor
        tau: Target network update rate

    Returns:
        info: Dictionary with all loss components and diagnostics
    """
    alpha = log_alpha.exp().item()

    # --- Q-network update ---
    q_loss, q_info = compute_q_loss(
        q_network, target_q_network, policy, batch, gamma, alpha
    )
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    # --- Policy update ---
    actor_loss, actor_info = compute_actor_loss(
        policy, q_network, batch["obs"], alpha
    )
    policy_optimizer.zero_grad()
    actor_loss.backward()
    policy_optimizer.step()

    # --- Temperature update ---
    with torch.no_grad():
        _, log_probs = policy(batch["obs"])
    alpha_loss, alpha_info = compute_temperature_loss(
        log_alpha, log_probs, target_entropy
    )
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    # --- Target network update (Polyak averaging) ---
    with torch.no_grad():
        for param, target_param in zip(q_network.parameters(), target_q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    return {**q_info, **actor_info, **alpha_info}
# --8<-- [end:sac_update]


# =============================================================================
# Verification (Sanity Checks)
# =============================================================================

def verify_q_network():
    """Verify Q-network forward pass."""
    print("Verifying Q-network...")

    obs_dim, act_dim = 10, 4
    batch_size = 32

    q_net = TwinQNetwork(obs_dim, act_dim)
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)

    q1, q2 = q_net(obs, actions)

    assert q1.shape == (batch_size,), f"Q1 shape mismatch: {q1.shape}"
    assert q2.shape == (batch_size,), f"Q2 shape mismatch: {q2.shape}"
    assert torch.isfinite(q1).all(), "Q1 contains NaN/Inf"
    assert torch.isfinite(q2).all(), "Q2 contains NaN/Inf"
    print(f"  Q1 mean: {q1.mean().item():.4f}")
    print(f"  Q2 mean: {q2.mean().item():.4f}")
    print("  [PASS] Q-network OK")


def verify_policy():
    """Verify policy forward pass."""
    print("Verifying policy...")

    obs_dim, act_dim = 10, 4
    batch_size = 32

    policy = GaussianPolicy(obs_dim, act_dim)
    obs = torch.randn(batch_size, obs_dim)

    actions, log_probs = policy(obs)

    assert actions.shape == (batch_size, act_dim), f"Action shape mismatch: {actions.shape}"
    assert log_probs.shape == (batch_size,), f"Log prob shape mismatch: {log_probs.shape}"
    assert (actions.abs() <= 1.0).all(), "Actions not in [-1, 1]"
    assert torch.isfinite(log_probs).all(), "Log probs contain NaN/Inf"
    print(f"  Actions in [{actions.min().item():.2f}, {actions.max().item():.2f}]")
    print(f"  Log prob mean: {log_probs.mean().item():.4f}")
    print("  [PASS] Policy OK")


def verify_sac_update():
    """Verify full SAC update step."""
    print("Verifying SAC update...")

    obs_dim, act_dim = 10, 4
    batch_size = 64
    device = torch.device("cpu")

    # Create networks
    policy = GaussianPolicy(obs_dim, act_dim)
    q_network = TwinQNetwork(obs_dim, act_dim)
    target_q_network = TwinQNetwork(obs_dim, act_dim)
    target_q_network.load_state_dict(q_network.state_dict())

    log_alpha = torch.tensor(0.0, requires_grad=True)
    target_entropy = -float(act_dim)

    # Optimizers
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

    # Mock batch
    batch = {
        "obs": torch.randn(batch_size, obs_dim),
        "actions": torch.randn(batch_size, act_dim).clamp(-1, 1),
        "rewards": torch.randn(batch_size),
        "next_obs": torch.randn(batch_size, obs_dim),
        "dones": torch.zeros(batch_size),
    }

    # Run updates
    initial_alpha = log_alpha.exp().item()
    for _ in range(20):
        info = sac_update(
            policy, q_network, target_q_network, log_alpha,
            policy_optimizer, q_optimizer, alpha_optimizer,
            batch, target_entropy
        )

    # Checks
    assert torch.isfinite(torch.tensor(info["q1_loss"])), "Q1 loss is NaN/Inf"
    assert torch.isfinite(torch.tensor(info["actor_loss"])), "Actor loss is NaN/Inf"
    assert info["alpha"] > 0, "Alpha should be positive"
    print(f"  Q1 loss: {info['q1_loss']:.4f}")
    print(f"  Actor loss: {info['actor_loss']:.4f}")
    print(f"  Alpha: {initial_alpha:.4f} -> {info['alpha']:.4f}")
    print("  [PASS] SAC update OK")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("SAC From Scratch -- Verification")
    print("=" * 60)

    verify_q_network()
    print()
    verify_policy()
    print()
    verify_sac_update()

    print()
    print("=" * 60)
    print("[ALL PASS] SAC implementation verified")
    print("=" * 60)


def run_demo(total_steps: int = 5000):
    """
    Train SAC from scratch on Pendulum-v1.

    Pendulum is a classic continuous control task:
    - Swing up and balance an inverted pendulum
    - Continuous action space (torque)
    - Dense reward (angle + velocity penalty)

    Args:
        total_steps: Training steps. Use 5000 for quick demo, 50000+ to solve.

    Performance guide:
    - Random policy: ~-1200 return
    - After 5k steps: ~-900 to -1000 (learning visible, not solved)
    - After 50k steps: ~-200 to -300 (solved)
    """
    import gymnasium as gym

    print("=" * 60)
    print("SAC From Scratch -- Pendulum Demo")
    print("=" * 60)
    print()

    # Hyperparameters
    buffer_size = 50000 if total_steps > 10000 else 10000
    batch_size = 256 if total_steps > 10000 else 128
    learning_starts = 1000 if total_steps > 10000 else 500
    lr = 3e-4
    gamma = 0.99
    tau = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Total steps: {total_steps}" + (" (use --steps 50000 to solve)" if total_steps < 20000 else ""))

    # Environment
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = env.action_space.high[0]  # Actions in [-2, 2] for Pendulum

    print(f"Environment: Pendulum-v1 (obs={obs_dim}, act={act_dim})")

    # Networks
    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    q_network = TwinQNetwork(obs_dim, act_dim).to(device)
    target_q_network = TwinQNetwork(obs_dim, act_dim).to(device)
    target_q_network.load_state_dict(q_network.state_dict())

    # Temperature (learnable)
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    target_entropy = -float(act_dim)

    # Optimizers
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr)

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

    # Training loop
    obs, _ = env.reset()
    episode_return = 0.0
    episode_returns = []

    for step in range(total_steps):
        # Select action
        if step < learning_starts:
            # Random exploration
            action = env.action_space.sample()
            action_for_buffer = action / act_scale
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action_tensor, _ = policy(obs_tensor)
                action_for_buffer = action_tensor.cpu().numpy().squeeze()  # [-1, 1]
                action = action_for_buffer * act_scale  # Scale to env range

        # Step environment - Pendulum expects shape (1,) not scalar
        action_env = np.atleast_1d(action)
        next_obs, reward, terminated, truncated, _ = env.step(action_env)
        done = terminated or truncated

        # Store transition (normalized action in [-1, 1])
        action_buffer = np.atleast_1d(action_for_buffer)
        replay_buffer.add(obs, action_buffer, reward, next_obs, float(done))

        episode_return += reward
        obs = next_obs

        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

        # Update (after learning_starts)
        if step >= learning_starts:
            batch = replay_buffer.sample(batch_size, device)
            # Note: actions in buffer are already in [-1, 1] (policy output scale)
            # Q-network was trained on this scale, so no rescaling needed

            info = sac_update(
                policy, q_network, target_q_network, log_alpha,
                policy_optimizer, q_optimizer, alpha_optimizer,
                batch, target_entropy, gamma, tau
            )

            # Logging (every 10% of training or every 500 steps, whichever is larger)
            log_interval = max(500, total_steps // 10)
            if (step + 1) % log_interval == 0:
                avg_return = np.mean(episode_returns[-10:]) if episode_returns else 0
                print(f"Step {step+1:5d} | "
                      f"Avg Return: {avg_return:7.1f} | "
                      f"Q1: {info['q1_mean']:6.2f} | "
                      f"Alpha: {info['alpha']:.3f} | "
                      f"Entropy: {info['entropy']:.2f}")

    env.close()

    final_return = np.mean(episode_returns[-10:]) if len(episode_returns) >= 10 else np.mean(episode_returns)
    print()
    print(f"Final average return: {final_return:.1f}")
    print(f"  Random policy: ~-1200")
    print(f"  Solved:        ~-200")
    print()

    if final_return > -300:
        print(f"[SOLVED] Pendulum solved with return {final_return:.0f}!")
    elif final_return > -800:
        print(f"Good progress! Return improved significantly from random.")
        print("For better results, try: --demo --steps 50000")
    else:
        print(f"Learning started. For solved policy: --demo --steps 50000")

    print()
    print("=" * 60)
    print("Demo complete. This is the same algorithm SB3 uses, just simpler.")
    print("=" * 60)

    return policy  # Return trained policy for recording


def record_episode(policy: GaussianPolicy, output_path: str = "videos/sac_pendulum_demo.gif"):
    """
    Record a GIF of the trained policy solving Pendulum.

    Args:
        policy: Trained Gaussian policy
        output_path: Where to save the GIF
    """
    import gymnasium as gym
    from pathlib import Path
    from PIL import Image

    print()
    print("Recording episode...")

    # Create environment with rendering
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    act_scale = env.action_space.high[0]
    device = next(policy.parameters()).device

    frames = []
    obs, _ = env.reset()
    total_reward = 0

    for step in range(200):  # One episode (Pendulum has 200 max steps)
        # Render frame
        frame = env.render()
        frames.append(Image.fromarray(frame))

        # Get action from policy (use mean for deterministic)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = policy(obs_tensor)
            action_np = action.cpu().numpy().squeeze() * act_scale

        action_env = np.atleast_1d(action_np)
        obs, reward, terminated, truncated, _ = env.step(action_env)
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

    print(f"Recorded {len(frames)} frames, total reward: {total_reward:.1f}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAC From Scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --verify              Run sanity checks (~10 seconds)
  --demo                Quick demo, shows learning (~30 seconds, 5k steps)
  --demo --steps 50000  Full demo, actually solves Pendulum (~5 minutes)
  --demo --steps 50000 --record   Solve and save a GIF
        """
    )
    parser.add_argument("--verify", action="store_true", help="Run verification checks")
    parser.add_argument("--demo", action="store_true", help="Run demo training on Pendulum")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Training steps for demo (default: 5000, use 50000+ to solve)")
    parser.add_argument("--record", action="store_true", help="Record a GIF after training")
    args = parser.parse_args()

    if args.verify:
        run_verification()
    elif args.demo:
        policy = run_demo(total_steps=args.steps)
        if args.record and policy is not None:
            record_episode(policy, "videos/sac_pendulum_demo.gif")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
