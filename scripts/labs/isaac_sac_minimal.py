#!/usr/bin/env python3
"""
Isaac SAC Minimal -- Build-It module for Appendix E.

Pedagogical goal:
- Show the core SAC equations in compact PyTorch code for dict observations.
- Keep the implementation small and explicit (not production-ready).
- Provide snippet regions for chapter/tutorial includes.

Usage:
    # Run lightweight sanity checks (~10-20 seconds CPU)
    python scripts/labs/isaac_sac_minimal.py --verify

    # Print how this maps to SB3 Appendix E pipeline
    python scripts/labs/isaac_sac_minimal.py --bridge

Notes:
- This file does NOT boot Isaac Sim during --verify.
- It uses synthetic batches to validate update math and tensor flow.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _mlp(input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for width in hidden_dims:
        layers.append(nn.Linear(last, width))
        layers.append(nn.ReLU())
        last = width
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


# --8<-- [start:dict_flatten_encoder]
class DictFlattenEncoder(nn.Module):
    """
    Flatten selected dict-observation keys into one feature vector.

    Supports both observation conventions:
    - Gymnasium-Robotics goal-conditioned: keys=["observation", "achieved_goal", "desired_goal"]
    - Isaac Lab flat-dict: keys=["policy"]

    The keys parameter makes this encoder agnostic to the observation layout.
    """

    def __init__(self, keys: list[str]):
        super().__init__()
        if not keys:
            raise ValueError("keys must be non-empty")
        self.keys = keys

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for key in self.keys:
            x = obs[key]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            parts.append(x.float())
        return torch.cat(parts, dim=-1)
# --8<-- [end:dict_flatten_encoder]


# --8<-- [start:squashed_gaussian_actor]
class SquashedGaussianActor(nn.Module):
    """SAC actor with tanh squashing and log-prob correction."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden = hidden_dims or [256, 256]
        self.backbone = _mlp(obs_dim, hidden[-1], hidden[:-1])
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs_flat)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mu, std)
        u = dist.rsample()  # reparameterization trick
        action = torch.tanh(u)

        # log pi(a|s) with tanh correction
        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob
# --8<-- [end:squashed_gaussian_actor]


# --8<-- [start:twin_q_critic]
class TwinQCritic(nn.Module):
    """Twin Q critics used by SAC to reduce overestimation bias."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden = hidden_dims or [256, 256]
        self.q1 = _mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = _mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs_flat: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_flat, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
# --8<-- [end:twin_q_critic]


@dataclass
class Batch:
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: dict[str, torch.Tensor]
    dones: torch.Tensor


def _polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.copy_(tau * src.data + (1.0 - tau) * tgt.data)


# --8<-- [start:sac_losses]
def critic_loss(
    encoder: DictFlattenEncoder,
    actor: SquashedGaussianActor,
    critic: TwinQCritic,
    critic_target: TwinQCritic,
    batch: Batch,
    gamma: float,
    alpha: torch.Tensor,
) -> torch.Tensor:
    obs_flat = encoder(batch.obs)
    next_flat = encoder(batch.next_obs)

    q1, q2 = critic(obs_flat, batch.actions)

    with torch.no_grad():
        next_a, next_logp = actor(next_flat)
        tq1, tq2 = critic_target(next_flat, next_a)
        tq = torch.min(tq1, tq2)
        target = batch.rewards + gamma * (1.0 - batch.dones) * (tq - alpha * next_logp)

    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def actor_loss(
    encoder: DictFlattenEncoder,
    actor: SquashedGaussianActor,
    critic: TwinQCritic,
    batch: Batch,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_flat = encoder(batch.obs)
    act, logp = actor(obs_flat)
    q1, q2 = critic(obs_flat, act)
    q = torch.min(q1, q2)
    loss = (alpha * logp - q).mean()
    return loss, logp


def temperature_loss(log_alpha: torch.Tensor, logp: torch.Tensor, target_entropy: float) -> torch.Tensor:
    alpha = log_alpha.exp()
    return -(alpha * (logp.detach() + target_entropy)).mean()
# --8<-- [end:sac_losses]


# --8<-- [start:sac_update_step]
def sac_update_step(
    encoder: DictFlattenEncoder,
    actor: SquashedGaussianActor,
    critic: TwinQCritic,
    critic_target: TwinQCritic,
    log_alpha: torch.Tensor,
    batch: Batch,
    *,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer,
    gamma: float = 0.99,
    tau: float = 0.005,
    target_entropy: float | None = None,
) -> dict[str, float]:
    if target_entropy is None:
        target_entropy = -float(batch.actions.shape[-1])

    alpha = log_alpha.exp()

    c_loss = critic_loss(encoder, actor, critic, critic_target, batch, gamma, alpha)
    critic_opt.zero_grad()
    c_loss.backward()
    critic_opt.step()

    a_loss, logp = actor_loss(encoder, actor, critic, batch, alpha)
    actor_opt.zero_grad()
    a_loss.backward()
    actor_opt.step()

    t_loss = temperature_loss(log_alpha, logp, target_entropy)
    alpha_opt.zero_grad()
    t_loss.backward()
    alpha_opt.step()

    _polyak_update(critic, critic_target, tau)

    return {
        "critic_loss": float(c_loss.item()),
        "actor_loss": float(a_loss.item()),
        "alpha_loss": float(t_loss.item()),
        "alpha": float(log_alpha.exp().item()),
        "entropy": float((-logp.mean()).item()),
    }
# --8<-- [end:sac_update_step]


def _make_synthetic_batch(batch_size: int, obs_dim: int, goal_dim: int, act_dim: int, device: torch.device) -> Batch:
    obs = {
        "observation": torch.randn(batch_size, obs_dim, device=device),
        "achieved_goal": torch.randn(batch_size, goal_dim, device=device),
        "desired_goal": torch.randn(batch_size, goal_dim, device=device),
    }
    next_obs = {
        "observation": torch.randn(batch_size, obs_dim, device=device),
        "achieved_goal": torch.randn(batch_size, goal_dim, device=device),
        "desired_goal": torch.randn(batch_size, goal_dim, device=device),
    }

    # Actions are already in [-1, 1] scale expected by tanh-policy critics.
    actions = torch.rand(batch_size, act_dim, device=device) * 2.0 - 1.0
    rewards = torch.randn(batch_size, device=device)
    dones = torch.randint(0, 2, (batch_size,), device=device).float()
    return Batch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)


def _make_isaac_style_batch(batch_size: int, obs_dim: int, act_dim: int, device: torch.device) -> Batch:
    """Build a batch mimicking Isaac Lab's {'policy': (batch, obs_dim)} convention."""
    obs = {"policy": torch.randn(batch_size, obs_dim, device=device)}
    next_obs = {"policy": torch.randn(batch_size, obs_dim, device=device)}
    actions = torch.rand(batch_size, act_dim, device=device) * 2.0 - 1.0
    rewards = torch.randn(batch_size, device=device)
    dones = torch.randint(0, 2, (batch_size,), device=device).float()
    return Batch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)


def run_verification(seed: int = 0) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = 18
    goal_dim = 3
    act_dim = 4
    batch_size = 128

    encoder = DictFlattenEncoder(["observation", "achieved_goal", "desired_goal"]).to(device)
    flat_dim = obs_dim + goal_dim + goal_dim
    actor = SquashedGaussianActor(flat_dim, act_dim).to(device)
    critic = TwinQCritic(flat_dim, act_dim).to(device)
    critic_target = TwinQCritic(flat_dim, act_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)

    print("=" * 60)
    print("Isaac SAC Minimal -- Verification")
    print("=" * 60)
    print(f"device={device}")

    # Shape checks
    batch = _make_synthetic_batch(batch_size, obs_dim, goal_dim, act_dim, device)
    obs_flat = encoder(batch.obs)
    assert obs_flat.shape == (batch_size, flat_dim), f"Unexpected flat shape: {obs_flat.shape}"
    actions, logp = actor(obs_flat)
    assert actions.shape == (batch_size, act_dim), f"Unexpected action shape: {actions.shape}"
    assert logp.shape == (batch_size,), f"Unexpected logp shape: {logp.shape}"
    assert torch.isfinite(actions).all(), "Actor output has NaN/Inf"
    assert torch.isfinite(logp).all(), "Actor logp has NaN/Inf"
    print("[PASS] Encoder/actor shape checks")

    # Update checks
    last_info: dict[str, float] = {}
    for _ in range(25):
        batch = _make_synthetic_batch(batch_size, obs_dim, goal_dim, act_dim, device)
        last_info = sac_update_step(
            encoder,
            actor,
            critic,
            critic_target,
            log_alpha,
            batch,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            alpha_opt=alpha_opt,
            gamma=0.99,
            tau=0.005,
            target_entropy=-float(act_dim),
        )

    for name, value in last_info.items():
        assert torch.isfinite(torch.tensor(value)), f"{name} became non-finite: {value}"
    assert last_info["alpha"] > 0.0, "alpha must stay positive"

    print(f"critic_loss={last_info['critic_loss']:.4f}")
    print(f"actor_loss={last_info['actor_loss']:.4f}")
    print(f"alpha={last_info['alpha']:.4f}")
    print("[PASS] SAC update checks")

    # Test 2: Isaac Lab convention (single 'policy' key).
    # Isaac Lab envs expose {'policy': (num_envs, obs_dim)} instead of
    # {observation, achieved_goal, desired_goal}. Verify encoder handles it.
    print()
    print("--- Isaac Lab observation convention ---")
    isaac_obs_dim = 32
    encoder_isaac = DictFlattenEncoder(["policy"]).to(device)
    batch_isaac = _make_isaac_style_batch(batch_size, isaac_obs_dim, act_dim, device)
    obs_flat_isaac = encoder_isaac(batch_isaac.obs)
    assert obs_flat_isaac.shape == (batch_size, isaac_obs_dim), (
        f"Expected ({batch_size}, {isaac_obs_dim}), got {obs_flat_isaac.shape}"
    )
    assert torch.isfinite(obs_flat_isaac).all(), "Isaac-style encoding has NaN/Inf"

    # Build actor/critic for Isaac obs dim and run one update.
    actor_isaac = SquashedGaussianActor(isaac_obs_dim, act_dim).to(device)
    critic_isaac = TwinQCritic(isaac_obs_dim, act_dim).to(device)
    critic_target_isaac = TwinQCritic(isaac_obs_dim, act_dim).to(device)
    critic_target_isaac.load_state_dict(critic_isaac.state_dict())
    log_alpha_isaac = torch.tensor(0.0, requires_grad=True, device=device)

    isaac_info = sac_update_step(
        encoder_isaac, actor_isaac, critic_isaac, critic_target_isaac,
        log_alpha_isaac, batch_isaac,
        actor_opt=torch.optim.Adam(actor_isaac.parameters(), lr=3e-4),
        critic_opt=torch.optim.Adam(critic_isaac.parameters(), lr=3e-4),
        alpha_opt=torch.optim.Adam([log_alpha_isaac], lr=3e-4),
    )
    for name, value in isaac_info.items():
        assert torch.isfinite(torch.tensor(value)), f"Isaac {name} non-finite: {value}"
    print("[PASS] Isaac-style observation encoding + SAC update")

    print("=" * 60)
    print("[ALL PASS] Isaac SAC minimal verified")
    print("=" * 60)


def run_bridge() -> None:
    print("=" * 60)
    print("Isaac SAC Minimal -- Bridge to SB3 Appendix E")
    print("=" * 60)
    print("From-scratch component -> SB3/Run-It counterpart")
    print("  DictFlattenEncoder    -> MultiInputPolicy feature concat (goal-conditioned)")
    print("                        -> MlpPolicy feature extractor (Isaac flat-dict)")
    print("  SquashedGaussianActor -> SAC actor network")
    print("  TwinQCritic           -> SAC critic/target critics")
    print("  temperature_loss      -> SAC ent_coef auto tuning")
    print("  sac_update_step       -> model.learn() gradient step internals")
    print()
    print("Observation conventions:")
    print("  Gymnasium-Robotics: {observation, achieved_goal, desired_goal} -> MultiInputPolicy + HER")
    print("  Isaac Lab:          {policy: flat_vector}                      -> MlpPolicy, no HER")
    print("  The Run-It adapter detects the convention and picks accordingly.")
    print()
    print("Run-It script: scripts/appendix_e_isaac_manipulation.py")
    print("Build-It add-on: this file + isaac_goal_relabeler.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac SAC Minimal (Build-It)")
    parser.add_argument("--verify", action="store_true", help="Run synthetic sanity checks")
    parser.add_argument("--bridge", action="store_true", help="Print Build-It <-> Run-It mapping")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --verify")
    args = parser.parse_args()

    if args.verify:
        run_verification(seed=args.seed)
    elif args.bridge:
        run_bridge()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
