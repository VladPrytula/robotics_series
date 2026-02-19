from __future__ import annotations

"""
SB3 comparison helpers (optional).

These utilities are used by the lab modules' `--compare-sb3` mode to validate
that our from-scratch implementations match Stable-Baselines3 (SB3) on core
invariants.

Notes:
- The core lab implementations remain SB3-free. This file is only imported
  when the user explicitly requests SB3 comparison.
- Intended to be run inside the project's Docker environment where SB3 is
  installed (see `bash docker/dev.sh ...`).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    passed: bool
    metrics: dict[str, float]
    notes: str | None = None


def compare_ppo_gae_to_sb3(
    *,
    seed: int = 0,
    horizon: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    atol: float = 1e-6,
) -> ComparisonResult:
    """Compare our GAE computation against SB3 RolloutBuffer."""
    from scripts.labs.ppo_from_scratch import compute_gae

    try:
        from gymnasium import spaces
        from stable_baselines3.common.buffers import RolloutBuffer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SB3 comparison requires gymnasium + stable-baselines3. "
            "Run inside Docker: `bash docker/dev.sh python scripts/labs/ppo_from_scratch.py --compare-sb3`."
        ) from exc

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    rewards = torch.from_numpy(rng.standard_normal(horizon).astype(np.float32))
    values = torch.from_numpy(rng.standard_normal(horizon).astype(np.float32))
    dones = torch.from_numpy((rng.random(horizon) < 0.15).astype(np.float32))
    next_value = torch.from_numpy(rng.standard_normal(1).astype(np.float32)).squeeze(0)

    ours_adv, ours_returns = compute_gae(
        rewards=rewards,
        values=values,
        next_value=next_value,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    sb3_buf = RolloutBuffer(
        buffer_size=horizon,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=1,
    )

    sb3_buf.rewards[:] = rewards.cpu().numpy().reshape(horizon, 1)
    sb3_buf.values[:] = values.cpu().numpy().reshape(horizon, 1)

    episode_starts = np.zeros((horizon, 1), dtype=bool)
    episode_starts[0, 0] = True
    if horizon > 1:
        episode_starts[1:, 0] = dones.cpu().numpy()[:-1].astype(bool)
    sb3_buf.episode_starts[:] = episode_starts

    last_values = torch.tensor([float(next_value.item())], dtype=torch.float32)
    dones_last = np.asarray([float(dones[-1].item())], dtype=np.float32)

    sb3_buf.compute_returns_and_advantage(last_values=last_values, dones=dones_last)

    sb3_adv = torch.from_numpy(sb3_buf.advantages.squeeze(-1)).to(dtype=torch.float32)
    sb3_returns = torch.from_numpy(sb3_buf.returns.squeeze(-1)).to(dtype=torch.float32)

    max_abs_adv_diff = float((ours_adv.cpu() - sb3_adv).abs().max().item())
    max_abs_returns_diff = float((ours_returns.cpu() - sb3_returns).abs().max().item())

    passed = (max_abs_adv_diff <= atol) and (max_abs_returns_diff <= atol)
    return ComparisonResult(
        name="ppo_gae_vs_sb3",
        passed=passed,
        metrics={
            "max_abs_adv_diff": max_abs_adv_diff,
            "max_abs_returns_diff": max_abs_returns_diff,
            "atol": float(atol),
        },
        notes=None if passed else "Mismatch suggests a bug in masking or episode boundary handling.",
    )


def compare_sac_squashed_gaussian_log_prob_to_sb3(
    *,
    seed: int = 0,
    batch_size: int = 256,
    act_dim: int = 4,
    atol: float = 5e-2,
) -> ComparisonResult:
    """Compare our tanh-squashed Gaussian log-prob against SB3 distribution."""
    try:
        from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SB3 comparison requires stable-baselines3. "
            "Run inside Docker: `bash docker/dev.sh python scripts/labs/sac_from_scratch.py --compare-sb3`."
        ) from exc

    torch.manual_seed(seed)

    mean = torch.randn(batch_size, act_dim)
    # Clamp log_std to [-2, 0.5] so samples stay in the regime where both
    # Jacobian formulas agree.  Our numerically stable formula gives exact
    # log(1 - tanh^2(u)), while SB3 uses log(1 - a^2 + eps) with eps=1e-6.
    # For |pre_tanh| > ~10, tanh saturates in float32 and SB3's epsilon
    # dominates, causing an expected divergence of ~14 nats.  Clamping std
    # to <= 1.65 keeps nearly all samples below |pre_tanh| < 6 where the
    # epsilon is negligible.
    log_std = torch.clamp(torch.randn(batch_size, act_dim), min=-2.0, max=0.5)
    std = log_std.exp()
    base_dist = torch.distributions.Normal(mean, std)

    pre_tanh = base_dist.rsample()
    actions = torch.tanh(pre_tanh)

    ours_log_prob = base_dist.log_prob(pre_tanh).sum(dim=-1)
    ours_log_prob -= (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1)

    sb3_dist = SquashedDiagGaussianDistribution(act_dim).proba_distribution(mean, log_std)
    # Pass gaussian_actions=pre_tanh so SB3 skips atanh(tanh(x)), which is
    # numerically lossy for large |x| (tanh saturates, atanh clips).
    sb3_log_prob = sb3_dist.log_prob(actions, gaussian_actions=pre_tanh).reshape(-1)

    max_abs_log_prob_diff = float((ours_log_prob.reshape(-1) - sb3_log_prob).abs().max().item())
    passed = max_abs_log_prob_diff <= atol
    return ComparisonResult(
        name="sac_log_prob_vs_sb3",
        passed=passed,
        metrics={
            "max_abs_log_prob_diff": max_abs_log_prob_diff,
            "atol": float(atol),
        },
        notes=None if passed else "Mismatch suggests a bug in the tanh Jacobian correction term.",
    )


def compare_her_relabeling_to_sb3(
    *,
    seed: int = 0,
    obs_dim: int = 10,
    act_dim: int = 4,
    goal_dim: int = 3,
    n_steps: int = 64,
    batch_size: int = 512,
    n_sampled_goal: int = 4,
    goal_selection_strategy: str = "future",
    distance_threshold: float = 0.05,
) -> ComparisonResult:
    """
    Compare HER reward recomputation invariants against SB3 HerReplayBuffer.

    This is not a training comparison. It checks that when SB3 relabels goals,
    the sampled rewards are exactly those returned by `env.compute_reward(...)`.
    """
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SB3 HER comparison requires gymnasium + stable-baselines3. "
            "Run inside Docker: `bash docker/dev.sh python scripts/labs/her_relabeler.py --compare-sb3`."
        ) from exc

    class GoalEnvStub(gym.Env):
        metadata: dict[str, Any] = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                    ),
                    "achieved_goal": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                    ),
                    "desired_goal": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                    ),
                }
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
            )
            self.distance_threshold = float(distance_threshold)
            self._rng = np.random.default_rng(seed)
            self._desired_goal = (self._rng.standard_normal(goal_dim) + 10.0).astype(np.float32)
            self._achieved_goal = self._rng.standard_normal(goal_dim).astype(np.float32)

        def compute_reward(self, achieved_goal, desired_goal, _info):  # type: ignore[override]
            # Note: real Fetch environments ignore info in compute_reward.
            # SB3's HerReplayBuffer passes info as a list (one per vec-env),
            # so we must not call info.get(...) here.
            achieved = np.asarray(achieved_goal, dtype=np.float32)
            desired = np.asarray(desired_goal, dtype=np.float32)
            distances = np.linalg.norm(achieved - desired, axis=-1)
            return np.where(distances < self.distance_threshold, 0.0, -1.0).astype(np.float32)

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            obs = self._rng.standard_normal(obs_dim).astype(np.float32)
            return (
                {
                    "observation": obs,
                    "achieved_goal": self._achieved_goal.copy(),
                    "desired_goal": self._desired_goal.copy(),
                },
                {},
            )

        def step(self, action):
            obs, info = self.reset()
            reward = float(
                self.compute_reward(
                    obs["achieved_goal"],
                    obs["desired_goal"],
                    {"distance_threshold": self.distance_threshold},
                )
            )
            terminated = False
            truncated = False
            return obs, reward, terminated, truncated, info

    env = GoalEnvStub()
    vec_env = DummyVecEnv([lambda: env])

    buf = HerReplayBuffer(
        buffer_size=max(1024, n_steps * 4),
        observation_space=env.observation_space,
        action_space=env.action_space,
        env=vec_env,
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy=goal_selection_strategy,
    )

    rng = np.random.default_rng(seed)

    desired_goal = (rng.standard_normal(goal_dim) + 10.0).astype(np.float32)
    achieved_goal = rng.standard_normal(goal_dim).astype(np.float32)
    if np.linalg.norm(achieved_goal - desired_goal) < distance_threshold:
        desired_goal = desired_goal + 10.0

    for step_idx in range(n_steps):
        obs = {
            "observation": rng.standard_normal(obs_dim).astype(np.float32),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": desired_goal.copy(),
        }
        next_obs = {
            "observation": rng.standard_normal(obs_dim).astype(np.float32),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": desired_goal.copy(),
        }
        action = rng.uniform(low=-1.0, high=1.0, size=(act_dim,)).astype(np.float32)
        reward = env.compute_reward(
            next_obs["achieved_goal"],
            obs["desired_goal"],
            {"distance_threshold": distance_threshold},
        )
        done = bool(step_idx == n_steps - 1)

        buf.add(
            {k: v[None, :] for k, v in obs.items()},
            {k: v[None, :] for k, v in next_obs.items()},
            action[None, :],
            np.asarray([reward], dtype=np.float32),
            np.asarray([done], dtype=bool),
            infos=[{}],
        )

    samples = buf.sample(batch_size)

    if isinstance(samples.observations, dict) and isinstance(samples.next_observations, dict):
        achieved = samples.next_observations["achieved_goal"].cpu().numpy()
        desired = samples.observations["desired_goal"].cpu().numpy()
        sampled_rewards = samples.rewards.cpu().numpy().reshape(-1)
    else:  # pragma: no cover
        raise RuntimeError("Unexpected SB3 sample format: expected dict observations for goal environments.")

    recomputed = env.compute_reward(
        achieved,
        desired,
        {"distance_threshold": distance_threshold},
    ).reshape(-1)

    max_abs_reward_diff = float(np.max(np.abs(sampled_rewards - recomputed)))
    success_fraction = float(np.mean(sampled_rewards >= 0.0))

    passed = (max_abs_reward_diff == 0.0) and (success_fraction > 0.0)
    return ComparisonResult(
        name="her_relabeling_vs_sb3",
        passed=passed,
        metrics={
            "max_abs_reward_diff": max_abs_reward_diff,
            "success_fraction": success_fraction,
            "batch_size": float(batch_size),
            "n_sampled_goal": float(n_sampled_goal),
        },
        notes=None
        if passed
        else "Mismatch suggests reward recomputation inconsistency or HER relabeling not applied during sampling.",
    )


def compare_nature_cnn_to_sb3(
    *,
    seed: int = 0,
    batch_size: int = 8,
    features_dim: int = 512,
    atol: float = 1e-5,
) -> ComparisonResult:
    """Compare our NatureCNN output against SB3's NatureCnn.

    Creates both implementations, copies weights from SB3 to ours,
    feeds identical random input, and checks output difference.
    """
    try:
        from gymnasium import spaces
        from stable_baselines3.common.torch_layers import NatureCNN as SB3NatureCNN
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SB3 comparison requires gymnasium + stable-baselines3. "
            "Run inside Docker: `bash docker/dev.sh python scripts/labs/visual_encoder.py --compare-sb3`."
        ) from exc

    from scripts.labs.visual_encoder import NatureCNN

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create SB3 NatureCNN (needs a Box observation space)
    obs_space = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
    sb3_cnn = SB3NatureCNN(obs_space, features_dim=features_dim)
    sb3_cnn.eval()

    # Create our NatureCNN
    ours = NatureCNN(in_channels=3, features_dim=features_dim)
    ours.eval()

    # Copy weights from SB3 to ours
    # SB3 layout: self.cnn = Sequential(Conv, ReLU, Conv, ReLU, Conv, ReLU, Flatten)
    #             self.linear = Sequential(Linear, ReLU)
    # Ours:       self.conv = Sequential(Conv, ReLU, Conv, ReLU, Conv, ReLU, Flatten)
    #             self.fc   = Sequential(Linear, ReLU)
    with torch.no_grad():
        # Conv layers (indices 0, 2, 4 in both Sequentials)
        for idx in [0, 2, 4]:
            ours.conv[idx].weight.copy_(sb3_cnn.cnn[idx].weight)
            ours.conv[idx].bias.copy_(sb3_cnn.cnn[idx].bias)
        # FC layer (index 0 in both Sequentials)
        ours.fc[0].weight.copy_(sb3_cnn.linear[0].weight)
        ours.fc[0].bias.copy_(sb3_cnn.linear[0].bias)

    # Feed identical input (normalized float32, as SB3 does internally)
    dummy_input = torch.rand(batch_size, 3, 84, 84)

    with torch.no_grad():
        sb3_out = sb3_cnn(dummy_input)
        our_out = ours(dummy_input)

    max_abs_diff = float((sb3_out - our_out).abs().max().item())
    passed = max_abs_diff <= atol

    return ComparisonResult(
        name="nature_cnn_vs_sb3",
        passed=passed,
        metrics={
            "max_abs_output_diff": max_abs_diff,
            "atol": float(atol),
            "sb3_output_mean": float(sb3_out.mean().item()),
            "our_output_mean": float(our_out.mean().item()),
        },
        notes=None if passed else "Mismatch suggests a weight copy error or architectural difference.",
    )
