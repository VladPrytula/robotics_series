"""SAC-specific diagnostics callback for replay buffer analysis.

Logs to TensorBoard:
- Q-value statistics (mean, std, min, max) from sampled transitions
- Entropy coefficient (alpha) over training
- Reward distribution from replay buffer
- Achieved vs desired goal distance distribution (for goal-conditioned envs)

These diagnostics help identify:
- Value overestimation (Q-values growing unbounded -> divergence)
- Entropy collapse (alpha -> 0 means exploration stopped)
- Reward sparsity (histogram reveals learning signal quality)
- Goal-reaching progress (distance distribution should shift left over training)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from stable_baselines3 import SAC


class SACDiagnosticsCallback(BaseCallback):
    """Callback for logging SAC-specific replay buffer diagnostics.

    Args:
        log_freq: How often (in timesteps) to log diagnostics.
        n_samples: Number of transitions to sample from replay buffer for statistics.
        verbose: Verbosity level (0=silent, 1=info).
    """

    def __init__(self, log_freq: int = 10_000, n_samples: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.n_samples = n_samples
        self._last_log_step = 0

    def _on_step(self) -> bool:
        # Only log at specified frequency and after enough samples exist
        if self.num_timesteps - self._last_log_step < self.log_freq:
            return True

        model: SAC = self.model
        replay_buffer = model.replay_buffer

        # Need enough samples in buffer
        if replay_buffer.size() < self.n_samples:
            return True

        self._last_log_step = self.num_timesteps
        self._log_q_value_stats(model, replay_buffer)
        self._log_entropy_coefficient(model)
        self._log_reward_stats(replay_buffer)
        self._log_goal_distance_stats(replay_buffer)

        return True

    def _log_q_value_stats(self, model: SAC, replay_buffer) -> None:
        """Sample transitions and compute Q-value statistics."""
        import torch

        # Sample a batch from replay buffer
        replay_data = replay_buffer.sample(self.n_samples, env=model._vec_normalize_env)

        # Get Q-values from both critics
        with torch.no_grad():
            obs = replay_data.observations
            actions = replay_data.actions

            # Evaluate Q-values using both critics
            q1, q2 = model.critic(obs, actions)
            q_min = torch.min(q1, q2)

            q1_np = q1.cpu().numpy().flatten()
            q2_np = q2.cpu().numpy().flatten()
            q_min_np = q_min.cpu().numpy().flatten()

        # Log Q-value statistics
        self.logger.record("replay/q1_mean", float(np.mean(q1_np)))
        self.logger.record("replay/q1_std", float(np.std(q1_np)))
        self.logger.record("replay/q2_mean", float(np.mean(q2_np)))
        self.logger.record("replay/q2_std", float(np.std(q2_np)))
        self.logger.record("replay/q_min_mean", float(np.mean(q_min_np)))
        self.logger.record("replay/q_min_std", float(np.std(q_min_np)))
        self.logger.record("replay/q_min_max", float(np.max(q_min_np)))
        self.logger.record("replay/q_min_min", float(np.min(q_min_np)))

        if self.verbose > 0:
            print(f"[SACDiag] Q-values: mean={np.mean(q_min_np):.2f}, std={np.std(q_min_np):.2f}")

    def _log_entropy_coefficient(self, model: SAC) -> None:
        """Log the current entropy coefficient (alpha)."""
        import torch

        ent_coef = None

        # SB3 SAC stores log_ent_coef when using auto-tuning
        if hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
            with torch.no_grad():
                if isinstance(model.log_ent_coef, torch.Tensor):
                    ent_coef = model.log_ent_coef.exp().item()
                else:
                    ent_coef = float(model.log_ent_coef)
        elif hasattr(model, "ent_coef") and not isinstance(model.ent_coef, str):
            ent_coef = float(model.ent_coef)

        if ent_coef is not None:
            self.logger.record("replay/ent_coef", ent_coef)
            if self.verbose > 0:
                print(f"[SACDiag] Entropy coef: {ent_coef:.4f}")

    def _log_reward_stats(self, replay_buffer) -> None:
        """Log reward distribution statistics from replay buffer."""
        # Sample rewards from buffer
        n_samples = min(self.n_samples, replay_buffer.size())

        # Access raw buffer data (position-aware for circular buffer)
        if hasattr(replay_buffer, "rewards"):
            # Standard replay buffer
            pos = replay_buffer.pos
            full = replay_buffer.full
            if full:
                rewards = replay_buffer.rewards.flatten()
            else:
                rewards = replay_buffer.rewards[:pos].flatten()
        else:
            # HER replay buffer or other structure - sample instead
            try:
                replay_data = replay_buffer.sample(n_samples, env=None)
                rewards = replay_data.rewards.cpu().numpy().flatten()
            except Exception:
                return

        if len(rewards) == 0:
            return

        # Compute statistics
        self.logger.record("replay/reward_mean", float(np.mean(rewards)))
        self.logger.record("replay/reward_std", float(np.std(rewards)))
        self.logger.record("replay/reward_min", float(np.min(rewards)))
        self.logger.record("replay/reward_max", float(np.max(rewards)))

        # Histogram bins for sparse rewards (typically -1 or 0)
        # Count proportion of "successful" transitions (reward > -0.5 for sparse)
        success_frac = float(np.mean(rewards > -0.5))
        self.logger.record("replay/reward_positive_frac", success_frac)

        if self.verbose > 0:
            print(f"[SACDiag] Rewards: mean={np.mean(rewards):.3f}, positive_frac={success_frac:.3f}")

    def _log_goal_distance_stats(self, replay_buffer) -> None:
        """Log achieved vs desired goal distance for goal-conditioned envs."""
        # Check if this is a goal-conditioned buffer (has achieved_goal, desired_goal)
        if not hasattr(replay_buffer, "observations"):
            return

        n_samples = min(self.n_samples, replay_buffer.size())

        try:
            # Try to access goal information
            # For DictReplayBuffer, observations is a dict
            obs = replay_buffer.observations

            if isinstance(obs, dict):
                achieved = obs.get("achieved_goal")
                desired = obs.get("desired_goal")
            elif hasattr(replay_buffer, "achieved_goal") and hasattr(replay_buffer, "desired_goal"):
                achieved = replay_buffer.achieved_goal
                desired = replay_buffer.desired_goal
            else:
                return

            if achieved is None or desired is None:
                return

            # Get valid data (accounting for circular buffer)
            pos = replay_buffer.pos
            full = replay_buffer.full

            if full:
                achieved_np = achieved[:n_samples]
                desired_np = desired[:n_samples]
            else:
                n_valid = min(pos, n_samples)
                achieved_np = achieved[:n_valid]
                desired_np = desired[:n_valid]

            if len(achieved_np) == 0:
                return

            # Compute Euclidean distances
            # Handle potential extra dimensions (n_envs, horizon, etc.)
            achieved_flat = achieved_np.reshape(-1, achieved_np.shape[-1])
            desired_flat = desired_np.reshape(-1, desired_np.shape[-1])

            distances = np.linalg.norm(achieved_flat - desired_flat, axis=-1)

            self.logger.record("replay/goal_distance_mean", float(np.mean(distances)))
            self.logger.record("replay/goal_distance_std", float(np.std(distances)))
            self.logger.record("replay/goal_distance_min", float(np.min(distances)))
            self.logger.record("replay/goal_distance_max", float(np.max(distances)))

            # Fraction within typical success threshold (0.05 for Fetch)
            success_threshold = 0.05
            within_threshold = float(np.mean(distances < success_threshold))
            self.logger.record("replay/goal_within_threshold", within_threshold)

            if self.verbose > 0:
                print(f"[SACDiag] Goal distance: mean={np.mean(distances):.4f}, within_thresh={within_threshold:.3f}")

        except Exception as e:
            if self.verbose > 0:
                print(f"[SACDiag] Could not compute goal distances: {e}")
