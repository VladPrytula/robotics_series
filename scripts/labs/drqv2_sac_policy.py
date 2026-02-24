#!/usr/bin/env python3
"""
DrQ-v2-Style SAC Policy -- Critic Updates Encoder, Actor Gets Detached Features

SB3's default gradient routing for shared encoders is backwards from DrQ-v2:

    SB3 share_features_extractor=True:
        - Encoder in ACTOR optimizer only
        - Critic forward uses set_grad_enabled(False) on encoder
        - Actor's noisy policy gradient is the only signal for representation learning

    DrQ-v2 (Yarats et al. 2021):
        - One shared encoder
        - Encoder in CRITIC optimizer (rich TD signal)
        - Actor receives obs.detach() (no encoder gradient from policy loss)

This module implements the DrQ-v2 pattern as an SB3-compatible custom policy.
The key insight: the critic's TD loss provides a stable, information-rich gradient
for the encoder, while the actor's policy gradient is noisy (especially early
in training when success rate is near zero).

Classes:
    CriticEncoderCritic -- ContinuousCritic with gradients ENABLED through encoder
    CriticEncoderActor  -- Actor with features.detach() (stop-gradient for encoder)
    DrQv2SACPolicy      -- SACPolicy with encoder in critic optimizer

Usage:
    # Verify gradient flow, optimizer params, and short training
    python scripts/labs/drqv2_sac_policy.py --verify

References:
    - Yarats et al. (2021), "Mastering Visual Continuous Control: Improved
      Data-Augmented Reinforcement Learning", arXiv:2107.09645
    - SB3 SAC: stable_baselines3/sac/policies.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Union

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy, LOG_STD_MAX, LOG_STD_MIN

# Ensure project root is on sys.path for cross-module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# CriticEncoderCritic: enable gradients through shared encoder
# =============================================================================

# --8<-- [start:critic_encoder_critic]
class CriticEncoderCritic(ContinuousCritic):
    """ContinuousCritic that always enables gradients through the features extractor.

    SB3's default ContinuousCritic gates encoder gradients with:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)

    When share_features_extractor=True, this DISABLES gradients -- the encoder
    gets no signal from the critic's TD loss. We override forward() to always
    enable gradients, so the critic's optimizer can update the shared encoder.
    """

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        # CHANGED: Always enable gradients through features_extractor.
        # The critic optimizer includes encoder params, so TD loss updates the encoder.
        features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)
# --8<-- [end:critic_encoder_critic]


# =============================================================================
# CriticEncoderActor: detach features (stop-gradient for encoder)
# =============================================================================

# --8<-- [start:critic_encoder_actor]
class CriticEncoderActor(Actor):
    """Actor that detaches features before passing to the policy MLP.

    In DrQ-v2, the actor receives obs.detach() so the policy gradient does
    not flow back into the shared encoder. This prevents noisy policy gradients
    (especially at low success rates) from corrupting the encoder's
    representation, which is being learned by the more stable critic TD loss.
    """

    def get_action_dist_params(
        self, obs: PyTorchObs
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        # CHANGED: Detach features so actor loss does not update the encoder.
        features = features.detach()
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}
# --8<-- [end:critic_encoder_actor]


# =============================================================================
# DrQv2SACPolicy: shared encoder in critic optimizer
# =============================================================================

# --8<-- [start:drqv2_sac_policy]
class DrQv2SACPolicy(SACPolicy):
    """SAC policy with DrQ-v2-style gradient routing.

    Architecture:
        - One shared encoder (actor and critic share the same features_extractor)
        - Encoder parameters in CRITIC optimizer (updated by TD loss)
        - Encoder parameters excluded from ACTOR optimizer
        - Actor receives detached features (via CriticEncoderActor)
        - Critic enables gradients through encoder (via CriticEncoderCritic)

    This matches the gradient flow in facebookresearch/drqv2:
        update_critic(): encoder_opt.step()  (encoder updated by critic loss)
        update_actor():  obs = obs.detach()  (no encoder gradient from actor)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        # Force share_features_extractor=True -- the whole point of this policy
        # is a shared encoder with critic-driven gradient routing.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor=True,  # always shared
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CriticEncoderActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CriticEncoderActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CriticEncoderCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CriticEncoderCritic(**critic_kwargs).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        # 1. Create actor (with its own features_extractor initially)
        self.actor = self.make_actor()

        # 2. Create critic that SHARES the actor's features_extractor
        self.critic = self.make_critic(features_extractor=self.actor.features_extractor)

        # 3. REVERSED from SB3 default:
        #    - Encoder params in CRITIC optimizer (rich TD signal)
        #    - Encoder params EXCLUDED from actor optimizer (detached in forward anyway)
        # Use identity-based filtering (by id()) rather than name-based to avoid
        # fragility if parameter naming conventions change across SB3 versions.
        encoder_param_ids = {id(p) for p in self.actor.features_extractor.parameters()}
        actor_parameters = [
            p for p in self.actor.parameters()
            if id(p) not in encoder_param_ids
        ]
        critic_parameters = list(self.critic.parameters())  # includes shared encoder

        self.actor.optimizer = self.optimizer_class(
            actor_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # 4. Target critic gets its own encoder copy (not shared)
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)
# --8<-- [end:drqv2_sac_policy]


# =============================================================================
# Verification Functions
# =============================================================================

def verify_optimizer_params():
    """Confirm encoder params are in critic optimizer and NOT in actor optimizer."""
    import numpy as np
    from scripts.labs.manipulation_encoder import ManipulationExtractor

    print("Verifying optimizer parameter assignment...")

    obs_space = spaces.Dict({
        "pixels": spaces.Box(0, 255, (12, 84, 84), dtype=np.uint8),
        "proprioception": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float64),
        "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
        "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
    })
    act_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    policy = DrQv2SACPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 1e-4,
        features_extractor_class=ManipulationExtractor,
        features_extractor_kwargs={"spatial_softmax": True},
    )

    # Collect param ids in each optimizer
    actor_opt_ids = set()
    for pg in policy.actor.optimizer.param_groups:
        for p in pg["params"]:
            actor_opt_ids.add(id(p))

    critic_opt_ids = set()
    for pg in policy.critic.optimizer.param_groups:
        for p in pg["params"]:
            critic_opt_ids.add(id(p))

    # Check encoder params
    encoder_param_ids = set()
    encoder_param_names = []
    for n, p in policy.actor.named_parameters():
        if "features_extractor" in n:
            encoder_param_ids.add(id(p))
            encoder_param_names.append(n)

    enc_in_actor = encoder_param_ids & actor_opt_ids
    enc_in_critic = encoder_param_ids & critic_opt_ids

    assert len(enc_in_actor) == 0, (
        f"Encoder params should NOT be in actor optimizer, found {len(enc_in_actor)}"
    )
    assert len(enc_in_critic) == len(encoder_param_ids), (
        f"All encoder params should be in critic optimizer, "
        f"found {len(enc_in_critic)}/{len(encoder_param_ids)}"
    )

    # Verify shared encoder (same object)
    assert policy.actor.features_extractor is policy.critic.features_extractor, (
        "Actor and critic should share the same features_extractor instance"
    )

    # Verify target critic has its own encoder (different object)
    assert policy.critic_target.features_extractor is not policy.critic.features_extractor, (
        "Target critic should have its own features_extractor (not shared)"
    )

    print(f"  Encoder param groups: {len(encoder_param_names)}")
    print(f"  Encoder in actor optimizer:  {len(enc_in_actor)} (expected: 0)")
    print(f"  Encoder in critic optimizer: {len(enc_in_critic)} (expected: {len(encoder_param_ids)})")
    print(f"  Shared encoder: {policy.actor.features_extractor is policy.critic.features_extractor}")
    print(f"  Target has own encoder: {policy.critic_target.features_extractor is not policy.critic.features_extractor}")
    print("  [PASS] Optimizer params OK")


def verify_gradient_flow():
    """Verify encoder gets grad from critic but NOT from actor."""
    import numpy as np
    from scripts.labs.manipulation_encoder import ManipulationExtractor

    print("Verifying gradient flow...")

    obs_space = spaces.Dict({
        "pixels": spaces.Box(0, 255, (12, 84, 84), dtype=np.uint8),
        "proprioception": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float64),
        "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
        "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
    })
    act_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    policy = DrQv2SACPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 1e-4,
        features_extractor_class=ManipulationExtractor,
        features_extractor_kwargs={"spatial_softmax": True},
    )
    policy.set_training_mode(True)

    # Create mock batch
    batch_size = 4
    mock_obs = {
        "pixels": th.randint(0, 256, (batch_size, 12, 84, 84), dtype=th.uint8),
        "proprioception": th.randn(batch_size, 10),
        "achieved_goal": th.randn(batch_size, 3),
        "desired_goal": th.randn(batch_size, 3),
    }
    mock_actions = th.randn(batch_size, 4)

    # --- Step 1: Critic forward + backward ---
    # Snapshot encoder params before
    enc_params_before = {
        n: p.clone().detach()
        for n, p in policy.critic.named_parameters()
        if "features_extractor" in n
    }

    policy.critic.optimizer.zero_grad()
    q_values = policy.critic(mock_obs, mock_actions)
    critic_loss = sum(q.mean() for q in q_values)
    critic_loss.backward()

    # Check encoder has gradient from critic
    enc_has_grad = {}
    for n, p in policy.critic.named_parameters():
        if "features_extractor" in n:
            enc_has_grad[n] = p.grad is not None and p.grad.abs().sum() > 0

    assert all(enc_has_grad.values()), (
        f"Encoder should have gradients after critic backward. "
        f"Params with grad: {sum(enc_has_grad.values())}/{len(enc_has_grad)}"
    )

    policy.critic.optimizer.step()

    # Check encoder params changed
    enc_changed_critic = {}
    for n, p in policy.critic.named_parameters():
        if "features_extractor" in n:
            enc_changed_critic[n] = not th.allclose(p, enc_params_before[n])

    assert all(enc_changed_critic.values()), (
        "Encoder params should change after critic optimizer step"
    )

    print(f"  After critic step: encoder has grad = True (all {len(enc_has_grad)} params)")
    print(f"  After critic step: encoder params changed = True")

    # --- Step 2: Actor forward + backward ---
    # Snapshot encoder params
    enc_params_before_actor = {
        n: p.clone().detach()
        for n, p in policy.actor.named_parameters()
        if "features_extractor" in n
    }

    policy.actor.optimizer.zero_grad()
    # Also zero encoder grads (they're not in actor optimizer but could accumulate)
    for n, p in policy.actor.named_parameters():
        if "features_extractor" in n and p.grad is not None:
            p.grad.zero_()

    actions_pi, log_prob = policy.actor.action_log_prob(mock_obs)
    actor_loss = -log_prob.mean()
    actor_loss.backward()

    # Check encoder does NOT have gradient from actor (features were detached)
    enc_has_grad_actor = {}
    for n, p in policy.actor.named_parameters():
        if "features_extractor" in n:
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            enc_has_grad_actor[n] = has_grad

    assert not any(enc_has_grad_actor.values()), (
        f"Encoder should NOT have gradients after actor backward (features detached). "
        f"Params with grad: {sum(enc_has_grad_actor.values())}/{len(enc_has_grad_actor)}"
    )

    policy.actor.optimizer.step()

    # Check encoder params did NOT change
    enc_changed_actor = {}
    for n, p in policy.actor.named_parameters():
        if "features_extractor" in n:
            enc_changed_actor[n] = not th.allclose(p, enc_params_before_actor[n])

    assert not any(enc_changed_actor.values()), (
        "Encoder params should NOT change after actor optimizer step"
    )

    print(f"  After actor step: encoder has grad = False (all {len(enc_has_grad_actor)} params)")
    print(f"  After actor step: encoder params changed = False")

    # --- Step 3: Actor loss through critic (SB3's real pattern) ---
    # SB3 computes actor_loss = -Q(s, a_pi) + alpha * log_pi. The critic forward
    # runs WITH gradients (our CriticEncoderCritic always enables them), so encoder
    # grads WILL be computed during the actor step. But since encoder is NOT in
    # actor.optimizer, those grads are never stepped. Verify: after actor step,
    # encoder params are unchanged, and critic.optimizer.zero_grad() clears them.
    enc_params_before_step3 = {
        n: p.clone().detach()
        for n, p in policy.critic.named_parameters()
        if "features_extractor" in n
    }

    policy.actor.optimizer.zero_grad()
    policy.critic.optimizer.zero_grad()

    # Simulate SB3's actor loss: get actions, evaluate Q
    actions_pi, log_prob = policy.actor.action_log_prob(mock_obs)
    q_values_pi = policy.critic(mock_obs, actions_pi)
    min_q = th.min(q_values_pi[0], q_values_pi[1])
    actor_loss_full = (0.05 * log_prob - min_q).mean()
    actor_loss_full.backward()

    # Encoder may have grads from critic(obs, a_pi) -- that's expected (wasted compute).
    # But actor.optimizer.step() must NOT change encoder params.
    policy.actor.optimizer.step()

    enc_changed_step3 = {}
    for n, p in policy.critic.named_parameters():
        if "features_extractor" in n:
            enc_changed_step3[n] = not th.allclose(p, enc_params_before_step3[n])

    assert not any(enc_changed_step3.values()), (
        "Encoder params should NOT change from actor.optimizer.step() "
        "even when critic forward computes encoder grads during actor loss"
    )

    # Confirm critic.optimizer.zero_grad() clears the stale encoder grads
    policy.critic.optimizer.zero_grad()
    enc_has_stale_grad = 0
    for n, p in policy.critic.named_parameters():
        if "features_extractor" in n:
            if p.grad is not None and p.grad.abs().sum() > 0:
                enc_has_stale_grad += 1

    assert enc_has_stale_grad == 0, (
        f"critic.optimizer.zero_grad() should clear encoder grads, "
        f"but {enc_has_stale_grad} params still have grads"
    )

    print(f"  Actor-through-critic: encoder unchanged after actor.optimizer.step() = True")
    print(f"  critic.optimizer.zero_grad() clears stale encoder grads = True")
    print("  [PASS] Gradient flow OK")


def verify_target_critic():
    """Verify target critic has its own encoder and soft-updates work."""
    import numpy as np
    from stable_baselines3.common.utils import polyak_update
    from scripts.labs.manipulation_encoder import ManipulationExtractor

    print("Verifying target critic...")

    obs_space = spaces.Dict({
        "pixels": spaces.Box(0, 255, (12, 84, 84), dtype=np.uint8),
        "proprioception": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float64),
        "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
        "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
    })
    act_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    policy = DrQv2SACPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 1e-4,
        features_extractor_class=ManipulationExtractor,
        features_extractor_kwargs={"spatial_softmax": True},
    )

    # Verify initial state dict match
    critic_sd = policy.critic.state_dict()
    target_sd = policy.critic_target.state_dict()
    for key in critic_sd:
        assert th.allclose(critic_sd[key], target_sd[key]), (
            f"Initial mismatch at {key}"
        )

    # Perturb critic params
    with th.no_grad():
        for p in policy.critic.parameters():
            p.add_(th.randn_like(p) * 0.1)

    # Snapshot target encoder params before soft update
    target_enc_before = {
        n: p.clone().detach()
        for n, p in policy.critic_target.named_parameters()
        if "features_extractor" in n
    }

    # Soft update
    tau = 0.005
    polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)

    # Target encoder should have changed
    target_enc_changed = {}
    for n, p in policy.critic_target.named_parameters():
        if "features_extractor" in n:
            target_enc_changed[n] = not th.allclose(p, target_enc_before[n])

    assert all(target_enc_changed.values()), (
        "Target encoder should change after polyak_update"
    )

    print(f"  Target encoder params changed after soft update: True")
    print(f"  Target has separate encoder instance: True")
    print("  [PASS] Target critic OK")


def verify_sb3_training():
    """Verify 50-step training with DrQv2SACPolicy (no crash)."""
    import numpy as np
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from scripts.labs.manipulation_encoder import ManipulationExtractor
    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    print("Verifying SB3 training integration (50 steps)...")

    push_proprio = [0, 1, 2, 9, 10, 20, 21, 22, 23, 24]
    env = gym.make("FetchPush-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(
        env, image_size=(84, 84), goal_mode="both",
        frame_stack=4, proprio_indices=push_proprio,
    )

    model = SAC(
        DrQv2SACPolicy,
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

    # Verify the policy type
    assert isinstance(model.policy, DrQv2SACPolicy), (
        f"Expected DrQv2SACPolicy, got {type(model.policy).__name__}"
    )
    assert isinstance(model.policy.actor, CriticEncoderActor), (
        f"Expected CriticEncoderActor, got {type(model.policy.actor).__name__}"
    )
    assert isinstance(model.policy.critic, CriticEncoderCritic), (
        f"Expected CriticEncoderCritic, got {type(model.policy.critic).__name__}"
    )

    model.learn(total_timesteps=50)

    # Check encoder grad norms after training
    enc_grad_norms = {}
    for n, p in model.policy.critic.named_parameters():
        if "features_extractor" in n and p.grad is not None:
            enc_grad_norms[n] = p.grad.norm().item()

    env.close()

    print(f"  Policy type: {type(model.policy).__name__}")
    print(f"  Actor type: {type(model.policy.actor).__name__}")
    print(f"  Critic type: {type(model.policy.critic).__name__}")
    if enc_grad_norms:
        max_norm = max(enc_grad_norms.values())
        print(f"  Encoder grad norms: {len(enc_grad_norms)} params, max={max_norm:.6f}")
    else:
        print("  Encoder grad norms: (no grads retained after training -- expected)")
    print("  50-step training completed without crashes")
    print("  [PASS] SB3 training OK")


def verify_save_load():
    """Verify model save/load with DrQv2SACPolicy."""
    import tempfile
    import numpy as np
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC
    from scripts.labs.manipulation_encoder import ManipulationExtractor
    from scripts.labs.pixel_wrapper import PixelObservationWrapper

    print("Verifying save/load...")

    push_proprio = [0, 1, 2, 9, 10, 20, 21, 22, 23, 24]
    env = gym.make("FetchPush-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(
        env, image_size=(84, 84), goal_mode="both",
        frame_stack=4, proprio_indices=push_proprio,
    )

    model = SAC(
        DrQv2SACPolicy,
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
    model.learn(total_timesteps=50)

    # Get prediction before save
    obs, _ = env.reset()
    obs_th = {k: th.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
    with th.no_grad():
        pred_before = model.policy.actor(obs_th, deterministic=True)

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.zip"
        model.save(str(save_path))

        loaded = SAC.load(
            str(save_path),
            env=env,
            device="cpu",
            custom_objects={
                "policy_class": DrQv2SACPolicy,
            },
        )

        # Verify loaded model type
        assert isinstance(loaded.policy, DrQv2SACPolicy), (
            f"Loaded policy type: {type(loaded.policy).__name__}"
        )

        # Verify predictions match
        with th.no_grad():
            pred_after = loaded.policy.actor(obs_th, deterministic=True)

        assert th.allclose(pred_before, pred_after, atol=1e-5), (
            "Predictions changed after save/load"
        )

    env.close()
    print("  Save/load round-trip: predictions match")
    print(f"  Loaded policy type: {type(loaded.policy).__name__}")
    print("  [PASS] Save/load OK")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("DrQ-v2 SAC Policy -- Verification")
    print("=" * 60)

    verify_optimizer_params()
    print()
    verify_gradient_flow()
    print()
    verify_target_critic()
    print()
    verify_sb3_training()
    print()
    verify_save_load()

    print()
    print("=" * 60)
    print("[ALL PASS] DrQ-v2 SAC policy verified")
    print("=" * 60)


# =============================================================================
# SB3 Gradient Probe (Step 0a)
# =============================================================================

def probe_sb3_gradients():
    """Empirically confirm SB3's default gradient routing.

    This probe creates SAC with share_features_extractor=True and False,
    then manually runs critic and actor steps to verify which optimizer
    updates the encoder. This de-risks the DrQv2SACPolicy implementation
    by confirming the SB3 behavior we're trying to fix.
    """
    import numpy as np
    from scripts.labs.manipulation_encoder import ManipulationExtractor

    print("=" * 60)
    print("SB3 Gradient Probe -- Confirming Default Behavior")
    print("=" * 60)

    obs_space = spaces.Dict({
        "pixels": spaces.Box(0, 255, (12, 84, 84), dtype=np.uint8),
        "proprioception": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float64),
        "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
        "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
    })
    act_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    batch_size = 4

    mock_obs = {
        "pixels": th.randint(0, 256, (batch_size, 12, 84, 84), dtype=th.uint8),
        "proprioception": th.randn(batch_size, 10),
        "achieved_goal": th.randn(batch_size, 3),
        "desired_goal": th.randn(batch_size, 3),
    }
    mock_actions = th.randn(batch_size, 4)

    for shared in [True, False]:
        print(f"\n  --- share_features_extractor={shared} ---")

        policy = SACPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 1e-4,
            features_extractor_class=ManipulationExtractor,
            features_extractor_kwargs={"spatial_softmax": True},
            share_features_extractor=shared,
        )
        policy.set_training_mode(True)

        # Check which optimizer has encoder params
        actor_opt_ids = set()
        for pg in policy.actor.optimizer.param_groups:
            for p in pg["params"]:
                actor_opt_ids.add(id(p))

        critic_opt_ids = set()
        for pg in policy.critic.optimizer.param_groups:
            for p in pg["params"]:
                critic_opt_ids.add(id(p))

        # Actor's encoder params
        actor_enc_ids = set()
        for n, p in policy.actor.named_parameters():
            if "features_extractor" in n:
                actor_enc_ids.add(id(p))

        enc_in_actor = len(actor_enc_ids & actor_opt_ids)
        enc_in_critic = len(actor_enc_ids & critic_opt_ids)
        total_enc = len(actor_enc_ids)

        print(f"  Actor encoder params in actor optimizer:  {enc_in_actor}/{total_enc}")
        print(f"  Actor encoder params in critic optimizer: {enc_in_critic}/{total_enc}")

        if shared:
            # Critic step: expect NO encoder gradient (set_grad_enabled(False))
            policy.critic.optimizer.zero_grad()
            q_values = policy.critic(mock_obs, mock_actions)
            critic_loss = sum(q.mean() for q in q_values)
            critic_loss.backward()

            enc_has_grad_critic = 0
            for n, p in policy.critic.named_parameters():
                if "features_extractor" in n:
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        enc_has_grad_critic += 1

            print(f"  After critic backward: encoder params with grad = {enc_has_grad_critic}/{total_enc}")

            # Actor step: expect encoder gradient (no gate)
            policy.actor.optimizer.zero_grad()
            for p in policy.actor.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            actions_pi, log_prob = policy.actor.action_log_prob(mock_obs)
            actor_loss = -log_prob.mean()
            actor_loss.backward()

            enc_has_grad_actor = 0
            for n, p in policy.actor.named_parameters():
                if "features_extractor" in n:
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        enc_has_grad_actor += 1

            print(f"  After actor backward:  encoder params with grad = {enc_has_grad_actor}/{total_enc}")

    print()
    print("  Summary:")
    print("    shared=True:  encoder in actor opt only, critic disables gradients")
    print("    shared=False: separate encoders, each gets its own loss")
    print("    -> Confirms: SB3's default shared mode is BACKWARDS from DrQ-v2")
    print()
    print("=" * 60)
    print("[DONE] SB3 gradient probe complete")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DrQ-v2-Style SAC Policy (critic updates encoder)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --verify          Run all verification checks (~60 seconds)
  --probe           Run SB3 gradient probe (confirm default behavior)
        """
    )
    parser.add_argument("--verify", action="store_true",
                        help="Run verification checks")
    parser.add_argument("--probe", action="store_true",
                        help="Run SB3 gradient probe (Step 0a)")
    args = parser.parse_args()

    if args.verify:
        run_verification()
    elif args.probe:
        probe_sb3_gradients()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
