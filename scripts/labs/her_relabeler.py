#!/usr/bin/env python3
"""
HER Relabeler -- Pedagogical Implementation

This module implements Hindsight Experience Replay (HER) goal relabeling
with explicit operations to show how the math maps to code.

HER is a technique that retroactively relabels failed episodes by substituting
the originally desired goal with a goal that was actually achieved. This turns
failures into successes for learning.

Usage:
    # Run verification (sanity checks, ~1 minute)
    python scripts/labs/her_relabeler.py --verify

    # Demonstrate relabeling on synthetic data
    python scripts/labs/her_relabeler.py --demo

Key regions exported for tutorials:
    - goal_sampling: Different HER goal sampling strategies
    - relabel_transition: How to relabel a single transition
    - her_buffer_insert: Inserting relabeled transitions into replay
"""
from __future__ import annotations

import argparse
from enum import Enum
from typing import NamedTuple

import numpy as np


# =============================================================================
# Data Structures
# =============================================================================

class Transition(NamedTuple):
    """A single environment transition in a goal-conditioned setting."""
    obs: np.ndarray           # observation (e.g., robot state)
    action: np.ndarray        # action taken
    reward: float             # reward received
    next_obs: np.ndarray      # next observation
    done: bool                # episode terminated?
    achieved_goal: np.ndarray # goal actually achieved at next_obs
    desired_goal: np.ndarray  # goal the agent was trying to reach


class Episode(NamedTuple):
    """A complete episode of transitions."""
    transitions: list[Transition]

    def __len__(self) -> int:
        return len(self.transitions)


# =============================================================================
# Goal Sampling Strategies
# =============================================================================

class GoalStrategy(Enum):
    """HER goal sampling strategies from Andrychowicz et al. (2017)."""
    FINAL = "final"      # Use final achieved goal
    FUTURE = "future"    # Sample from future timesteps
    EPISODE = "episode"  # Sample from anywhere in episode
    RANDOM = "random"    # Sample from replay buffer (requires buffer access)


# --8<-- [start:goal_sampling]
def sample_her_goals(
    episode: Episode,
    transition_idx: int,
    strategy: GoalStrategy = GoalStrategy.FUTURE,
    k: int = 4,  # number of goals to sample
) -> list[np.ndarray]:
    """
    Sample alternative goals for HER relabeling.

    The key insight of HER: even if we failed to reach the desired goal,
    we DID reach some goal (the achieved_goal). By relabeling the transition
    with achieved_goal as the desired_goal, we create a "success" example.

    Different strategies determine where to sample alternative goals from:
    - FINAL: Always use the final achieved goal of the episode
    - FUTURE: Sample from achieved goals at timesteps > current
    - EPISODE: Sample from any achieved goal in the episode

    Args:
        episode: The episode containing this transition
        transition_idx: Index of the current transition
        strategy: Which HER strategy to use
        k: Number of alternative goals to sample

    Returns:
        List of k alternative goals (each is an np.ndarray)
    """
    n = len(episode)
    goals = []

    if strategy == GoalStrategy.FINAL:
        # Always use the final achieved goal
        final_goal = episode.transitions[-1].achieved_goal
        goals = [final_goal.copy() for _ in range(k)]

    elif strategy == GoalStrategy.FUTURE:
        # Sample from future timesteps (most common strategy)
        # This ensures temporal consistency: we only relabel with
        # goals that could have been reached from the current state
        future_indices = list(range(transition_idx + 1, n))

        if len(future_indices) == 0:
            # At last timestep, use own achieved goal
            goals = [episode.transitions[-1].achieved_goal.copy() for _ in range(k)]
        else:
            # Sample k indices from future (with replacement if needed)
            sampled_indices = np.random.choice(
                future_indices,
                size=min(k, len(future_indices)),
                replace=False,
            ).tolist()
            # Pad with final goal if not enough future timesteps
            while len(sampled_indices) < k:
                sampled_indices.append(n - 1)

            goals = [episode.transitions[i].achieved_goal.copy() for i in sampled_indices]

    elif strategy == GoalStrategy.EPISODE:
        # Sample from anywhere in the episode
        all_indices = list(range(n))
        sampled_indices = np.random.choice(all_indices, size=k, replace=True).tolist()
        goals = [episode.transitions[i].achieved_goal.copy() for i in sampled_indices]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return goals
# --8<-- [end:goal_sampling]


# =============================================================================
# Relabeling Logic
# =============================================================================

# --8<-- [start:relabel_transition]
def relabel_transition(
    transition: Transition,
    new_goal: np.ndarray,
    compute_reward_fn: callable,
    threshold: float = 0.05,
) -> Transition:
    """
    Relabel a transition with a new goal and recompute the reward.

    This is the core HER operation:
    1. Replace the desired_goal with new_goal (typically an achieved_goal)
    2. Recompute the reward based on whether achieved_goal == new_goal

    The crucial requirement: the environment must support reward recomputation
    via compute_reward(achieved_goal, desired_goal). Gymnasium-Robotics Fetch
    environments provide this.

    Args:
        transition: Original transition to relabel
        new_goal: The new goal to substitute
        compute_reward_fn: Function(achieved, desired, info) -> reward
        threshold: Distance threshold for success (default 0.05m for Fetch)

    Returns:
        New transition with updated desired_goal and recomputed reward
    """
    # Compute new reward: did the achieved goal match the new desired goal?
    # For Fetch environments, reward = 0 if ||achieved - desired|| < threshold, else -1
    new_reward = compute_reward_fn(
        transition.achieved_goal,
        new_goal,
        {"distance_threshold": threshold},
    )

    # Create relabeled transition (achieved_goal stays the same!)
    return Transition(
        obs=transition.obs,
        action=transition.action,
        reward=new_reward,
        next_obs=transition.next_obs,
        done=transition.done,
        achieved_goal=transition.achieved_goal,  # unchanged
        desired_goal=new_goal,  # replaced with new goal
    )


def sparse_reward(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    info: dict,
) -> float:
    """
    Compute sparse reward: success (0) or failure (-1).

    This is the standard Gymnasium-Robotics reward function.
    HER makes this learnable by turning failures into successes through relabeling.
    """
    threshold = info.get("distance_threshold", 0.05)
    distance = np.linalg.norm(achieved_goal - desired_goal)
    return 0.0 if distance < threshold else -1.0
# --8<-- [end:relabel_transition]


# =============================================================================
# Buffer Integration
# =============================================================================

# --8<-- [start:her_buffer_insert]
def process_episode_with_her(
    episode: Episode,
    compute_reward_fn: callable,
    strategy: GoalStrategy = GoalStrategy.FUTURE,
    k: int = 4,
    her_ratio: float = 0.8,
) -> list[Transition]:
    """
    Process an episode and generate HER-augmented transitions.

    For each transition in the episode:
    1. Always add the original transition (with real goal)
    2. With probability her_ratio, also add k relabeled transitions

    This dramatically increases the number of "successful" transitions,
    which is crucial for learning with sparse rewards.

    Example: A 50-step episode with k=4 and her_ratio=0.8 generates:
    - 50 original transitions
    - ~50 * 0.8 * 4 = 160 relabeled transitions
    Total: ~210 transitions, most of which are "successes"

    Args:
        episode: Episode to process
        compute_reward_fn: Function to recompute rewards
        strategy: HER goal sampling strategy
        k: Number of relabeled goals per transition
        her_ratio: Probability of adding relabeled transitions

    Returns:
        List of all transitions (original + relabeled)
    """
    all_transitions = []

    for idx, transition in enumerate(episode.transitions):
        # Always add original transition
        all_transitions.append(transition)

        # With probability her_ratio, add relabeled transitions
        if np.random.random() < her_ratio:
            # Sample alternative goals
            her_goals = sample_her_goals(episode, idx, strategy, k)

            # Create relabeled transitions
            for new_goal in her_goals:
                relabeled = relabel_transition(
                    transition, new_goal, compute_reward_fn
                )
                all_transitions.append(relabeled)

    return all_transitions


def compute_success_fraction(transitions: list[Transition]) -> float:
    """Compute fraction of transitions with non-negative reward (successes)."""
    if not transitions:
        return 0.0
    successes = sum(1 for t in transitions if t.reward >= 0)
    return successes / len(transitions)
# --8<-- [end:her_buffer_insert]


# =============================================================================
# Verification (Sanity Checks)
# =============================================================================

def create_synthetic_episode(
    n_steps: int = 50,
    obs_dim: int = 10,
    act_dim: int = 4,
    goal_dim: int = 3,
) -> Episode:
    """Create a synthetic episode for testing."""
    transitions = []

    # Simulate a trajectory that gradually approaches a random goal
    desired_goal = np.random.randn(goal_dim)
    achieved = np.random.randn(goal_dim) * 2  # Start far from goal

    for t in range(n_steps):
        obs = np.random.randn(obs_dim)
        action = np.random.randn(act_dim)

        # Move achieved goal slightly toward desired (but never quite reach)
        achieved = achieved + 0.02 * (desired_goal - achieved) + 0.01 * np.random.randn(goal_dim)

        # Sparse reward: almost always -1 (failure)
        reward = sparse_reward(achieved, desired_goal, {"distance_threshold": 0.05})

        transitions.append(Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=np.random.randn(obs_dim),
            done=(t == n_steps - 1),
            achieved_goal=achieved.copy(),
            desired_goal=desired_goal.copy(),
        ))

    return Episode(transitions=transitions)


def verify_goal_sampling():
    """Verify goal sampling strategies."""
    print("Verifying goal sampling...")

    episode = create_synthetic_episode(n_steps=20)

    for strategy in [GoalStrategy.FINAL, GoalStrategy.FUTURE, GoalStrategy.EPISODE]:
        goals = sample_her_goals(episode, transition_idx=5, strategy=strategy, k=4)
        assert len(goals) == 4, f"Expected 4 goals, got {len(goals)}"
        for g in goals:
            assert g.shape == episode.transitions[0].achieved_goal.shape
        print(f"  {strategy.value}: sampled {len(goals)} goals")

    print("  [PASS] Goal sampling OK")


def verify_relabeling():
    """Verify transition relabeling."""
    print("Verifying relabeling...")

    episode = create_synthetic_episode(n_steps=10)
    transition = episode.transitions[5]

    # Original transition should have negative reward (failure)
    assert transition.reward < 0, "Original should be failure"

    # Relabel with own achieved goal -> should be success
    relabeled = relabel_transition(
        transition,
        transition.achieved_goal,  # Use own achieved goal
        sparse_reward,
    )
    assert relabeled.reward == 0.0, "Relabeled with own goal should be success"
    assert np.array_equal(relabeled.achieved_goal, transition.achieved_goal), "Achieved goal should be unchanged"
    assert np.array_equal(relabeled.desired_goal, transition.achieved_goal), "Desired goal should be relabeled"

    print(f"  Original reward: {transition.reward}")
    print(f"  Relabeled reward: {relabeled.reward}")
    print("  [PASS] Relabeling OK")


def verify_her_processing():
    """Verify full HER episode processing."""
    print("Verifying HER processing...")

    episode = create_synthetic_episode(n_steps=50)

    # Check original success rate
    original_success = compute_success_fraction(episode.transitions)
    print(f"  Original success rate: {original_success:.1%}")

    # Process with HER
    her_transitions = process_episode_with_her(
        episode,
        sparse_reward,
        strategy=GoalStrategy.FUTURE,
        k=4,
        her_ratio=0.8,
    )

    # Check HER success rate
    her_success = compute_success_fraction(her_transitions)
    print(f"  HER success rate: {her_success:.1%}")
    print(f"  Transitions: {len(episode)} -> {len(her_transitions)}")

    # HER should dramatically increase success rate
    assert her_success > original_success, "HER should increase success rate"
    assert len(her_transitions) > len(episode.transitions), "HER should increase data"

    print("  [PASS] HER processing OK")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("HER Relabeler -- Verification")
    print("=" * 60)

    verify_goal_sampling()
    print()
    verify_relabeling()
    print()
    verify_her_processing()

    print()
    print("=" * 60)
    print("[ALL PASS] HER implementation verified")
    print("=" * 60)


def run_demo():
    """Run a demonstration of HER relabeling."""
    print("=" * 60)
    print("HER Relabeler -- Demo")
    print("=" * 60)

    # Create synthetic episode
    episode = create_synthetic_episode(n_steps=50)
    print(f"\nSynthetic episode: {len(episode)} steps")

    # Original metrics
    original_success = compute_success_fraction(episode.transitions)
    print(f"Original success rate: {original_success:.1%}")

    # Process with different strategies
    for strategy in [GoalStrategy.FINAL, GoalStrategy.FUTURE, GoalStrategy.EPISODE]:
        her_transitions = process_episode_with_her(
            episode, sparse_reward, strategy=strategy, k=4, her_ratio=0.8
        )
        success_rate = compute_success_fraction(her_transitions)
        print(f"  {strategy.value}: {len(her_transitions)} transitions, {success_rate:.1%} success")

    print("\nKey insight: HER turns ~0% success into ~60-80% success")
    print("by relabeling failures with goals that were actually achieved.")


def main():
    parser = argparse.ArgumentParser(description="HER Relabeler")
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
