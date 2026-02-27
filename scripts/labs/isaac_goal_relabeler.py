#!/usr/bin/env python3
"""
Isaac Goal Relabeler -- Build-It module for Appendix E.

Pedagogical goal:
- Show HER-style goal relabeling mechanics on insertion-like trajectories.
- Keep logic explicit and simulator-agnostic.
- Provide snippet regions for chapter/tutorial includes.

Usage:
    # Run lightweight sanity checks (~1-5 seconds)
    python scripts/labs/isaac_goal_relabeler.py --verify

    # Demonstrate data amplification on synthetic insertion trajectories
    python scripts/labs/isaac_goal_relabeler.py --demo

    # Print SB3 mapping for Appendix E run-it pipeline
    python scripts/labs/isaac_goal_relabeler.py --bridge
"""
from __future__ import annotations

import argparse
from enum import Enum
from typing import NamedTuple

import numpy as np


# --8<-- [start:goal_transition_structs]
class GoalTransition(NamedTuple):
    """
    Goal-conditioned transition used for HER-style relabeling.

    We keep observations/actions generic because Appendix E tasks may expose
    different state layouts across Isaac releases.
    """

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    achieved_goal: np.ndarray
    desired_goal: np.ndarray


class GoalEpisode(NamedTuple):
    transitions: list[GoalTransition]

    def __len__(self) -> int:
        return len(self.transitions)


class GoalStrategy(Enum):
    FINAL = "final"
    FUTURE = "future"
    EPISODE = "episode"
# --8<-- [end:goal_transition_structs]


# --8<-- [start:isaac_goal_sampling]
def sample_goals(
    episode: GoalEpisode,
    t: int,
    strategy: GoalStrategy = GoalStrategy.FUTURE,
    k: int = 4,
) -> list[np.ndarray]:
    """
    Sample alternative desired goals from achieved goals in the same episode.
    """
    n = len(episode)
    if n == 0:
        return []

    if strategy == GoalStrategy.FINAL:
        g = episode.transitions[-1].achieved_goal
        return [g.copy() for _ in range(k)]

    if strategy == GoalStrategy.FUTURE:
        future = list(range(t + 1, n))
        if not future:
            return [episode.transitions[-1].achieved_goal.copy() for _ in range(k)]
        picked = np.random.choice(future, size=min(k, len(future)), replace=False).tolist()
        while len(picked) < k:
            picked.append(future[-1])
        return [episode.transitions[i].achieved_goal.copy() for i in picked]

    if strategy == GoalStrategy.EPISODE:
        all_idx = list(range(n))
        picked = np.random.choice(all_idx, size=k, replace=True).tolist()
        return [episode.transitions[i].achieved_goal.copy() for i in picked]

    raise ValueError(f"Unknown strategy: {strategy}")
# --8<-- [end:isaac_goal_sampling]


def sparse_goal_reward(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """
    Generic sparse success reward for insertion-like precision tasks.

    Returns 0 on success and -1 on failure, matching Gym robotics convention.
    """
    dist = float(np.linalg.norm(achieved_goal - desired_goal))
    return 0.0 if dist <= threshold else -1.0


# --8<-- [start:isaac_relabel_transition]
def relabel_transition(
    transition: GoalTransition,
    new_goal: np.ndarray,
    reward_fn: callable,
) -> GoalTransition:
    """
    HER core operation:
    - keep observed state/action/achieved_goal unchanged
    - replace desired_goal
    - recompute reward from (achieved_goal, new desired_goal)
    """
    new_reward = reward_fn(transition.achieved_goal, new_goal)
    return GoalTransition(
        obs=transition.obs,
        action=transition.action,
        reward=float(new_reward),
        next_obs=transition.next_obs,
        done=transition.done,
        achieved_goal=transition.achieved_goal,
        desired_goal=new_goal,
    )
# --8<-- [end:isaac_relabel_transition]


# --8<-- [start:isaac_her_episode_processing]
def process_episode_with_her(
    episode: GoalEpisode,
    reward_fn: callable,
    *,
    strategy: GoalStrategy = GoalStrategy.FUTURE,
    k: int = 4,
    her_ratio: float = 0.8,
) -> list[GoalTransition]:
    """
    Convert one episode into original + relabeled transitions.
    """
    out: list[GoalTransition] = []
    for t, tr in enumerate(episode.transitions):
        out.append(tr)
        if np.random.random() >= her_ratio:
            continue
        goals = sample_goals(episode, t=t, strategy=strategy, k=k)
        for g in goals:
            out.append(relabel_transition(tr, g, reward_fn))
    return out


def success_fraction(transitions: list[GoalTransition]) -> float:
    if not transitions:
        return 0.0
    return float(sum(1 for t in transitions if t.reward >= 0.0) / len(transitions))
# --8<-- [end:isaac_her_episode_processing]


def _make_synthetic_insertion_episode(
    n_steps: int = 60,
    obs_dim: int = 20,
    act_dim: int = 7,
    goal_dim: int = 3,
) -> GoalEpisode:
    """
    Build a trajectory that gradually approaches target but rarely reaches it.

    This mimics early insertion training: many near-misses, very few successes.
    """
    desired = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    achieved = np.array([0.03, -0.03, 0.04], dtype=np.float32)

    transitions: list[GoalTransition] = []
    for t in range(n_steps):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(act_dim).astype(np.float32) * 0.1

        # Drift toward target with noise; threshold is tight (1 cm),
        # so random episodes remain mostly failures.
        achieved = achieved + 0.05 * (desired - achieved) + 0.002 * np.random.randn(goal_dim).astype(np.float32)
        reward = sparse_goal_reward(achieved, desired, threshold=0.01)

        transitions.append(
            GoalTransition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=np.random.randn(obs_dim).astype(np.float32),
                done=(t == n_steps - 1),
                achieved_goal=achieved.copy(),
                desired_goal=desired.copy(),
            )
        )

    return GoalEpisode(transitions)


def run_verification(seed: int = 0) -> None:
    np.random.seed(seed)

    print("=" * 60)
    print("Isaac Goal Relabeler -- Verification")
    print("=" * 60)

    ep = _make_synthetic_insertion_episode()
    original_sr = success_fraction(ep.transitions)
    print(f"original success fraction: {original_sr:.2%}")

    # Sampling invariants
    goals = sample_goals(ep, t=10, strategy=GoalStrategy.FUTURE, k=4)
    assert len(goals) == 4, f"expected 4 goals, got {len(goals)}"
    for g in goals:
        assert g.shape == ep.transitions[0].achieved_goal.shape
    print("[PASS] goal sampling")

    # Relabel invariant: relabel with own achieved goal should be success
    tr = ep.transitions[15]
    relabeled = relabel_transition(tr, tr.achieved_goal, lambda ag, dg: sparse_goal_reward(ag, dg, threshold=0.01))
    assert relabeled.reward == 0.0, "self-achieved goal relabel should be success"
    assert np.allclose(relabeled.achieved_goal, tr.achieved_goal)
    print("[PASS] single-transition relabeling")

    # Episode processing should increase success signal and transition count
    her_transitions = process_episode_with_her(
        ep,
        reward_fn=lambda ag, dg: sparse_goal_reward(ag, dg, threshold=0.01),
        strategy=GoalStrategy.FUTURE,
        k=4,
        her_ratio=0.8,
    )
    her_sr = success_fraction(her_transitions)
    assert len(her_transitions) > len(ep.transitions), "HER should increase transition count"
    assert her_sr > original_sr, "HER should increase success fraction on sparse trajectories"
    print(f"HER transitions: {len(ep.transitions)} -> {len(her_transitions)}")
    print(f"HER success fraction: {her_sr:.2%}")
    print("[PASS] episode-level HER processing")

    print("=" * 60)
    print("[ALL PASS] Isaac goal relabeler verified")
    print("=" * 60)


def run_demo(seed: int = 0) -> None:
    np.random.seed(seed)

    print("=" * 60)
    print("Isaac Goal Relabeler -- Demo")
    print("=" * 60)

    ep = _make_synthetic_insertion_episode(n_steps=50)
    print(f"episode length: {len(ep)}")
    print(f"original success fraction: {success_fraction(ep.transitions):.2%}")

    for strategy in [GoalStrategy.FINAL, GoalStrategy.FUTURE, GoalStrategy.EPISODE]:
        aug = process_episode_with_her(
            ep,
            reward_fn=lambda ag, dg: sparse_goal_reward(ag, dg, threshold=0.01),
            strategy=strategy,
            k=4,
            her_ratio=0.8,
        )
        print(f"{strategy.value:>7}: transitions={len(aug):3d}, success_fraction={success_fraction(aug):.2%}")

    print("Key idea: relabeling converts precision near-misses into supervised signal.")


def run_bridge() -> None:
    print("=" * 60)
    print("Isaac Goal Relabeler -- Bridge to SB3")
    print("=" * 60)
    print("From-scratch component -> SB3 HerReplayBuffer concept")
    print("  sample_goals         -> goal_selection_strategy")
    print("  k                    -> n_sampled_goal")
    print("  relabel_transition   -> virtual transition synthesis")
    print("  reward_fn            -> env.compute_reward invariant")
    print("  process_episode_with_her -> real + virtual sample mix")
    print()
    print("Run-It integration point: scripts/appendix_e_isaac_peg.py --her auto|on|off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Goal Relabeler (Build-It)")
    parser.add_argument("--verify", action="store_true", help="Run synthetic sanity checks")
    parser.add_argument("--demo", action="store_true", help="Run a synthetic demonstration")
    parser.add_argument("--bridge", action="store_true", help="Print Build-It <-> SB3 mapping")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    if args.verify:
        run_verification(seed=args.seed)
    elif args.demo:
        run_demo(seed=args.seed)
    elif args.bridge:
        run_bridge()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
