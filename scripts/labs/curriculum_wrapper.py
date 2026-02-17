#!/usr/bin/env python3
"""
Curriculum Goal Wrapper -- Pedagogical Implementation

This module implements a curriculum learning wrapper for goal-conditioned
environments. It gradually increases task difficulty by controlling the
distribution of goals during training.

For PickAndPlace, difficulty has two dimensions:
1. Goal range: how far from the initial position goals can be
2. Air probability: how often goals are above the table (requiring grasping)

Easy goals (difficulty=0): small range, all on table (just pushing)
Hard goals (difficulty=1): full range, ~50% in air (full pick-and-place)

Usage:
    # Run verification (sanity checks, ~30 seconds)
    python scripts/labs/curriculum_wrapper.py --verify

    # Demonstrate goal distributions at different difficulty levels
    python scripts/labs/curriculum_wrapper.py --demo

Key regions exported for tutorials:
    - curriculum_wrapper: CurriculumGoalWrapper class
    - curriculum_schedule: CurriculumScheduleCallback class
    - difficulty_demo: Demonstration of difficulty levels
    - integration_example: How to integrate with SB3 make_vec_env
"""
from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym
import numpy as np


# =============================================================================
# Curriculum Goal Wrapper
# =============================================================================

# --8<-- [start:curriculum_wrapper]
class CurriculumGoalWrapper(gym.Wrapper):
    """Wrapper that controls goal difficulty during training.

    Overrides reset() to replace the environment's sampled goal with a
    curriculum-controlled goal. The difficulty parameter (0.0 to 1.0)
    controls two aspects:

    1. **Goal range:** How far goals can be from the gripper's initial
       position. At difficulty=0, goals are within 2cm. At difficulty=1,
       goals span the full 15cm range.

    2. **Air probability:** How often goals are placed above the table.
       At difficulty=0, all goals are on the table (z = height_offset).
       At difficulty=1, ~50% of goals are elevated (requiring grasp+lift).

    The wrapper does NOT modify the environment dynamics -- only the goal
    distribution. This means the physics, rewards, and success criteria
    remain unchanged.

    Args:
        env: A goal-conditioned Gymnasium environment.
        initial_difficulty: Starting difficulty level (0.0 to 1.0).
        min_range: Goal range at difficulty=0 (meters).
        max_range: Goal range at difficulty=1 (meters).
        max_air_prob: Air goal probability at difficulty=1.
        height_offset: Table surface height (z coordinate for table goals).
        max_air_height: Maximum height above table for air goals.
    """

    def __init__(
        self,
        env: gym.Env,
        initial_difficulty: float = 0.0,
        min_range: float = 0.02,
        max_range: float = 0.15,
        max_air_prob: float = 0.5,
        height_offset: float = 0.42,
        max_air_height: float = 0.15,
    ):
        super().__init__(env)
        self._difficulty = float(np.clip(initial_difficulty, 0.0, 1.0))
        self._min_range = min_range
        self._max_range = max_range
        self._max_air_prob = max_air_prob
        self._height_offset = height_offset
        self._max_air_height = max_air_height
        self._rng = np.random.default_rng()

    @property
    def difficulty(self) -> float:
        """Current difficulty level (0.0 = easy, 1.0 = full distribution)."""
        return self._difficulty

    def set_difficulty(self, fraction: float) -> None:
        """Set the difficulty level.

        Args:
            fraction: Difficulty from 0.0 (easy) to 1.0 (full distribution).
        """
        self._difficulty = float(np.clip(fraction, 0.0, 1.0))

    def _current_goal_range(self) -> float:
        """Interpolate goal range based on difficulty."""
        return self._min_range + self._difficulty * (self._max_range - self._min_range)

    def _current_air_prob(self) -> float:
        """Interpolate air goal probability based on difficulty."""
        return self._difficulty * self._max_air_prob

    def _sample_curriculum_goal(self, base_pos: np.ndarray) -> np.ndarray:
        """Sample a goal position according to current difficulty.

        The goal is sampled relative to a base position (typically the
        initial gripper or object position):
        - XY offset: uniform in [-range, +range]
        - Z: either on the table or elevated (based on air probability)

        Args:
            base_pos: Reference position [x, y, z] for goal sampling.

        Returns:
            Goal position [x, y, z].
        """
        goal_range = self._current_goal_range()
        air_prob = self._current_air_prob()

        # Sample XY offset
        xy_offset = self._rng.uniform(-goal_range, goal_range, size=2)

        # Decide air vs table
        is_air = self._rng.random() < air_prob

        if is_air:
            # Air goal: z above the table surface
            z = self._height_offset + self._rng.uniform(0.02, self._max_air_height)
        else:
            # Table goal: z at the table surface
            z = self._height_offset

        goal = np.array([
            base_pos[0] + xy_offset[0],
            base_pos[1] + xy_offset[1],
            z,
        ])

        return goal

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset and replace the goal with a curriculum-controlled one."""
        obs, info = super().reset(seed=seed, options=options)

        # Use the object's initial position as base for goal sampling.
        # For PickAndPlace, the object starts on the table.
        # achieved_goal gives us the current object position.
        base_pos = obs["achieved_goal"].copy()

        # Sample new goal
        new_goal = self._sample_curriculum_goal(base_pos)

        # Replace goal in observation
        obs["desired_goal"] = new_goal.copy()

        # Also update the environment's internal goal if possible.
        # This ensures compute_reward uses the correct goal.
        if hasattr(self.unwrapped, "goal"):
            self.unwrapped.goal = new_goal.copy()

        return obs, info
# --8<-- [end:curriculum_wrapper]


# =============================================================================
# Schedule Callback
# =============================================================================

# --8<-- [start:curriculum_schedule]
class CurriculumScheduleCallback:
    """SB3 callback that advances curriculum difficulty during training.

    Supports two scheduling modes:

    1. **Linear:** Difficulty increases linearly with training progress.
       At step 0, difficulty = 0. At final step, difficulty = 1.
       Simple and predictable.

    2. **Success-gated:** Difficulty advances only when the agent achieves
       a minimum success rate over a sliding window. This is adaptive --
       the agent stays at each difficulty level until it masters it.

    Usage with SB3's model.learn():
        callback = CurriculumScheduleCallback(
            wrapper=wrapped_env,
            total_timesteps=5_000_000,
            mode="linear",
        )
        model.learn(total_timesteps=5_000_000, callback=callback)

    Args:
        wrapper: The CurriculumGoalWrapper to control.
        total_timesteps: Total training steps (for linear mode).
        mode: "linear" or "success_gated".
        success_threshold: Success rate to advance (success_gated mode).
        advance_increment: How much to increase difficulty (success_gated).
        window_size: Episode window for success rate (success_gated).
        check_freq: How often to check/update (in timesteps).
        verbose: Print difficulty updates.
    """

    def __init__(
        self,
        wrapper: CurriculumGoalWrapper,
        total_timesteps: int,
        mode: str = "linear",
        success_threshold: float = 0.7,
        advance_increment: float = 0.05,
        window_size: int = 100,
        check_freq: int = 5000,
        verbose: int = 1,
    ):
        if mode not in ("linear", "success_gated"):
            raise ValueError(f"Unknown mode: {mode}. Use 'linear' or 'success_gated'.")
        self._wrapper = wrapper
        self._total_timesteps = total_timesteps
        self._mode = mode
        self._success_threshold = success_threshold
        self._advance_increment = advance_increment
        self._window_size = window_size
        self._check_freq = max(1, check_freq)
        self._verbose = verbose
        self._n_calls = 0
        self._successes: list[float] = []
        self._last_logged_difficulty = -1.0

    def __call__(self, locals_dict: dict, globals_dict: dict) -> bool:
        """Called at each training step by SB3."""
        self._n_calls += 1

        if self._n_calls % self._check_freq != 0:
            return True

        if self._mode == "linear":
            progress = self._n_calls / max(1, self._total_timesteps)
            new_difficulty = min(1.0, progress)
            self._wrapper.set_difficulty(new_difficulty)

        elif self._mode == "success_gated":
            # Collect success signals from info dicts
            infos = locals_dict.get("infos") or []
            dones = locals_dict.get("dones")

            if isinstance(infos, list):
                for idx, info in enumerate(infos):
                    if not isinstance(info, dict):
                        continue
                    done = False
                    if dones is not None:
                        try:
                            done = bool(dones[idx])
                        except (IndexError, TypeError):
                            done = bool(dones)
                    if "episode" in info:
                        done = True
                    if done and "is_success" in info:
                        self._successes.append(float(info["is_success"]))

            # Check if we should advance
            window = self._successes[-self._window_size:]
            if len(window) >= self._window_size:
                success_rate = sum(window) / len(window)
                if success_rate >= self._success_threshold:
                    current = self._wrapper.difficulty
                    new_difficulty = min(1.0, current + self._advance_increment)
                    self._wrapper.set_difficulty(new_difficulty)
                    if self._verbose > 0:
                        print(
                            f"[Curriculum] Success rate {success_rate:.1%} >= "
                            f"{self._success_threshold:.0%}, advancing: "
                            f"difficulty {current:.2f} -> {new_difficulty:.2f}"
                        )
                    # Reset window after advancing
                    self._successes.clear()

        # Log difficulty changes
        current = self._wrapper.difficulty
        if self._verbose > 0 and abs(current - self._last_logged_difficulty) >= 0.1:
            pct = int(100 * self._n_calls / self._total_timesteps)
            print(f"[Curriculum] {pct}% complete, difficulty={current:.2f}")
            self._last_logged_difficulty = current

        return True
# --8<-- [end:curriculum_schedule]


# =============================================================================
# Integration Example
# =============================================================================

# --8<-- [start:integration_example]
def make_curriculum_env(
    env_id: str = "FetchPickAndPlace-v4",
    initial_difficulty: float = 0.0,
    **wrapper_kwargs: Any,
) -> CurriculumGoalWrapper:
    """Create a curriculum-wrapped environment.

    This is the simplest integration: wrap a single environment instance.
    For vectorized environments (make_vec_env), you would pass this as
    the env constructor.

    Args:
        env_id: Gymnasium environment ID.
        initial_difficulty: Starting difficulty (0.0 to 1.0).
        **wrapper_kwargs: Additional args for CurriculumGoalWrapper.

    Returns:
        Wrapped environment with curriculum control.

    Example:
        env = make_curriculum_env("FetchPickAndPlace-v4", initial_difficulty=0.0)
        obs, info = env.reset()
        # All goals will be on the table, very close to the object
        env.set_difficulty(1.0)
        obs, info = env.reset()
        # Goals can now be anywhere, including in the air
    """
    env = gym.make(env_id)
    return CurriculumGoalWrapper(env, initial_difficulty=initial_difficulty, **wrapper_kwargs)
# --8<-- [end:integration_example]


# =============================================================================
# Verification
# =============================================================================

def _classify_goal(goal: np.ndarray, height_offset: float = 0.42) -> str:
    """Classify a goal as 'air' or 'table' based on z coordinate."""
    return "air" if goal[2] > height_offset + 0.02 else "table"


def verify_wrapper_basics():
    """Verify CurriculumGoalWrapper basic functionality."""
    print("Verifying wrapper basics...")

    env = gym.make("FetchPickAndPlace-v4")
    wrapped = CurriculumGoalWrapper(env, initial_difficulty=0.0)

    # Check difficulty property
    assert wrapped.difficulty == 0.0, "Initial difficulty should be 0.0"
    wrapped.set_difficulty(0.5)
    assert wrapped.difficulty == 0.5, "Difficulty should be 0.5"
    wrapped.set_difficulty(-1.0)
    assert wrapped.difficulty == 0.0, "Negative difficulty should clamp to 0.0"
    wrapped.set_difficulty(2.0)
    assert wrapped.difficulty == 1.0, "Difficulty > 1 should clamp to 1.0"

    wrapped.close()
    print("  [PASS] Wrapper basics OK")


def verify_easy_goals():
    """Verify that difficulty=0 produces table-only, close goals."""
    print("Verifying easy goals (difficulty=0.0)...")

    env = gym.make("FetchPickAndPlace-v4")
    wrapped = CurriculumGoalWrapper(env, initial_difficulty=0.0)

    n_resets = 100
    air_count = 0
    distances = []

    for _ in range(n_resets):
        obs, _ = wrapped.reset()
        goal = obs["desired_goal"]
        achieved = obs["achieved_goal"]

        if _classify_goal(goal) == "air":
            air_count += 1
        distances.append(np.linalg.norm(goal[:2] - achieved[:2]))

    air_frac = air_count / n_resets
    mean_dist = np.mean(distances)

    print(f"  Air goals: {air_frac:.0%} (expected: 0%)")
    print(f"  Mean XY distance: {mean_dist:.4f}m (expected: <0.02m)")

    assert air_frac == 0.0, f"Easy mode should have 0% air goals, got {air_frac:.0%}"
    assert mean_dist < 0.03, f"Easy mode goals should be close, got {mean_dist:.4f}m"

    wrapped.close()
    print("  [PASS] Easy goals OK")


def verify_hard_goals():
    """Verify that difficulty=1 produces a mix of air and table goals."""
    print("Verifying hard goals (difficulty=1.0)...")

    env = gym.make("FetchPickAndPlace-v4")
    wrapped = CurriculumGoalWrapper(env, initial_difficulty=1.0)

    n_resets = 200
    air_count = 0
    distances = []

    for _ in range(n_resets):
        obs, _ = wrapped.reset()
        goal = obs["desired_goal"]
        achieved = obs["achieved_goal"]

        if _classify_goal(goal) == "air":
            air_count += 1
        distances.append(np.linalg.norm(goal[:2] - achieved[:2]))

    air_frac = air_count / n_resets
    mean_dist = np.mean(distances)

    print(f"  Air goals: {air_frac:.0%} (expected: ~50%)")
    print(f"  Mean XY distance: {mean_dist:.4f}m (expected: wider spread)")

    # Allow some statistical tolerance
    assert 0.3 <= air_frac <= 0.7, f"Hard mode should have ~50% air goals, got {air_frac:.0%}"
    assert mean_dist > 0.03, f"Hard mode should have wider spread, got {mean_dist:.4f}m"

    wrapped.close()
    print("  [PASS] Hard goals OK")


def verify_callback():
    """Verify CurriculumScheduleCallback in linear mode."""
    print("Verifying schedule callback (linear mode)...")

    env = gym.make("FetchPickAndPlace-v4")
    wrapped = CurriculumGoalWrapper(env, initial_difficulty=0.0)

    callback = CurriculumScheduleCallback(
        wrapper=wrapped,
        total_timesteps=1000,
        mode="linear",
        check_freq=100,
        verbose=0,
    )

    # Simulate 1000 steps
    for step in range(1000):
        callback({"self": None}, {})

    final_difficulty = wrapped.difficulty
    print(f"  Final difficulty: {final_difficulty:.2f} (expected: ~1.0)")

    assert final_difficulty >= 0.9, f"After full training, difficulty should be ~1.0, got {final_difficulty:.2f}"

    wrapped.close()
    print("  [PASS] Schedule callback OK")


def run_verification():
    """Run all verification checks."""
    import gymnasium_robotics  # noqa: F401 -- registers Fetch envs

    print("=" * 60)
    print("Curriculum Wrapper -- Verification")
    print("=" * 60)

    verify_wrapper_basics()
    print()
    verify_easy_goals()
    print()
    verify_hard_goals()
    print()
    verify_callback()

    print()
    print("=" * 60)
    print("[ALL PASS] Curriculum wrapper verified")
    print("=" * 60)


# =============================================================================
# Demo
# =============================================================================

# --8<-- [start:difficulty_demo]
def run_demo():
    """Demonstrate goal distributions at different difficulty levels."""
    import gymnasium_robotics  # noqa: F401 -- registers Fetch envs

    print("=" * 60)
    print("Curriculum Wrapper -- Demo")
    print("=" * 60)

    env = gym.make("FetchPickAndPlace-v4")
    wrapped = CurriculumGoalWrapper(env, initial_difficulty=0.0)

    difficulty_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_samples = 100

    print(f"\nSampling {n_samples} goals at each difficulty level:\n")
    print(f"{'Difficulty':>12} | {'Air %':>8} | {'Table %':>8} | {'Mean XY dist':>14} | {'Mean Z':>10}")
    print("-" * 65)

    for diff in difficulty_levels:
        wrapped.set_difficulty(diff)

        air_count = 0
        xy_dists = []
        z_vals = []

        for _ in range(n_samples):
            obs, _ = wrapped.reset()
            goal = obs["desired_goal"]
            achieved = obs["achieved_goal"]

            if _classify_goal(goal) == "air":
                air_count += 1
            xy_dists.append(np.linalg.norm(goal[:2] - achieved[:2]))
            z_vals.append(goal[2])

        air_pct = 100 * air_count / n_samples
        table_pct = 100 - air_pct
        mean_xy = np.mean(xy_dists)
        mean_z = np.mean(z_vals)

        print(f"{diff:>12.2f} | {air_pct:>7.0f}% | {table_pct:>7.0f}% | {mean_xy:>13.4f}m | {mean_z:>9.4f}m")

    print()
    print("Key observations:")
    print("  - At difficulty=0, all goals are on the table and very close")
    print("  - Air % increases linearly with difficulty")
    print("  - XY spread increases as goal range widens")
    print("  - Mean Z rises as air goals appear more frequently")

    wrapped.close()
# --8<-- [end:difficulty_demo]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Curriculum Goal Wrapper")
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
