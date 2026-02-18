#!/usr/bin/env python3
"""
Action Interface Engineering -- Pedagogical Wrappers and Metrics

This module provides eval-time action wrappers and a proportional controller
baseline for analyzing learned policies as controllers. No retraining is
needed -- we evaluate the same checkpoint under different action interfaces.

Components:
    - ActionScalingWrapper: Scale actions by a constant factor
    - LowPassFilterWrapper: Exponential moving average smoothing
    - ProportionalController: Simple P-controller baseline (SB3-compatible)
    - ControllerMetrics: Per-episode engineering metrics dataclass
    - compute_controller_metrics: Extract metrics from a trajectory
    - run_controller_eval: Unified eval loop for SB3 models and controllers

Usage:
    # Run verification (sanity checks, ~30 seconds)
    python scripts/labs/action_interface.py --verify

    # Demonstrate wrappers on FetchReach-v4
    python scripts/labs/action_interface.py --demo

Key regions exported for tutorials:
    - action_scaling_wrapper: ActionScalingWrapper class
    - lowpass_filter_wrapper: LowPassFilterWrapper class
    - proportional_controller: ProportionalController class
    - controller_metrics: ControllerMetrics dataclass + compute function
    - run_controller_eval: Unified evaluation loop
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


# =============================================================================
# Action Scaling Wrapper
# =============================================================================

# --8<-- [start:action_scaling_wrapper]
class ActionScalingWrapper(gym.ActionWrapper):
    """Scale actions by a constant factor before sending to the environment.

    scale < 1.0 -> finer, slower control (actions shrink toward zero)
    scale > 1.0 -> coarser control (actions saturate at bounds faster)

    The output is always clipped to the original action space bounds,
    so the environment never sees out-of-range actions.
    """

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def action(self, act: np.ndarray) -> np.ndarray:
        scaled = act * self.scale
        return np.clip(scaled, self.action_space.low, self.action_space.high)
# --8<-- [end:action_scaling_wrapper]


# =============================================================================
# Low-Pass Filter Wrapper
# =============================================================================

# --8<-- [start:lowpass_filter_wrapper]
class LowPassFilterWrapper(gym.ActionWrapper):
    """Smooth actions with an exponential moving average (EMA).

    a_out = alpha * a_raw + (1 - alpha) * a_prev

    alpha = 1.0 -> no filtering (pass-through, identity)
    alpha = 0.0 -> frozen at initial action (never updates)

    Lower alpha = heavier smoothing = smoother but more sluggish.
    Filter state resets on each episode via reset().
    """

    def __init__(self, env: gym.Env, alpha: float = 1.0):
        super().__init__(env)
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
        self.alpha = alpha
        self._prev_action: np.ndarray | None = None

    def action(self, act: np.ndarray) -> np.ndarray:
        if self._prev_action is None:
            self._prev_action = act.copy()
            return act
        smoothed = self.alpha * act + (1.0 - self.alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return np.clip(smoothed, self.action_space.low, self.action_space.high)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._prev_action = None
        return super().reset(**kwargs)
# --8<-- [end:lowpass_filter_wrapper]


# =============================================================================
# Proportional Controller
# =============================================================================

# --8<-- [start:proportional_controller]
class ProportionalController:
    """Simple proportional controller for Fetch environments.

    Implements predict(obs, deterministic=True) -> (action, None) to match
    the SB3 model interface, so the same evaluation loop works for both.

    For Reach:  a_xyz = Kp * (desired_goal - grip_pos), gripper = 0
    For Push:   two-phase -- (1) approach object, (2) push toward goal

    The MuJoCo environment already has internal PD controllers on the joints,
    so we only need proportional control in Cartesian space (Kd=0).
    """

    def __init__(self, env_id: str, kp: float = 10.0):
        self.env_id = env_id
        self.kp = kp
        self._is_push = "Push" in env_id

    def predict(
        self, obs: dict[str, np.ndarray], deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        grip_pos = obs["observation"][:3]
        desired_goal = obs["desired_goal"]

        if self._is_push:
            return self._push_action(obs, grip_pos, desired_goal), None
        else:
            return self._reach_action(grip_pos, desired_goal), None

    def _reach_action(
        self, grip_pos: np.ndarray, desired_goal: np.ndarray
    ) -> np.ndarray:
        """Drive gripper toward the desired goal."""
        error = desired_goal - grip_pos
        a_xyz = self.kp * error
        # 4D action: [dx, dy, dz, gripper]; gripper=0 means no grip change
        action = np.zeros(4, dtype=np.float32)
        action[:3] = a_xyz
        return np.clip(action, -1.0, 1.0)

    def _push_action(
        self,
        obs: dict[str, np.ndarray],
        grip_pos: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        """Two-phase push: (1) move above object, (2) push object to goal.

        Phase 1: If gripper is far from the object, move toward object.
        Phase 2: Once near the object, push in the direction of the goal.
        """
        # Object position is at obs["observation"][3:6] in Fetch envs
        obj_pos = obs["observation"][3:6]
        action = np.zeros(4, dtype=np.float32)

        # Phase 1: approach the object from above
        grip_to_obj = obj_pos - grip_pos
        dist_to_obj = np.linalg.norm(grip_to_obj)

        if dist_to_obj > 0.05:
            # Move toward object
            approach = grip_to_obj.copy()
            approach[2] = max(approach[2], 0.0)  # stay above table
            action[:3] = self.kp * approach
        else:
            # Phase 2: push object toward goal
            obj_to_goal = desired_goal - obj_pos
            action[:3] = self.kp * obj_to_goal

        return np.clip(action, -1.0, 1.0)
# --8<-- [end:proportional_controller]


# =============================================================================
# Controller Metrics
# =============================================================================

# --8<-- [start:controller_metrics]
@dataclass
class ControllerMetrics:
    """Per-episode engineering metrics for evaluating a controller.

    These go beyond success/failure to measure movement quality:
    - smoothness: mean squared action change per step (lower = smoother)
    - peak_action: maximum absolute action component (saturation indicator)
    - path_length: gripper travel distance in meters
    - action_energy: total squared action magnitude (effort proxy)
    """
    success: bool = False
    time_to_success: int | None = None
    episode_length: int = 0
    episode_return: float = 0.0
    final_distance: float = 0.0
    smoothness: float = 0.0
    peak_action: float = 0.0
    path_length: float = 0.0
    action_energy: float = 0.0


def compute_controller_metrics(
    actions: list[np.ndarray],
    grip_positions: list[np.ndarray],
    rewards: list[float],
    successes: list[bool],
    final_achieved: np.ndarray,
    final_desired: np.ndarray,
) -> ControllerMetrics:
    """Compute engineering metrics from a single episode trajectory.

    Args:
        actions: List of actions taken (T elements).
        grip_positions: Gripper positions at each step (T+1 elements,
            including initial position before first action).
        rewards: Rewards received at each step.
        successes: Per-step is_success flags.
        final_achieved: Achieved goal at episode end.
        final_desired: Desired goal for this episode.

    Returns:
        ControllerMetrics with all fields populated.
    """
    T = len(actions)
    if T == 0:
        return ControllerMetrics()

    # Success and timing
    success = any(successes)
    time_to_success = None
    for t, s in enumerate(successes):
        if s:
            time_to_success = t + 1  # 1-indexed step count
            break

    # Smoothness: mean squared action difference per step
    smoothness = 0.0
    if T > 1:
        diffs = [np.sum((actions[t] - actions[t - 1]) ** 2) for t in range(1, T)]
        smoothness = float(np.mean(diffs))

    # Peak action: max absolute value across entire episode
    peak_action = float(max(np.max(np.abs(a)) for a in actions))

    # Path length: total gripper travel
    path_length = 0.0
    for t in range(1, len(grip_positions)):
        path_length += float(np.linalg.norm(
            grip_positions[t] - grip_positions[t - 1]
        ))

    # Action energy: total squared magnitude
    action_energy = float(sum(np.sum(a ** 2) for a in actions))

    # Final distance
    final_distance = float(np.linalg.norm(final_achieved - final_desired))

    return ControllerMetrics(
        success=success,
        time_to_success=time_to_success,
        episode_length=T,
        episode_return=sum(rewards),
        final_distance=final_distance,
        smoothness=smoothness,
        peak_action=peak_action,
        path_length=path_length,
        action_energy=action_energy,
    )
# --8<-- [end:controller_metrics]


# =============================================================================
# Unified Evaluation Loop
# =============================================================================

# --8<-- [start:run_controller_eval]
def run_controller_eval(
    policy: Any,
    env_id: str,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 0,
    wrappers: list[tuple[type, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Evaluate a policy (SB3 model or ProportionalController) with metrics.

    Args:
        policy: Object with predict(obs, deterministic) -> (action, _).
        env_id: Gymnasium environment ID.
        n_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic actions.
        seed: Base seed for environment resets.
        wrappers: List of (WrapperClass, kwargs) to apply around the env.
            Applied in order, so the last wrapper is outermost.

    Returns:
        Dictionary with 'aggregate' and 'episodes' keys, JSON-serializable.
    """
    import gymnasium_robotics  # noqa: F401

    env: gym.Env = gym.make(env_id)
    if wrappers:
        for wrapper_cls, wrapper_kwargs in wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    all_metrics: list[ControllerMetrics] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        desired_goal = obs["desired_goal"].copy()

        actions: list[np.ndarray] = []
        grip_positions: list[np.ndarray] = [obs["observation"][:3].copy()]
        rewards: list[float] = []
        successes: list[bool] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            successes.append(bool(info.get("is_success", False)))
            grip_positions.append(obs["observation"][:3].copy())

        metrics = compute_controller_metrics(
            actions=actions,
            grip_positions=grip_positions,
            rewards=rewards,
            successes=successes,
            final_achieved=obs["achieved_goal"],
            final_desired=desired_goal,
        )
        all_metrics.append(metrics)

    env.close()

    # Aggregate
    n = len(all_metrics)
    success_rate = sum(m.success for m in all_metrics) / n
    tts_values = [m.time_to_success for m in all_metrics if m.time_to_success is not None]

    aggregate = {
        "success_rate": success_rate,
        "return_mean": float(np.mean([m.episode_return for m in all_metrics])),
        "ep_len_mean": float(np.mean([m.episode_length for m in all_metrics])),
        "final_distance_mean": float(np.mean([m.final_distance for m in all_metrics])),
        "time_to_success_mean": float(np.mean(tts_values)) if tts_values else None,
        "smoothness_mean": float(np.mean([m.smoothness for m in all_metrics])),
        "peak_action_mean": float(np.mean([m.peak_action for m in all_metrics])),
        "path_length_mean": float(np.mean([m.path_length for m in all_metrics])),
        "action_energy_mean": float(np.mean([m.action_energy for m in all_metrics])),
    }

    episodes = [
        {
            "episode": ep,
            "success": m.success,
            "time_to_success": m.time_to_success,
            "episode_length": m.episode_length,
            "episode_return": m.episode_return,
            "final_distance": m.final_distance,
            "smoothness": m.smoothness,
            "peak_action": m.peak_action,
            "path_length": m.path_length,
            "action_energy": m.action_energy,
        }
        for ep, m in enumerate(all_metrics)
    ]

    return {"aggregate": aggregate, "episodes": episodes}
# --8<-- [end:run_controller_eval]


# =============================================================================
# Video Recording
# =============================================================================

# --8<-- [start:record_episode]
def record_episode(
    policy: Any,
    env_id: str,
    seed: int = 0,
    deterministic: bool = True,
    wrappers: list[tuple[type, dict[str, Any]]] | None = None,
    label: str = "",
) -> tuple[list[np.ndarray], ControllerMetrics]:
    """Record a single episode with RGB frames for video generation.

    Creates the environment with render_mode="rgb_array" so MuJoCo renders
    each frame to an offscreen buffer (via EGL in Docker).

    Args:
        policy: Object with predict(obs, deterministic) -> (action, _).
        env_id: Gymnasium environment ID.
        seed: Seed for the environment reset.
        deterministic: Whether to use deterministic actions.
        wrappers: List of (WrapperClass, kwargs) to apply.
        label: Optional label for logging.

    Returns:
        Tuple of (frames, metrics) where frames is a list of RGB arrays.
    """
    import gymnasium_robotics  # noqa: F401

    env: gym.Env = gym.make(env_id, render_mode="rgb_array")
    if wrappers:
        for wrapper_cls, wrapper_kwargs in wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    obs, info = env.reset(seed=seed)
    desired_goal = obs["desired_goal"].copy()

    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    grip_positions: list[np.ndarray] = [obs["observation"][:3].copy()]
    rewards: list[float] = []
    successes: list[bool] = []

    # Capture initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32)
        actions.append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        successes.append(bool(info.get("is_success", False)))
        grip_positions.append(obs["observation"][:3].copy())

        frame = env.render()
        if frame is not None:
            frames.append(frame)

    metrics = compute_controller_metrics(
        actions=actions,
        grip_positions=grip_positions,
        rewards=rewards,
        successes=successes,
        final_achieved=obs["achieved_goal"],
        final_desired=desired_goal,
    )

    env.close()

    if label:
        status = "SUCCESS" if metrics.success else "FAIL"
        print(f"  [{status}] {label}: {len(frames)} frames, "
              f"TTS={metrics.time_to_success}, smooth={metrics.smoothness:.4f}")

    return frames, metrics
# --8<-- [end:record_episode]


def save_video(
    frames: list[np.ndarray],
    path: str | Path,
    fps: int = 20,
) -> Path:
    """Save frames as an MP4 video using imageio.

    Args:
        frames: List of RGB numpy arrays (H, W, 3).
        path: Output file path.
        fps: Frames per second.

    Returns:
        Resolved path to the saved video.
    """
    import imageio.v3 as iio

    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames, fps=fps)
    print(f"  Saved video: {out_path} ({len(frames)} frames, {fps} fps)")
    return out_path


def save_comparison_video(
    frame_lists: list[tuple[str, list[np.ndarray]]],
    path: str | Path,
    fps: int = 20,
) -> Path:
    """Create a side-by-side comparison video from multiple episodes.

    Pads shorter episodes by repeating the last frame so all columns
    have equal length.

    Args:
        frame_lists: List of (label, frames) tuples.
        path: Output file path.
        fps: Frames per second.

    Returns:
        Resolved path to the saved video.
    """
    import imageio.v3 as iio

    if not frame_lists:
        raise ValueError("No frame lists provided")

    # Pad to equal length
    max_len = max(len(frames) for _, frames in frame_lists)
    padded = []
    for label, frames in frame_lists:
        if len(frames) < max_len:
            frames = frames + [frames[-1]] * (max_len - len(frames))
        padded.append((label, frames))

    # Concatenate frames horizontally for each timestep
    combined_frames = []
    for t in range(max_len):
        row = [frames[t] for _, frames in padded]
        combined = np.concatenate(row, axis=1)
        combined_frames.append(combined)

    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, combined_frames, fps=fps)
    n_panels = len(frame_lists)
    labels = ", ".join(label for label, _ in frame_lists)
    print(f"  Saved comparison video: {out_path} ({n_panels} panels: {labels})")
    return out_path


# =============================================================================
# Verification
# =============================================================================

def verify_action_scaling():
    """Verify ActionScalingWrapper preserves space and scales correctly."""
    print("Verifying ActionScalingWrapper...")

    import gymnasium_robotics  # noqa: F401

    env = gym.make("FetchReach-v4")
    wrapped = ActionScalingWrapper(env, scale=0.5)

    # Action space should be unchanged
    assert wrapped.action_space == env.action_space, "Action space changed"
    print("  action_space preserved: OK")

    # Scaling math: 0.5 * [1, 1, 1, 1] should give [0.5, 0.5, 0.5, 0.5]
    test_action = np.ones(4, dtype=np.float32)
    result = wrapped.action(test_action)
    expected = np.full(4, 0.5, dtype=np.float32)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("  scale=0.5 math: OK")

    # Clipping: scale=2.0 with action=1.0 should clip to 1.0
    wrapped_big = ActionScalingWrapper(env, scale=2.0)
    result_big = wrapped_big.action(test_action)
    assert np.allclose(result_big, np.ones(4)), f"Clipping failed: {result_big}"
    print("  scale=2.0 clipping: OK")

    # Scale=1.0 should be identity
    wrapped_id = ActionScalingWrapper(env, scale=1.0)
    result_id = wrapped_id.action(test_action)
    assert np.allclose(result_id, test_action), f"Identity failed: {result_id}"
    print("  scale=1.0 identity: OK")

    env.close()
    print("  [PASS] ActionScalingWrapper OK")


def verify_lowpass_filter():
    """Verify LowPassFilterWrapper identity and smoothing behavior."""
    print("Verifying LowPassFilterWrapper...")

    import gymnasium_robotics  # noqa: F401

    env = gym.make("FetchReach-v4")

    # alpha=1.0 should be identity (pass-through)
    wrapped = LowPassFilterWrapper(env, alpha=1.0)
    obs, _ = wrapped.reset(seed=0)
    a1 = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)
    a2 = np.array([-0.5, -0.5, -0.5, 0.0], dtype=np.float32)
    r1 = wrapped.action(a1)
    r2 = wrapped.action(a2)
    assert np.allclose(r1, a1), f"alpha=1.0 should pass through, got {r1}"
    assert np.allclose(r2, a2), f"alpha=1.0 should pass through, got {r2}"
    print("  alpha=1.0 identity: OK")

    # alpha=0.5: a_out = 0.5*a_raw + 0.5*a_prev
    wrapped_half = LowPassFilterWrapper(env, alpha=0.5)
    wrapped_half.reset(seed=0)
    r1 = wrapped_half.action(a1)  # first action: no prev -> pass through
    assert np.allclose(r1, a1), f"First action should pass through, got {r1}"
    r2 = wrapped_half.action(a2)  # 0.5*(-0.5) + 0.5*(0.5) = 0.0
    expected = 0.5 * a2 + 0.5 * a1
    assert np.allclose(r2, expected), f"Expected {expected}, got {r2}"
    print(f"  alpha=0.5 smoothing: {r2[:3]} (expected {expected[:3]}): OK")

    # Reset clears filter state
    wrapped_half.reset(seed=1)
    r3 = wrapped_half.action(a1)
    assert np.allclose(r3, a1), f"After reset, first action should pass through, got {r3}"
    print("  reset clears state: OK")

    env.close()
    print("  [PASS] LowPassFilterWrapper OK")


def verify_proportional_controller():
    """Verify PD controller converges on FetchReach-v4."""
    print("Verifying ProportionalController on FetchReach-v4...")

    import gymnasium_robotics  # noqa: F401

    env = gym.make("FetchReach-v4")
    controller = ProportionalController("FetchReach-v4", kp=10.0)

    obs, info = env.reset(seed=42)
    success = False
    steps = 0
    max_steps = 50

    while steps < max_steps:
        action, _ = controller.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    env.close()

    print(f"  converged: {success} in {steps} steps")
    assert success, f"PD controller failed to reach goal in {max_steps} steps"
    assert steps < 35, f"PD controller too slow: {steps} steps (expected <35)"
    print("  [PASS] ProportionalController OK")


def verify_metrics_finite():
    """Verify compute_controller_metrics returns finite values."""
    print("Verifying compute_controller_metrics...")

    actions = [np.random.randn(4).astype(np.float32) for _ in range(10)]
    grips = [np.random.randn(3).astype(np.float32) for _ in range(11)]
    rewards = [-1.0] * 9 + [0.0]
    successes = [False] * 9 + [True]
    final_a = np.zeros(3, dtype=np.float32)
    final_d = np.ones(3, dtype=np.float32)

    m = compute_controller_metrics(actions, grips, rewards, successes, final_a, final_d)

    assert np.isfinite(m.smoothness), f"smoothness not finite: {m.smoothness}"
    assert np.isfinite(m.peak_action), f"peak_action not finite: {m.peak_action}"
    assert np.isfinite(m.path_length), f"path_length not finite: {m.path_length}"
    assert np.isfinite(m.action_energy), f"action_energy not finite: {m.action_energy}"
    assert np.isfinite(m.final_distance), f"final_distance not finite: {m.final_distance}"
    assert m.success is True, "Expected success=True"
    assert m.time_to_success == 10, f"Expected TTS=10, got {m.time_to_success}"
    assert m.episode_length == 10, f"Expected length=10, got {m.episode_length}"

    print(f"  smoothness: {m.smoothness:.4f}")
    print(f"  peak_action: {m.peak_action:.4f}")
    print(f"  path_length: {m.path_length:.4f}")
    print(f"  action_energy: {m.action_energy:.4f}")
    print(f"  final_distance: {m.final_distance:.4f}")
    print(f"  time_to_success: {m.time_to_success}")
    print("  [PASS] ControllerMetrics OK")


def verify_run_controller_eval():
    """Verify run_controller_eval returns correct structure."""
    print("Verifying run_controller_eval with PD controller...")

    controller = ProportionalController("FetchReach-v4", kp=10.0)
    result = run_controller_eval(
        policy=controller,
        env_id="FetchReach-v4",
        n_episodes=5,
        deterministic=True,
        seed=0,
    )

    assert "aggregate" in result, "Missing 'aggregate' key"
    assert "episodes" in result, "Missing 'episodes' key"
    assert len(result["episodes"]) == 5, f"Expected 5 episodes, got {len(result['episodes'])}"

    agg = result["aggregate"]
    assert agg["success_rate"] >= 0.8, f"PD success rate too low: {agg['success_rate']}"
    assert np.isfinite(agg["smoothness_mean"]), "smoothness_mean not finite"
    assert np.isfinite(agg["path_length_mean"]), "path_length_mean not finite"

    print(f"  success_rate: {agg['success_rate']:.0%}")
    print(f"  smoothness_mean: {agg['smoothness_mean']:.4f}")
    print(f"  path_length_mean: {agg['path_length_mean']:.4f}")
    print(f"  peak_action_mean: {agg['peak_action_mean']:.4f}")
    print("  [PASS] run_controller_eval OK")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("Action Interface Lab -- Verification")
    print("=" * 60)

    verify_action_scaling()
    print()
    verify_lowpass_filter()
    print()
    verify_metrics_finite()
    print()
    verify_proportional_controller()
    print()
    verify_run_controller_eval()

    print()
    print("=" * 60)
    print("[ALL PASS] Action interface lab verified")
    print("=" * 60)


# =============================================================================
# Demo
# =============================================================================

def run_demo():
    """Demonstrate wrappers on FetchReach-v4 with a PD controller."""
    import gymnasium_robotics  # noqa: F401

    print("=" * 60)
    print("Action Interface Lab -- Demo")
    print("=" * 60)

    controller = ProportionalController("FetchReach-v4", kp=10.0)

    configs = [
        ("No wrapper", []),
        ("Scale=0.5", [(ActionScalingWrapper, {"scale": 0.5})]),
        ("Scale=2.0", [(ActionScalingWrapper, {"scale": 2.0})]),
        ("LPF alpha=0.4", [(LowPassFilterWrapper, {"alpha": 0.4})]),
        ("LPF alpha=0.8", [(LowPassFilterWrapper, {"alpha": 0.8})]),
    ]

    print(f"\n{'Config':<18} | {'Success':>8} | {'Smooth':>8} | {'Peak |a|':>9} | {'Path (m)':>9} | {'TTS':>6}")
    print("-" * 70)

    for label, wrappers in configs:
        result = run_controller_eval(
            policy=controller,
            env_id="FetchReach-v4",
            n_episodes=20,
            seed=0,
            wrappers=wrappers,
        )
        agg = result["aggregate"]
        tts = agg["time_to_success_mean"]
        tts_str = f"{tts:.1f}" if tts is not None else "N/A"
        print(
            f"{label:<18} | {agg['success_rate']:>7.0%} | "
            f"{agg['smoothness_mean']:>8.4f} | {agg['peak_action_mean']:>9.3f} | "
            f"{agg['path_length_mean']:>9.4f} | {tts_str:>6}"
        )

    print("\nKey observations:")
    print("  - Scale=0.5 makes the PD controller slower but smoother")
    print("  - Scale=2.0 clips most actions to bounds (peak ~1.0)")
    print("  - Low-pass filter reduces jitter at the cost of responsiveness")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Action Interface Lab Module")
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
