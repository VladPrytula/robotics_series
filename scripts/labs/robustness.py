#!/usr/bin/env python3
"""
Robustness Curves -- Noise Injection and Degradation Analysis

This module provides tools for quantifying policy brittleness under noise.
The core idea: sweep noise levels and measure how success rate degrades,
producing degradation curves with confidence intervals.

Components:
    - NoisyEvalWrapper: gym.Wrapper adding Gaussian noise to obs/actions
    - NoiseSweepResult: Per-noise-level evaluation result
    - run_noise_sweep: Sweep noise levels, return list of results
    - aggregate_across_seeds: Combine per-seed results with CI
    - compute_degradation_summary: Critical sigma, AUC, slope

Usage:
    # Run verification (sanity checks, ~60 seconds)
    python scripts/labs/robustness.py --verify

Key regions exported for tutorials:
    - noisy_eval_wrapper: NoisyEvalWrapper class
    - noise_sweep_result: NoiseSweepResult dataclass
    - run_noise_sweep: Noise level sweep function
    - aggregate_across_seeds: Cross-seed aggregation
    - compute_degradation_summary: Summary statistics
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

# Ensure project root is on sys.path so `from scripts.labs...` imports work
# when this module is run directly (e.g., python scripts/labs/robustness.py --verify)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# Noisy Evaluation Wrapper
# =============================================================================

# --8<-- [start:noisy_eval_wrapper]
class NoisyEvalWrapper(gym.Wrapper):
    """Add Gaussian noise to observations and/or actions during evaluation.

    This wrapper simulates two types of real-world imprecision:
    - Observation noise (sensor noise): Gaussian perturbation of the
      'observation' key only. Goals (desired_goal, achieved_goal) are left
      untouched -- they represent ground truth targets, not sensor readings.
    - Action noise (actuator noise): Gaussian perturbation of actions,
      clipped to the action space bounds so the environment never sees
      out-of-range commands.

    Setting both noise stds to 0.0 produces an identity wrapper.

    Args:
        env: The environment to wrap.
        obs_noise_std: Standard deviation of observation noise.
        act_noise_std: Standard deviation of action noise.
        seed: Random seed for the noise generator (independent of env seed).
    """

    def __init__(
        self,
        env: gym.Env,
        obs_noise_std: float = 0.0,
        act_noise_std: float = 0.0,
        seed: int = 0,
    ):
        super().__init__(env)
        self.obs_noise_std = obs_noise_std
        self.act_noise_std = act_noise_std
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._add_obs_noise(obs), info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        if self.act_noise_std > 0.0:
            noise = self._rng.normal(0.0, self.act_noise_std, size=action.shape)
            action = action + noise.astype(action.dtype)
            action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_obs_noise(obs), reward, terminated, truncated, info

    def _add_obs_noise(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Add noise to 'observation' key only -- goals are ground truth."""
        if self.obs_noise_std <= 0.0:
            return obs
        noisy_obs = {}
        for key, val in obs.items():
            if key == "observation":
                noise = self._rng.normal(0.0, self.obs_noise_std, size=val.shape)
                noisy_obs[key] = val + noise.astype(val.dtype)
            else:
                noisy_obs[key] = val
        return noisy_obs
# --8<-- [end:noisy_eval_wrapper]


# =============================================================================
# Noise Sweep Result
# =============================================================================

# --8<-- [start:noise_sweep_result]
@dataclass
class NoiseSweepResult:
    """Result from evaluating a policy at one noise level.

    Stores the noise configuration, aggregate metrics from
    run_controller_eval, and per-episode success flags for
    cross-seed confidence interval computation.
    """
    noise_type: str                    # "obs" or "act"
    noise_std: float                   # sigma used for this level
    env_id: str = ""
    seed: int = 0
    n_episodes: int = 0

    # Aggregate metrics (from run_controller_eval)
    success_rate: float = 0.0
    return_mean: float = 0.0
    ep_len_mean: float = 0.0
    smoothness_mean: float = 0.0
    peak_action_mean: float = 0.0
    path_length_mean: float = 0.0
    action_energy_mean: float = 0.0
    time_to_success_mean: float | None = None
    final_distance_mean: float = 0.0

    # Per-episode data for cross-seed CI
    episode_successes: list[bool] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "noise_type": self.noise_type,
            "noise_std": self.noise_std,
            "env_id": self.env_id,
            "seed": self.seed,
            "n_episodes": self.n_episodes,
            "success_rate": self.success_rate,
            "return_mean": self.return_mean,
            "ep_len_mean": self.ep_len_mean,
            "smoothness_mean": self.smoothness_mean,
            "peak_action_mean": self.peak_action_mean,
            "path_length_mean": self.path_length_mean,
            "action_energy_mean": self.action_energy_mean,
            "time_to_success_mean": self.time_to_success_mean,
            "final_distance_mean": self.final_distance_mean,
            "episode_successes": self.episode_successes,
        }
# --8<-- [end:noise_sweep_result]


# =============================================================================
# Noise Sweep
# =============================================================================

# --8<-- [start:run_noise_sweep]
def run_noise_sweep(
    policy: Any,
    env_id: str,
    noise_type: str,
    noise_levels: list[float],
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 0,
    extra_wrappers: list[tuple[type, dict[str, Any]]] | None = None,
) -> list[NoiseSweepResult]:
    """Evaluate a policy across a range of noise levels.

    For each sigma in noise_levels, creates a NoisyEvalWrapper with the
    appropriate noise_std and calls run_controller_eval from the action
    interface lab. All 8 controller metrics come for free.

    Args:
        policy: Object with predict(obs, deterministic) -> (action, _).
        env_id: Gymnasium environment ID.
        noise_type: "obs" for observation noise, "act" for action noise.
        noise_levels: List of noise standard deviations to sweep.
        n_episodes: Episodes per noise level.
        deterministic: Whether to use deterministic actions.
        seed: Base seed for evaluation.
        extra_wrappers: Additional wrappers to apply after NoisyEvalWrapper.
            Useful for mitigation experiments (e.g., LowPassFilterWrapper).

    Returns:
        List of NoiseSweepResult, one per noise level.
    """
    from scripts.labs.action_interface import run_controller_eval

    if noise_type not in ("obs", "act"):
        raise ValueError(f"noise_type must be 'obs' or 'act', got '{noise_type}'")

    results: list[NoiseSweepResult] = []

    for sigma in noise_levels:
        # Build wrapper kwargs based on noise type
        wrapper_kwargs: dict[str, Any] = {"seed": seed}
        if noise_type == "obs":
            wrapper_kwargs["obs_noise_std"] = sigma
            wrapper_kwargs["act_noise_std"] = 0.0
        else:
            wrapper_kwargs["obs_noise_std"] = 0.0
            wrapper_kwargs["act_noise_std"] = sigma

        wrappers: list[tuple[type, dict[str, Any]]] = [
            (NoisyEvalWrapper, wrapper_kwargs),
        ]
        if extra_wrappers:
            wrappers.extend(extra_wrappers)

        eval_result = run_controller_eval(
            policy=policy,
            env_id=env_id,
            n_episodes=n_episodes,
            deterministic=deterministic,
            seed=seed,
            wrappers=wrappers,
        )

        agg = eval_result["aggregate"]
        episode_successes = [ep["success"] for ep in eval_result["episodes"]]

        results.append(NoiseSweepResult(
            noise_type=noise_type,
            noise_std=sigma,
            env_id=env_id,
            seed=seed,
            n_episodes=n_episodes,
            success_rate=agg["success_rate"],
            return_mean=agg["return_mean"],
            ep_len_mean=agg["ep_len_mean"],
            smoothness_mean=agg["smoothness_mean"],
            peak_action_mean=agg["peak_action_mean"],
            path_length_mean=agg["path_length_mean"],
            action_energy_mean=agg["action_energy_mean"],
            time_to_success_mean=agg["time_to_success_mean"],
            final_distance_mean=agg["final_distance_mean"],
            episode_successes=episode_successes,
        ))

    return results
# --8<-- [end:run_noise_sweep]


# =============================================================================
# Cross-Seed Aggregation
# =============================================================================

def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, and 95% CI for a list of values."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
        ci95 = 1.96 * std / math.sqrt(n) if n >= 30 else 2.0 * std / math.sqrt(n)
    else:
        std = 0.0
        ci95 = 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "n": n}


# --8<-- [start:aggregate_across_seeds]
def aggregate_across_seeds(
    seed_results: list[list[NoiseSweepResult]],
) -> list[dict[str, Any]]:
    """Aggregate sweep results across multiple seeds with confidence intervals.

    Takes a list of per-seed sweep results (each is a list of NoiseSweepResult
    with the same noise levels) and computes mean/std/ci95 for each metric
    at each noise level.

    Args:
        seed_results: List of per-seed results. Each element is a list of
            NoiseSweepResult from run_noise_sweep, with matching noise levels.

    Returns:
        List of dicts, one per noise level, with aggregated statistics.
    """
    if not seed_results:
        return []

    # Use the first seed's noise levels as reference
    n_levels = len(seed_results[0])
    aggregated: list[dict[str, Any]] = []

    for level_idx in range(n_levels):
        ref = seed_results[0][level_idx]
        level_results = [sr[level_idx] for sr in seed_results]

        # Collect per-seed values for each metric
        sr_values = [r.success_rate for r in level_results]
        return_values = [r.return_mean for r in level_results]
        ep_len_values = [r.ep_len_mean for r in level_results]
        smooth_values = [r.smoothness_mean for r in level_results]
        peak_values = [r.peak_action_mean for r in level_results]
        path_values = [r.path_length_mean for r in level_results]
        energy_values = [r.action_energy_mean for r in level_results]
        dist_values = [r.final_distance_mean for r in level_results]
        tts_values = [
            r.time_to_success_mean for r in level_results
            if r.time_to_success_mean is not None
        ]

        aggregated.append({
            "noise_type": ref.noise_type,
            "noise_std": ref.noise_std,
            "n_seeds": len(seed_results),
            "success_rate": _compute_stats(sr_values),
            "return_mean": _compute_stats(return_values),
            "ep_len_mean": _compute_stats(ep_len_values),
            "smoothness_mean": _compute_stats(smooth_values),
            "peak_action_mean": _compute_stats(peak_values),
            "path_length_mean": _compute_stats(path_values),
            "action_energy_mean": _compute_stats(energy_values),
            "final_distance_mean": _compute_stats(dist_values),
            "time_to_success_mean": _compute_stats(tts_values) if tts_values else None,
        })

    return aggregated
# --8<-- [end:aggregate_across_seeds]


# =============================================================================
# Degradation Summary
# =============================================================================

# --8<-- [start:compute_degradation_summary]
def compute_degradation_summary(
    aggregated: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute summary statistics from aggregated degradation curves.

    Three key robustness metrics:
    - critical_sigma: First noise level where mean SR drops below 50%.
      If SR never drops below 50%, returns None (policy is highly robust).
    - degradation_slope: Linear regression slope of SR vs sigma.
      More negative = faster degradation = more brittle.
    - robustness_auc: Area under the SR-vs-sigma curve (trapezoidal rule).
      Higher AUC = more robust. Normalized by the sigma range.

    Args:
        aggregated: Output of aggregate_across_seeds.

    Returns:
        Dict with critical_sigma, degradation_slope, robustness_auc.
    """
    if not aggregated:
        return {
            "critical_sigma": None,
            "degradation_slope": 0.0,
            "robustness_auc": 0.0,
        }

    sigmas = [a["noise_std"] for a in aggregated]
    sr_means = [a["success_rate"]["mean"] for a in aggregated]

    # Critical sigma: first sigma where SR < 50%
    critical_sigma: float | None = None
    for sigma, sr in zip(sigmas, sr_means):
        if sr < 0.50:
            critical_sigma = sigma
            break

    # Degradation slope: linear regression SR = m * sigma + b
    n = len(sigmas)
    if n >= 2:
        x_arr = np.array(sigmas, dtype=np.float64)
        y_arr = np.array(sr_means, dtype=np.float64)
        x_mean = x_arr.mean()
        y_mean = y_arr.mean()
        numerator = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)))
        denominator = float(np.sum((x_arr - x_mean) ** 2))
        degradation_slope = numerator / denominator if denominator > 1e-12 else 0.0
    else:
        degradation_slope = 0.0

    # Robustness AUC: trapezoidal integration of SR over sigma
    robustness_auc = 0.0
    for i in range(1, n):
        robustness_auc += 0.5 * (sr_means[i - 1] + sr_means[i]) * (sigmas[i] - sigmas[i - 1])

    return {
        "critical_sigma": critical_sigma,
        "degradation_slope": round(degradation_slope, 4),
        "robustness_auc": round(robustness_auc, 4),
    }
# --8<-- [end:compute_degradation_summary]


# =============================================================================
# Verification
# =============================================================================

def verify_noisy_wrapper():
    """Verify NoisyEvalWrapper preserves goals and adds correct noise."""
    print("Verifying NoisyEvalWrapper...")

    import gymnasium_robotics  # noqa: F401

    # --- Zero noise should be identity ---
    env_zero = gym.make("FetchReach-v4")
    wrapped_zero = NoisyEvalWrapper(env_zero, obs_noise_std=0.0, act_noise_std=0.0, seed=0)
    # gym.Wrapper wraps env_zero, so both reset the same underlying sim
    obs_wrap, _ = wrapped_zero.reset(seed=42)
    wrapped_zero.close()

    env_ref = gym.make("FetchReach-v4")
    obs_raw, _ = env_ref.reset(seed=42)
    env_ref.close()

    for key in obs_raw:
        assert np.allclose(obs_raw[key], obs_wrap[key]), (
            f"Zero noise changed '{key}': {obs_raw[key]} vs {obs_wrap[key]}"
        )
    print("  zero noise = identity: OK")

    # --- Obs noise changes observations but not goals ---
    env_noisy = gym.make("FetchReach-v4")
    wrapped_noisy = NoisyEvalWrapper(env_noisy, obs_noise_std=0.1, act_noise_std=0.0, seed=0)
    obs_noisy, _ = wrapped_noisy.reset(seed=42)

    assert not np.allclose(obs_noisy["observation"], obs_raw["observation"]), (
        "Obs noise did not change 'observation'"
    )
    assert np.allclose(obs_noisy["desired_goal"], obs_raw["desired_goal"]), (
        "Obs noise changed 'desired_goal'"
    )
    assert np.allclose(obs_noisy["achieved_goal"], obs_raw["achieved_goal"]), (
        "Obs noise changed 'achieved_goal'"
    )
    wrapped_noisy.close()
    print("  obs noise: observation changed, goals preserved: OK")

    # --- Action noise clips to bounds ---
    env_act = gym.make("FetchReach-v4")
    wrapped_act = NoisyEvalWrapper(env_act, obs_noise_std=0.0, act_noise_std=10.0, seed=0)
    obs, _ = wrapped_act.reset(seed=42)
    big_action = np.ones(4, dtype=np.float32)
    obs_after, _, _, _, _ = wrapped_act.step(big_action)
    wrapped_act.close()
    print("  action noise with large sigma: env stable: OK")

    # --- Wrapper is a proper gym.Wrapper ---
    env_iface = gym.make("FetchReach-v4")
    wrapped_iface = NoisyEvalWrapper(env_iface)
    assert isinstance(wrapped_iface, gym.Wrapper), "Not a gym.Wrapper subclass"
    assert hasattr(wrapped_iface, "observation_space"), "Missing observation_space"
    assert hasattr(wrapped_iface, "action_space"), "Missing action_space"
    wrapped_iface.close()
    print("  gym.Wrapper interface: OK")

    print("  [PASS] NoisyEvalWrapper OK")


def verify_sweep():
    """Verify run_noise_sweep returns correct structure and ordering."""
    print("Verifying run_noise_sweep...")

    from scripts.labs.action_interface import ProportionalController

    controller = ProportionalController("FetchReach-v4", kp=10.0)
    levels = [0.0, 0.05, 0.1]
    results = run_noise_sweep(
        policy=controller,
        env_id="FetchReach-v4",
        noise_type="obs",
        noise_levels=levels,
        n_episodes=10,
        seed=0,
    )

    # Correct number of results
    assert len(results) == len(levels), (
        f"Expected {len(levels)} results, got {len(results)}"
    )
    print(f"  result count: {len(results)} (expected {len(levels)}): OK")

    # Each result has matching noise_std
    for r, sigma in zip(results, levels):
        assert r.noise_std == sigma, f"noise_std mismatch: {r.noise_std} vs {sigma}"
        assert r.noise_type == "obs", f"noise_type mismatch: {r.noise_type}"
        assert len(r.episode_successes) == 10, (
            f"Expected 10 episode_successes, got {len(r.episode_successes)}"
        )

    # sigma=0 should have highest (or tied highest) SR
    assert results[0].success_rate >= results[-1].success_rate, (
        f"sigma=0 SR ({results[0].success_rate}) < sigma={levels[-1]} SR ({results[-1].success_rate})"
    )

    for r in results:
        print(f"  sigma={r.noise_std:.3f}: SR={r.success_rate:.0%}")

    print("  [PASS] run_noise_sweep OK")


def verify_aggregation():
    """Verify aggregate_across_seeds computes correct statistics."""
    print("Verifying aggregate_across_seeds...")

    # Create two identical seed results -- CI should be 0
    r1 = NoiseSweepResult(
        noise_type="obs", noise_std=0.0, success_rate=1.0,
        return_mean=-8.0, smoothness_mean=0.01, peak_action_mean=0.9,
        path_length_mean=0.15, action_energy_mean=25.0,
        final_distance_mean=0.02, time_to_success_mean=9.0,
        episode_successes=[True] * 10,
    )
    r2 = NoiseSweepResult(
        noise_type="obs", noise_std=0.1, success_rate=0.7,
        return_mean=-15.0, smoothness_mean=0.05, peak_action_mean=0.95,
        path_length_mean=0.20, action_energy_mean=30.0,
        final_distance_mean=0.05, time_to_success_mean=12.0,
        episode_successes=[True] * 7 + [False] * 3,
    )

    # Two identical seeds
    seed_results = [[r1, r2], [r1, r2]]
    agg = aggregate_across_seeds(seed_results)

    assert len(agg) == 2, f"Expected 2 levels, got {len(agg)}"

    # Identical seeds -> std = 0, ci95 = 0
    sr_stats = agg[0]["success_rate"]
    assert sr_stats["mean"] == 1.0, f"Expected SR mean 1.0, got {sr_stats['mean']}"
    assert sr_stats["std"] == 0.0, f"Expected SR std 0.0, got {sr_stats['std']}"
    print(f"  identical seeds -> mean={sr_stats['mean']}, std={sr_stats['std']}: OK")

    # Different seeds -> nonzero std
    r1_alt = NoiseSweepResult(
        noise_type="obs", noise_std=0.1, success_rate=0.5,
        return_mean=-20.0, smoothness_mean=0.08, peak_action_mean=0.98,
        path_length_mean=0.25, action_energy_mean=35.0,
        final_distance_mean=0.08, time_to_success_mean=15.0,
        episode_successes=[True] * 5 + [False] * 5,
    )
    seed_results_diff = [[r1, r2], [r1, r1_alt]]
    agg_diff = aggregate_across_seeds(seed_results_diff)

    sr_stats_diff = agg_diff[1]["success_rate"]
    assert sr_stats_diff["std"] > 0.0, (
        f"Different seeds should have nonzero std: {sr_stats_diff['std']}"
    )
    print(f"  different seeds -> std={sr_stats_diff['std']:.4f}: OK")

    print("  [PASS] aggregate_across_seeds OK")


def verify_degradation_summary():
    """Verify compute_degradation_summary computes correct metrics."""
    print("Verifying compute_degradation_summary...")

    # Create aggregated data with known properties
    aggregated = [
        {"noise_std": 0.0, "success_rate": {"mean": 1.0}},
        {"noise_std": 0.05, "success_rate": {"mean": 0.8}},
        {"noise_std": 0.1, "success_rate": {"mean": 0.4}},
    ]

    summary = compute_degradation_summary(aggregated)

    # Critical sigma: SR drops below 50% at sigma=0.1
    assert summary["critical_sigma"] == 0.1, (
        f"Expected critical_sigma=0.1, got {summary['critical_sigma']}"
    )
    print(f"  critical_sigma: {summary['critical_sigma']}: OK")

    # Slope should be negative (SR decreases with noise)
    assert summary["degradation_slope"] < 0, (
        f"Expected negative slope, got {summary['degradation_slope']}"
    )
    print(f"  degradation_slope: {summary['degradation_slope']}: OK")

    # AUC should be positive
    assert summary["robustness_auc"] > 0, (
        f"Expected positive AUC, got {summary['robustness_auc']}"
    )
    # Manual trapezoidal: 0.5*(1.0+0.8)*0.05 + 0.5*(0.8+0.4)*0.05 = 0.045 + 0.030 = 0.075
    expected_auc = 0.075
    assert abs(summary["robustness_auc"] - expected_auc) < 0.001, (
        f"Expected AUC ~{expected_auc}, got {summary['robustness_auc']}"
    )
    print(f"  robustness_auc: {summary['robustness_auc']} (expected ~{expected_auc}): OK")

    # Edge case: SR never drops below 50%
    agg_robust = [
        {"noise_std": 0.0, "success_rate": {"mean": 1.0}},
        {"noise_std": 0.1, "success_rate": {"mean": 0.9}},
    ]
    summary_robust = compute_degradation_summary(agg_robust)
    assert summary_robust["critical_sigma"] is None, (
        f"Expected None critical_sigma for robust policy, got {summary_robust['critical_sigma']}"
    )
    print(f"  robust policy -> critical_sigma=None: OK")

    print("  [PASS] compute_degradation_summary OK")


def run_verification():
    """Run all verification checks."""
    print("=" * 60)
    print("Robustness Lab -- Verification")
    print("=" * 60)

    verify_noisy_wrapper()
    print()
    verify_aggregation()
    print()
    verify_degradation_summary()
    print()
    verify_sweep()

    print()
    print("=" * 60)
    print("[ALL PASS] Robustness lab verified")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robustness Lab Module")
    parser.add_argument("--verify", action="store_true", help="Run verification checks")
    args = parser.parse_args()

    if args.verify:
        run_verification()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
