#!/usr/bin/env python3
"""
Environment Anatomy -- Pedagogical Inspection Lab

This module provides from-scratch inspection components for Fetch
environment observations, actions, rewards, and goals. Each component
is a self-contained function suitable for inclusion as a book listing.

Usage:
    # Run verification (all 7 component checks, < 2 min CPU)
    python scripts/labs/env_anatomy.py --verify

    # Bridge: manual reward vs compute_reward vs env.step() reward
    python scripts/labs/env_anatomy.py --bridge

Key regions exported for tutorials:
    - obs_inspector: inspect obs dict structure, shapes, dtypes, bounds
    - action_explorer: step with axis-aligned actions, measure displacement
    - goal_space: sample goals via reset, check bounds, verify phi
    - dense_reward_check: verify dense reward = -||ag-dg||
    - sparse_reward_check: verify sparse reward = 0/-1 threshold
    - relabel_check: call compute_reward with arbitrary goals
    - cross_env_compare: compare obs dims across Reach/Push/PickAndPlace
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np


def _make_env(env_id: str, seed: int = 0):
    """Create a Fetch environment with rendering disabled."""
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401 -- registers Fetch envs

    os.environ.setdefault("MUJOCO_GL", "disable")
    env = gym.make(env_id)
    return env


# =============================================================================
# Component 1: Observation Dictionary Inspector
# =============================================================================

# --8<-- [start:obs_inspector]
def obs_inspector(env_id: str = "FetchReachDense-v4", seed: int = 0) -> dict:
    """Inspect obs dict: keys, shapes, dtypes, workspace bounds."""
    env = _make_env(env_id, seed)
    obs, info = env.reset(seed=seed)
    summary: dict[str, Any] = {}
    for key in ["observation", "achieved_goal", "desired_goal"]:
        arr = np.asarray(obs[key])
        summary[key] = {
            "shape": arr.shape, "dtype": str(arr.dtype),
            "min": float(arr.min()), "max": float(arr.max()),
            "finite": bool(np.all(np.isfinite(arr))),
        }
    # Workspace bounds (m): x:[1.1,1.5], y:[0.5,1.0], z:[0.35,0.75]
    dg = np.asarray(obs["desired_goal"])
    summary["desired_goal_in_workspace"] = bool(
        1.1 <= dg[0] <= 1.5 and 0.5 <= dg[1] <= 1.0 and 0.35 <= dg[2] <= 0.75
    )
    env.close()
    return summary
# --8<-- [end:obs_inspector]


# =============================================================================
# Component 2: Action Space Explorer
# =============================================================================

# --8<-- [start:action_explorer]
def action_explorer(env_id: str = "FetchReachDense-v4", seed: int = 0) -> dict:
    """Step with axis-aligned actions, measure end-effector displacement."""
    env = _make_env(env_id, seed)
    results: dict[str, Any] = {
        "action_shape": env.action_space.shape,
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "displacements": {},
    }
    axis_names = ["x", "y", "z"]
    for axis_idx in range(3):
        obs, _ = env.reset(seed=seed)
        grip_before = np.asarray(obs["achieved_goal"]).copy()
        action = np.zeros(4, dtype=np.float32)
        action[axis_idx] = 1.0
        obs, _, _, _, _ = env.step(action)
        grip_after = np.asarray(obs["achieved_goal"])
        disp = grip_after - grip_before
        results["displacements"][axis_names[axis_idx]] = {
            "action": action.tolist(), "displacement": disp.tolist(),
            "moved_positive": bool(disp[axis_idx] > 0),
        }
    env.close()
    return results
# --8<-- [end:action_explorer]


# =============================================================================
# Component 3: Goal Space
# =============================================================================

# --8<-- [start:goal_space]
def goal_space(
    env_id: str = "FetchReachDense-v4", n_resets: int = 100, seed: int = 0,
) -> dict:
    """Sample goals via reset; check bounds; verify phi(s) = obs[:3]."""
    env = _make_env(env_id, seed)
    desired_goals, phi_matches = [], []
    for i in range(n_resets):
        obs, _ = env.reset(seed=seed + i)
        desired_goals.append(np.asarray(obs["desired_goal"]))
        ag = np.asarray(obs["achieved_goal"])
        obs_vec = np.asarray(obs["observation"])
        # phi(s) = grip_pos = obs["observation"][:3] for FetchReach
        phi_matches.append(bool(np.allclose(ag, obs_vec[:3])))
    dg_arr = np.stack(desired_goals)
    env.close()
    return {
        "n_resets": n_resets,
        "goal_dim": int(dg_arr.shape[1]),
        "desired_goal_bounds": {
            "min": dg_arr.min(axis=0).tolist(),
            "max": dg_arr.max(axis=0).tolist(),
        },
        "all_phi_match": all(phi_matches),
    }
# --8<-- [end:goal_space]


# =============================================================================
# Component 4: Dense Reward Check
# =============================================================================

# --8<-- [start:dense_reward_check]
def dense_reward_check(
    env_id: str = "FetchReachDense-v4", n_steps: int = 100,
    seed: int = 0, atol: float = 1e-10,
) -> dict:
    """Verify R = -||ag - dg|| matches step_reward and compute_reward."""
    env = _make_env(env_id, seed)
    obs, info = env.reset(seed=seed)
    mismatches = 0
    for t in range(n_steps):
        action = env.action_space.sample()
        obs, step_reward, terminated, truncated, info = env.step(action)
        ag = np.asarray(obs["achieved_goal"])
        dg = np.asarray(obs["desired_goal"])
        manual = -float(np.linalg.norm(ag - dg))
        cr = float(env.unwrapped.compute_reward(ag, dg, info))
        step_r = float(step_reward)
        if abs(manual - cr) > atol or abs(manual - step_r) > atol:
            mismatches += 1
        if terminated or truncated:
            obs, info = env.reset(seed=seed + t + 1)
    env.close()
    return {"n_steps": n_steps, "mismatches": mismatches, "atol": atol}
# --8<-- [end:dense_reward_check]


# =============================================================================
# Component 5: Sparse Reward Check
# =============================================================================

# --8<-- [start:sparse_reward_check]
def sparse_reward_check(
    env_id: str = "FetchReach-v4", n_steps: int = 100,
    seed: int = 0, atol: float = 1e-10,
) -> dict:
    """Verify R = 0 if ||ag-dg|| <= eps else -1; check threshold."""
    env = _make_env(env_id, seed)
    eps = float(env.unwrapped.distance_threshold)
    obs, info = env.reset(seed=seed)
    mismatches, non_binary = 0, 0
    for t in range(n_steps):
        action = env.action_space.sample()
        obs, step_reward, terminated, truncated, info = env.step(action)
        ag = np.asarray(obs["achieved_goal"])
        dg = np.asarray(obs["desired_goal"])
        dist = float(np.linalg.norm(ag - dg))
        expected = 0.0 if dist <= eps else -1.0
        step_r, cr_r = float(step_reward), float(
            env.unwrapped.compute_reward(ag, dg, info))
        if step_r not in (0.0, -1.0):
            non_binary += 1
        if abs(step_r - expected) > atol or abs(cr_r - expected) > atol:
            mismatches += 1
        if terminated or truncated:
            obs, info = env.reset(seed=seed + t + 1)
    env.close()
    return {"n_steps": n_steps, "threshold": eps,
            "mismatches": mismatches, "non_binary": non_binary}
# --8<-- [end:sparse_reward_check]


# =============================================================================
# Component 6: Goal Relabeling Check
# =============================================================================

# --8<-- [start:relabel_check]
def relabel_check(
    env_id: str = "FetchReachDense-v4", n_goals: int = 10,
    seed: int = 0, atol: float = 1e-10,
) -> dict:
    """Call compute_reward with arbitrary goals (HER simulation)."""
    env = _make_env(env_id, seed)
    obs, info = env.reset(seed=seed)
    goal_sp = env.observation_space.spaces["desired_goal"]
    # Take one step to get a non-trivial achieved_goal
    obs, _, _, _, info = env.step(env.action_space.sample())
    ag = np.asarray(obs["achieved_goal"])
    reward_type = getattr(env.unwrapped, "reward_type", "dense")
    eps = float(env.unwrapped.distance_threshold)
    mismatches = 0
    for _ in range(n_goals):
        random_goal = goal_sp.sample()
        cr = float(env.unwrapped.compute_reward(ag, random_goal, info))
        dist = float(np.linalg.norm(ag - random_goal))
        expected = -dist if reward_type == "dense" else (
            0.0 if dist <= eps else -1.0)
        if abs(cr - expected) > atol:
            mismatches += 1
    env.close()
    return {"n_goals": n_goals, "mismatches": mismatches, "total": n_goals}
# --8<-- [end:relabel_check]


# =============================================================================
# Component 7: Cross-Environment Comparison
# =============================================================================

# --8<-- [start:cross_env_compare]
def cross_env_compare(seed: int = 0) -> dict:
    """Compare obs dims across FetchReach, FetchPush, FetchPickAndPlace.

    Reach has 10D obs (no object); Push and PickAndPlace have 25D.
    For Push, achieved_goal is the object position, not the gripper.
    """
    env_configs = {
        "FetchReach-v4": {"expected_obs_dim": 10, "ag_is_grip": True},
        "FetchPush-v4": {"expected_obs_dim": 25, "ag_is_grip": False},
        "FetchPickAndPlace-v4": {"expected_obs_dim": 25, "ag_is_grip": False},
    }
    results: dict[str, Any] = {}

    for env_id, cfg in env_configs.items():
        env = _make_env(env_id, seed)
        obs, _ = env.reset(seed=seed)
        obs_vec = np.asarray(obs["observation"])
        ag = np.asarray(obs["achieved_goal"])
        grip_pos = obs_vec[:3]
        ag_matches_grip = bool(np.allclose(ag, grip_pos))
        results[env_id] = {
            "obs_dim": obs_vec.shape[0],
            "expected_dim": cfg["expected_obs_dim"],
            "dim_ok": obs_vec.shape[0] == cfg["expected_obs_dim"],
            "ag_matches_grip": ag_matches_grip,
        }
        env.close()

    return results
# --8<-- [end:cross_env_compare]


# =============================================================================
# Verification Mode (--verify)
# =============================================================================

def run_verification() -> None:
    """Run all 7 component checks. Target: < 2 min on CPU."""
    print("=" * 60)
    print("Environment Anatomy -- Verification")
    print("=" * 60)

    # 1. Observation inspector
    print("\n1. Observation dictionary inspector...")
    result = obs_inspector()
    obs_shape = result["observation"]["shape"]
    ag_shape = result["achieved_goal"]["shape"]
    dg_shape = result["desired_goal"]["shape"]
    assert obs_shape == (10,), f"obs shape {obs_shape} != (10,)"
    assert ag_shape == (3,), f"ag shape {ag_shape} != (3,)"
    assert dg_shape == (3,), f"dg shape {dg_shape} != (3,)"
    assert result["observation"]["finite"], "obs contains non-finite values"
    assert result["desired_goal_in_workspace"], "desired_goal outside workspace"
    print(f"  obs={obs_shape}, ag={ag_shape}, dg={dg_shape}")
    print(f"  desired_goal in workspace: {result['desired_goal_in_workspace']}")
    print("  [OK]")

    # 2. Action space explorer
    print("\n2. Action space explorer...")
    result = action_explorer()
    assert result["action_shape"] == (4,), f"action shape {result['action_shape']}"
    assert result["action_low"] == [-1.0, -1.0, -1.0, -1.0], "bad low bounds"
    assert result["action_high"] == [1.0, 1.0, 1.0, 1.0], "bad high bounds"
    for axis in ["x", "y", "z"]:
        disp = result["displacements"][axis]
        assert disp["moved_positive"], f"action along +{axis} did not move +{axis}"
    print(f"  action_shape={result['action_shape']}, bounds=[-1, 1]")
    for axis in ["x", "y", "z"]:
        d = result["displacements"][axis]["displacement"]
        print(f"  +{axis} displacement: [{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}]")
    print("  [OK]")

    # 3. Goal space
    print("\n3. Goal space (100 resets)...")
    result = goal_space(n_resets=100)
    assert result["goal_dim"] == 3, f"goal dim {result['goal_dim']} != 3"
    assert result["all_phi_match"], "phi(s) != obs[:3] for some resets"
    bounds = result["desired_goal_bounds"]
    print(f"  goal_dim={result['goal_dim']}, phi matches obs[:3]: {result['all_phi_match']}")
    print(f"  desired_goal range: x=[{bounds['min'][0]:.3f},{bounds['max'][0]:.3f}]"
          f"  y=[{bounds['min'][1]:.3f},{bounds['max'][1]:.3f}]"
          f"  z=[{bounds['min'][2]:.3f},{bounds['max'][2]:.3f}]")
    print("  [OK]")

    # 4. Dense reward check
    print("\n4. Dense reward check (100 steps)...")
    result = dense_reward_check(n_steps=100)
    assert result["mismatches"] == 0, f"dense reward mismatches: {result['mismatches']}"
    print(f"  steps={result['n_steps']}, mismatches={result['mismatches']}, atol={result['atol']}")
    print("  [OK]")

    # 5. Sparse reward check
    print("\n5. Sparse reward check (100 steps)...")
    result = sparse_reward_check(n_steps=100)
    assert result["mismatches"] == 0, f"sparse reward mismatches: {result['mismatches']}"
    assert result["non_binary"] == 0, f"non-binary sparse rewards: {result['non_binary']}"
    print(f"  steps={result['n_steps']}, threshold={result['threshold']}")
    print(f"  mismatches={result['mismatches']}, non_binary={result['non_binary']}")
    print("  [OK]")

    # 6. Goal relabeling check
    print("\n6. Goal relabeling check (10 random goals)...")
    result = relabel_check(n_goals=10)
    assert result["mismatches"] == 0, f"relabel mismatches: {result['mismatches']}"
    print(f"  n_goals={result['n_goals']}, mismatches={result['mismatches']}")
    print("  [OK]")

    # 7. Cross-environment comparison
    print("\n7. Cross-environment comparison...")
    result = cross_env_compare()
    for env_id, data in result.items():
        assert data["dim_ok"], f"{env_id}: obs_dim={data['obs_dim']} != {data['expected_dim']}"
        print(f"  {env_id}: obs_dim={data['obs_dim']}, ag_matches_grip={data['ag_matches_grip']}")
    # For FetchPush, achieved_goal should NOT be the gripper position
    assert not result["FetchPush-v4"]["ag_matches_grip"], \
        "FetchPush ag should be object pos, not grip pos"
    print("  [OK]")

    print()
    print("=" * 60)
    print("[ALL OK] All 7 environment anatomy checks passed")
    print("=" * 60)


# =============================================================================
# Bridge Mode (--bridge)
# =============================================================================

def run_bridge() -> None:
    """Bridging proof: manual reward vs compute_reward vs env.step() reward.

    100 steps on FetchReachDense-v4 with seed=42. For each step:
      - Compare manual_reward = -||ag - dg||
      - Compare cr_reward = compute_reward(ag, dg, info)
      - Compare step_reward from env.step()
    Plus 5 randomly sampled alternative goals per step (HER simulation).
    """
    print("=" * 60)
    print("Environment Anatomy -- Bridging Proof")
    print("=" * 60)

    env_id = "FetchReachDense-v4"
    seed = 42
    n_steps = 100
    n_relabel_goals = 5
    atol = 1e-10

    env = _make_env(env_id, seed)
    obs, info = env.reset(seed=seed)
    goal_space_box = env.observation_space.spaces["desired_goal"]

    step_mismatches = 0
    relabel_mismatches = 0
    total_relabel_checks = 0

    print(f"\nEnv: {env_id}, seed={seed}, steps={n_steps}")
    print(f"Relabeled goals per step: {n_relabel_goals}")
    print(f"Tolerance: atol={atol}")
    print()

    for t in range(n_steps):
        action = env.action_space.sample()
        obs, step_reward, terminated, truncated, info = env.step(action)
        ag = np.asarray(obs["achieved_goal"])
        dg = np.asarray(obs["desired_goal"])

        # Three-way comparison
        manual_reward = -float(np.linalg.norm(ag - dg))
        cr_reward = float(env.unwrapped.compute_reward(ag, dg, info))
        step_r = float(step_reward)

        if (abs(manual_reward - step_r) > atol
                or abs(cr_reward - step_r) > atol
                or abs(manual_reward - cr_reward) > atol):
            step_mismatches += 1
            print(f"  [MISMATCH] step {t}: manual={manual_reward:.12f} "
                  f"cr={cr_reward:.12f} step={step_r:.12f}")

        # HER-style relabeling: 5 random goals per step
        for _ in range(n_relabel_goals):
            random_goal = goal_space_box.sample()
            relabel_cr = float(env.unwrapped.compute_reward(ag, random_goal, info))
            relabel_expected = -float(np.linalg.norm(ag - random_goal))
            if abs(relabel_cr - relabel_expected) > atol:
                relabel_mismatches += 1
            total_relabel_checks += 1

        if terminated or truncated:
            obs, info = env.reset(seed=seed + t + 1)

    env.close()

    # Report
    print(f"Step reward checks:    {n_steps} steps, "
          f"mismatches={step_mismatches}")
    print(f"Relabel reward checks: {total_relabel_checks} checks, "
          f"mismatches={relabel_mismatches}")

    ok = (step_mismatches == 0 and relabel_mismatches == 0)
    print()
    if ok:
        print("[MATCH] All rewards match within atol={:.0e}".format(atol))
        print("  manual_reward == compute_reward == step_reward  (100 steps)")
        print(f"  relabel_reward == -||ag - random_goal||  ({total_relabel_checks} checks)")
    else:
        print(f"[MISMATCH] step_mismatches={step_mismatches}, "
              f"relabel_mismatches={relabel_mismatches}")
        raise SystemExit(1)

    print()
    print("=" * 60)
    print("[BRIDGE OK] Bridging proof passed")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Environment Anatomy -- Pedagogical Inspection Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --verify    Run all 7 component checks (< 2 min CPU)
  --bridge    Bridging proof: manual vs compute_reward vs env.step()
        """,
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run all verification checks",
    )
    parser.add_argument(
        "--bridge", action="store_true",
        help="Run bridging proof (manual reward vs compute_reward vs step)",
    )
    args = parser.parse_args()

    if args.verify:
        run_verification()
    elif args.bridge:
        run_bridge()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
