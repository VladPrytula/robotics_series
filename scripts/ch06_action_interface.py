#!/usr/bin/env python3
"""Chapter 06: Action-Interface Engineering -- Policies as Controllers

This chapter treats learned policies as controllers and evaluates them with
engineering metrics: smoothness, peak effort, path efficiency, and
time-to-success. No new training is needed -- we reuse ch04 checkpoints
and apply eval-time wrappers to study how the action interface affects
movement quality.

Experiments:
    1. Action scaling: sweep scale factors {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
    2. Low-pass filtering: sweep alpha {0.2, 0.4, 0.6, 0.8, 1.0}
    3. PD baseline: proportional controller at Kp {5.0, 10.0, 20.0}
    4. Comparison: table + planning-vs-control decomposition

Usage:
    # Train SAC+HER if checkpoints missing (fallback)
    python scripts/ch06_action_interface.py train --env FetchReach-v4 --seed 0

    # Action scaling sweep
    python scripts/ch06_action_interface.py scaling --seed 0

    # Low-pass filter sweep
    python scripts/ch06_action_interface.py filter --seed 0

    # PD baseline
    python scripts/ch06_action_interface.py baseline

    # Compare all results
    python scripts/ch06_action_interface.py compare

    # Full pipeline
    python scripts/ch06_action_interface.py all --seed 0

    # Include Push environment
    python scripts/ch06_action_interface.py all --seed 0 --include-push
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `from scripts.labs...` imports work
# when this script is invoked as `python scripts/ch06_action_interface.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Experiment configuration for ch06.

    Reuses ch04 checkpoints -- no training hyperparameters needed unless
    fallback training is triggered.
    """
    # Environments
    env: str = "FetchReach-v4"
    include_push: bool = False
    seed: int = 0

    # Evaluation
    n_eval_episodes: int = 100
    deterministic: bool = True

    # Training fallback (only used if checkpoint missing)
    n_envs: int = 8
    total_steps: int = 1_000_000
    device: str = "auto"

    # Action scaling sweep
    scale_factors: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    )

    # Low-pass filter sweep
    filter_alphas: list[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # PD baseline
    pd_kp_values: list[float] = field(
        default_factory=lambda: [5.0, 10.0, 20.0]
    )

    # Paths
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create config from parsed CLI arguments."""
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if hasattr(args, f):
                kwargs[f] = getattr(args, f)
        return cls(**kwargs)


DEFAULT_CONFIG = Config()


# =============================================================================
# Utilities
# =============================================================================

def _gather_versions() -> dict[str, str]:
    """Collect package versions for reproducibility."""
    versions: dict[str, str] = {"python": sys.version.replace("\n", " ")}
    try:
        import torch
        versions["torch"] = getattr(torch, "__version__", "unknown")
        versions["torch_cuda"] = str(getattr(torch.version, "cuda", "unknown"))
    except Exception:
        pass
    for module_name in ["gymnasium", "gymnasium_robotics", "mujoco", "stable_baselines3"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            continue
    return versions


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


def _ckpt_path(env_id: str, seed: int, checkpoints_dir: str = "checkpoints") -> Path:
    """Path to SAC+HER checkpoint.

    Looks for the standard checkpoint first, then falls back to nsg8 variant
    (ch04 found that Push benefits from n_sampled_goal=8).
    """
    ckpt_dir = Path(checkpoints_dir)
    primary = ckpt_dir / f"sac_her_{env_id}_seed{seed}.zip"
    if primary.exists():
        return primary

    # Fallback: try nsg8 variants
    nsg8 = ckpt_dir / f"sac_her_{env_id}_seed{seed}_nsg8.zip"
    if nsg8.exists():
        print(f"[ch06] Using nsg8 variant: {nsg8}")
        return nsg8

    # Try any nsg8 seed as last resort
    for candidate in sorted(ckpt_dir.glob(f"sac_her_{env_id}_seed*_nsg8.zip")):
        print(f"[ch06] Primary not found, using: {candidate}")
        return candidate

    return primary


def _load_sac_model(ckpt: str, env_id: str, device: str = "auto") -> Any:
    """Load a SAC checkpoint, handling HER replay buffer requirement."""
    import gymnasium
    import gymnasium_robotics  # noqa: F401
    from stable_baselines3 import SAC

    try:
        return SAC.load(ckpt, device=device)
    except (AssertionError, Exception) as e:
        if "HerReplayBuffer" in str(e) or "env" in str(e).lower():
            env = gymnasium.make(env_id)
            model = SAC.load(ckpt, env=env, device=device)
            env.close()
            return model
        raise


def _ensure_writable_dir(path: str | Path, label: str) -> Path:
    """Create directory if needed, return Path."""
    target = Path(path).expanduser().resolve()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise SystemExit(f"[ch06] {label} not writable: {target}")
    if not os.access(target, os.W_OK):
        raise SystemExit(f"[ch06] {label} not writable: {target}")
    return target


def _envs_for_config(cfg: Config) -> list[str]:
    """Return list of environments to evaluate."""
    envs = [cfg.env]
    if cfg.include_push:
        envs.append("FetchPush-v4")
    return envs


def _save_json(data: dict, path: Path) -> None:
    """Write JSON with trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ch06] Saved: {path}")


# =============================================================================
# Commands
# =============================================================================

def cmd_train(args: argparse.Namespace) -> int:
    """Train SAC+HER if checkpoint is missing (fallback for ch06)."""
    cfg = Config.from_args(args)
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")
    checkpoints_dir = _ensure_writable_dir(cfg.checkpoints_dir, "Checkpoints dir")

    for env_id in _envs_for_config(cfg):
        ckpt = _ckpt_path(env_id, cfg.seed, cfg.checkpoints_dir)
        if ckpt.exists():
            print(f"[ch06] Checkpoint exists: {ckpt} -- skipping training")
            continue

        print(f"[ch06] Checkpoint not found: {ckpt}")
        print(f"[ch06] Training SAC+HER on {env_id} (seed {cfg.seed})...")

        cmd = [
            sys.executable, "train.py",
            "--algo", "sac",
            "--her",
            "--env", env_id,
            "--seed", str(cfg.seed),
            "--n-envs", str(cfg.n_envs),
            "--total-steps", str(cfg.total_steps),
            "--device", cfg.device,
            "--gamma", "0.95",
            "--ent-coef", "0.05",
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            print(f"[ch06] Training failed for {env_id} seed {cfg.seed}")
            return result.returncode

    return 0


def cmd_scaling(args: argparse.Namespace) -> int:
    """Action scaling sweep: evaluate checkpoint with different scale factors."""
    from scripts.labs.action_interface import (
        ActionScalingWrapper,
        run_controller_eval,
    )

    cfg = Config.from_args(args)
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")

    for env_id in _envs_for_config(cfg):
        ckpt = _ckpt_path(env_id, cfg.seed, cfg.checkpoints_dir)
        if not ckpt.exists():
            print(f"[ch06] Checkpoint not found: {ckpt}")
            print(f"[ch06] Run 'train' subcommand first, or provide ch04 checkpoints.")
            return 1

        print(f"\n{'='*70}")
        print(f"Action Scaling Sweep: {env_id} (seed {cfg.seed})")
        print(f"{'='*70}")

        model = _load_sac_model(str(ckpt), env_id, cfg.device)
        all_results: list[dict[str, Any]] = []

        print(f"\n{'Scale':>8} | {'Success':>8} | {'Smooth':>8} | {'Peak |a|':>9} | "
              f"{'Path (m)':>9} | {'Energy':>8} | {'TTS':>6}")
        print("-" * 75)

        for scale in cfg.scale_factors:
            wrappers = [(ActionScalingWrapper, {"scale": scale})]
            result = run_controller_eval(
                policy=model,
                env_id=env_id,
                n_episodes=cfg.n_eval_episodes,
                deterministic=cfg.deterministic,
                seed=cfg.seed,
                wrappers=wrappers,
            )
            agg = result["aggregate"]
            tts = agg["time_to_success_mean"]
            tts_str = f"{tts:.1f}" if tts is not None else "N/A"

            print(
                f"{scale:>8.2f} | {agg['success_rate']:>7.0%} | "
                f"{agg['smoothness_mean']:>8.4f} | {agg['peak_action_mean']:>9.3f} | "
                f"{agg['path_length_mean']:>9.4f} | {agg['action_energy_mean']:>8.2f} | "
                f"{tts_str:>6}"
            )

            all_results.append({
                "scale": scale,
                "env_id": env_id,
                "seed": cfg.seed,
                "n_episodes": cfg.n_eval_episodes,
                **result,
            })

        # Save results
        out_data = {
            "experiment": "scaling",
            "env_id": env_id,
            "seed": cfg.seed,
            "n_episodes": cfg.n_eval_episodes,
            "checkpoint": str(ckpt),
            "scale_factors": cfg.scale_factors,
            "results": all_results,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "versions": _gather_versions(),
        }
        out_path = results_dir / f"ch06_scaling_{env_id.lower()}_seed{cfg.seed}.json"
        _save_json(out_data, out_path)

    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    """Low-pass filter sweep: evaluate checkpoint with different alpha values."""
    from scripts.labs.action_interface import (
        LowPassFilterWrapper,
        run_controller_eval,
    )

    cfg = Config.from_args(args)
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")

    for env_id in _envs_for_config(cfg):
        ckpt = _ckpt_path(env_id, cfg.seed, cfg.checkpoints_dir)
        if not ckpt.exists():
            print(f"[ch06] Checkpoint not found: {ckpt}")
            return 1

        print(f"\n{'='*70}")
        print(f"Low-Pass Filter Sweep: {env_id} (seed {cfg.seed})")
        print(f"{'='*70}")

        model = _load_sac_model(str(ckpt), env_id, cfg.device)
        all_results: list[dict[str, Any]] = []

        print(f"\n{'Alpha':>8} | {'Success':>8} | {'Smooth':>8} | {'Peak |a|':>9} | "
              f"{'Path (m)':>9} | {'Energy':>8} | {'TTS':>6}")
        print("-" * 75)

        for alpha in cfg.filter_alphas:
            wrappers = [(LowPassFilterWrapper, {"alpha": alpha})]
            result = run_controller_eval(
                policy=model,
                env_id=env_id,
                n_episodes=cfg.n_eval_episodes,
                deterministic=cfg.deterministic,
                seed=cfg.seed,
                wrappers=wrappers,
            )
            agg = result["aggregate"]
            tts = agg["time_to_success_mean"]
            tts_str = f"{tts:.1f}" if tts is not None else "N/A"

            print(
                f"{alpha:>8.2f} | {agg['success_rate']:>7.0%} | "
                f"{agg['smoothness_mean']:>8.4f} | {agg['peak_action_mean']:>9.3f} | "
                f"{agg['path_length_mean']:>9.4f} | {agg['action_energy_mean']:>8.2f} | "
                f"{tts_str:>6}"
            )

            all_results.append({
                "alpha": alpha,
                "env_id": env_id,
                "seed": cfg.seed,
                "n_episodes": cfg.n_eval_episodes,
                **result,
            })

        out_data = {
            "experiment": "filter",
            "env_id": env_id,
            "seed": cfg.seed,
            "n_episodes": cfg.n_eval_episodes,
            "checkpoint": str(ckpt),
            "filter_alphas": cfg.filter_alphas,
            "results": all_results,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "versions": _gather_versions(),
        }
        out_path = results_dir / f"ch06_filter_{env_id.lower()}_seed{cfg.seed}.json"
        _save_json(out_data, out_path)

    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """PD controller baseline: evaluate proportional controller at multiple Kp."""
    from scripts.labs.action_interface import (
        ProportionalController,
        run_controller_eval,
    )

    cfg = Config.from_args(args)
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")

    for env_id in _envs_for_config(cfg):
        print(f"\n{'='*70}")
        print(f"PD Baseline: {env_id}")
        print(f"{'='*70}")

        all_results: list[dict[str, Any]] = []

        print(f"\n{'Kp':>8} | {'Success':>8} | {'Smooth':>8} | {'Peak |a|':>9} | "
              f"{'Path (m)':>9} | {'Energy':>8} | {'TTS':>6}")
        print("-" * 75)

        for kp in cfg.pd_kp_values:
            controller = ProportionalController(env_id, kp=kp)
            result = run_controller_eval(
                policy=controller,
                env_id=env_id,
                n_episodes=cfg.n_eval_episodes,
                deterministic=True,
                seed=cfg.seed,
            )
            agg = result["aggregate"]
            tts = agg["time_to_success_mean"]
            tts_str = f"{tts:.1f}" if tts is not None else "N/A"

            print(
                f"{kp:>8.1f} | {agg['success_rate']:>7.0%} | "
                f"{agg['smoothness_mean']:>8.4f} | {agg['peak_action_mean']:>9.3f} | "
                f"{agg['path_length_mean']:>9.4f} | {agg['action_energy_mean']:>8.2f} | "
                f"{tts_str:>6}"
            )

            all_results.append({
                "kp": kp,
                "env_id": env_id,
                "n_episodes": cfg.n_eval_episodes,
                **result,
            })

        out_data = {
            "experiment": "baseline",
            "env_id": env_id,
            "seed": cfg.seed,
            "n_episodes": cfg.n_eval_episodes,
            "kp_values": cfg.pd_kp_values,
            "results": all_results,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "versions": _gather_versions(),
        }
        out_path = results_dir / f"ch06_baseline_{env_id.lower()}.json"
        _save_json(out_data, out_path)

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Load all ch06 results and print comparison tables."""
    cfg = Config.from_args(args)
    results_dir = Path(cfg.results_dir)

    if not results_dir.exists():
        print(f"[ch06] Results directory not found: {results_dir}")
        return 1

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environments": {},
    }

    for env_id in _envs_for_config(cfg):
        env_key = env_id.lower()

        print(f"\n{'='*70}")
        print(f"Chapter 6: Action Interface Comparison -- {env_id}")
        print(f"{'='*70}")

        env_report: dict[str, Any] = {}

        # --- Scaling results ---
        scaling_path = results_dir / f"ch06_scaling_{env_key}_seed{cfg.seed}.json"
        if scaling_path.exists():
            with open(scaling_path) as f:
                scaling_data = json.load(f)

            print(f"\n--- Action Scaling (seed {cfg.seed}) ---")
            print(f"{'Scale':>8} | {'Success':>8} | {'Smooth':>8} | "
                  f"{'Peak |a|':>9} | {'TTS':>6}")
            print("-" * 55)

            for r in scaling_data["results"]:
                agg = r["aggregate"]
                tts = agg["time_to_success_mean"]
                tts_str = f"{tts:.1f}" if tts is not None else "N/A"
                marker = " <-- trained" if r["scale"] == 1.0 else ""
                print(
                    f"{r['scale']:>8.2f} | {agg['success_rate']:>7.0%} | "
                    f"{agg['smoothness_mean']:>8.4f} | "
                    f"{agg['peak_action_mean']:>9.3f} | {tts_str:>6}{marker}"
                )

            env_report["scaling"] = scaling_data["results"]

            # Check monotonicity: smoothness should increase with scale
            smoothness_vals = [r["aggregate"]["smoothness_mean"] for r in scaling_data["results"]]
            monotone = all(smoothness_vals[i] <= smoothness_vals[i+1] + 0.01
                          for i in range(len(smoothness_vals) - 1))
            status = "PASS" if monotone else "SOFT FAIL (not strictly monotone)"
            print(f"\n  Smoothness monotonicity: [{status}]")
        else:
            print(f"\n  [skip] No scaling results found: {scaling_path}")

        # --- Filter results ---
        filter_path = results_dir / f"ch06_filter_{env_key}_seed{cfg.seed}.json"
        if filter_path.exists():
            with open(filter_path) as f:
                filter_data = json.load(f)

            print(f"\n--- Low-Pass Filter (seed {cfg.seed}) ---")
            print(f"{'Alpha':>8} | {'Success':>8} | {'Smooth':>8} | "
                  f"{'Peak |a|':>9} | {'TTS':>6}")
            print("-" * 55)

            for r in filter_data["results"]:
                agg = r["aggregate"]
                tts = agg["time_to_success_mean"]
                tts_str = f"{tts:.1f}" if tts is not None else "N/A"
                marker = " <-- no filter" if r["alpha"] == 1.0 else ""
                print(
                    f"{r['alpha']:>8.2f} | {agg['success_rate']:>7.0%} | "
                    f"{agg['smoothness_mean']:>8.4f} | "
                    f"{agg['peak_action_mean']:>9.3f} | {tts_str:>6}{marker}"
                )

            env_report["filter"] = filter_data["results"]

            # Check: smoothness should decrease as alpha increases (heavier filter = lower alpha = smoother)
            smoothness_vals = [r["aggregate"]["smoothness_mean"] for r in filter_data["results"]]
            monotone_dec = all(smoothness_vals[i] >= smoothness_vals[i+1] - 0.01
                              for i in range(len(smoothness_vals) - 1))
            status = "PASS" if monotone_dec else "SOFT FAIL (not strictly monotone)"
            print(f"\n  Smoothness decreases with alpha: [{status}]")
        else:
            print(f"\n  [skip] No filter results found: {filter_path}")

        # --- Baseline results ---
        baseline_path = results_dir / f"ch06_baseline_{env_key}.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline_data = json.load(f)

            print(f"\n--- PD Baseline ---")
            print(f"{'Kp':>8} | {'Success':>8} | {'Smooth':>8} | "
                  f"{'Peak |a|':>9} | {'TTS':>6}")
            print("-" * 55)

            for r in baseline_data["results"]:
                agg = r["aggregate"]
                tts = agg["time_to_success_mean"]
                tts_str = f"{tts:.1f}" if tts is not None else "N/A"
                print(
                    f"{r['kp']:>8.1f} | {agg['success_rate']:>7.0%} | "
                    f"{agg['smoothness_mean']:>8.4f} | "
                    f"{agg['peak_action_mean']:>9.3f} | {tts_str:>6}"
                )

            env_report["baseline"] = baseline_data["results"]
        else:
            print(f"\n  [skip] No baseline results found: {baseline_path}")

        # --- Planning vs Control decomposition ---
        has_scaling = scaling_path.exists()
        has_baseline = baseline_path.exists()

        if has_scaling and has_baseline:
            # RL at scale=1.0
            rl_result = next(
                (r for r in scaling_data["results"] if r["scale"] == 1.0), None
            )
            # PD at best Kp
            best_pd = max(baseline_data["results"],
                          key=lambda r: r["aggregate"]["success_rate"])

            if rl_result:
                rl_sr = rl_result["aggregate"]["success_rate"]
                pd_sr = best_pd["aggregate"]["success_rate"]
                gap = rl_sr - pd_sr

                print(f"\n--- Planning vs Control Decomposition ---")
                print(f"  RL (SAC+HER, scale=1.0):  {rl_sr:.0%} success")
                print(f"  PD (best Kp={best_pd['kp']:.0f}):     {pd_sr:.0%} success")
                print(f"  Gap (RL - PD):            {gap:+.0%}")

                if rl_sr < 0.20 and pd_sr < 0.20:
                    print(f"  -> NEITHER solves {env_id}. Checkpoint may need retraining.")
                elif gap < 0.05 and rl_sr >= 0.80:
                    print(f"  -> {env_id} is a pure CONTROL task: PD solves it.")
                elif gap > 0.30:
                    print(f"  -> {env_id} requires PLANNING: RL learns strategies PD cannot.")
                else:
                    print(f"  -> {env_id} has moderate planning complexity.")

                env_report["decomposition"] = {
                    "rl_success_rate": rl_sr,
                    "pd_success_rate": pd_sr,
                    "pd_best_kp": best_pd["kp"],
                    "gap": gap,
                }

        report["environments"][env_id] = env_report

    # Save comparison report
    results_dir_path = _ensure_writable_dir(cfg.results_dir, "Results dir")
    report_path = results_dir_path / "ch06_comparison.json"
    _save_json(report, report_path)

    print(f"\n{'='*70}")
    print("Comparison complete.")
    print(f"{'='*70}")

    return 0


def cmd_video(args: argparse.Namespace) -> int:
    """Record comparison videos: RL vs PD, scaling effects."""
    from scripts.labs.action_interface import (
        ActionScalingWrapper,
        LowPassFilterWrapper,
        ProportionalController,
        record_episode,
        save_comparison_video,
        save_video,
    )

    cfg = Config.from_args(args)
    videos_dir = _ensure_writable_dir(getattr(args, "videos_dir", "videos"), "Videos dir")

    for env_id in _envs_for_config(cfg):
        ckpt = _ckpt_path(env_id, cfg.seed, cfg.checkpoints_dir)
        has_ckpt = ckpt.exists()

        if has_ckpt:
            model = _load_sac_model(str(ckpt), env_id, cfg.device)
        else:
            print(f"[ch06] No checkpoint for {env_id} -- recording PD only")
            model = None

        env_tag = env_id.lower().replace("-", "")
        controller = ProportionalController(env_id, kp=10.0)

        # --- Video 1: RL vs PD side-by-side ---
        print(f"\n{'='*70}")
        print(f"Recording: RL vs PD -- {env_id}")
        print(f"{'='*70}")

        pd_frames, pd_m = record_episode(
            controller, env_id, seed=cfg.seed,
            label=f"PD (Kp=10)",
        )

        if model is not None:
            rl_frames, rl_m = record_episode(
                model, env_id, seed=cfg.seed,
                label=f"SAC+HER",
            )
            save_comparison_video(
                [("SAC+HER", rl_frames), ("PD Kp=10", pd_frames)],
                videos_dir / f"ch06_{env_tag}_rl_vs_pd.mp4",
            )
        else:
            save_video(pd_frames, videos_dir / f"ch06_{env_tag}_pd_only.mp4")

        # --- Video 2: Action scaling comparison (RL only) ---
        if model is not None:
            print(f"\nRecording: Scaling comparison -- {env_id}")
            scale_vids: list[tuple[str, list]] = []
            for scale in [0.25, 1.0, 2.0]:
                wrappers = [(ActionScalingWrapper, {"scale": scale})]
                frames, m = record_episode(
                    model, env_id, seed=cfg.seed,
                    wrappers=wrappers,
                    label=f"scale={scale}",
                )
                scale_vids.append((f"scale={scale}", frames))

            save_comparison_video(
                scale_vids,
                videos_dir / f"ch06_{env_tag}_scaling.mp4",
            )

            # --- Video 3: Filter comparison (RL only) ---
            print(f"\nRecording: Filter comparison -- {env_id}")
            filter_vids: list[tuple[str, list]] = []
            for alpha in [0.2, 0.6, 1.0]:
                wrappers = [(LowPassFilterWrapper, {"alpha": alpha})]
                frames, m = record_episode(
                    model, env_id, seed=cfg.seed,
                    wrappers=wrappers,
                    label=f"alpha={alpha}",
                )
                filter_vids.append((f"alpha={alpha}", frames))

            save_comparison_video(
                filter_vids,
                videos_dir / f"ch06_{env_tag}_filter.mp4",
            )

    print(f"\n[ch06] All videos saved to: {videos_dir}/")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Full pipeline: scaling -> filter -> baseline -> compare -> video."""
    cfg = Config.from_args(args)

    print("=" * 70)
    print("Chapter 6: Action Interface Engineering -- Full Pipeline")
    print(f"Environments: {_envs_for_config(cfg)}")
    print(f"Seed: {cfg.seed}, Episodes per condition: {cfg.n_eval_episodes}")
    print("=" * 70)

    # Step 1: Ensure checkpoints exist
    print("\n[1/5] Checking checkpoints...")
    ret = cmd_train(args)
    if ret != 0:
        print("[ch06] Checkpoint preparation failed. Aborting.")
        return ret

    # Step 2: Action scaling sweep
    print("\n[2/5] Action scaling sweep...")
    ret = cmd_scaling(args)
    if ret != 0:
        print("[ch06] Scaling sweep failed.")
        return ret

    # Step 3: Low-pass filter sweep
    print("\n[3/5] Low-pass filter sweep...")
    ret = cmd_filter(args)
    if ret != 0:
        print("[ch06] Filter sweep failed.")
        return ret

    # Step 4: PD baseline
    print("\n[4/5] PD controller baseline...")
    ret = cmd_baseline(args)
    if ret != 0:
        print("[ch06] Baseline evaluation failed.")
        return ret

    # Step 5: Comparison + videos
    print("\n[5/5] Comparison and video recording...")
    ret = cmd_compare(args)
    if ret != 0:
        return ret

    return cmd_video(args)


# =============================================================================
# CLI
# =============================================================================

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
    defaults = DEFAULT_CONFIG
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="Random seed / checkpoint seed")
    parser.add_argument("--n-eval-episodes", type=int, default=defaults.n_eval_episodes,
                        help="Episodes per evaluation condition")
    parser.add_argument("--device", default=defaults.device,
                        help="Device for model inference: auto, cpu, cuda")
    parser.add_argument("--include-push", action="store_true", default=defaults.include_push,
                        help="Also evaluate on FetchPush-v4")
    parser.add_argument("--checkpoints-dir", default=defaults.checkpoints_dir,
                        help="Directory with ch04 checkpoints")
    parser.add_argument("--results-dir", default=defaults.results_dir,
                        help="Directory for output JSON")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter 06: Action-Interface Engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Action scaling sweep (FetchReach-v4, seed 0)
  python scripts/ch06_action_interface.py scaling --seed 0

  # Low-pass filter sweep
  python scripts/ch06_action_interface.py filter --seed 0

  # PD baseline on Reach and Push
  python scripts/ch06_action_interface.py baseline --include-push

  # Full pipeline
  python scripts/ch06_action_interface.py all --seed 0

  # Quick smoke test (fewer episodes)
  python scripts/ch06_action_interface.py scaling --seed 0 --n-eval-episodes 20

  # Record comparison videos (requires MUJOCO_GL=egl)
  python scripts/ch06_action_interface.py video --seed 0 --include-push
""",
    )
    parser.add_argument("--mujoco-gl", default=None,
                        help="Override MUJOCO_GL for this run")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train (fallback)
    p_train = sub.add_parser("train",
                             help="Train SAC+HER if checkpoint missing",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_train.add_argument("--env", default=DEFAULT_CONFIG.env,
                         help="Environment ID")
    p_train.add_argument("--n-envs", type=int, default=DEFAULT_CONFIG.n_envs,
                         help="Number of parallel envs for training")
    p_train.add_argument("--total-steps", type=int, default=DEFAULT_CONFIG.total_steps,
                         help="Total training timesteps")
    _add_common_args(p_train)

    # scaling
    p_scale = sub.add_parser("scaling",
                             help="Action scaling sweep",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_scale)

    # filter
    p_filter = sub.add_parser("filter",
                              help="Low-pass filter sweep",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_filter)

    # baseline
    p_base = sub.add_parser("baseline",
                            help="PD controller baseline",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_base)

    # compare
    p_cmp = sub.add_parser("compare",
                           help="Compare all results",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_cmp)

    # video
    p_vid = sub.add_parser("video",
                           help="Record comparison videos (RL vs PD, scaling, filter)",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_vid.add_argument("--videos-dir", default="videos",
                       help="Directory for video output")
    _add_common_args(p_vid)

    # all
    p_all = sub.add_parser("all",
                           help="Full pipeline: scaling -> filter -> baseline -> compare",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_all.add_argument("--env", default=DEFAULT_CONFIG.env,
                       help="Primary environment ID")
    p_all.add_argument("--n-envs", type=int, default=DEFAULT_CONFIG.n_envs,
                       help="Number of parallel envs for training fallback")
    p_all.add_argument("--total-steps", type=int, default=DEFAULT_CONFIG.total_steps,
                       help="Total training timesteps for training fallback")
    p_all.add_argument("--videos-dir", default="videos",
                       help="Directory for video output")
    _add_common_args(p_all)

    args = parser.parse_args()

    if args.mujoco_gl is None:
        os.environ.setdefault("MUJOCO_GL", "disable")
    else:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    handlers = {
        "train": cmd_train,
        "scaling": cmd_scaling,
        "filter": cmd_filter,
        "baseline": cmd_baseline,
        "compare": cmd_compare,
        "video": cmd_video,
        "all": cmd_all,
    }

    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
