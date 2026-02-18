#!/usr/bin/env python3
"""Chapter 07: Robustness Curves -- Quantify Brittleness

After ch06 (action interface engineering), we can measure policy quality
under ideal conditions. This chapter asks: how brittle are our policies?

The core deliverable is degradation curves -- success_rate(sigma) with 95%
confidence bands across seeds -- turning the binary stress test from ch05
into a parametric, quantitative analysis.

Experiments:
    1. Observation noise sweep: 6 levels, per env/seed
    2. Action noise sweep: 6 levels, per env/seed
    3. Mitigation: low-pass filter under action noise
    4. Cross-env comparison: Reach vs Push brittleness

Usage:
    # Ensure checkpoints exist
    python scripts/ch07_robustness_curves.py train --seeds 0

    # Observation noise sweep
    python scripts/ch07_robustness_curves.py obs-sweep --seeds 0

    # Action noise sweep
    python scripts/ch07_robustness_curves.py act-sweep --seeds 0

    # Mitigation experiment
    python scripts/ch07_robustness_curves.py mitigate --seeds 0

    # Compare all results
    python scripts/ch07_robustness_curves.py compare --seeds 0

    # Full pipeline
    python scripts/ch07_robustness_curves.py all --seeds 0

    # Quick smoke test
    python scripts/ch07_robustness_curves.py all --seeds 0 --n-eval-episodes 20

    # Multi-seed with Push
    python scripts/ch07_robustness_curves.py all --seeds 0,1,2 --include-push
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `from scripts.labs...` imports work
# when this script is invoked as `python scripts/ch07_robustness_curves.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Experiment configuration for ch07.

    Reuses ch04 checkpoints (SAC+HER) -- no training hyperparameters needed
    unless fallback training is triggered.
    """
    # Environments
    env: str = "FetchReach-v4"
    include_push: bool = True

    # Seeds (parsed from comma-separated or range string)
    seeds: str = "0"

    # Evaluation
    n_eval_episodes: int = 100
    deterministic: bool = True

    # Training fallback (only used if checkpoint missing)
    n_envs: int = 8
    total_steps: int = 1_000_000
    device: str = "auto"

    # Observation noise sweep
    obs_noise_levels: list[float] = field(
        default_factory=lambda: [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    )

    # Action noise sweep
    act_noise_levels: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.025, 0.05, 0.1, 0.2]
    )

    # Mitigation experiment
    mitigation_alphas: list[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.7, 1.0]
    )
    mitigation_act_noise: float = 0.1

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


def _ckpt_path(env_id: str, seed: int, checkpoints_dir: str = "checkpoints") -> Path:
    """Path to SAC+HER checkpoint.

    Looks for the standard checkpoint first, then falls back to nsg8 variant
    (ch04 found that Push benefits from n_sampled_goal=8).
    """
    ckpt_dir = Path(checkpoints_dir)
    primary = ckpt_dir / f"sac_her_{env_id}_seed{seed}.zip"
    if primary.exists():
        return primary

    nsg8 = ckpt_dir / f"sac_her_{env_id}_seed{seed}_nsg8.zip"
    if nsg8.exists():
        print(f"[ch07] Using nsg8 variant: {nsg8}")
        return nsg8

    for candidate in sorted(ckpt_dir.glob(f"sac_her_{env_id}_seed*_nsg8.zip")):
        print(f"[ch07] Primary not found, using: {candidate}")
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
        raise SystemExit(f"[ch07] {label} not writable: {target}")
    if not os.access(target, os.W_OK):
        raise SystemExit(f"[ch07] {label} not writable: {target}")
    return target


def _save_json(data: dict, path: Path) -> None:
    """Write JSON with trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ch07] Saved: {path}")


def _parse_seeds(seeds_str: str) -> list[int]:
    """Parse seed string: '0', '0,1,2', or '0-4'."""
    seeds_str = seeds_str.strip()
    if "-" in seeds_str and "," not in seeds_str:
        parts = seeds_str.split("-")
        if len(parts) == 2:
            return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(s.strip()) for s in seeds_str.split(",")]


def _envs_for_config(cfg: Config) -> list[str]:
    """Return list of environments to evaluate."""
    envs = [cfg.env]
    if cfg.include_push:
        if "FetchPush-v4" not in envs:
            envs.append("FetchPush-v4")
    return envs


# =============================================================================
# Commands
# =============================================================================

def cmd_train(args: argparse.Namespace) -> int:
    """Train SAC+HER if checkpoint is missing (fallback for ch07)."""
    cfg = Config.from_args(args)
    _ensure_writable_dir(cfg.results_dir, "Results dir")
    _ensure_writable_dir(cfg.checkpoints_dir, "Checkpoints dir")

    seeds = _parse_seeds(cfg.seeds)
    for env_id in _envs_for_config(cfg):
        for seed in seeds:
            ckpt = _ckpt_path(env_id, seed, cfg.checkpoints_dir)
            if ckpt.exists():
                print(f"[ch07] Checkpoint exists: {ckpt} -- skipping training")
                continue

            print(f"[ch07] Checkpoint not found: {ckpt}")
            print(f"[ch07] Training SAC+HER on {env_id} (seed {seed})...")

            cmd = [
                sys.executable, "train.py",
                "--algo", "sac",
                "--her",
                "--env", env_id,
                "--seed", str(seed),
                "--n-envs", str(cfg.n_envs),
                "--total-steps", str(cfg.total_steps),
                "--device", cfg.device,
                "--gamma", "0.95",
                "--ent-coef", "0.05",
            ]
            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
            if result.returncode != 0:
                print(f"[ch07] Training failed for {env_id} seed {seed}")
                return result.returncode

    return 0


def cmd_obs_sweep(args: argparse.Namespace) -> int:
    """Observation noise sweep: evaluate at multiple obs noise levels."""
    from scripts.labs.robustness import (
        aggregate_across_seeds,
        compute_degradation_summary,
        run_noise_sweep,
    )

    cfg = Config.from_args(args)
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")
    seeds = _parse_seeds(cfg.seeds)

    for env_id in _envs_for_config(cfg):
        print(f"\n{'='*70}")
        print(f"Observation Noise Sweep: {env_id}")
        print(f"Seeds: {seeds}, Episodes/condition: {cfg.n_eval_episodes}")
        print(f"Noise levels: {cfg.obs_noise_levels}")
        print(f"{'='*70}")

        all_seed_results: list[list] = []

        for seed in seeds:
            ckpt = _ckpt_path(env_id, seed, cfg.checkpoints_dir)
            if not ckpt.exists():
                print(f"[ch07] Checkpoint not found: {ckpt}")
                print("[ch07] Run 'train' subcommand first.")
                return 1

            print(f"\n--- Seed {seed} ---")
            model = _load_sac_model(str(ckpt), env_id, cfg.device)

            results = run_noise_sweep(
                policy=model,
                env_id=env_id,
                noise_type="obs",
                noise_levels=cfg.obs_noise_levels,
                n_episodes=cfg.n_eval_episodes,
                deterministic=cfg.deterministic,
                seed=seed,
            )

            # Print per-seed table
            print(f"\n{'sigma':>8} | {'Success':>8} | {'Return':>8} | "
                  f"{'Smooth':>8} | {'Dist':>8} | {'TTS':>6}")
            print("-" * 65)
            for r in results:
                tts = r.time_to_success_mean
                tts_str = f"{tts:.1f}" if tts is not None else "N/A"
                print(
                    f"{r.noise_std:>8.4f} | {r.success_rate:>7.0%} | "
                    f"{r.return_mean:>8.2f} | {r.smoothness_mean:>8.4f} | "
                    f"{r.final_distance_mean:>8.4f} | {tts_str:>6}"
                )

            all_seed_results.append(results)

            # Save per-seed JSON
            out_data = {
                "experiment": "obs_sweep",
                "env_id": env_id,
                "seed": seed,
                "n_episodes": cfg.n_eval_episodes,
                "checkpoint": str(ckpt),
                "noise_levels": cfg.obs_noise_levels,
                "results": [r.to_dict() for r in results],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "versions": _gather_versions(),
            }
            out_path = results_dir / f"ch07_obs_sweep_{env_id.lower()}_seed{seed}.json"
            _save_json(out_data, out_path)

        # Aggregate across seeds if multiple
        if len(seeds) > 1:
            aggregated = aggregate_across_seeds(all_seed_results)
            summary = compute_degradation_summary(aggregated)

            print(f"\n--- Aggregated ({len(seeds)} seeds) ---")
            print(f"{'sigma':>8} | {'SR mean':>8} | {'SR ci95':>8}")
            print("-" * 30)
            for a in aggregated:
                sr = a["success_rate"]
                print(f"{a['noise_std']:>8.4f} | {sr['mean']:>7.0%} | "
                      f"+/-{sr['ci95']:.0%}")

            print(f"\n  Critical sigma: {summary['critical_sigma']}")
            print(f"  Degradation slope: {summary['degradation_slope']}")
            print(f"  Robustness AUC: {summary['robustness_auc']}")

            agg_data = {
                "experiment": "obs_sweep_aggregated",
                "env_id": env_id,
                "seeds": seeds,
                "aggregated": aggregated,
                "summary": summary,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            agg_path = results_dir / f"ch07_obs_sweep_{env_id.lower()}_aggregated.json"
            _save_json(agg_data, agg_path)

    return 0


def cmd_act_sweep(args: argparse.Namespace) -> int:
    """Action noise sweep: evaluate at multiple action noise levels."""
    from scripts.labs.robustness import (
        aggregate_across_seeds,
        compute_degradation_summary,
        run_noise_sweep,
    )

    cfg = Config.from_args(args)
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")
    seeds = _parse_seeds(cfg.seeds)

    for env_id in _envs_for_config(cfg):
        print(f"\n{'='*70}")
        print(f"Action Noise Sweep: {env_id}")
        print(f"Seeds: {seeds}, Episodes/condition: {cfg.n_eval_episodes}")
        print(f"Noise levels: {cfg.act_noise_levels}")
        print(f"{'='*70}")

        all_seed_results: list[list] = []

        for seed in seeds:
            ckpt = _ckpt_path(env_id, seed, cfg.checkpoints_dir)
            if not ckpt.exists():
                print(f"[ch07] Checkpoint not found: {ckpt}")
                print("[ch07] Run 'train' subcommand first.")
                return 1

            print(f"\n--- Seed {seed} ---")
            model = _load_sac_model(str(ckpt), env_id, cfg.device)

            results = run_noise_sweep(
                policy=model,
                env_id=env_id,
                noise_type="act",
                noise_levels=cfg.act_noise_levels,
                n_episodes=cfg.n_eval_episodes,
                deterministic=cfg.deterministic,
                seed=seed,
            )

            # Print per-seed table
            print(f"\n{'sigma':>8} | {'Success':>8} | {'Return':>8} | "
                  f"{'Smooth':>8} | {'Energy':>8} | {'TTS':>6}")
            print("-" * 65)
            for r in results:
                tts = r.time_to_success_mean
                tts_str = f"{tts:.1f}" if tts is not None else "N/A"
                print(
                    f"{r.noise_std:>8.4f} | {r.success_rate:>7.0%} | "
                    f"{r.return_mean:>8.2f} | {r.smoothness_mean:>8.4f} | "
                    f"{r.action_energy_mean:>8.2f} | {tts_str:>6}"
                )

            all_seed_results.append(results)

            # Save per-seed JSON
            out_data = {
                "experiment": "act_sweep",
                "env_id": env_id,
                "seed": seed,
                "n_episodes": cfg.n_eval_episodes,
                "checkpoint": str(ckpt),
                "noise_levels": cfg.act_noise_levels,
                "results": [r.to_dict() for r in results],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "versions": _gather_versions(),
            }
            out_path = results_dir / f"ch07_act_sweep_{env_id.lower()}_seed{seed}.json"
            _save_json(out_data, out_path)

        # Aggregate across seeds if multiple
        if len(seeds) > 1:
            aggregated = aggregate_across_seeds(all_seed_results)
            summary = compute_degradation_summary(aggregated)

            print(f"\n--- Aggregated ({len(seeds)} seeds) ---")
            print(f"{'sigma':>8} | {'SR mean':>8} | {'SR ci95':>8}")
            print("-" * 30)
            for a in aggregated:
                sr = a["success_rate"]
                print(f"{a['noise_std']:>8.4f} | {sr['mean']:>7.0%} | "
                      f"+/-{sr['ci95']:.0%}")

            print(f"\n  Critical sigma: {summary['critical_sigma']}")
            print(f"  Degradation slope: {summary['degradation_slope']}")
            print(f"  Robustness AUC: {summary['robustness_auc']}")

            agg_data = {
                "experiment": "act_sweep_aggregated",
                "env_id": env_id,
                "seeds": seeds,
                "aggregated": aggregated,
                "summary": summary,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            agg_path = results_dir / f"ch07_act_sweep_{env_id.lower()}_aggregated.json"
            _save_json(agg_data, agg_path)

    return 0


def cmd_mitigate(args: argparse.Namespace) -> int:
    """Test low-pass filter mitigation under action noise.

    Wrapper ordering: NoisyEvalWrapper is innermost (wraps env first),
    LowPassFilterWrapper is outermost (wraps NoisyEvalWrapper). So the
    data flow is: policy -> LPF smooths -> NoisyEvalWrapper adds noise -> env.
    This matches the physical model: controller smooths, actuator still noisy.
    """
    from scripts.labs.action_interface import LowPassFilterWrapper
    from scripts.labs.robustness import run_noise_sweep

    cfg = Config.from_args(args)
    results_dir = _ensure_writable_dir(cfg.results_dir, "Results dir")
    seeds = _parse_seeds(cfg.seeds)
    act_sigma = cfg.mitigation_act_noise

    for env_id in _envs_for_config(cfg):
        print(f"\n{'='*70}")
        print(f"Mitigation Experiment: {env_id}")
        print(f"Fixed action noise: sigma={act_sigma}")
        print(f"LPF alphas: {cfg.mitigation_alphas}")
        print(f"{'='*70}")

        for seed in seeds:
            ckpt = _ckpt_path(env_id, seed, cfg.checkpoints_dir)
            if not ckpt.exists():
                print(f"[ch07] Checkpoint not found: {ckpt}")
                return 1

            print(f"\n--- Seed {seed} ---")
            model = _load_sac_model(str(ckpt), env_id, cfg.device)

            all_results: list[dict[str, Any]] = []

            print(f"\n{'Alpha':>8} | {'Success':>8} | {'Return':>8} | "
                  f"{'Smooth':>8} | {'Energy':>8} | {'TTS':>6}")
            print("-" * 65)

            for alpha in cfg.mitigation_alphas:
                # LPF as extra_wrappers: applied after NoisyEvalWrapper
                # run_noise_sweep creates NoisyEvalWrapper first, then appends extra_wrappers
                # So: env -> NoisyEvalWrapper -> LowPassFilterWrapper
                # Call: LPF.step(action) -> NoisyEvalWrapper.step(smoothed) -> env.step(noisy)
                extra_wrappers: list[tuple[type, dict[str, Any]]] | None = None
                if alpha < 1.0:
                    extra_wrappers = [(LowPassFilterWrapper, {"alpha": alpha})]

                results = run_noise_sweep(
                    policy=model,
                    env_id=env_id,
                    noise_type="act",
                    noise_levels=[act_sigma],
                    n_episodes=cfg.n_eval_episodes,
                    deterministic=cfg.deterministic,
                    seed=seed,
                    extra_wrappers=extra_wrappers,
                )

                r = results[0]
                tts = r.time_to_success_mean
                tts_str = f"{tts:.1f}" if tts is not None else "N/A"
                marker = " <-- no filter" if alpha == 1.0 else ""
                print(
                    f"{alpha:>8.2f} | {r.success_rate:>7.0%} | "
                    f"{r.return_mean:>8.2f} | {r.smoothness_mean:>8.4f} | "
                    f"{r.action_energy_mean:>8.2f} | {tts_str:>6}{marker}"
                )

                all_results.append({
                    "alpha": alpha,
                    "act_noise_std": act_sigma,
                    **r.to_dict(),
                })

            # Save per-seed mitigation results
            out_data = {
                "experiment": "mitigation",
                "env_id": env_id,
                "seed": seed,
                "n_episodes": cfg.n_eval_episodes,
                "checkpoint": str(ckpt),
                "act_noise_std": act_sigma,
                "alphas": cfg.mitigation_alphas,
                "results": all_results,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "versions": _gather_versions(),
            }
            out_path = results_dir / f"ch07_mitigation_{env_id.lower()}_seed{seed}.json"
            _save_json(out_data, out_path)

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Load all ch07 results and print comparison tables."""
    from scripts.labs.robustness import (
        aggregate_across_seeds,
        compute_degradation_summary,
        NoiseSweepResult,
    )

    cfg = Config.from_args(args)
    results_dir = Path(cfg.results_dir)
    seeds = _parse_seeds(cfg.seeds)

    if not results_dir.exists():
        print(f"[ch07] Results directory not found: {results_dir}")
        return 1

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "environments": {},
    }

    env_summaries: dict[str, dict[str, Any]] = {}

    for env_id in _envs_for_config(cfg):
        env_key = env_id.lower()

        print(f"\n{'='*70}")
        print(f"Chapter 7: Robustness Comparison -- {env_id}")
        print(f"{'='*70}")

        env_report: dict[str, Any] = {}

        # --- Obs noise results ---
        _print_sweep_comparison(
            results_dir, env_key, env_id, seeds, "obs_sweep",
            cfg.obs_noise_levels, "Observation Noise", env_report,
        )

        # --- Act noise results ---
        _print_sweep_comparison(
            results_dir, env_key, env_id, seeds, "act_sweep",
            cfg.act_noise_levels, "Action Noise", env_report,
        )

        # --- Mitigation results ---
        _print_mitigation_comparison(
            results_dir, env_key, env_id, seeds, cfg, env_report,
        )

        report["environments"][env_id] = env_report

        # Store summaries for cross-env comparison
        env_summaries[env_id] = env_report

    # --- Cross-environment comparison ---
    if len(env_summaries) > 1:
        _print_cross_env_comparison(env_summaries)

    # Save comparison report
    results_dir_path = _ensure_writable_dir(cfg.results_dir, "Results dir")
    report_path = results_dir_path / "ch07_comparison.json"
    _save_json(report, report_path)

    print(f"\n{'='*70}")
    print("Comparison complete.")
    print(f"{'='*70}")

    return 0


def _print_sweep_comparison(
    results_dir: Path,
    env_key: str,
    env_id: str,
    seeds: list[int],
    experiment: str,
    noise_levels: list[float],
    label: str,
    env_report: dict[str, Any],
) -> None:
    """Print a sweep comparison table from saved JSON files."""
    from scripts.labs.robustness import (
        NoiseSweepResult,
        aggregate_across_seeds,
        compute_degradation_summary,
    )

    all_seed_results: list[list[NoiseSweepResult]] = []
    single_seed_data: dict | None = None

    for seed in seeds:
        path = results_dir / f"ch07_{experiment}_{env_key}_seed{seed}.json"
        if not path.exists():
            print(f"\n  [skip] No {experiment} results: {path}")
            return

        with open(path) as f:
            data = json.load(f)

        # Reconstruct NoiseSweepResult objects
        seed_results: list[NoiseSweepResult] = []
        for r in data["results"]:
            seed_results.append(NoiseSweepResult(
                noise_type=r["noise_type"],
                noise_std=r["noise_std"],
                env_id=r.get("env_id", env_id),
                seed=r.get("seed", seed),
                n_episodes=r.get("n_episodes", 0),
                success_rate=r["success_rate"],
                return_mean=r["return_mean"],
                ep_len_mean=r.get("ep_len_mean", 0.0),
                smoothness_mean=r["smoothness_mean"],
                peak_action_mean=r["peak_action_mean"],
                path_length_mean=r["path_length_mean"],
                action_energy_mean=r["action_energy_mean"],
                time_to_success_mean=r.get("time_to_success_mean"),
                final_distance_mean=r["final_distance_mean"],
                episode_successes=r.get("episode_successes", []),
            ))
        all_seed_results.append(seed_results)
        if single_seed_data is None:
            single_seed_data = data

    # Print table
    noise_type_label = "obs" if "obs" in experiment else "act"
    print(f"\n--- {label} Sweep ---")

    # Always aggregate (works for single seed too -- CI=0)
    aggregated = aggregate_across_seeds(all_seed_results)
    summary = compute_degradation_summary(aggregated)

    if len(seeds) == 1:
        # Single seed: print raw results
        print(f"{'sigma':>8} | {'Success':>8} | {'Return':>8} | "
              f"{'Smooth':>8} | {'Dist':>8}")
        print("-" * 55)
        for r in all_seed_results[0]:
            marker = " <-- baseline" if r.noise_std == 0.0 else ""
            print(
                f"{r.noise_std:>8.4f} | {r.success_rate:>7.0%} | "
                f"{r.return_mean:>8.2f} | {r.smoothness_mean:>8.4f} | "
                f"{r.final_distance_mean:>8.4f}{marker}"
            )
    else:
        # Multi-seed: print aggregated with CI
        print(f"{'sigma':>8} | {'SR mean':>8} | {'SR ci95':>8} | "
              f"{'Return':>8} | {'Dist':>8}")
        print("-" * 55)
        for a in aggregated:
            sr = a["success_rate"]
            ret = a["return_mean"]
            dist = a["final_distance_mean"]
            marker = " <-- baseline" if a["noise_std"] == 0.0 else ""
            print(
                f"{a['noise_std']:>8.4f} | {sr['mean']:>7.0%} | "
                f"+/-{sr['ci95']:.0%}  | {ret['mean']:>8.2f} | "
                f"{dist['mean']:>8.4f}{marker}"
            )

    print(f"\n  Critical sigma: {summary['critical_sigma']}")
    print(f"  Degradation slope: {summary['degradation_slope']}")
    print(f"  Robustness AUC: {summary['robustness_auc']}")

    # Check approximate monotonicity of SR using aggregated means
    sr_values = [a["success_rate"]["mean"] for a in aggregated]
    approx_monotone = all(
        sr_values[i] >= sr_values[i + 1] - 0.05
        for i in range(len(sr_values) - 1)
    )
    status = "PASS" if approx_monotone else "SOFT FAIL (not approximately monotone)"
    print(f"  SR monotonicity: [{status}]")

    env_report[experiment] = {
        "summary": summary,
        "monotone": approx_monotone,
    }


def _print_mitigation_comparison(
    results_dir: Path,
    env_key: str,
    env_id: str,
    seeds: list[int],
    cfg: Config,
    env_report: dict[str, Any],
) -> None:
    """Print mitigation comparison table."""
    # Use first seed's results for display
    seed = seeds[0]
    path = results_dir / f"ch07_mitigation_{env_key}_seed{seed}.json"
    if not path.exists():
        print(f"\n  [skip] No mitigation results: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    print(f"\n--- Mitigation: LPF under act_noise={cfg.mitigation_act_noise} ---")
    print(f"{'Alpha':>8} | {'Success':>8} | {'Smooth':>8} | "
          f"{'Energy':>8} | {'TTS':>6}")
    print("-" * 55)

    baseline_sr = None
    best_sr = 0.0
    best_alpha = 1.0

    for r in data["results"]:
        alpha = r["alpha"]
        sr = r["success_rate"]
        tts = r.get("time_to_success_mean")
        tts_str = f"{tts:.1f}" if tts is not None else "N/A"
        marker = " <-- no filter" if alpha == 1.0 else ""

        if alpha == 1.0:
            baseline_sr = sr
        if sr > best_sr:
            best_sr = sr
            best_alpha = alpha

        print(
            f"{alpha:>8.2f} | {sr:>7.0%} | "
            f"{r['smoothness_mean']:>8.4f} | "
            f"{r['action_energy_mean']:>8.2f} | {tts_str:>6}{marker}"
        )

    if baseline_sr is not None and best_alpha < 1.0:
        improvement = best_sr - baseline_sr
        print(f"\n  Best filter: alpha={best_alpha} ({improvement:+.0%} vs unfiltered)")
    elif baseline_sr is not None:
        print(f"\n  No filter improved over baseline ({baseline_sr:.0%})")

    env_report["mitigation"] = {
        "baseline_sr": baseline_sr,
        "best_alpha": best_alpha,
        "best_sr": best_sr,
    }


def _print_cross_env_comparison(env_summaries: dict[str, dict[str, Any]]) -> None:
    """Print cross-environment robustness comparison."""
    print(f"\n{'='*70}")
    print("Cross-Environment Robustness Comparison")
    print(f"{'='*70}")

    print(f"\n{'Env':<20} | {'Noise':>6} | {'Crit sigma':>11} | "
          f"{'Slope':>8} | {'AUC':>8}")
    print("-" * 65)

    for env_id, data in env_summaries.items():
        for sweep_type in ["obs_sweep", "act_sweep"]:
            if sweep_type not in data:
                continue
            summary = data[sweep_type]["summary"]
            noise_label = "obs" if "obs" in sweep_type else "act"
            crit = summary["critical_sigma"]
            crit_str = f"{crit:.4f}" if crit is not None else "N/A"
            print(
                f"{env_id:<20} | {noise_label:>6} | {crit_str:>11} | "
                f"{summary['degradation_slope']:>8.4f} | "
                f"{summary['robustness_auc']:>8.4f}"
            )

    # Verdict
    envs = list(env_summaries.keys())
    if len(envs) >= 2:
        env_a, env_b = envs[0], envs[1]
        print(f"\n--- Verdict ---")
        for sweep_type, noise_label in [("obs_sweep", "observation"), ("act_sweep", "action")]:
            if sweep_type in env_summaries.get(env_a, {}) and sweep_type in env_summaries.get(env_b, {}):
                auc_a = env_summaries[env_a][sweep_type]["summary"]["robustness_auc"]
                auc_b = env_summaries[env_b][sweep_type]["summary"]["robustness_auc"]
                more_brittle = env_b if auc_b < auc_a else env_a
                print(f"  {noise_label.capitalize()} noise: {more_brittle} is more brittle "
                      f"(AUC: {min(auc_a, auc_b):.4f} vs {max(auc_a, auc_b):.4f})")


def cmd_all(args: argparse.Namespace) -> int:
    """Full pipeline: train -> obs-sweep -> act-sweep -> mitigate -> compare."""
    cfg = Config.from_args(args)
    seeds = _parse_seeds(cfg.seeds)

    print("=" * 70)
    print("Chapter 7: Robustness Curves -- Full Pipeline")
    print(f"Environments: {_envs_for_config(cfg)}")
    print(f"Seeds: {seeds}, Episodes per condition: {cfg.n_eval_episodes}")
    print("=" * 70)

    # Step 1: Ensure checkpoints exist
    print("\n[1/5] Checking checkpoints...")
    ret = cmd_train(args)
    if ret != 0:
        print("[ch07] Checkpoint preparation failed. Aborting.")
        return ret

    # Step 2: Observation noise sweep
    print("\n[2/5] Observation noise sweep...")
    ret = cmd_obs_sweep(args)
    if ret != 0:
        print("[ch07] Obs sweep failed.")
        return ret

    # Step 3: Action noise sweep
    print("\n[3/5] Action noise sweep...")
    ret = cmd_act_sweep(args)
    if ret != 0:
        print("[ch07] Act sweep failed.")
        return ret

    # Step 4: Mitigation experiment
    print("\n[4/5] Mitigation experiment...")
    ret = cmd_mitigate(args)
    if ret != 0:
        print("[ch07] Mitigation failed.")
        return ret

    # Step 5: Comparison
    print("\n[5/5] Comparison...")
    return cmd_compare(args)


# =============================================================================
# CLI
# =============================================================================

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
    defaults = DEFAULT_CONFIG
    parser.add_argument("--seeds", default=defaults.seeds,
                        help="Seeds: '0', '0,1,2', or '0-4'")
    parser.add_argument("--n-eval-episodes", type=int, default=defaults.n_eval_episodes,
                        help="Episodes per evaluation condition")
    parser.add_argument("--device", default=defaults.device,
                        help="Device for model inference: auto, cpu, cuda")
    parser.add_argument("--include-push", action="store_true", default=defaults.include_push,
                        help="Also evaluate on FetchPush-v4")
    parser.add_argument("--no-push", action="store_false", dest="include_push",
                        help="Skip FetchPush-v4 (Reach only)")
    parser.add_argument("--checkpoints-dir", default=defaults.checkpoints_dir,
                        help="Directory with checkpoints")
    parser.add_argument("--results-dir", default=defaults.results_dir,
                        help="Directory for output JSON")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter 07: Robustness Curves -- Quantify Brittleness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Observation noise sweep (FetchReach-v4, seed 0)
  python scripts/ch07_robustness_curves.py obs-sweep --seeds 0

  # Action noise sweep
  python scripts/ch07_robustness_curves.py act-sweep --seeds 0

  # Mitigation experiment
  python scripts/ch07_robustness_curves.py mitigate --seeds 0

  # Compare all results
  python scripts/ch07_robustness_curves.py compare --seeds 0

  # Full pipeline (Reach + Push)
  python scripts/ch07_robustness_curves.py all --seeds 0

  # Quick smoke test (fewer episodes)
  python scripts/ch07_robustness_curves.py all --seeds 0 --n-eval-episodes 20

  # Multi-seed experiment
  python scripts/ch07_robustness_curves.py all --seeds 0,1,2 --include-push
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

    # obs-sweep
    p_obs = sub.add_parser("obs-sweep",
                           help="Observation noise sweep",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_obs)

    # act-sweep
    p_act = sub.add_parser("act-sweep",
                           help="Action noise sweep",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_act)

    # mitigate
    p_mit = sub.add_parser("mitigate",
                           help="LPF mitigation under action noise",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_mit.add_argument("--mitigation-act-noise", type=float,
                       default=DEFAULT_CONFIG.mitigation_act_noise,
                       help="Fixed action noise for mitigation test")
    _add_common_args(p_mit)

    # compare
    p_cmp = sub.add_parser("compare",
                           help="Compare all results",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(p_cmp)

    # all
    p_all = sub.add_parser("all",
                           help="Full pipeline: train -> obs -> act -> mitigate -> compare",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_all.add_argument("--env", default=DEFAULT_CONFIG.env,
                       help="Primary environment ID")
    p_all.add_argument("--n-envs", type=int, default=DEFAULT_CONFIG.n_envs,
                       help="Number of parallel envs for training fallback")
    p_all.add_argument("--total-steps", type=int, default=DEFAULT_CONFIG.total_steps,
                       help="Total training timesteps for training fallback")
    p_all.add_argument("--mitigation-act-noise", type=float,
                       default=DEFAULT_CONFIG.mitigation_act_noise,
                       help="Fixed action noise for mitigation test")
    _add_common_args(p_all)

    args = parser.parse_args()

    if args.mujoco_gl is None:
        os.environ.setdefault("MUJOCO_GL", "disable")
    else:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    handlers = {
        "train": cmd_train,
        "obs-sweep": cmd_obs_sweep,
        "act-sweep": cmd_act_sweep,
        "mitigate": cmd_mitigate,
        "compare": cmd_compare,
        "all": cmd_all,
    }

    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
