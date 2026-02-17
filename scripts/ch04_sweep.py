#!/usr/bin/env python3
"""Hyperparameter sweep for finding robust SAC+HER settings on FetchPush.

Goal: Find hyperparameters that achieve high success rate with LOW VARIANCE
across seeds. A good configuration should work reliably, not just with lucky seeds.

NOTE: By default uses ch04_her_sparse_reach_push-wokred-once.py (the proven working script).
      Use --script to select which training script to use:
        --script old    Uses ch04_her_sparse_reach_push-wokred-once.py (default)
        --script new    Uses ch04_her_sparse_reach_push.py
        --script both   Runs both scripts for direct comparison

Usage:
    # Run full sweep (12 configs × 5 seeds = 60 runs)
    python scripts/ch04_sweep.py run --env FetchPush-v4 --total-steps 500000

    # Run quick test (2 configs × 2 seeds = 4 runs)
    python scripts/ch04_sweep.py run --env FetchPush-v4 --total-steps 100000 --quick

    # Analyze results after sweep completes
    python scripts/ch04_sweep.py analyze

    # Resume interrupted sweep (skips completed runs)
    python scripts/ch04_sweep.py run --env FetchPush-v4 --total-steps 500000 --resume
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


SCRIPT_OLD = "scripts/ch04_her_sparse_reach_push-wokred-once.py"
SCRIPT_NEW = "scripts/ch04_her_sparse_reach_push.py"


@dataclass
class SweepConfig:
    """Single configuration to test."""
    ent_coef: float
    n_sampled_goal: int
    learning_starts: int
    gamma: float = 0.98
    script: str = "old"  # "old", "new", or path

    @property
    def script_path(self) -> str:
        if self.script == "old":
            return SCRIPT_OLD
        elif self.script == "new":
            return SCRIPT_NEW
        return self.script

    @property
    def script_tag(self) -> str:
        if self.script == "old":
            return "old"
        elif self.script == "new":
            return "new"
        return "custom"

    @property
    def name(self) -> str:
        base = f"ent{self.ent_coef}_nsg{self.n_sampled_goal}_ls{self.learning_starts}_g{self.gamma}"
        return f"{base}_{self.script_tag}"


def get_sweep_configs(quick: bool = False, script: str = "old") -> list[SweepConfig]:
    """Generate all configurations to sweep.

    Args:
        quick: If True, only generate 2 configs for quick testing
        script: Which script(s) to use - "old", "new", or "both"
    """
    scripts = ["old", "new"] if script == "both" else [script]

    if quick:
        # Quick test: just 2 configs per script
        configs = []
        for s in scripts:
            configs.extend([
                SweepConfig(ent_coef=0.1, n_sampled_goal=4, learning_starts=5000, gamma=0.98, script=s),
                SweepConfig(ent_coef=0.1, n_sampled_goal=8, learning_starts=5000, gamma=0.98, script=s),
            ])
        return configs

    # Full sweep grid
    ent_coefs = [0.05, 0.1, 0.2]
    n_sampled_goals = [4, 8]
    learning_starts_values = [1000, 5000]
    gammas = [0.95, 0.98]  # 0.99 is SB3 default but often too high for 50-step Fetch

    configs = []
    for s in scripts:
        for ent, nsg, ls, gamma in itertools.product(ent_coefs, n_sampled_goals, learning_starts_values, gammas):
            configs.append(SweepConfig(ent_coef=ent, n_sampled_goal=nsg, learning_starts=ls, gamma=gamma, script=s))

    return configs


def get_seeds(quick: bool = False) -> list[int]:
    """Seeds to test each configuration with."""
    if quick:
        return [0, 42]
    return [0, 1, 2, 42, 77]


def run_single(
    env: str,
    cfg: SweepConfig,
    seed: int,
    total_steps: int,
    sweep_dir: Path,
    parallel: bool = False,
) -> dict | None:
    """Run a single train + eval experiment."""

    ckpt_name = f"sweep_{cfg.name}_seed{seed}"
    ckpt_path = sweep_dir / "checkpoints" / f"{ckpt_name}.zip"
    eval_path = sweep_dir / "results" / f"{ckpt_name}_eval.json"

    # Skip if already completed
    if eval_path.exists():
        print(f"[sweep] Skipping {cfg.name} seed={seed} (already done)")
        with open(eval_path) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"[sweep] Config: {cfg.name}, Seed: {seed}")
    print(f"[sweep] ent_coef={cfg.ent_coef}, n_sampled_goal={cfg.n_sampled_goal}, learning_starts={cfg.learning_starts}")
    print(f"{'='*70}\n")

    # Ensure directories exist
    (sweep_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "results").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "runs").mkdir(parents=True, exist_ok=True)

    # Train using selected script
    # Use script-specific log dir to avoid conflicts when running old/new in parallel
    # Isolate TB logs per config to avoid SB3's auto-increment (seed0 -> seed0_1)
    log_dir = sweep_dir / "runs" / cfg.script_tag / cfg.name

    train_cmd = [
        sys.executable,
        cfg.script_path,
        "train",
        "--env", env,
        "--seed", str(seed),
        "--total-steps", str(total_steps),
        "--her",
        "--ent-coef", str(cfg.ent_coef),
        "--n-sampled-goal", str(cfg.n_sampled_goal),
        "--learning-starts", str(cfg.learning_starts),
        "--gamma", str(cfg.gamma),
        "--out", str(ckpt_path),  # include .zip to avoid pathlib suffix bugs with dotted names
        "--log-dir", str(log_dir),
    ]

    # When running in parallel, redirect subprocess output to log files
    (sweep_dir / "logs").mkdir(parents=True, exist_ok=True)
    run_log = sweep_dir / "logs" / f"{ckpt_name}.log"
    log_fh = open(run_log, "w") if parallel else None
    sub_kwargs: dict = {"cwd": Path(__file__).parent.parent}
    if log_fh:
        sub_kwargs["stdout"] = log_fh
        sub_kwargs["stderr"] = subprocess.STDOUT

    t0 = time.perf_counter()
    result = subprocess.run(train_cmd, **sub_kwargs)
    train_time = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"[sweep] Training FAILED for {cfg.name} seed={seed}" +
              (f" (see {run_log})" if log_fh else ""))
        if log_fh:
            log_fh.close()
        return None

    # Eval
    eval_cmd = [
        sys.executable,
        "eval.py",
        "--ckpt", str(ckpt_path),
        "--env", env,
        "--n-episodes", "100",
        "--seeds", "0-99",
        "--deterministic",
        "--json-out", str(eval_path),
    ]

    result = subprocess.run(eval_cmd, **sub_kwargs)
    if log_fh:
        log_fh.close()

    if result.returncode != 0:
        print(f"[sweep] Evaluation FAILED for {cfg.name} seed={seed}" +
              (f" (see {run_log})" if log_fh else ""))
        return None

    # Load and return results
    with open(eval_path) as f:
        eval_data = json.load(f)

    # Add sweep metadata
    eval_data["sweep_config"] = {
        "ent_coef": cfg.ent_coef,
        "n_sampled_goal": cfg.n_sampled_goal,
        "learning_starts": cfg.learning_starts,
        "gamma": cfg.gamma,
        "script": cfg.script_tag,
        "script_path": cfg.script_path,
        "seed": seed,
        "total_steps": total_steps,
        "train_time_sec": train_time,
    }

    # Rewrite with metadata
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    return eval_data


def cmd_run(args: argparse.Namespace) -> int:
    """Run the sweep."""
    env = args.env
    total_steps = args.total_steps
    quick = args.quick
    script = args.script
    parallel = args.parallel

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)

    configs = get_sweep_configs(quick=quick, script=script)
    seeds = get_seeds(quick=quick)

    total_runs = len(configs) * len(seeds)
    est_time_per_run = total_steps / 500 / 60  # rough estimate: 500 steps/sec
    est_total_hours = total_runs * est_time_per_run / 60 / max(1, parallel)

    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SWEEP: {env}")
    print(f"{'='*70}")
    print(f"Configurations: {len(configs)}")
    print(f"Seeds per config: {len(seeds)}")
    print(f"Total runs: {total_runs}")
    print(f"Steps per run: {total_steps:,}")
    print(f"Parallel workers: {parallel}")
    print(f"Script mode: {script}")
    print(f"Estimated time: ~{est_total_hours:.1f} hours")
    print(f"Output directory: {sweep_dir}")
    print(f"{'='*70}\n")

    # Save sweep plan
    plan = {
        "env": env,
        "total_steps": total_steps,
        "script_mode": script,
        "parallel": parallel,
        "configs": [
            {"ent_coef": c.ent_coef, "n_sampled_goal": c.n_sampled_goal, "learning_starts": c.learning_starts, "gamma": c.gamma, "script": c.script_tag}
            for c in configs
        ],
        "seeds": seeds,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (sweep_dir / "sweep_plan.json").write_text(json.dumps(plan, indent=2))

    # Build list of all (config, seed) pairs to run
    jobs = [(cfg, seed) for cfg in configs for seed in seeds]
    all_results = []
    completed = 0

    if parallel <= 1:
        # Sequential execution
        for cfg, seed in jobs:
            completed += 1
            print(f"\n[sweep] Progress: {completed}/{total_runs}")

            result = run_single(env, cfg, seed, total_steps, sweep_dir)
            if result:
                all_results.append({
                    "config": cfg.name,
                    "ent_coef": cfg.ent_coef,
                    "n_sampled_goal": cfg.n_sampled_goal,
                    "learning_starts": cfg.learning_starts,
                    "gamma": cfg.gamma,
                    "script": cfg.script_tag,
                    "seed": seed,
                    "success_rate": result["aggregate"]["success_rate"],
                    "return_mean": result["aggregate"]["return_mean"],
                })
    else:
        # Parallel execution
        print(f"[sweep] Running {parallel} experiments in parallel...")
        print(f"[sweep] Per-run logs: {sweep_dir / 'logs' / '*.log'}\n")

        def run_job(job_tuple):
            cfg, seed = job_tuple
            result = run_single(env, cfg, seed, total_steps, sweep_dir, parallel=True)
            return cfg, seed, result

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(run_job, job): job for job in jobs}

            for future in as_completed(futures):
                completed += 1
                cfg, seed, result = future.result()
                print(f"[sweep] Completed {completed}/{total_runs}: {cfg.name} seed={seed}")

                if result:
                    all_results.append({
                        "config": cfg.name,
                        "ent_coef": cfg.ent_coef,
                        "n_sampled_goal": cfg.n_sampled_goal,
                        "learning_starts": cfg.learning_starts,
                        "gamma": cfg.gamma,
                        "script": cfg.script_tag,
                        "seed": seed,
                        "success_rate": result["aggregate"]["success_rate"],
                        "return_mean": result["aggregate"]["return_mean"],
                    })

    # Save raw results
    (sweep_dir / "sweep_results_raw.json").write_text(json.dumps(all_results, indent=2))

    print(f"\n[sweep] Complete! Results saved to {sweep_dir}")
    print(f"[sweep] Run 'python scripts/ch04_sweep.py analyze --sweep-dir {sweep_dir}' to analyze")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze sweep results."""
    import math

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()

    # Load all eval results
    results_dir = sweep_dir / "results"
    if not results_dir.exists():
        print(f"[sweep] No results found in {results_dir}")
        return 1

    # Group results by config
    config_results: dict[str, list[dict]] = {}

    for eval_file in results_dir.glob("sweep_*_eval.json"):
        with open(eval_file) as f:
            data = json.load(f)

        if "sweep_config" not in data:
            continue

        cfg = data["sweep_config"]
        gamma = cfg.get("gamma", 0.98)  # default for backwards compat
        script = cfg.get("script", "old")  # default for backwards compat
        config_name = f"ent{cfg['ent_coef']}_nsg{cfg['n_sampled_goal']}_ls{cfg['learning_starts']}_g{gamma}_{script}"

        if config_name not in config_results:
            config_results[config_name] = []

        config_results[config_name].append({
            "seed": cfg["seed"],
            "success_rate": data["aggregate"]["success_rate"],
            "return_mean": data["aggregate"]["return_mean"],
            "ent_coef": cfg["ent_coef"],
            "n_sampled_goal": cfg["n_sampled_goal"],
            "learning_starts": cfg["learning_starts"],
            "gamma": gamma,
            "script": script,
        })

    if not config_results:
        print("[sweep] No sweep results found")
        return 1

    # Compute statistics for each config
    def compute_stats(values: list[float]) -> dict:
        n = len(values)
        if n == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}
        mean = sum(values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0
        return {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "n": n,
        }

    summary = []
    for config_name, results in config_results.items():
        success_rates = [r["success_rate"] for r in results]
        stats = compute_stats(success_rates)

        # First result has the config details
        cfg = results[0]

        summary.append({
            "config": config_name,
            "ent_coef": cfg["ent_coef"],
            "n_sampled_goal": cfg["n_sampled_goal"],
            "learning_starts": cfg["learning_starts"],
            "gamma": cfg["gamma"],
            "script": cfg["script"],
            "success_rate_mean": stats["mean"],
            "success_rate_std": stats["std"],
            "success_rate_min": stats["min"],
            "success_rate_max": stats["max"],
            "n_seeds": stats["n"],
            "spread": stats["max"] - stats["min"],  # Range of results
        })

    # Sort by mean success rate (descending), then by std (ascending for stability)
    summary.sort(key=lambda x: (-x["success_rate_mean"], x["success_rate_std"]))

    # Print results
    print(f"\n{'='*95}")
    print("SWEEP RESULTS: Ranked by Success Rate (higher = better) and Stability (lower std = better)")
    print(f"{'='*95}\n")

    print(f"{'Rank':<5} {'Config':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Seeds':>6}")
    print("-" * 95)

    for i, s in enumerate(summary, 1):
        stability = "✓ STABLE" if s["success_rate_std"] < 0.15 and s["success_rate_mean"] > 0.7 else ""
        print(f"{i:<5} {s['config']:<35} {s['success_rate_mean']:>7.1%} {s['success_rate_std']:>7.1%} "
              f"{s['success_rate_min']:>7.1%} {s['success_rate_max']:>7.1%} {s['n_seeds']:>6}  {stability}")

    print("-" * 90)

    # Find best stable config
    stable_configs = [s for s in summary if s["success_rate_std"] < 0.15 and s["success_rate_mean"] > 0.7]

    if stable_configs:
        best = stable_configs[0]
        print(f"\n✓ RECOMMENDED CONFIG (high success + low variance):")
        print(f"  --ent-coef {best['ent_coef']} --n-sampled-goal {best['n_sampled_goal']} --learning-starts {best['learning_starts']} --gamma {best['gamma']}")
        print(f"  Script: {best['script']}")
        print(f"  Expected success rate: {best['success_rate_mean']:.1%} ± {best['success_rate_std']:.1%}")

    # If we have both scripts, show comparison
    scripts_used = set(s["script"] for s in summary)
    if len(scripts_used) > 1:
        print(f"\n{'='*95}")
        print("SCRIPT COMPARISON (same hyperparams, different scripts)")
        print(f"{'='*95}\n")

        # Group by hyperparams (without script)
        from collections import defaultdict
        by_params = defaultdict(dict)
        for s in summary:
            param_key = f"ent{s['ent_coef']}_nsg{s['n_sampled_goal']}_ls{s['learning_starts']}_g{s['gamma']}"
            by_params[param_key][s["script"]] = s

        print(f"{'Hyperparams':<30} | {'Old Mean':>10} | {'New Mean':>10} | {'Diff':>10} | {'Winner':>8}")
        print("-" * 80)

        for param_key, scripts in sorted(by_params.items()):
            if "old" in scripts and "new" in scripts:
                old_sr = scripts["old"]["success_rate_mean"]
                new_sr = scripts["new"]["success_rate_mean"]
                diff = new_sr - old_sr
                winner = "OLD" if old_sr > new_sr + 0.02 else ("NEW" if new_sr > old_sr + 0.02 else "TIE")
                print(f"{param_key:<30} | {old_sr:>9.1%} | {new_sr:>9.1%} | {diff:>+9.1%} | {winner:>8}")
    else:
        print("\n⚠ No stable configuration found (all have high variance or low success rate)")
        print("  Consider: longer training, different hyperparameter ranges, or curriculum learning")

    # Save analysis
    analysis = {
        "summary": summary,
        "recommended": stable_configs[0] if stable_configs else None,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    (sweep_dir / "sweep_analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"\n[sweep] Analysis saved to {sweep_dir / 'sweep_analysis.json'}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for robust SAC+HER settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser("run", help="Run the sweep")
    p_run.add_argument("--env", default="FetchPush-v4", help="Environment")
    p_run.add_argument("--total-steps", type=int, default=500_000, help="Steps per run")
    p_run.add_argument("--sweep-dir", default="results/sweep", help="Output directory")
    p_run.add_argument("--quick", action="store_true", help="Quick test (2 configs × 2 seeds)")
    p_run.add_argument("--resume", action="store_true", help="Skip completed runs")
    p_run.add_argument("--script", choices=["old", "new", "both"], default="old",
                       help="Which training script to use: old (proven), new, or both (for comparison)")
    import os
    n_cpus = os.cpu_count() or 1
    n_envs_per_run = 8  # each run uses --n-envs 8
    auto_parallel = max(1, n_cpus // (n_envs_per_run + 1))  # +1 for the main training process
    p_run.add_argument("--parallel", type=int, default=auto_parallel, metavar="N",
                       help=f"Run N experiments in parallel (auto-detected: {auto_parallel} for {n_cpus} CPUs).")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze sweep results")
    p_analyze.add_argument("--sweep-dir", default="results/sweep", help="Sweep directory")

    args = parser.parse_args()

    if args.cmd == "run":
        return cmd_run(args)
    elif args.cmd == "analyze":
        return cmd_analyze(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
