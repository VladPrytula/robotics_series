#!/usr/bin/env python3
"""
Chapter 2: PPO on Dense Reach -- Pipeline Truth Serum

This script validates the training infrastructure by running PPO on
FetchReachDense-v4, the simplest task with continuous reward signal.

Usage:
    # Full pipeline: train + evaluate + report
    python scripts/ch02_ppo_dense_reach.py all --seed 0

    # Training only (quick sanity check)
    python scripts/ch02_ppo_dense_reach.py train --total-steps 100000

    # Evaluation only (requires existing checkpoint)
    python scripts/ch02_ppo_dense_reach.py eval --seed 0

    # Multi-seed experiment
    for seed in 0 1 2 3 4; do
        python scripts/ch02_ppo_dense_reach.py all --seed $seed
    done
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Configuration
ENV_ID = "FetchReachDense-v4"
ALGO = "ppo"
DEFAULT_TOTAL_STEPS = 1_000_000
DEFAULT_N_ENVS = 8
DEFAULT_N_EVAL_EPISODES = 100

# Expected results (for validation)
EXPECTED_SUCCESS_RATE_MIN = 0.90
EXPECTED_MEAN_RETURN_MIN = -10.0


def get_checkpoint_path(seed: int) -> Path:
    """Return the checkpoint path for a given seed."""
    return Path("checkpoints") / f"{ALGO}_{ENV_ID}_seed{seed}.zip"


def get_eval_output_path(seed: int) -> Path:
    """Return the evaluation output path for a given seed."""
    return Path("results") / f"ch02_{ALGO}_{ENV_ID.lower()}_seed{seed}_eval.json"


def check_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            print(f"[ch02] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[ch02] WARNING: No GPU detected, training will be slow")
        return available
    except ImportError:
        print("[ch02] WARNING: PyTorch not available, cannot check GPU")
        return False


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"[ch02] {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[ch02] ERROR: {description} failed with code {result.returncode}")
    else:
        print(f"\n[ch02] OK: {description} completed successfully")

    return result.returncode


def cmd_train(args: argparse.Namespace) -> int:
    """Train PPO on FetchReachDense-v4."""
    check_gpu()
    checkpoint_path = get_checkpoint_path(args.seed)

    cmd = [
        sys.executable, "train.py",
        "--algo", ALGO,
        "--env", ENV_ID,
        "--seed", str(args.seed),
        "--n-envs", str(args.n_envs),
        "--total-steps", str(args.total_steps),
        "--out", str(checkpoint_path.with_suffix("")),  # train.py adds .zip
        "--track", "tb",
        "--ppo-n-steps", "1024",
        "--ppo-batch-size", "256",
    ]

    return run_command(cmd, f"Training {ALGO.upper()} on {ENV_ID} (seed={args.seed})")


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate the trained checkpoint."""
    checkpoint_path = get_checkpoint_path(args.seed)
    eval_output = get_eval_output_path(args.seed)

    if not checkpoint_path.exists():
        print(f"[ch02] ERROR: Checkpoint not found: {checkpoint_path}")
        print(f"[ch02] Run 'python scripts/ch02_ppo_dense_reach.py train --seed {args.seed}' first.")
        return 1

    # Ensure output directory exists
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "eval.py",
        "--ckpt", str(checkpoint_path),
        "--env", ENV_ID,
        "--n-episodes", str(args.n_eval_episodes),
        "--seeds", f"0-{args.n_eval_episodes - 1}",
        "--deterministic",
        "--json-out", str(eval_output),
    ]

    return run_command(cmd, f"Evaluating checkpoint (seed={args.seed})")


def cmd_report(args: argparse.Namespace) -> int:
    """Generate and display a summary report."""
    eval_output = get_eval_output_path(args.seed)
    checkpoint_path = get_checkpoint_path(args.seed)
    meta_path = checkpoint_path.with_suffix(".meta.json")

    print(f"\n{'='*60}")
    print(f"[ch02] Chapter 2 Summary Report")
    print(f"{'='*60}")

    # Load evaluation results
    if not eval_output.exists():
        print(f"[ch02] WARNING: Evaluation output not found: {eval_output}")
        return 1

    with open(eval_output) as f:
        eval_data = json.load(f)

    # Extract metrics
    aggregate = eval_data.get("aggregate", {})
    success_rate = aggregate.get("success_rate", 0.0)
    mean_return = aggregate.get("return_mean", float("-inf"))
    mean_goal_dist = aggregate.get("final_distance_mean", float("inf"))
    n_episodes = eval_data.get("n_episodes", 0)

    # Load training metadata
    training_info: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            training_info = json.load(f)

    # Display results
    print(f"\nEnvironment: {ENV_ID}")
    print(f"Algorithm: {ALGO.upper()}")
    print(f"Seed: {args.seed}")
    print(f"Total training steps: {training_info.get('total_steps', 'unknown')}")
    print(f"Evaluation episodes: {n_episodes}")

    print(f"\n--- Results ---")
    print(f"Success Rate:       {success_rate:.1%} (expected > {EXPECTED_SUCCESS_RATE_MIN:.0%})")
    print(f"Mean Return:        {mean_return:.2f} (expected > {EXPECTED_MEAN_RETURN_MIN:.1f})")
    print(f"Mean Goal Distance: {mean_goal_dist:.4f}m")

    # Validation
    print(f"\n--- Validation ---")
    passed = True

    if success_rate >= EXPECTED_SUCCESS_RATE_MIN:
        print(f"[PASS] Success rate >= {EXPECTED_SUCCESS_RATE_MIN:.0%}")
    else:
        print(f"[FAIL] Success rate {success_rate:.1%} < {EXPECTED_SUCCESS_RATE_MIN:.0%}")
        passed = False

    if mean_return >= EXPECTED_MEAN_RETURN_MIN:
        print(f"[PASS] Mean return >= {EXPECTED_MEAN_RETURN_MIN:.1f}")
    else:
        print(f"[FAIL] Mean return {mean_return:.2f} < {EXPECTED_MEAN_RETURN_MIN:.1f}")
        passed = False

    # Summary
    print(f"\n--- Files Generated ---")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metadata:   {meta_path}")
    print(f"Eval JSON:  {eval_output}")
    print(f"TB Logs:    runs/{ALGO}/{ENV_ID}/seed{args.seed}/")

    if passed:
        print(f"\n[ch02] SUCCESS: PPO baseline validated. Pipeline is working correctly.")
        print(f"[ch02] You can now proceed to Chapter 3 (SAC on dense Reach).")
        return 0
    else:
        print(f"\n[ch02] FAILURE: Results below expected thresholds.")
        print(f"[ch02] Check TensorBoard logs for diagnostic signals.")
        print(f"[ch02] See tutorials/ch02_ppo_dense_reach.md Section 5 for debugging.")
        return 1


def cmd_all(args: argparse.Namespace) -> int:
    """Run full pipeline: train + eval + report."""
    # Train
    ret = cmd_train(args)
    if ret != 0:
        return ret

    # Evaluate
    ret = cmd_eval(args)
    if ret != 0:
        return ret

    # Report
    return cmd_report(args)


def cmd_multi_seed(args: argparse.Namespace) -> int:
    """Run experiment across multiple seeds and aggregate results."""
    seeds = list(range(args.seeds))
    results: list[dict[str, Any]] = []

    print(f"\n[ch02] Multi-seed experiment: seeds {seeds}")

    for seed in seeds:
        args.seed = seed
        ret = cmd_all(args)

        # Collect results even if validation failed
        eval_output = get_eval_output_path(seed)
        if eval_output.exists():
            with open(eval_output) as f:
                data = json.load(f)
            results.append({
                "seed": seed,
                "success_rate": data.get("summary", {}).get("success_rate", 0.0),
                "return_mean": data.get("summary", {}).get("return_mean", 0.0),
            })

    # Aggregate
    if results:
        import statistics

        success_rates = [r["success_rate"] for r in results]
        returns = [r["return_mean"] for r in results]

        print(f"\n{'='*60}")
        print(f"[ch02] Multi-Seed Aggregate Results ({len(results)} seeds)")
        print(f"{'='*60}")
        print(f"Success Rate: {statistics.mean(success_rates):.1%} +/- {statistics.stdev(success_rates):.1%}")
        print(f"Mean Return:  {statistics.mean(returns):.2f} +/- {statistics.stdev(returns):.2f}")

        # Save aggregate
        aggregate_path = Path("results") / "ch02_ppo_reachdense_aggregate.json"
        aggregate_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "env_id": ENV_ID,
            "algo": ALGO,
            "seeds": seeds,
            "per_seed": results,
            "aggregate": {
                "success_rate_mean": statistics.mean(success_rates),
                "success_rate_std": statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
                "return_mean": statistics.mean(returns),
                "return_std": statistics.stdev(returns) if len(returns) > 1 else 0.0,
            },
        }
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(json.dumps(aggregate_data, indent=2) + "\n")
        print(f"\nAggregate saved to: {aggregate_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter 2: PPO on Dense Reach -- Pipeline Truth Serum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=0, help="Random seed")
        p.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS, help="Number of parallel envs")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train PPO on FetchReachDense-v4")
    add_common_args(train_parser)
    train_parser.add_argument(
        "--total-steps", type=int, default=DEFAULT_TOTAL_STEPS,
        help="Total training timesteps"
    )
    train_parser.set_defaults(func=cmd_train)

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained checkpoint")
    add_common_args(eval_parser)
    eval_parser.add_argument(
        "--n-eval-episodes", type=int, default=DEFAULT_N_EVAL_EPISODES,
        help="Number of evaluation episodes"
    )
    eval_parser.set_defaults(func=cmd_eval)

    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Generate summary report")
    add_common_args(report_parser)
    report_parser.set_defaults(func=cmd_report)

    # All subcommand
    all_parser = subparsers.add_parser("all", help="Run full pipeline: train + eval + report")
    add_common_args(all_parser)
    all_parser.add_argument(
        "--total-steps", type=int, default=DEFAULT_TOTAL_STEPS,
        help="Total training timesteps"
    )
    all_parser.add_argument(
        "--n-eval-episodes", type=int, default=DEFAULT_N_EVAL_EPISODES,
        help="Number of evaluation episodes"
    )
    all_parser.set_defaults(func=cmd_all)

    # Multi-seed subcommand
    multi_parser = subparsers.add_parser("multi-seed", help="Run experiment across multiple seeds")
    multi_parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (0 to N-1)")
    multi_parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    multi_parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    multi_parser.add_argument("--n-eval-episodes", type=int, default=DEFAULT_N_EVAL_EPISODES)
    multi_parser.set_defaults(func=cmd_multi_seed)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
