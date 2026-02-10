#!/usr/bin/env python3
"""Chapter 04: Sparse Reach + Push -- Introduce HER Where It Matters

Week 4 goals:
1. Demonstrate that HER is the difference-maker on sparse goals
2. Train SAC on sparse Reach/Push WITHOUT HER (baseline difficulty)
3. Train SAC + HER on sparse Reach/Push (the solution)
4. Compare with clear separation in success rates
5. Multi-seed experiments for statistical validity

Usage:
    # Full pipeline for Reach (train no-HER, train HER, compare)
    python scripts/ch04_her_sparse_reach_push.py reach-all --seeds 0,1,2

    # Full pipeline for Push
    python scripts/ch04_her_sparse_reach_push.py push-all --seeds 0,1,2

    # Train single experiment
    python scripts/ch04_her_sparse_reach_push.py train --env FetchReach-v4 --her --seed 0

    # Evaluate checkpoint
    python scripts/ch04_her_sparse_reach_push.py eval --ckpt checkpoints/sac_her_FetchReach-v4_seed0.zip

    # Compare HER vs no-HER results
    python scripts/ch04_her_sparse_reach_push.py compare --env FetchReach-v4 --seeds 0,1,2

    # Ablation: sweep n_sampled_goal values
    python scripts/ch04_her_sparse_reach_push.py ablation --env FetchPush-v4 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Disable rendering for training
os.environ.setdefault("MUJOCO_GL", "disable")


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


def _parse_seeds(seeds_str: str) -> list[int]:
    """Parse seeds string like '0,1,2' or '0-4' into list of ints."""
    seeds = []
    for part in seeds_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _ckpt_name(env_id: str, her: bool, seed: int, n_sampled_goal: int | None = None) -> str:
    """Generate checkpoint name."""
    base = f"sac_{'her_' if her else ''}{env_id}_seed{seed}"
    if her and n_sampled_goal is not None and n_sampled_goal != 4:
        base += f"_nsg{n_sampled_goal}"
    return base


def _result_name(env_id: str, her: bool, seed: int, n_sampled_goal: int | None = None) -> str:
    """Generate result JSON name."""
    base = f"ch04_sac_{'her_' if her else ''}{env_id.lower()}_seed{seed}"
    if her and n_sampled_goal is not None and n_sampled_goal != 4:
        base += f"_nsg{n_sampled_goal}"
    return base + "_eval.json"


def cmd_train(args: argparse.Namespace) -> int:
    """Train SAC on sparse env, optionally with HER."""
    import gymnasium_robotics  # noqa: F401

    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    env_id = args.env
    seed = args.seed
    n_envs = args.n_envs
    total_steps = args.total_steps
    use_her = args.her
    n_sampled_goal = args.n_sampled_goal
    goal_strategy = args.goal_selection_strategy

    device = "cuda"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    her_str = f"HER (n_sampled_goal={n_sampled_goal}, strategy={goal_strategy})" if use_her else "no-HER"
    print(f"[ch04] Training SAC on {env_id} with {her_str}")
    print(f"[ch04] seed={seed}, n_envs={n_envs}, total_steps={total_steps}, device={device}")

    set_random_seed(seed)

    # Setup paths
    out_path = Path(args.out) if args.out else Path("checkpoints") / _ckpt_name(
        env_id, use_her, seed, n_sampled_goal if use_her else None
    )
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir).expanduser().resolve()
    her_tag = f"her_nsg{n_sampled_goal}" if use_her else "noher"
    run_id = f"sac_{her_tag}/{env_id}/seed{seed}"

    # Create environment
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed)

    try:
        # Setup HER if requested
        replay_buffer_class = None
        replay_buffer_kwargs = None

        if use_her:
            try:
                from stable_baselines3 import HerReplayBuffer
            except ImportError:
                from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

            replay_buffer_class = HerReplayBuffer
            replay_buffer_kwargs = {
                "n_sampled_goal": n_sampled_goal,
                "goal_selection_strategy": goal_strategy,
            }

        # Create SAC model
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            ent_coef="auto",
            learning_rate=3e-4,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
        )

        # Train
        print(f"[ch04] Starting training for {total_steps} steps...")
        t0 = time.perf_counter()
        model.learn(total_timesteps=total_steps, tb_log_name=run_id)
        elapsed = time.perf_counter() - t0
        steps_per_sec = total_steps / elapsed

        print(f"[ch04] Training complete in {elapsed:.1f}s ({steps_per_sec:.0f} steps/sec)")

        # Save checkpoint
        model.save(str(out_path))
        print(f"[ch04] Saved checkpoint: {out_path}.zip")

    finally:
        env.close()

    # Write metadata
    suffix = "" if str(out_path).endswith(".zip") else ".zip"
    meta_path = out_path.with_suffix(".meta.json")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algo": "sac",
        "env_id": env_id,
        "seed": seed,
        "device": device,
        "n_envs": n_envs,
        "total_steps": total_steps,
        "training_time_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "checkpoint": str(out_path) + suffix,
        "her": use_her,
        "hyperparams": {
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "ent_coef": "auto",
            "learning_rate": 3e-4,
        },
        "versions": _gather_versions(),
    }
    if use_her:
        metadata["her_config"] = {
            "n_sampled_goal": n_sampled_goal,
            "goal_selection_strategy": goal_strategy,
        }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ch04] Wrote metadata: {meta_path}")

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate checkpoint on sparse env."""
    ckpt = args.ckpt
    if not ckpt:
        print("[ch04] Error: --ckpt required")
        return 1

    # Infer env from checkpoint name if not specified
    env_id = args.env
    if not env_id:
        ckpt_name = Path(ckpt).stem
        if "FetchReach-v4" in ckpt_name:
            env_id = "FetchReach-v4"
        elif "FetchPush-v4" in ckpt_name:
            env_id = "FetchPush-v4"
        else:
            print("[ch04] Error: Could not infer env from checkpoint name. Use --env")
            return 1

    n_episodes = args.n_episodes
    out_json = args.json_out
    if not out_json:
        # Generate from checkpoint name
        ckpt_stem = Path(ckpt).stem
        out_json = f"results/ch04_{ckpt_stem}_eval.json"

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "eval.py",
        "--ckpt",
        ckpt,
        "--env",
        env_id,
        "--n-episodes",
        str(n_episodes),
        "--seeds",
        f"0-{n_episodes - 1}",
        "--deterministic",
        "--json-out",
        out_json,
    ]

    print(f"[ch04] Evaluating: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        print(f"[ch04] Evaluation complete: {out_json}")
    return result.returncode


def _load_eval_results(pattern: str) -> list[dict[str, Any]]:
    """Load all eval JSON files matching pattern."""
    results_dir = Path("results")
    files = sorted(results_dir.glob(pattern))
    results = []
    for f in files:
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except Exception as e:
            print(f"[ch04] Warning: Could not load {f}: {e}")
    return results


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, and 95% CI for a list of values."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}

    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
        # 95% CI using t-distribution approximation (for small n, use 2.0 as rough multiplier)
        ci95 = 1.96 * std / math.sqrt(n) if n >= 30 else 2.0 * std / math.sqrt(n)
    else:
        std = 0.0
        ci95 = 0.0

    return {"mean": mean, "std": std, "ci95": ci95, "n": n}


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare HER vs no-HER results across seeds."""
    env_id = args.env
    seeds = _parse_seeds(args.seeds)

    env_lower = env_id.lower()

    print(f"\n[ch04] Comparing HER vs no-HER on {env_id}")
    print(f"[ch04] Seeds: {seeds}")

    # Load no-HER results
    noher_results = []
    for seed in seeds:
        pattern = f"ch04_sac_{env_lower}_seed{seed}_eval.json"
        files = list(Path("results").glob(pattern))
        if files:
            with open(files[0]) as f:
                noher_results.append(json.load(f))

    # Load HER results
    her_results = []
    for seed in seeds:
        pattern = f"ch04_sac_her_{env_lower}_seed{seed}_eval.json"
        files = list(Path("results").glob(pattern))
        if files:
            with open(files[0]) as f:
                her_results.append(json.load(f))

    if not noher_results and not her_results:
        print(f"[ch04] No results found for {env_id}. Run training first.")
        return 1

    print("\n" + "=" * 70)
    print(f"Week 4: HER vs No-HER Comparison ({env_id})")
    print("=" * 70)

    # Compute statistics
    def get_success_rates(results: list[dict]) -> list[float]:
        return [r["aggregate"]["success_rate"] for r in results]

    def get_returns(results: list[dict]) -> list[float]:
        return [r["aggregate"]["return_mean"] for r in results]

    def get_distances(results: list[dict]) -> list[float]:
        return [r["aggregate"]["final_distance_mean"] for r in results]

    noher_sr = _compute_stats(get_success_rates(noher_results)) if noher_results else None
    her_sr = _compute_stats(get_success_rates(her_results)) if her_results else None

    noher_ret = _compute_stats(get_returns(noher_results)) if noher_results else None
    her_ret = _compute_stats(get_returns(her_results)) if her_results else None

    noher_dist = _compute_stats(get_distances(noher_results)) if noher_results else None
    her_dist = _compute_stats(get_distances(her_results)) if her_results else None

    # Print comparison table
    print(f"\n{'Metric':<25} | {'No-HER':>20} | {'HER':>20} | {'Delta':>12}")
    print("-" * 85)

    def fmt_stat(stat: dict | None, fmt: str = ".1%") -> str:
        if stat is None:
            return "N/A"
        if fmt == ".1%":
            return f"{stat['mean']:.1%} +/- {stat['ci95']:.1%}"
        elif fmt == ".3f":
            return f"{stat['mean']:.3f} +/- {stat['ci95']:.3f}"
        elif fmt == ".4f":
            return f"{stat['mean']:.4f} +/- {stat['ci95']:.4f}"
        return f"{stat['mean']} +/- {stat['ci95']}"

    # Success rate
    noher_str = fmt_stat(noher_sr, ".1%")
    her_str = fmt_stat(her_sr, ".1%")
    if noher_sr and her_sr:
        delta = her_sr["mean"] - noher_sr["mean"]
        delta_str = f"{delta:+.1%}"
    else:
        delta_str = "N/A"
    print(f"{'Success Rate':<25} | {noher_str:>20} | {her_str:>20} | {delta_str:>12}")

    # Return
    noher_str = fmt_stat(noher_ret, ".3f")
    her_str = fmt_stat(her_ret, ".3f")
    if noher_ret and her_ret:
        delta = her_ret["mean"] - noher_ret["mean"]
        delta_str = f"{delta:+.3f}"
    else:
        delta_str = "N/A"
    print(f"{'Return (mean)':<25} | {noher_str:>20} | {her_str:>20} | {delta_str:>12}")

    # Final distance
    noher_str = fmt_stat(noher_dist, ".4f")
    her_str = fmt_stat(her_dist, ".4f")
    if noher_dist and her_dist:
        delta = her_dist["mean"] - noher_dist["mean"]
        delta_str = f"{delta:+.4f}"
    else:
        delta_str = "N/A"
    print(f"{'Final Distance (mean)':<25} | {noher_str:>20} | {her_str:>20} | {delta_str:>12}")

    print("-" * 85)
    print(f"{'Seeds evaluated':<25} | {noher_sr['n'] if noher_sr else 0:>20} | {her_sr['n'] if her_sr else 0:>20} |")

    # Summary
    print("\n" + "=" * 70)
    if her_sr and noher_sr:
        improvement = her_sr["mean"] - noher_sr["mean"]
        if improvement > 0.5:  # >50% improvement
            print("CLEAR SEPARATION: HER dramatically outperforms no-HER on sparse rewards.")
            print(f"Success rate improvement: {improvement:.1%}")
        elif improvement > 0.1:
            print("MODERATE SEPARATION: HER shows improvement over no-HER.")
            print(f"Success rate improvement: {improvement:.1%}")
        else:
            print("WEAK/NO SEPARATION: HER and no-HER perform similarly.")
            print("This may indicate the task is too easy or training was insufficient.")

    # Save comparison report
    report = {
        "env_id": env_id,
        "seeds": seeds,
        "no_her": {
            "n_seeds": noher_sr["n"] if noher_sr else 0,
            "success_rate": noher_sr if noher_sr else None,
            "return": noher_ret if noher_ret else None,
            "final_distance": noher_dist if noher_dist else None,
        },
        "her": {
            "n_seeds": her_sr["n"] if her_sr else 0,
            "success_rate": her_sr if her_sr else None,
            "return": her_ret if her_ret else None,
            "final_distance": her_dist if her_dist else None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path(f"results/ch04_{env_lower}_comparison.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch04] Comparison report saved: {report_path}")

    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """Ablation study: sweep n_sampled_goal values."""
    env_id = args.env
    seed = args.seed
    nsg_values = [int(x) for x in args.nsg_values.split(",")]
    total_steps = args.total_steps

    print(f"\n[ch04] Ablation study: n_sampled_goal on {env_id}")
    print(f"[ch04] Values: {nsg_values}, seed={seed}, total_steps={total_steps}")

    results = []

    for nsg in nsg_values:
        print(f"\n{'='*60}")
        print(f"[ch04] Training with n_sampled_goal={nsg}")
        print("=" * 60)

        # Train
        train_args = argparse.Namespace(
            env=env_id,
            seed=seed,
            n_envs=args.n_envs,
            total_steps=total_steps,
            her=True,
            n_sampled_goal=nsg,
            goal_selection_strategy="future",
            out=None,
            log_dir=args.log_dir,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
        )
        ret = cmd_train(train_args)
        if ret != 0:
            print(f"[ch04] Training failed for nsg={nsg}")
            continue

        # Eval
        ckpt = f"checkpoints/{_ckpt_name(env_id, True, seed, nsg)}.zip"
        out_json = f"results/{_result_name(env_id, True, seed, nsg)}"

        eval_args = argparse.Namespace(
            ckpt=ckpt,
            env=env_id,
            n_episodes=100,
            json_out=out_json,
        )
        ret = cmd_eval(eval_args)
        if ret != 0:
            print(f"[ch04] Evaluation failed for nsg={nsg}")
            continue

        # Load results
        with open(out_json) as f:
            eval_data = json.load(f)

        results.append({
            "n_sampled_goal": nsg,
            "success_rate": eval_data["aggregate"]["success_rate"],
            "return_mean": eval_data["aggregate"]["return_mean"],
            "final_distance_mean": eval_data["aggregate"]["final_distance_mean"],
        })

    # Print summary
    print("\n" + "=" * 70)
    print(f"Ablation Results: n_sampled_goal on {env_id}")
    print("=" * 70)
    print(f"\n{'n_sampled_goal':>15} | {'Success Rate':>15} | {'Return':>12} | {'Final Dist':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['n_sampled_goal']:>15} | {r['success_rate']:>14.1%} | {r['return_mean']:>12.3f} | {r['final_distance_mean']:>12.4f}")

    # Save ablation report
    report = {
        "env_id": env_id,
        "seed": seed,
        "total_steps": total_steps,
        "results": results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = Path(f"results/ch04_{env_id.lower()}_ablation_nsg.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch04] Ablation report saved: {report_path}")

    return 0


def cmd_reach_all(args: argparse.Namespace) -> int:
    """Full pipeline for sparse Reach: no-HER baseline + HER + comparison."""
    env_id = "FetchReach-v4"
    seeds = _parse_seeds(args.seeds)
    total_steps = args.total_steps

    print("=" * 70)
    print("Week 4: Sparse Reach - Full Pipeline")
    print(f"Seeds: {seeds}, Total steps per run: {total_steps}")
    print("=" * 70)

    # Train no-HER baselines
    print("\n[1/3] Training SAC WITHOUT HER (baseline)...")
    for seed in seeds:
        print(f"\n--- Seed {seed} (no-HER) ---")
        train_args = argparse.Namespace(
            env=env_id,
            seed=seed,
            n_envs=args.n_envs,
            total_steps=total_steps,
            her=False,
            n_sampled_goal=4,
            goal_selection_strategy="future",
            out=None,
            log_dir=args.log_dir,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
        )
        ret = cmd_train(train_args)
        if ret != 0:
            print(f"[ch04] Training failed for seed {seed}")

        # Eval
        ckpt = f"checkpoints/{_ckpt_name(env_id, False, seed)}.zip"
        out_json = f"results/{_result_name(env_id, False, seed)}"
        eval_args = argparse.Namespace(ckpt=ckpt, env=env_id, n_episodes=100, json_out=out_json)
        cmd_eval(eval_args)

    # Train HER
    print("\n[2/3] Training SAC WITH HER...")
    for seed in seeds:
        print(f"\n--- Seed {seed} (HER) ---")
        train_args = argparse.Namespace(
            env=env_id,
            seed=seed,
            n_envs=args.n_envs,
            total_steps=total_steps,
            her=True,
            n_sampled_goal=4,
            goal_selection_strategy="future",
            out=None,
            log_dir=args.log_dir,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
        )
        ret = cmd_train(train_args)
        if ret != 0:
            print(f"[ch04] Training failed for seed {seed}")

        # Eval
        ckpt = f"checkpoints/{_ckpt_name(env_id, True, seed)}.zip"
        out_json = f"results/{_result_name(env_id, True, seed)}"
        eval_args = argparse.Namespace(ckpt=ckpt, env=env_id, n_episodes=100, json_out=out_json)
        cmd_eval(eval_args)

    # Compare
    print("\n[3/3] Comparing results...")
    compare_args = argparse.Namespace(env=env_id, seeds=args.seeds)
    return cmd_compare(compare_args)


def cmd_push_all(args: argparse.Namespace) -> int:
    """Full pipeline for sparse Push: no-HER baseline + HER + comparison."""
    env_id = "FetchPush-v4"
    seeds = _parse_seeds(args.seeds)
    total_steps = args.total_steps

    print("=" * 70)
    print("Week 4: Sparse Push - Full Pipeline")
    print(f"Seeds: {seeds}, Total steps per run: {total_steps}")
    print("=" * 70)

    # Train no-HER baselines
    print("\n[1/3] Training SAC WITHOUT HER (baseline)...")
    for seed in seeds:
        print(f"\n--- Seed {seed} (no-HER) ---")
        train_args = argparse.Namespace(
            env=env_id,
            seed=seed,
            n_envs=args.n_envs,
            total_steps=total_steps,
            her=False,
            n_sampled_goal=4,
            goal_selection_strategy="future",
            out=None,
            log_dir=args.log_dir,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
        )
        ret = cmd_train(train_args)
        if ret != 0:
            print(f"[ch04] Training failed for seed {seed}")

        # Eval
        ckpt = f"checkpoints/{_ckpt_name(env_id, False, seed)}.zip"
        out_json = f"results/{_result_name(env_id, False, seed)}"
        eval_args = argparse.Namespace(ckpt=ckpt, env=env_id, n_episodes=100, json_out=out_json)
        cmd_eval(eval_args)

    # Train HER
    print("\n[2/3] Training SAC WITH HER...")
    for seed in seeds:
        print(f"\n--- Seed {seed} (HER) ---")
        train_args = argparse.Namespace(
            env=env_id,
            seed=seed,
            n_envs=args.n_envs,
            total_steps=total_steps,
            her=True,
            n_sampled_goal=4,
            goal_selection_strategy="future",
            out=None,
            log_dir=args.log_dir,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
        )
        ret = cmd_train(train_args)
        if ret != 0:
            print(f"[ch04] Training failed for seed {seed}")

        # Eval
        ckpt = f"checkpoints/{_ckpt_name(env_id, True, seed)}.zip"
        out_json = f"results/{_result_name(env_id, True, seed)}"
        eval_args = argparse.Namespace(ckpt=ckpt, env=env_id, n_episodes=100, json_out=out_json)
        cmd_eval(eval_args)

    # Compare
    print("\n[3/3] Comparing results...")
    compare_args = argparse.Namespace(env=env_id, seeds=args.seeds)
    return cmd_compare(compare_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter 04: Sparse Reach + Push -- Introduce HER Where It Matters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train SAC on sparse env (with or without HER)")
    p_train.add_argument("--env", required=True, help="Environment ID (e.g., FetchReach-v4, FetchPush-v4)")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--n-envs", type=int, default=8)
    p_train.add_argument("--total-steps", type=int, default=1_000_000)
    p_train.add_argument("--her", action="store_true", help="Enable HER")
    p_train.add_argument("--n-sampled-goal", type=int, default=4)
    p_train.add_argument("--goal-selection-strategy", choices=["future", "final", "episode"], default="future")
    p_train.add_argument("--out", default=None, help="Checkpoint output path prefix")
    p_train.add_argument("--log-dir", default="runs")
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--buffer-size", type=int, default=1_000_000)
    p_train.add_argument("--learning-starts", type=int, default=10_000)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate checkpoint")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--env", default=None, help="Environment ID (inferred from ckpt if not specified)")
    p_eval.add_argument("--n-episodes", type=int, default=100)
    p_eval.add_argument("--json-out", default=None)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare HER vs no-HER results")
    p_cmp.add_argument("--env", required=True, help="Environment ID")
    p_cmp.add_argument("--seeds", default="0,1,2", help="Seeds to compare (comma-separated or range)")

    # ablation
    p_abl = sub.add_parser("ablation", help="Ablation study: sweep n_sampled_goal")
    p_abl.add_argument("--env", required=True, help="Environment ID")
    p_abl.add_argument("--seed", type=int, default=0)
    p_abl.add_argument("--n-envs", type=int, default=8)
    p_abl.add_argument("--total-steps", type=int, default=1_000_000)
    p_abl.add_argument("--nsg-values", default="2,4,8", help="n_sampled_goal values to test")
    p_abl.add_argument("--log-dir", default="runs")

    # reach-all
    p_reach = sub.add_parser("reach-all", help="Full pipeline for sparse Reach")
    p_reach.add_argument("--seeds", default="0,1,2", help="Seeds (comma-separated or range)")
    p_reach.add_argument("--n-envs", type=int, default=8)
    p_reach.add_argument("--total-steps", type=int, default=1_000_000)
    p_reach.add_argument("--log-dir", default="runs")

    # push-all
    p_push = sub.add_parser("push-all", help="Full pipeline for sparse Push")
    p_push.add_argument("--seeds", default="0,1,2", help="Seeds (comma-separated or range)")
    p_push.add_argument("--n-envs", type=int, default=8)
    p_push.add_argument("--total-steps", type=int, default=1_000_000)
    p_push.add_argument("--log-dir", default="runs")

    args = parser.parse_args()

    handlers = {
        "train": cmd_train,
        "eval": cmd_eval,
        "compare": cmd_compare,
        "ablation": cmd_ablation,
        "reach-all": cmd_reach_all,
        "push-all": cmd_push_all,
    }

    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
