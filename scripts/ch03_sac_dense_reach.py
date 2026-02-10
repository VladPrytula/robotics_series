#!/usr/bin/env python3
"""Chapter 03: SAC on Dense Reach + Replay Diagnostics

Week 3 goals:
1. Validate off-policy stack (SAC) before adding HER
2. Add replay buffer diagnostics (Q-values, entropy, rewards, goal distances)
3. Throughput scaling (n_envs sweep)

Usage:
    # Full pipeline (train + eval + compare to PPO)
    python scripts/ch03_sac_dense_reach.py all

    # Train SAC only (with diagnostics callback)
    python scripts/ch03_sac_dense_reach.py train --total-steps 1000000

    # Evaluate existing checkpoint
    python scripts/ch03_sac_dense_reach.py eval --ckpt checkpoints/sac_FetchReachDense-v4_seed0.zip

    # Throughput scaling experiment
    python scripts/ch03_sac_dense_reach.py throughput --n-envs-list 1,2,4,8,16

    # Compare SAC vs PPO results
    python scripts/ch03_sac_dense_reach.py compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


def cmd_train(args: argparse.Namespace) -> int:
    """Train SAC on FetchReachDense-v4 with replay diagnostics."""
    import gymnasium_robotics  # noqa: F401

    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    # Import our custom callback
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from callbacks.sac_diagnostics import SACDiagnosticsCallback

    env_id = "FetchReachDense-v4"
    seed = args.seed
    n_envs = args.n_envs
    total_steps = args.total_steps

    try:
        import torch

        if args.device == "mps":
            if not torch.backends.mps.is_available():
                print("[ch03] WARNING: MPS requested but not available, falling back to CPU", file=sys.stderr)
                device = "cpu"
            else:
                print("[ch03] WARNING: MPS support is experimental. If you encounter issues, use --device cpu", file=sys.stderr)
                device = "mps"
        elif args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
    except Exception:
        device = "cpu"

    print(f"[ch03] Training SAC on {env_id}")
    print(f"[ch03] seed={seed}, n_envs={n_envs}, total_steps={total_steps}, device={device}")

    set_random_seed(seed)

    # Setup paths
    out_path = Path(args.out) if args.out else Path("checkpoints") / f"sac_{env_id}_seed{seed}"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir).expanduser().resolve()
    run_id = f"sac/{env_id}/seed{seed}"

    # Create environment
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed)

    try:
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
            # SAC-specific: auto-tuned entropy coefficient
            ent_coef="auto",
            # Learning rates (defaults work well, but exposed for ablations)
            learning_rate=3e-4,
        )

        # Create diagnostics callback
        diag_callback = SACDiagnosticsCallback(
            log_freq=args.diag_freq,
            n_samples=1000,
            verbose=1 if args.verbose else 0,
        )

        # Train
        print(f"[ch03] Starting training for {total_steps} steps...")
        t0 = time.perf_counter()
        model.learn(
            total_timesteps=total_steps,
            tb_log_name=run_id,
            callback=diag_callback,
        )
        elapsed = time.perf_counter() - t0
        steps_per_sec = total_steps / elapsed

        print(f"[ch03] Training complete in {elapsed:.1f}s ({steps_per_sec:.0f} steps/sec)")

        # Save checkpoint
        model.save(str(out_path))
        print(f"[ch03] Saved checkpoint: {out_path}.zip")

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
        "hyperparams": {
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "ent_coef": "auto",
            "learning_rate": 3e-4,
        },
        "versions": _gather_versions(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ch03] Wrote metadata: {meta_path}")

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate SAC checkpoint."""
    import subprocess

    ckpt = args.ckpt or "checkpoints/sac_FetchReachDense-v4_seed0.zip"
    env_id = "FetchReachDense-v4"
    n_episodes = args.n_episodes
    out_json = args.json_out or f"results/ch03_sac_fetchreachdense-v4_seed{args.seed}_eval.json"

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

    print(f"[ch03] Evaluating: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        print(f"[ch03] Evaluation complete: {out_json}")
    return result.returncode


def cmd_throughput(args: argparse.Namespace) -> int:
    """Throughput scaling experiment: sweep n_envs."""
    import gymnasium_robotics  # noqa: F401

    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    env_id = "FetchReachDense-v4"
    n_envs_list = [int(x) for x in args.n_envs_list.split(",")]
    warmup_steps = args.warmup_steps
    measure_steps = args.measure_steps

    device = "cuda"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    print(f"[ch03] Throughput scaling experiment on {env_id}")
    print(f"[ch03] n_envs: {n_envs_list}, warmup={warmup_steps}, measure={measure_steps}, device={device}")

    results = []

    for n_envs in n_envs_list:
        print(f"\n[ch03] Testing n_envs={n_envs}...")
        set_random_seed(0)

        env = make_vec_env(env_id, n_envs=n_envs, seed=0)
        try:
            model = SAC(
                "MultiInputPolicy",
                env,
                verbose=0,
                device=device,
                buffer_size=100_000,  # Smaller buffer for quick test
                learning_starts=1000,
            )

            # Warmup
            if warmup_steps > 0:
                model.learn(total_timesteps=warmup_steps)

            # Measure
            t0 = time.perf_counter()
            model.learn(total_timesteps=measure_steps, reset_num_timesteps=False)
            elapsed = time.perf_counter() - t0

            steps_per_sec = measure_steps / elapsed
            print(f"[ch03] n_envs={n_envs}: {steps_per_sec:.0f} steps/sec ({elapsed:.1f}s)")

            results.append(
                {
                    "n_envs": n_envs,
                    "steps_per_sec": steps_per_sec,
                    "elapsed_sec": elapsed,
                    "measure_steps": measure_steps,
                }
            )

        finally:
            env.close()

    # Save results
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "env_id": env_id,
        "device": device,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "results": results,
        "versions": _gather_versions(),
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ch03] Throughput results saved: {out_path}")

    # Print summary
    print("\n[ch03] Throughput Summary:")
    print(f"{'n_envs':>8} | {'steps/sec':>12} | {'speedup':>8}")
    print("-" * 35)
    baseline = results[0]["steps_per_sec"] if results else 1
    for r in results:
        speedup = r["steps_per_sec"] / baseline
        print(f"{r['n_envs']:>8} | {r['steps_per_sec']:>12.0f} | {speedup:>7.2f}x")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare SAC vs PPO evaluation results."""
    ppo_path = Path(args.ppo_json)
    sac_path = Path(args.sac_json)

    if not ppo_path.exists():
        print(f"[ch03] PPO results not found: {ppo_path}")
        print("[ch03] Run Week 2 first: python scripts/ch02_ppo_dense_reach.py all")
        return 1

    if not sac_path.exists():
        print(f"[ch03] SAC results not found: {sac_path}")
        print("[ch03] Run SAC training and eval first: python scripts/ch03_sac_dense_reach.py train && python scripts/ch03_sac_dense_reach.py eval")
        return 1

    with open(ppo_path) as f:
        ppo = json.load(f)
    with open(sac_path) as f:
        sac = json.load(f)

    print("\n" + "=" * 60)
    print("Week 3: SAC vs PPO Comparison (FetchReachDense-v4)")
    print("=" * 60)

    ppo_agg = ppo["aggregate"]
    sac_agg = sac["aggregate"]

    metrics = [
        ("Success Rate", "success_rate", "{:.1%}"),
        ("Return (mean)", "return_mean", "{:.3f}"),
        ("Return (std)", "return_std", "{:.3f}"),
        ("Final Distance (mean)", "final_distance_mean", "{:.4f}"),
        ("Final Distance (std)", "final_distance_std", "{:.4f}"),
        ("Action Smoothness", "action_smoothness_mean", "{:.3f}"),
    ]

    print(f"\n{'Metric':<25} | {'PPO':>12} | {'SAC':>12} | {'Delta':>10}")
    print("-" * 65)

    for name, key, fmt in metrics:
        ppo_val = ppo_agg.get(key, 0)
        sac_val = sac_agg.get(key, 0)
        delta = sac_val - ppo_val

        ppo_str = fmt.format(ppo_val)
        sac_str = fmt.format(sac_val)

        # For success rate and distance, indicate if better/worse
        if key == "success_rate":
            delta_str = f"{delta:+.1%}"
        elif "distance" in key:
            delta_str = f"{delta:+.4f}" + (" (better)" if delta < 0 else " (worse)" if delta > 0 else "")
        else:
            delta_str = f"{delta:+.3f}"

        print(f"{name:<25} | {ppo_str:>12} | {sac_str:>12} | {delta_str:>10}")

    print("\n" + "=" * 60)

    # Summary
    both_succeed = ppo_agg["success_rate"] >= 0.95 and sac_agg["success_rate"] >= 0.95
    if both_succeed:
        print("PASS: Both PPO and SAC achieve high success on dense Reach.")
        print("      Off-policy stack (SAC) is validated. Ready for Week 4 (HER).")
    else:
        print("WARNING: One or both algorithms did not reach 95% success.")
        print("         Debug before proceeding to Week 4.")

    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run full Week 3 pipeline: train -> eval -> compare."""
    print("=" * 60)
    print("Week 3: SAC on Dense Reach - Full Pipeline")
    print("=" * 60)

    # Train
    print("\n[1/3] Training SAC...")
    train_args = argparse.Namespace(
        seed=args.seed,
        n_envs=args.n_envs,
        total_steps=args.total_steps,
        device=args.device,
        out=None,
        log_dir=args.log_dir,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=10_000,
        diag_freq=10_000,
        verbose=True,
    )
    ret = cmd_train(train_args)
    if ret != 0:
        return ret

    # Eval
    print("\n[2/3] Evaluating SAC...")
    eval_args = argparse.Namespace(
        ckpt=f"checkpoints/sac_FetchReachDense-v4_seed{args.seed}.zip",
        n_episodes=100,
        json_out=f"results/ch03_sac_fetchreachdense-v4_seed{args.seed}_eval.json",
        seed=args.seed,
    )
    ret = cmd_eval(eval_args)
    if ret != 0:
        return ret

    # Compare
    print("\n[3/3] Comparing SAC vs PPO...")
    compare_args = argparse.Namespace(
        ppo_json="results/ch02_ppo_fetchreachdense-v4_seed0_eval.json",
        sac_json=f"results/ch03_sac_fetchreachdense-v4_seed{args.seed}_eval.json",
    )
    ret = cmd_compare(compare_args)

    return ret


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter 03: SAC on Dense Reach + Replay Diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train SAC with diagnostics callback")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--n-envs", type=int, default=8)
    p_train.add_argument("--total-steps", type=int, default=1_000_000)
    p_train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p_train.add_argument("--out", default=None, help="Checkpoint output path prefix")
    p_train.add_argument("--log-dir", default="runs")
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--buffer-size", type=int, default=1_000_000)
    p_train.add_argument("--learning-starts", type=int, default=10_000)
    p_train.add_argument("--diag-freq", type=int, default=10_000, help="Diagnostics logging frequency (steps)")
    p_train.add_argument("--verbose", action="store_true")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate SAC checkpoint")
    p_eval.add_argument("--ckpt", default=None)
    p_eval.add_argument("--n-episodes", type=int, default=100)
    p_eval.add_argument("--json-out", default=None)
    p_eval.add_argument("--seed", type=int, default=0)

    # throughput
    p_tp = sub.add_parser("throughput", help="Throughput scaling experiment")
    p_tp.add_argument("--n-envs-list", default="1,2,4,8,16", help="Comma-separated n_envs values")
    p_tp.add_argument("--warmup-steps", type=int, default=5000)
    p_tp.add_argument("--measure-steps", type=int, default=20000)
    p_tp.add_argument("--json-out", default="results/ch03_throughput.json")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare SAC vs PPO results")
    p_cmp.add_argument("--ppo-json", default="results/ch02_ppo_fetchreachdense-v4_seed0_eval.json")
    p_cmp.add_argument("--sac-json", default="results/ch03_sac_fetchreachdense-v4_seed0_eval.json")

    # all
    p_all = sub.add_parser("all", help="Full pipeline: train -> eval -> compare")
    p_all.add_argument("--seed", type=int, default=0)
    p_all.add_argument("--n-envs", type=int, default=8)
    p_all.add_argument("--total-steps", type=int, default=1_000_000)
    p_all.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p_all.add_argument("--log-dir", default="runs")

    args = parser.parse_args()

    handlers = {
        "train": cmd_train,
        "eval": cmd_eval,
        "throughput": cmd_throughput,
        "compare": cmd_compare,
        "all": cmd_all,
    }

    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
