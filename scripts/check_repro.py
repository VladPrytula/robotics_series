#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _set_mujoco_gl(backend: str) -> None:
    os.environ["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproducibility check: train twice with the same seed, evaluate both, compare metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo", choices=["ppo", "sac", "td3"], default="ppo")
    parser.add_argument("--env", dest="env_id", default="FetchReachDense-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--out-dir", default="results/repro")
    parser.add_argument("--success-atol", type=float, default=0.05)
    parser.add_argument("--return-atol", type=float, default=5.0)
    args = parser.parse_args()

    _set_mujoco_gl("disable")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run1 = out_dir / "run1"
    run2 = out_dir / "run2"
    eval1 = out_dir / "eval_run1.json"
    eval2 = out_dir / "eval_run2.json"

    train_base = [
        sys.executable,
        "train.py",
        "--algo",
        args.algo,
        "--env",
        args.env_id,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--n-envs",
        str(args.n_envs),
        "--total-steps",
        str(args.total_steps),
        "--track",
        "none",
    ]

    _run([*train_base, "--out", str(run1)])
    _run([*train_base, "--out", str(run2)])

    eval_base = [
        sys.executable,
        "eval.py",
        "--env",
        args.env_id,
        "--n-episodes",
        str(args.n_eval_episodes),
        "--seed",
        "0",
        "--deterministic",
        "--device",
        args.device,
    ]

    _run([*eval_base, "--ckpt", str(run1) + ".zip", "--json-out", str(eval1)])
    _run([*eval_base, "--ckpt", str(run2) + ".zip", "--json-out", str(eval2)])

    m1 = json.loads(eval1.read_text(encoding="utf-8"))
    m2 = json.loads(eval2.read_text(encoding="utf-8"))

    s1 = float(m1["aggregate"]["success_rate"])
    s2 = float(m2["aggregate"]["success_rate"])
    r1 = float(m1["aggregate"]["return_mean"])
    r2 = float(m2["aggregate"]["return_mean"])

    ds = abs(s1 - s2)
    dr = abs(r1 - r2)
    print(f"success_rate: run1={s1:.3f} run2={s2:.3f} |Δ|={ds:.3f}")
    print(f"return_mean:  run1={r1:.3f} run2={r2:.3f} |Δ|={dr:.3f}")

    if ds > args.success_atol or dr > args.return_atol:
        print("FAIL: reproducibility deltas exceed tolerance", file=sys.stderr)
        return 2

    print("OK: reproducibility within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

