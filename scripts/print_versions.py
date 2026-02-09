#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _maybe_run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None) or "unknown"
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a reproducibility-friendly environment/version snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json-out", default="", help="Write JSON to this path (optional).")
    args = parser.parse_args()

    snapshot = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "env": {k: os.environ.get(k) for k in ["MUJOCO_GL", "PYOPENGL_PLATFORM", "CUDA_VISIBLE_DEVICES"] if os.environ.get(k)},
        "packages": {
            name: _module_version(name)
            for name in ["torch", "gymnasium", "gymnasium_robotics", "mujoco", "stable_baselines3"]
            if _module_version(name) is not None
        },
        "nvidia_smi": {
            "version": _maybe_run(["nvidia-smi", "--version"]),
            "list": _maybe_run(["nvidia-smi", "-L"]),
        },
    }

    print(json.dumps(snapshot, indent=2, sort_keys=True))
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"OK: wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

