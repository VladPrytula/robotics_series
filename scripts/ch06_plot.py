#!/usr/bin/env python3
"""Chapter 06: Generate publication-quality figures from action-interface results.

Reads JSON results produced by ch06_action_interface.py and generates
matplotlib figures for the tutorial.

Figures:
    1. scaling  -- Success rate + smoothness vs scale factor (dual-axis)
    2. filter   -- Success rate + smoothness vs alpha (dual-axis)
    3. decomp   -- RL vs PD success rate bar chart (planning-vs-control)
    4. metrics  -- Engineering metrics comparison (TTS, path length, energy)
    5. all      -- Generate all figures

Usage:
    python scripts/ch06_plot.py all
    python scripts/ch06_plot.py scaling --results-dir results --out-dir figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011) -- matches project convention
# ---------------------------------------------------------------------------
COLOR_BLUE = "#0072B2"
COLOR_ORANGE = "#E69F00"
COLOR_GREEN = "#009E73"
COLOR_VERMILLION = "#D55E00"
COLOR_GRAY = "#999999"
COLOR_PURPLE = "#CC79A7"

# Environment display names and colors
ENV_STYLE = {
    "FetchReach-v4": {"color": COLOR_BLUE, "label": "FetchReach", "marker": "o"},
    "FetchPush-v4": {"color": COLOR_VERMILLION, "label": "FetchPush", "marker": "s"},
}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _setup_style():
    """Set consistent plot style."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": True,  # needed for dual-axis
    })


# ---------------------------------------------------------------------------
# Figure 1: Scaling sweep (dual-axis: success + smoothness)
# ---------------------------------------------------------------------------
def plot_scaling(results_dir: Path, out_dir: Path, seed: int = 0) -> list[Path]:
    """Generate scaling sweep figures -- one per environment."""
    _setup_style()
    paths = []

    for env_id, style in ENV_STYLE.items():
        env_key = env_id.lower()
        fpath = results_dir / f"ch06_scaling_{env_key}_seed{seed}.json"
        if not fpath.exists():
            print(f"  [skip] {fpath}")
            continue

        data = _load_json(fpath)
        scales = [r["scale"] for r in data["results"]]
        success = [r["aggregate"]["success_rate"] * 100 for r in data["results"]]
        smooth = [r["aggregate"]["smoothness_mean"] for r in data["results"]]
        tts = [r["aggregate"]["time_to_success_mean"] for r in data["results"]]

        fig, ax1 = plt.subplots(figsize=(7, 4.5))

        # Success rate (left axis)
        ax1.plot(scales, success, color=style["color"], marker=style["marker"],
                 linewidth=2, markersize=8, label="Success rate (%)", zorder=5)
        ax1.set_xlabel("Action scale factor")
        ax1.set_ylabel("Success rate (%)", color=style["color"])
        ax1.tick_params(axis="y", labelcolor=style["color"])
        ax1.set_ylim(-5, 110)
        ax1.axhline(y=100, color=COLOR_GRAY, linestyle="--", alpha=0.4, linewidth=0.8)
        ax1.axvline(x=1.0, color=COLOR_GRAY, linestyle=":", alpha=0.5, linewidth=0.8,
                    label="Trained scale")

        # Smoothness (right axis)
        ax2 = ax1.twinx()
        ax2.plot(scales, smooth, color=COLOR_ORANGE, marker="^",
                 linewidth=2, markersize=7, linestyle="--", label="Smoothness", zorder=4)
        ax2.set_ylabel("Smoothness (mean sq. action diff)", color=COLOR_ORANGE)
        ax2.tick_params(axis="y", labelcolor=COLOR_ORANGE)

        # TTS as annotated text at each point
        for i, (s, sr, t) in enumerate(zip(scales, success, tts)):
            if t is not None and sr > 10:
                ax1.annotate(f"TTS={t:.0f}", (s, sr),
                             textcoords="offset points", xytext=(0, -18),
                             fontsize=8, ha="center", color=COLOR_GRAY)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left",
                   framealpha=0.9, edgecolor="none")

        ax1.set_title(f"Action Scaling -- {style['label']}")
        fig.tight_layout()

        out = out_dir / f"ch06_scaling_{env_key}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
        paths.append(out)

    return paths


# ---------------------------------------------------------------------------
# Figure 2: Filter sweep (dual-axis: success + smoothness)
# ---------------------------------------------------------------------------
def plot_filter(results_dir: Path, out_dir: Path, seed: int = 0) -> list[Path]:
    """Generate filter sweep figures -- one per environment."""
    _setup_style()
    paths = []

    for env_id, style in ENV_STYLE.items():
        env_key = env_id.lower()
        fpath = results_dir / f"ch06_filter_{env_key}_seed{seed}.json"
        if not fpath.exists():
            print(f"  [skip] {fpath}")
            continue

        data = _load_json(fpath)
        alphas = [r["alpha"] for r in data["results"]]
        success = [r["aggregate"]["success_rate"] * 100 for r in data["results"]]
        smooth = [r["aggregate"]["smoothness_mean"] for r in data["results"]]

        fig, ax1 = plt.subplots(figsize=(7, 4.5))

        # Success rate (left axis)
        ax1.plot(alphas, success, color=style["color"], marker=style["marker"],
                 linewidth=2, markersize=8, label="Success rate (%)", zorder=5)
        ax1.set_xlabel(r"Filter coefficient $\alpha$ (1.0 = no filter)")
        ax1.set_ylabel("Success rate (%)", color=style["color"])
        ax1.tick_params(axis="y", labelcolor=style["color"])
        ax1.set_ylim(-5, 110)
        ax1.axhline(y=100, color=COLOR_GRAY, linestyle="--", alpha=0.4, linewidth=0.8)

        # Smoothness (right axis)
        ax2 = ax1.twinx()
        ax2.plot(alphas, smooth, color=COLOR_ORANGE, marker="^",
                 linewidth=2, markersize=7, linestyle="--", label="Smoothness", zorder=4)
        ax2.set_ylabel("Smoothness (mean sq. action diff)", color=COLOR_ORANGE)
        ax2.tick_params(axis="y", labelcolor=COLOR_ORANGE)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
                   framealpha=0.9, edgecolor="none")

        ax1.set_title(f"Low-Pass Filter -- {style['label']}")
        fig.tight_layout()

        out = out_dir / f"ch06_filter_{env_key}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
        paths.append(out)

    return paths


# ---------------------------------------------------------------------------
# Figure 3: Planning-vs-Control decomposition (RL vs PD bar chart)
# ---------------------------------------------------------------------------
def plot_decomposition(results_dir: Path, out_dir: Path, seed: int = 0) -> list[Path]:
    """Generate RL vs PD success rate comparison."""
    _setup_style()

    comp_path = results_dir / "ch06_comparison.json"
    if not comp_path.exists():
        print(f"  [skip] {comp_path}")
        return []

    comp = _load_json(comp_path)
    envs = list(comp["environments"].keys())

    env_labels = []
    rl_success = []
    pd_success = []

    for env_id in envs:
        env_data = comp["environments"][env_id]
        decomp = env_data.get("decomposition", {})
        rl_sr = decomp.get("rl_success_rate", 0) * 100
        pd_sr = decomp.get("pd_success_rate", 0) * 100
        label = ENV_STYLE.get(env_id, {}).get("label", env_id)

        env_labels.append(label)
        rl_success.append(rl_sr)
        pd_success.append(pd_sr)

    x = np.arange(len(env_labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars_rl = ax.bar(x - width / 2, rl_success, width, label="SAC + HER",
                     color=COLOR_BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    bars_pd = ax.bar(x + width / 2, pd_success, width, label="PD controller",
                     color=COLOR_ORANGE, edgecolor="white", linewidth=0.5, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars_rl, rl_success):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=COLOR_BLUE)
    for bar, val in zip(bars_pd, pd_success):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=COLOR_ORANGE)

    # Gap annotation
    for i, (rl, pd) in enumerate(zip(rl_success, pd_success)):
        gap = rl - pd
        if gap > 5:
            ax.annotate(
                f"gap = {gap:.0f}%",
                xy=(i, max(rl, pd) + 10), fontsize=9,
                ha="center", color=COLOR_VERMILLION,
                fontweight="bold",
            )

    ax.set_ylabel("Success rate (%)")
    ax.set_title("Planning vs Control Decomposition")
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels)
    ax.set_ylim(0, 125)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.axhline(y=100, color=COLOR_GRAY, linestyle="--", alpha=0.3, linewidth=0.8)
    fig.tight_layout()

    out = out_dir / "ch06_planning_vs_control.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return [out]


# ---------------------------------------------------------------------------
# Figure 4: Engineering metrics comparison
# ---------------------------------------------------------------------------
def plot_metrics(results_dir: Path, out_dir: Path, seed: int = 0) -> list[Path]:
    """Generate engineering metrics comparison across conditions."""
    _setup_style()
    paths = []

    comp_path = results_dir / "ch06_comparison.json"
    if not comp_path.exists():
        print(f"  [skip] {comp_path}")
        return []

    comp = _load_json(comp_path)

    for env_id in comp["environments"]:
        env_data = comp["environments"][env_id]
        style = ENV_STYLE.get(env_id, {"label": env_id, "color": COLOR_BLUE})
        env_key = env_id.lower()

        # Collect scaling results for metric comparison
        if "scaling" not in env_data:
            continue

        scales = [r["scale"] for r in env_data["scaling"]]
        tts_vals = [r["aggregate"].get("time_to_success_mean") for r in env_data["scaling"]]
        path_vals = [r["aggregate"]["path_length_mean"] for r in env_data["scaling"]]
        energy_vals = [r["aggregate"]["action_energy_mean"] for r in env_data["scaling"]]

        # Replace None TTS with max (50 = episode length)
        tts_vals = [t if t is not None else 50 for t in tts_vals]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # TTS
        axes[0].plot(scales, tts_vals, color=style["color"], marker="o",
                     linewidth=2, markersize=7)
        axes[0].set_xlabel("Action scale factor")
        axes[0].set_ylabel("Time to success (steps)")
        axes[0].set_title("Speed")
        axes[0].axvline(x=1.0, color=COLOR_GRAY, linestyle=":", alpha=0.5)

        # Path length
        axes[1].plot(scales, path_vals, color=COLOR_GREEN, marker="s",
                     linewidth=2, markersize=7)
        axes[1].set_xlabel("Action scale factor")
        axes[1].set_ylabel("Path length (m)")
        axes[1].set_title("Efficiency")
        axes[1].axvline(x=1.0, color=COLOR_GRAY, linestyle=":", alpha=0.5)

        # Energy
        axes[2].plot(scales, energy_vals, color=COLOR_VERMILLION, marker="^",
                     linewidth=2, markersize=7)
        axes[2].set_xlabel("Action scale factor")
        axes[2].set_ylabel("Action energy (sum ||a||^2)")
        axes[2].set_title("Effort")
        axes[2].axvline(x=1.0, color=COLOR_GRAY, linestyle=":", alpha=0.5)

        fig.suptitle(f"Engineering Metrics vs Action Scale -- {style['label']}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()

        out = out_dir / f"ch06_metrics_{env_key}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
        paths.append(out)

    return paths


# ---------------------------------------------------------------------------
# Figure 5: Combined scaling comparison (both envs on one plot)
# ---------------------------------------------------------------------------
def plot_scaling_combined(results_dir: Path, out_dir: Path, seed: int = 0) -> list[Path]:
    """Both environments on a single scaling success-rate plot."""
    _setup_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for env_id, style in ENV_STYLE.items():
        env_key = env_id.lower()
        fpath = results_dir / f"ch06_scaling_{env_key}_seed{seed}.json"
        if not fpath.exists():
            continue

        data = _load_json(fpath)
        scales = [r["scale"] for r in data["results"]]
        success = [r["aggregate"]["success_rate"] * 100 for r in data["results"]]

        ax.plot(scales, success, color=style["color"], marker=style["marker"],
                linewidth=2, markersize=8, label=style["label"])

    ax.set_xlabel("Action scale factor")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Action Scaling: Reach vs Push")
    ax.set_ylim(-5, 110)
    ax.axhline(y=100, color=COLOR_GRAY, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axvline(x=1.0, color=COLOR_GRAY, linestyle=":", alpha=0.5, linewidth=0.8,
               label="Trained scale")
    ax.legend(loc="lower left", framealpha=0.9, edgecolor="none")
    fig.tight_layout()

    out = out_dir / "ch06_scaling_combined.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return [out]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate ch06 figures")
    parser.add_argument("command", choices=["scaling", "filter", "decomp", "metrics",
                                            "combined", "all"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    dispatch = {
        "scaling": plot_scaling,
        "filter": plot_filter,
        "decomp": plot_decomposition,
        "metrics": plot_metrics,
        "combined": plot_scaling_combined,
    }

    if args.command == "all":
        all_paths = []
        for name, func in dispatch.items():
            print(f"\n--- {name} ---")
            all_paths.extend(func(results_dir, out_dir, args.seed))
        print(f"\nGenerated {len(all_paths)} figures in {out_dir}/")
    else:
        paths = dispatch[args.command](results_dir, out_dir, args.seed)
        print(f"\nGenerated {len(paths)} figures")

    return 0


if __name__ == "__main__":
    sys.exit(main())
