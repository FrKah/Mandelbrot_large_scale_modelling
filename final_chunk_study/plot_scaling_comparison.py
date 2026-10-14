#!/usr/bin/env python3
"""
Simple Mandelbrot Scaling & Speedup Plotter (No Pandas)
======================================================
Reads benchmark CSV files using built-in csv and matplotlib.
Plots: Total Runtime, Speedup S(P), and Parallel Efficiency E(P).
"""

import csv
import sys
import matplotlib.pyplot as plt


def load_data(filename):
    """Load (ranks, chunk_size) -> total_time mapping from CSV."""
    data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[(int(row["ranks"]), int(row["chunk_size"]))] = float(row["total_time"])
    return data


def get_curve(data, chunk):
    """Return sorted ranks, times, speedup, and efficiency for a given chunk size."""
    ranks = sorted(r for r, c in data if c == chunk)
    t1 = data.get((1, chunk), data.get((1, 4), 134.0))  # Baseline serial time T_1
    times = [data[(r, chunk)] for r in ranks]
    speedups = [t1 / t for t in times]
    efficiencies = [s / r for s, r in zip(speedups, ranks)]
    return ranks, times, speedups, efficiencies


def plot_benchmarks(f_1node="benchmark_sweep_1node_final.csv",
                    f_multi="benchmark_sweep_multinode_final.csv",
                    chunk=4,
                    out_fig="scaling_speedup_comparison.png"):
    """Generate 3-panel scaling comparison plot."""
    d_1node = load_data(f_1node)
    d_multi = load_data(f_multi)

    r1, t1, s1, e1 = get_curve(d_1node, chunk)
    rm, tm, sm, em = get_curve(d_multi, chunk)
    max_p = max(r1 + rm)

    fig, (ax_time, ax_speedup, ax_eff) = plt.subplots(1, 3, figsize=(16, 5))

    # Reference ideal curves
    ax_speedup.plot([1, max_p], [1, max_p], "k--", label="Ideal (S = P)", linewidth=1.5)
    ax_eff.axhline(1.0, color="black", linestyle="--", label="Ideal (E = 1)", linewidth=1.5)

    # 1-Node vs Multi-Node curves
    panels = [
        (ax_time, t1, tm, "Total time [s]", "Total Runtime vs. Processes"),
        (ax_speedup, s1, sm, "Speedup S(P)", "(a) Speedup S(P)"),
        (ax_eff, e1, em, "Parallel efficiency S(P) / P", "(b) Parallel Efficiency E(P)")
    ]

    for ax, y1, ym, ylabel, title in panels:
        ax.plot(r1, y1, "o-", label=f"1-Node (chunk={chunk})", color="#1f77b4", linewidth=2)
        ax.plot(rm, ym, "^-", label=f"Multi-Node (chunk={chunk})", color="#d62728", linewidth=2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Number of MPI processes (P)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(r1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    ax_eff.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f"[Success] Plot saved to: {out_fig}")


if __name__ == "__main__":
    c1 = sys.argv[1] if len(sys.argv) > 1 else "benchmark_sweep_1node_final.csv"
    c2 = sys.argv[2] if len(sys.argv) > 2 else "benchmark_sweep_multinode_final.csv"
    ch = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    plot_benchmarks(c1, c2, chunk=ch)
