#!/usr/bin/env python3
"""
Mandelbrot Chunk Size Optimization Plotter (No Pandas)
======================================================
Visualizes Total Runtime vs. Chunk Size for all process counts P in {2, 4, 8, 16}.
Compares Single-Node and Multi-Node side-by-side to identify the optimal chunk size.
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


def plot_chunk_study(f_1node="benchmark_sweep_1node_final.csv",
                     f_multi="benchmark_sweep_multinode_final.csv",
                     out_fig="chunk_optimization_analysis.png"):
    """Plot runtime vs chunk size across process counts for 1-node and multi-node."""
    d_1node = load_data(f_1node)
    d_multi = load_data(f_multi)

    # Process counts to plot (skip P=1 since serial has no rank distribution)
    ranks_to_plot = [2, 4, 8, 16]
    colors = {2: "#2ca02c", 4: "#ff7f0e", 8: "#9467bd", 16: "#1f77b4"}
    markers = {2: "s", 4: "d", 8: "^", 16: "o"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    plt.rcParams.update({"font.size": 11})

    for ax, data, title in [(ax1, d_1node, "Single-Node (hosts=1)"),
                            (ax2, d_multi, "Multi-Node (ptile=4)")]:
        for p in ranks_to_plot:
            chunks = sorted(c for (r, c) in data if r == p)
            if not chunks:
                continue
            times = [data[(p, c)] for c in chunks]

            # Find optimal (minimum time) chunk for this rank count
            min_time = min(times)
            best_chunk = chunks[times.index(min_time)]

            # Plot curve
            ax.plot(chunks, times, marker=markers[p], color=colors[p],
                    linewidth=2.0, markersize=6, label=f"P = {p} (best: C={best_chunk})")

            # Highlight minimum with a star
            ax.scatter([best_chunk], [min_time], color="crimson", marker="*",
                       s=160, zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xlabel("Chunk Size (log2 scale)", fontsize=11)
        ax.set_ylabel("Total Execution Time [s]", fontsize=11)
        ax.set_xticks(sorted(list(set(c for r, c in data if r in ranks_to_plot))))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(frameon=True, framealpha=0.9)

    plt.suptitle("Mandelbrot Chunk Size vs. Runtime Across Process Counts",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"[Success] Chunk optimization plot saved to: {out_fig}")

    # Print summary table of optimal chunk sizes
    print("\n" + "=" * 65)
    print("Optimal Chunk Size Summary")
    print("=" * 65)
    print(f"{'Ranks (P)':<10} | {'1-Node Best Chunk':<18} | {'1-Node Time':<12} | {'Multi Best Chunk':<18} | {'Multi Time':<12}")
    print("-" * 75)
    for p in ranks_to_plot:
        c1_list = sorted(c for (r, c) in d_1node if r == p)
        t1_list = [d_1node[(p, c)] for c in c1_list]
        b1 = c1_list[t1_list.index(min(t1_list))] if c1_list else "N/A"
        tm1 = f"{min(t1_list):.2f}s" if t1_list else "N/A"

        cm_list = sorted(c for (r, c) in d_multi if r == p)
        tm_list = [d_multi[(p, c)] for c in cm_list]
        bm = cm_list[tm_list.index(min(tm_list))] if cm_list else "N/A"
        tmm = f"{min(tm_list):.2f}s" if tm_list else "N/A"

        print(f"{p:<10} | C = {b1:<14} | {tm1:<12} | C = {bm:<14} | {tmm:<12}")


if __name__ == "__main__":
    f1 = sys.argv[1] if len(sys.argv) > 1 else "benchmark_sweep_1node_final.csv"
    f2 = sys.argv[2] if len(sys.argv) > 2 else "benchmark_sweep_multinode_final.csv"
    plot_chunk_study(f1, f2)
