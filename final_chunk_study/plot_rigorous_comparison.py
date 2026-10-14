#!/usr/bin/env python3
"""
Rigorous Scaling and Heatmap Benchmark Plotter (No Pandas)
==========================================================
Generates a 2-part scientific figure:
  1. Dual Heatmap (1-Node vs. Multi-Node) highlighting the optimal chunk per rank
  2. Rigorous Best-vs-Best Speedup & Efficiency curves using optimal configurations
"""

import csv
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_data(filepath):
    """Load benchmark CSV into dictionary: (rank, chunk_size) -> total_time."""
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = int(row["ranks"].strip())
            c = int(row["chunk_size"].strip())
            t = float(row["total_time"].strip())
            data[(r, c)] = t
    return data


def plot_rigorous_analysis(f_1node="benchmark_sweep_1node_final.csv",
                           f_multi="benchmark_sweep_multinode_final.csv",
                           output_fig="rigorous_scaling_analysis.png"):
    d_1node = load_data(f_1node)
    d_multi = load_data(f_multi)

    ranks = [2, 4, 8, 16]
    chunks = [1, 2, 4, 8, 16, 32, 64]

    # Find the global absolute baseline serial time T_1 (best serial run across all data)
    all_t1 = [t for (r, c), t in {**d_1node, **d_multi}.items() if r == 1]
    t1_base = min(all_t1) if all_t1 else 128.61

    # Extract optimal time and best chunk per rank
    opt_1node = {}
    for r in ranks:
        times_c = [(d_1node[(r, c)], c) for c in chunks if (r, c) in d_1node]
        opt_1node[r] = min(times_c)  # (best_time, best_chunk)

    opt_multi = {}
    for r in ranks:
        times_c = [(d_multi[(r, c)], c) for c in chunks if (r, c) in d_multi]
        opt_multi[r] = min(times_c)

    # Setup 4-panel figure (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.rcParams.update({"font.size": 11})

    (ax_heat1, ax_heat2), (ax_speedup, ax_eff) = axes

    # ==========================================
    # Panel 1 & 2: Dual Heatmap (1-Node & Multi)
    # ==========================================
    for ax, data, opt_dict, title in [
        (ax_heat1, d_1node, opt_1node, "Single-Node Runtime Heatmap [s] (hosts=1)"),
        (ax_heat2, d_multi, opt_multi, "Multi-Node Runtime Heatmap [s] (ptile=4)")
    ]:
        matrix = [[data.get((r, c), float("nan")) for c in chunks] for r in ranks]
        im = ax.imshow(matrix, cmap="YlGnBu_r", aspect="auto")

        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels(chunks)
        ax.set_yticks(range(len(ranks)))
        ax.set_yticklabels(ranks)
        ax.set_xlabel("Chunk Size", fontsize=11, fontweight="bold")
        ax.set_ylabel("MPI Processes (P)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Annotate cell numbers and highlight the best chunk in each row
        for i, r in enumerate(ranks):
            best_time, best_chunk = opt_dict[r]
            for j, c in enumerate(chunks):
                val = data.get((r, c), None)
                if val is not None:
                    is_best = (c == best_chunk)
                    text_color = "white" if val < 30 else "black"
                    fontweight = "bold" if is_best else "normal"
                    ax.text(j, i, f"{val:.1f}s", ha="center", va="center",
                            color=text_color, fontsize=10, fontweight=fontweight)

                    # Highlight optimal cell with red border
                    if is_best:
                        rect = patches.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                                 linewidth=2.5, edgecolor="red", facecolor="none")
                        ax.add_patch(rect)

        plt.colorbar(im, ax=ax, label="Runtime [s]")

    # ==========================================
    # Panel 3: Rigorous Speedup (Best-vs-Best)
    # ==========================================
    all_p = [1] + ranks
    s_ideal = all_p
    ax_speedup.plot(all_p, s_ideal, "k--", label="Ideal (S = P)", linewidth=1.5)

    s_1node = [1.0] + [t1_base / opt_1node[r][0] for r in ranks]
    s_multi = [1.0] + [t1_base / opt_multi[r][0] for r in ranks]

    ax_speedup.plot(all_p, s_1node, "o-", color="#1f77b4", linewidth=2.2, markersize=7,
                    label="1-Node (Optimal Chunks)")
    ax_speedup.plot(all_p, s_multi, "^-", color="#d62728", linewidth=2.2, markersize=7,
                    label="Multi-Node (Optimal Chunks)")

    # Annotate optimal chunk labels at P=16
    ax_speedup.annotate(f"C={opt_1node[16][1]} ({s_1node[-1]:.2f}x)",
                        (16, s_1node[-1]), textcoords="offset points",
                        xytext=(-50, -15), color="#1f77b4", fontweight="bold")
    ax_speedup.annotate(f"C={opt_multi[16][1]} ({s_multi[-1]:.2f}x)",
                        (16, s_multi[-1]), textcoords="offset points",
                        xytext=(-50, 10), color="#d62728", fontweight="bold")

    ax_speedup.set_title("(a) Absolute Speedup (Best-vs-Best Frontier)", fontsize=12, fontweight="bold")
    ax_speedup.set_xlabel("Number of MPI processes (P)", fontsize=11, fontweight="bold")
    ax_speedup.set_ylabel(f"Speedup S(P)  [T1 ref = {t1_base:.1f}s]", fontsize=11, fontweight="bold")
    ax_speedup.set_xticks(all_p)
    ax_speedup.set_ylim(0, 18)
    ax_speedup.grid(True, linestyle="--", alpha=0.5)
    ax_speedup.legend(loc="upper left", frameon=True)

    # ==========================================
    # Panel 4: Parallel Efficiency
    # ==========================================
    e_ideal = [1.0] * len(all_p)
    ax_eff.axhline(1.0, color="black", linestyle="--", label="Ideal (E = 1)", linewidth=1.5)

    e_1node = [s / p for s, p in zip(s_1node, all_p)]
    e_multi = [s / p for s, p in zip(s_multi, all_p)]

    ax_eff.plot(all_p, e_1node, "o-", color="#1f77b4", linewidth=2.2, markersize=7,
                label="1-Node (Optimal Chunks)")
    ax_eff.plot(all_p, e_multi, "^-", color="#d62728", linewidth=2.2, markersize=7,
                label="Multi-Node (Optimal Chunks)")

    ax_eff.set_title("(b) Parallel Efficiency (Best-vs-Best)", fontsize=12, fontweight="bold")
    ax_eff.set_xlabel("Number of MPI processes (P)", fontsize=11, fontweight="bold")
    ax_eff.set_ylabel("Parallel efficiency S(P) / P", fontsize=11, fontweight="bold")
    ax_eff.set_xticks(all_p)
    ax_eff.set_ylim(0, 1.15)
    ax_eff.grid(True, linestyle="--", alpha=0.5)
    ax_eff.legend(loc="upper right", frameon=True)

    plt.suptitle("Rigorous Mandelbrot Scalability: Parameter Space & Optimal Frontier",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    print(f"\n[Success] Rigorous analysis figure saved to: {output_fig}")

    # Print summary
    print("\n" + "=" * 70)
    print("Optimal Configuration Summary")
    print("=" * 70)
    print(f"{'Ranks':<6} | {'1-Node Best':<14} | {'1-Node Speedup':<16} | {'Multi Best':<14} | {'Multi Speedup':<16}")
    print("-" * 70)
    for r in ranks:
        t1, c1 = opt_1node[r]
        s1 = t1_base / t1
        tm, cm = opt_multi[r]
        sm = t1_base / tm
        print(f"{r:<6} | C={c1:<3} ({t1:.2f}s) | {s1:.2f}x ({s1/r*100:.0f}%)      | C={cm:<3} ({tm:.2f}s) | {sm:.2f}x ({sm/r*100:.0f}%)")


if __name__ == "__main__":
    f1 = sys.argv[1] if len(sys.argv) > 1 else "benchmark_sweep_1node_final.csv"
    f2 = sys.argv[2] if len(sys.argv) > 2 else "benchmark_sweep_multinode_final.csv"
    plot_rigorous_analysis(f1, f2)
