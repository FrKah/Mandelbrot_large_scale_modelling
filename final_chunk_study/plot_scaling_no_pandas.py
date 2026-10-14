#!/usr/bin/env python3
"""
Mandelbrot Scaling, Speedup, and Parallel Efficiency Plotter (No Pandas)
=======================================================================
Reads benchmark CSV files using Python's built-in csv module and matplotlib.
Generates a 3-panel figure comparing:
  1. Total Runtime vs. MPI Processes (P)
  2. Speedup S(P) = T_1 / T_P vs. Ideal Linear Speedup (S = P)
  3. Parallel Efficiency E(P) = S(P) / P vs. Ideal Efficiency (E = 1.0)

Usage:
  python3 plot_scaling_no_pandas.py [csv_1node] [csv_multinode] [chunk_size]
"""

import csv
import sys
import os
import matplotlib.pyplot as plt


def load_csv_data(filepath):
    """
    Read CSV file into a dictionary: (rank, chunk_size) -> total_time
    """
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return None

    data = {}
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rank = int(row["ranks"].strip())
                chunk = int(row["chunk_size"].strip())
                total_time = float(row["total_time"].strip())
                data[(rank, chunk)] = total_time
            except (KeyError, ValueError) as e:
                continue
    return data


def get_metrics(data, chunk_size):
    """
    Compute ranks, execution times, speedup, and efficiency for a given chunk size.
    """
    if not data:
        return None

    # Determine baseline serial time T_1 (P = 1)
    t1 = data.get((1, chunk_size))
    if t1 is None:
        # Fallback to the average of any rank=1 runs in the dataset
        rank1_times = [t for (r, c), t in data.items() if r == 1]
        if not rank1_times:
            print(f"[Warning] No rank=1 found to compute speedup for chunk={chunk_size}.")
            return None
        t1 = sum(rank1_times) / len(rank1_times)

    # Collect and sort all ranks that ran this chunk size
    ranks = sorted([r for (r, c) in data.keys() if c == chunk_size])
    if not ranks:
        return None

    times = [data[(r, chunk_size)] for r in ranks]
    speedups = [t1 / t for t in times]
    efficiencies = [s / r for s, r in zip(speedups, ranks)]

    return ranks, times, speedups, efficiencies


def plot_scaling(csv_1node="benchmark_sweep_1node_final.csv",
                 csv_multinode="benchmark_sweep_multinode_final.csv",
                 chunk_size=4,
                 bad_chunk=64,
                 output_fig="scaling_speedup_comparison.png"):
    """
    Generate and save the 3-panel scaling comparison plot.
    """
    # Check current directory and final_chunk_study directory
    if not os.path.exists(csv_1node) and os.path.exists(os.path.join("final_chunk_study", csv_1node)):
        csv_1node = os.path.join("final_chunk_study", csv_1node)
    if not os.path.exists(csv_multinode) and os.path.exists(os.path.join("final_chunk_study", csv_multinode)):
        csv_multinode = os.path.join("final_chunk_study", csv_multinode)

    print("=" * 65)
    print("Mandelbrot Scaling & Speedup Benchmark Plotter")
    print("=" * 65)
    print(f"Loading 1-node data:     {csv_1node}")
    print(f"Loading multi-node data: {csv_multinode}")
    print(f"Optimal chunk size:      {chunk_size}")
    print(f"Comparison chunk size:   {bad_chunk}")

    data_1node = load_csv_data(csv_1node)
    data_multi = load_csv_data(csv_multinode)

    if not data_1node and not data_multi:
        print("[Fatal Error] Could not load any valid benchmark data.")
        return

    # Extract metrics
    m_1node = get_metrics(data_1node, chunk_size) if data_1node else None
    m_multi = get_metrics(data_multi, chunk_size) if data_multi else None
    m_1node_bad = get_metrics(data_1node, bad_chunk) if (data_1node and bad_chunk) else None

    # Find unique ranks to set consistent x-axis ticks
    all_ranks_set = set()
    for m in [m_1node, m_multi, m_1node_bad]:
        if m:
            all_ranks_set.update(m[0])

    if not all_ranks_set:
        print("[Fatal Error] No matching rank data found.")
        return

    all_ranks = sorted(list(all_ranks_set))
    max_rank = all_ranks[-1]

    # Create 3-panel figure
    fig, (ax_time, ax_speedup, ax_eff) = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.rcParams.update({"font.size": 11})

    # Reference ideal lines
    ax_speedup.plot([1, max_rank], [1, max_rank], "k--", linewidth=1.5, label="Ideal (S = P)")
    ax_eff.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Ideal (E = 1)")

    # 1. Single-node optimal chunk
    if m_1node:
        r, t, s, e = m_1node
        ax_time.plot(r, t, "o-", color="#1f77b4", linewidth=2.2, markersize=7, label=f"1-Node (chunk={chunk_size})")
        ax_speedup.plot(r, s, "o-", color="#1f77b4", linewidth=2.2, markersize=7, label=f"1-Node (chunk={chunk_size})")
        ax_eff.plot(r, e, "o-", color="#1f77b4", linewidth=2.2, markersize=7, label=f"1-Node (chunk={chunk_size})")

    # 2. Multi-node optimal chunk
    if m_multi:
        r, t, s, e = m_multi
        ax_time.plot(r, t, "^-", color="#d62728", linewidth=2.2, markersize=7, label=f"Multi-Node (chunk={chunk_size})")
        ax_speedup.plot(r, s, "^-", color="#d62728", linewidth=2.2, markersize=7, label=f"Multi-Node (chunk={chunk_size})")
        ax_eff.plot(r, e, "^-", color="#d62728", linewidth=2.2, markersize=7, label=f"Multi-Node (chunk={chunk_size})")

    # 3. Single-node bad chunk (shows severe load imbalance)
    if m_1node_bad:
        r, t, s, e = m_1node_bad
        ax_time.plot(r, t, "s--", color="#aec7e8", linewidth=1.8, markersize=6, label=f"1-Node (chunk={bad_chunk}, imbalanced)")
        ax_speedup.plot(r, s, "s--", color="#aec7e8", linewidth=1.8, markersize=6, label=f"1-Node (chunk={bad_chunk})")
        ax_eff.plot(r, e, "s--", color="#aec7e8", linewidth=1.8, markersize=6, label=f"1-Node (chunk={bad_chunk})")

    # Subplot 1: Total Runtime
    ax_time.set_title("Total Runtime vs. Processes", fontsize=13, fontweight="bold")
    ax_time.set_xlabel("Number of MPI processes (P)", fontsize=11)
    ax_time.set_ylabel("Total time [s]", fontsize=11)
    ax_time.set_xticks(all_ranks)
    ax_time.grid(True, linestyle="--", alpha=0.5)
    ax_time.legend(frameon=True, framealpha=0.9)

    # Subplot 2: Speedup S(P)
    ax_speedup.set_title("(a) Speedup S(P)", fontsize=13, fontweight="bold")
    ax_speedup.set_xlabel("Number of MPI processes (P)", fontsize=11)
    ax_speedup.set_ylabel("Speedup S(P)", fontsize=11)
    ax_speedup.set_xticks(all_ranks)
    ax_speedup.set_ylim(0, max_rank * 1.1)
    ax_speedup.grid(True, linestyle="--", alpha=0.5)
    ax_speedup.legend(frameon=True, framealpha=0.9)

    # Subplot 3: Parallel Efficiency E(P)
    ax_eff.set_title("(b) Parallel Efficiency E(P)", fontsize=13, fontweight="bold")
    ax_eff.set_xlabel("Number of MPI processes (P)", fontsize=11)
    ax_eff.set_ylabel("Parallel efficiency S(P) / P", fontsize=11)
    ax_eff.set_xticks(all_ranks)
    ax_eff.set_ylim(0, 1.15)
    ax_eff.grid(True, linestyle="--", alpha=0.5)
    ax_eff.legend(frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    print(f"\n[Success] Figure saved to: {output_fig}")

    # Print summary table in terminal
    print("\n" + "=" * 65)
    print(f"Summary Table (chunk_size = {chunk_size})")
    print("=" * 65)
    header = f"{'Ranks':<6} | {'1-Node Time':<12} | {'1-Node Speedup':<15} | {'Multi Time':<12} | {'Multi Speedup':<15}"
    print(header)
    print("-" * len(header))

    map_1node = {r: (t, s) for r, t, s in zip(m_1node[0], m_1node[1], m_1node[2])} if m_1node else {}
    map_multi = {r: (t, s) for r, t, s in zip(m_multi[0], m_multi[1], m_multi[2])} if m_multi else {}

    for r in all_ranks:
        t1_str = f"{map_1node[r][0]:.2f}s" if r in map_1node else "N/A"
        s1_str = f"{map_1node[r][1]:.2f}x" if r in map_1node else "N/A"
        tm_str = f"{map_multi[r][0]:.2f}s" if r in map_multi else "N/A"
        sm_str = f"{map_multi[r][1]:.2f}x" if r in map_multi else "N/A"
        print(f"{r:<6} | {t1_str:<12} | {s1_str:<15} | {tm_str:<12} | {sm_str:<15}")


if __name__ == "__main__":
    c1 = sys.argv[1] if len(sys.argv) > 1 else "benchmark_sweep_1node_final.csv"
    c2 = sys.argv[2] if len(sys.argv) > 2 else "benchmark_sweep_multinode_final.csv"
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    plot_scaling(csv_1node=c1, csv_multinode=c2, chunk_size=chunk)
