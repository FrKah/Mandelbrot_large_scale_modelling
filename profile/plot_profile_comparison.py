#!/usr/bin/env python3
"""
Compares Static Block vs. Cyclic Distribution side-by-side in SVG format.
"""

import csv
import matplotlib.pyplot as plt


def load_profile(filename):
    """Read CSV file returning lists of rank, compute_time, and idle_time."""
    ranks, comp, idle = [], [], []
    with open(filename, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ranks.append(int(row["rank"]))
            comp.append(float(row["compute_time"]))
            idle.append(float(row["idle_time"]))
    return ranks, comp, idle


def main():
    # Load profile data
    ranks, comp_block, idle_block = load_profile("profile_block.csv")
    _, comp_cyclic, idle_cyclic = load_profile("profile_cyclic_4.csv")

    n_procs = len(ranks)
    slow_block = max(comp_block)
    idle_pct_block = (sum(idle_block) / (n_procs * slow_block)) * 100

    slow_cyclic = max(comp_cyclic)
    idle_pct_cyclic = (sum(idle_cyclic) / (n_procs * slow_cyclic)) * 100

    # Create 1x2 side-by-side subplots sharing the Y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    plt.rcParams.update({"font.size": 11})

    # (a) Static Block Subplot
    ax1.bar(ranks, comp_block, label="Computing", color="#1f77b4", width=0.8)
    ax1.bar(ranks, idle_block, bottom=comp_block, label="Idle (waiting)",
            color="#e0e0e0", hatch="//", edgecolor="#707070", width=0.8)
    ax1.axhline(slow_block, color="black", linestyle="--", linewidth=1.5,
                label=f"Slowest rank ({slow_block:.1f}s)")
    ax1.set_title(f"(a) Static Block Distribution\n{idle_pct_block:.0f}% of core time idle",
                  fontweight="bold", fontsize=12)
    ax1.set_xlabel("Rank", fontweight="bold")
    ax1.set_ylabel("Time [s]", fontweight="bold")
    ax1.set_xticks(ranks)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.95)

    # (b) Cyclic Subplot
    ax2.bar(ranks, comp_cyclic, label="Computing", color="#1f77b4", width=0.8)
    ax2.bar(ranks, idle_cyclic, bottom=comp_cyclic, label="Idle (waiting)",
            color="#e0e0e0", hatch="//", edgecolor="#707070", width=0.8)
    ax2.axhline(slow_cyclic, color="black", linestyle="--", linewidth=1.5,
                label=f"Slowest rank ({slow_cyclic:.1f}s)")
    ax2.set_title(f"(b) Cyclic Distribution (chunk=4)\n{idle_pct_cyclic:.0f}% of core time idle",
                  fontweight="bold", fontsize=12)
    ax2.set_xlabel("Rank", fontweight="bold")
    ax2.set_xticks(ranks)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    out_svg = "load_balance_comparison.svg"
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    print(f"[Success] Comparison SVG saved to: {out_svg}")


if __name__ == "__main__":
    main()
