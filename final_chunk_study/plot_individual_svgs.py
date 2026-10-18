#!/usr/bin/env python3
import csv
import sys
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


def plot_row_normalized_heatmap(data, ranks, chunks, title, filename):
    """Plot a clean heatmap where color scale resets per rank (per row)."""
    # Build raw and row-normalized matrices
    raw_matrix = [[data.get((r, c), float("nan")) for c in chunks] for r in ranks]
    norm_matrix = []

    best_chunks = {}
    for i, row in enumerate(raw_matrix):
        valid = [v for v in row if v == v]
        r_min, r_max = min(valid), max(valid)
        best_chunks[ranks[i]] = chunks[row.index(r_min)]
        # Min is 0.0 (fastest/dark), Max is 1.0 (slowest/light)
        span = (r_max - r_min) if (r_max - r_min) > 0 else 1.0
        norm_matrix.append([(v - r_min) / span for v in row])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(norm_matrix, cmap="YlGnBu_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(chunks)))
    ax.set_xticklabels(chunks)
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels(ranks)
    ax.set_xlabel("Chunk Size")
    ax.set_ylabel("MPI Processes")
    ax.set_title(title, pad=10)

    # Annotate cell values
    for i, r in enumerate(ranks):
        best_c = best_chunks[r]
        for j, c in enumerate(chunks):
            val = data.get((r, c))
            if val is not None:
                is_best = (c == best_c)
                norm_v = norm_matrix[i][j]
                # High norm_matrix (darker) gets white text
                txt_color = "white" if norm_v < 0.4 else "black"
                ax.text(j, i, f"{val:.1f}s", ha="center", va="center",
                        color=txt_color, weight="bold" if is_best else "normal", fontsize=9)
                if is_best:
                    rect = patches.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                             fill=False, edgecolor="#d62728", lw=1.5)
                    ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(filename, format="svg", bbox_inches="tight")
    plt.close()
    print(f"[Saved] {filename}")


def plot_scaling_curves(ranks, opt_1node, opt_multi, t1_base):
    """Plot clean Speedup and Efficiency curves."""
    all_p = [1] + ranks
    s_1node = [1.0] + [t1_base / opt_1node[r][0] for r in ranks]
    s_multi = [1.0] + [t1_base / opt_multi[r][0] for r in ranks]

    # 1. Speedup
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(all_p, all_p, "k--", lw=1.2, label="Ideal")
    ax.plot(all_p, s_1node, "o-", color="#1f77b4", lw=1.8, label="1-Node")
    ax.plot(all_p, s_multi, "^-", color="#d62728", lw=1.8, label="Multi-Node")

    for p, s in zip(ranks, s_1node[1:]):
        c = opt_1node[p][1]
        ax.annotate(f"C={c}", (p, s), textcoords="offset points", xytext=(-10, 8), fontsize=8)

    for p, s in zip(ranks, s_multi[1:]):
        c = opt_multi[p][1]
        ax.annotate(f"C={c}", (p, s), textcoords="offset points", xytext=(6, -12), fontsize=8)

    ax.set_title("Mandelbrot Speedup S(P)", pad=10)
    ax.set_xlabel("MPI Processes (P)")
    ax.set_ylabel("Speedup S(P)")
    ax.set_xticks(all_p)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("mandelbrot_speedup_best.svg", format="svg")
    plt.close()
    print("[Saved] mandelbrot_speedup_best.svg")

    # 2. Efficiency
    e_1node = [s / p for s, p in zip(s_1node, all_p)]
    e_multi = [s / p for s, p in zip(s_multi, all_p)]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="Ideal")
    ax.plot(all_p, e_1node, "o-", color="#1f77b4", lw=1.8, label="1-Node")
    ax.plot(all_p, e_multi, "^-", color="#d62728", lw=1.8, label="Multi-Node")

    ax.set_title("Mandelbrot Parallel Efficiency E(P)", pad=10)
    ax.set_xlabel("MPI Processes (P)")
    ax.set_ylabel("Efficiency E(P)")
    ax.set_xticks(all_p)
    ax.set_ylim(0, 1.1)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("mandelbrot_efficiency_best.svg", format="svg")
    plt.close()
    print("[Saved] mandelbrot_efficiency_best.svg")


def main(f_1node="benchmark_sweep_1node_final.csv",
         f_multi="benchmark_sweep_multinode_final.csv"):
    d_1node = load_data(f_1node)
    d_multi = load_data(f_multi)

    ranks = [2, 4, 8, 16]
    chunks = [1, 2, 4, 8, 16, 32, 64]

    # Global serial baseline T1
    all_t1 = [t for (r, c), t in {**d_1node, **d_multi}.items() if r == 1]
    t1_base = min(all_t1) if all_t1 else 128.61

    opt_1node = {r: min((d_1node[(r, c)], c) for c in chunks if (r, c) in d_1node) for r in ranks}
    opt_multi = {r: min((d_multi[(r, c)], c) for c in chunks if (r, c) in d_multi) for r in ranks}

    plt.rcParams.update({"font.size": 10})

    # Heatmaps (row-normalized)
    plot_row_normalized_heatmap(
        d_1node, ranks, chunks,
        "Single-Node Relative Runtime (Row-Normalized)",
        "mandelbrot_heatmap_1node.svg"
    )
    plot_row_normalized_heatmap(
        d_multi, ranks, chunks,
        "Multi-Node Relative Runtime (Row-Normalized)",
        "mandelbrot_heatmap_multinode.svg"
    )

    # Scaling curves
    plot_scaling_curves(ranks, opt_1node, opt_multi, t1_base)


if __name__ == "__main__":
    f1 = sys.argv[1] if len(sys.argv) > 1 else "benchmark_sweep_1node_final.csv"
    f2 = sys.argv[2] if len(sys.argv) > 2 else "benchmark_sweep_multinode_final.csv"
    main(f1, f2)
