import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = np.loadtxt("run_results/row_major_results.csv", delimiter=",")
nprocs = results[:, 0].astype(int)
t_total = results[:, 1]
p_values = np.unique(nprocs)
t_mean = np.array([t_total[nprocs == p].mean() for p in p_values])

p0 = p_values[0]
t0 = t_mean[0]
speedup = p0 * t0 / t_mean
efficiency = speedup / p_values
title_suffix = "Mandelbrot, row-major distribution"

plt.figure()
plt.plot(p_values, t_mean, "o-")
plt.xlabel("Number of MPI processes")
plt.ylabel("Total time [s]")
plt.title(f"Total time — {title_suffix}")
plt.xticks(p_values)
plt.grid(True)
plt.savefig("plots/row_major_timing.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(p_values, speedup, "o-", label="Measured")
plt.plot(p_values, p_values, "k--", label="Ideal (S = p)")
plt.xlabel("Number of MPI processes")
plt.ylabel("Speedup")
plt.title(f"Speedup — {title_suffix}")
plt.xticks(p_values)
plt.legend()
plt.grid(True)
plt.savefig("plots/row_major_speedup.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(p_values, efficiency, "o-", label="Measured")
plt.axhline(1.0, color="k", linestyle="--", label="Ideal (E = 1)")
plt.xlabel("Number of MPI processes")
plt.ylabel("Parallel efficiency S(p) / p")
plt.title(f"Parallel efficiency — {title_suffix}")
plt.xticks(p_values)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)
plt.savefig("plots/row_major_efficiency.png", bbox_inches="tight")
plt.close()

rank_data = np.loadtxt(
    "run_results/row_major_rank_times.csv", delimiter=",", usecols=(1, 2, 3)
)
rank_nprocs = rank_data[:, 0].astype(int)
ranks = rank_data[:, 1].astype(int)
times = rank_data[:, 2]

for p in (8, 16):
    sel = rank_nprocs == p
    r_ids = np.arange(p)
    t = np.array([times[sel & (ranks == r)].mean() for r in r_ids])
    t_max = t.max()

    plt.figure()
    plt.bar(r_ids, t, label="Computing")
    plt.bar(
        r_ids,
        t_max - t,
        bottom=t,
        color="lightgrey",
        hatch="//",
        edgecolor="grey",
        label="Idle (waiting)",
    )
    plt.axhline(t_max, color="k", linestyle="--", label="Slowest rank")
    plt.xlabel("Rank")
    plt.ylabel("Compute time [s]")
    idle_frac = 1 - t.sum() / (p * t_max)
    plt.title(f"Row-major load balance, {p} processes — {idle_frac:.0%} idle")
    plt.xticks(r_ids)
    plt.legend()
    plt.savefig(f"plots/row_major_load_balance_{p}.png", bbox_inches="tight")
    plt.close()
