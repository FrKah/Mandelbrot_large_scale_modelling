import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Columns: RANK,nprocs,rank,t_compute
data = np.loadtxt("run_results/rank_times.csv", delimiter=",", usecols=(1, 2, 3))
nprocs = data[:, 0].astype(int)
ranks = data[:, 1].astype(int)
times = data[:, 2]

for p in np.unique(nprocs):
    if p == 1:
        continue                                  # nothing to balance
    sel = nprocs == p
    # Average the repetitions for each rank
    r_ids = np.arange(p)
    t = np.array([times[sel & (ranks == r)].mean() for r in r_ids])
    t_max = t.max()

    plt.figure()
    plt.bar(r_ids, t, label="Computing")
    plt.bar(r_ids, t_max - t, bottom=t, color="lightgrey",
            hatch="//", edgecolor="grey", label="Idle (waiting)")
    plt.axhline(t_max, color="k", linestyle="--", label="Slowest rank")
    plt.xlabel("Rank")
    plt.ylabel("Compute time [s]")
    idle_frac = 1 - t.sum() / (p * t_max)
    plt.title(f"Load balance, {p} processes — {idle_frac:.0%} of core time idle")
    plt.xticks(r_ids)
    plt.legend()
    plt.savefig(f"plots/load_balance_{p}.png", bbox_inches="tight")
    plt.close()