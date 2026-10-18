import numpy as np
import matplotlib
matplotlib.use("Agg")          # no screen needed, so it works on the cluster
import matplotlib.pyplot as plt

# grep -E '^[0-9]+,[0-9.]+$' run_results/Mandelbrot_bench<jobid>.out > run_results/results.csv

# Columns in results.csv: nprocs,t_total
data = np.loadtxt("run_results/results.csv", delimiter=",")
nprocs = data[:, 0].astype(int)
t_total = data[:, 1]

# Average the repetitions for each number of processes
p_values = np.unique(nprocs)
t_mean = np.array([t_total[nprocs == p].mean() for p in p_values])

# Reference run: normally p = 1. If 1 is missing, use the smallest p
# (relative speedup, scaled so that it equals p0 at p0).
p0 = p_values[0]
t0 = t_mean[0]
speedup = p0 * t0 / t_mean
efficiency = speedup / p_values

title_suffix = "Mandelbrot, block distribution"

# 1. Total time
plt.figure()
plt.plot(p_values, t_mean, "o-")
plt.xlabel("Number of MPI processes")
plt.ylabel("Total time [s]")
plt.title(f"Total time — {title_suffix}")
plt.xticks(p_values)
plt.grid(True)
plt.savefig("plots/timing.png", bbox_inches="tight")
plt.close()

# 2. Speedup with ideal line
plt.figure()
plt.plot(p_values, speedup, "o-", label="Measured")
plt.plot(p_values, p_values, "k--", label="Ideal (S = p)")
plt.xlabel("Number of MPI processes")
plt.ylabel("Speedup")
plt.title(f"Speedup — {title_suffix}")
plt.xticks(p_values)
plt.legend()
plt.grid(True)
plt.savefig("plots/speedup.png", bbox_inches="tight")
plt.close()

# 3. Parallel efficiency with ideal line at 1
plt.figure()
plt.plot(p_values, efficiency, "o-", label="Measured")
plt.axhline(1.0, color="k", linestyle="--", label="Ideal (E = 1)")
plt.xlabel("Number of MPI processes")
plt.ylabel("Parallel efficiency  S(p) / p")
plt.title(f"Parallel efficiency — {title_suffix}")
plt.xticks(p_values)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)
plt.savefig("plots/efficiency.png", bbox_inches="tight")
plt.close()