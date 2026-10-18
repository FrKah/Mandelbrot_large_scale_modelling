import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg") # Headless mode for cluster execution
import matplotlib.pyplot as plt

def load_stats(filename):
    data = defaultdict(lambda: {"total": [], "comm_pct": []})
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            size_str = row["size"]
            
            # Skip 500x500 to eliminate sub-second network noise
            if size_str == "500x500":
                continue
                
            total = float(row["total_time"])
            max_comp = float(row["max_comp"])
            
            # Communication percentage relative to total runtime
            comm_pct = ((total - max_comp) / total) * 100.0
            data[size_str]["total"].append(total)
            data[size_str]["comm_pct"].append(comm_pct)
            
    # Sort resolutions by total pixel count
    sizes = sorted(data.keys(), key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]))
    pixels = [int(s.split("x")[0]) * int(s.split("x")[1]) for s in sizes]
    
    mean_time = [np.mean(data[s]["total"]) for s in sizes]
    std_time = [np.std(data[s]["total"]) for s in sizes]
    
    # Calculate observed min/max errors around the mean
    mean_comm = [np.mean(data[s]["comm_pct"]) for s in sizes]
    err_lower = [mean_comm[i] - np.min(data[s]["comm_pct"]) for i, s in enumerate(sizes)]
    err_upper = [np.max(data[s]["comm_pct"]) - mean_comm[i] for i, s in enumerate(sizes)]
    comm_err = [err_lower, err_upper]
    
    return sizes, pixels, mean_time, std_time, mean_comm, comm_err

# Load benchmark data
sizes, pixels, t_1n, std_t_1n, comm_1n, err_c_1n = load_stats("benchmark_sizes_1node_stats.csv")
_, _, t_mn, std_t_mn, comm_mn, err_c_mn = load_stats("benchmark_sizes_multinode_stats.csv")

plt.rcParams.update({"font.size": 10})

# --- Figure 1: Total execution time vs. image resolution ---
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.errorbar(pixels, t_1n, yerr=std_t_1n, fmt="-o", capsize=4, label="1-Node (P=16)", color="#1f77b4")
ax.errorbar(pixels, t_mn, yerr=std_t_mn, fmt="-^", capsize=4, label="Multi-Node (P=16)", color="#d62728")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of Pixels (Log Scale)")
ax.set_ylabel("Execution Time [s] (Log Scale)")
ax.set_title("Execution Time vs. Image Size (P=16, Chunk=8)")
ax.set_xticks(pixels)
ax.set_xticklabels(sizes, rotation=25)
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.savefig("mandelbrot_size_scaling.svg", bbox_inches="tight")
plt.close()

# --- Figure 2: Communication overhead without 500x500 outlier ---
fig, ax = plt.subplots(figsize=(6.5, 4.0))
x = np.arange(len(sizes))
w = 0.35

ax.bar(x - w/2, comm_1n, w, yerr=err_c_1n, capsize=3, label="1-Node", color="#1f77b4", alpha=0.85)
ax.bar(x + w/2, comm_mn, w, yerr=err_c_mn, capsize=3, label="Multi-Node", color="#d62728", alpha=0.85)

ax.set_xlabel("Image Resolution")
ax.set_ylabel("Communication Overhead [%]")
ax.set_title("Communication Overhead vs. Image Resolution (Mean & Min/Max Range)")
ax.set_xticks(x)
ax.set_xticklabels(sizes, rotation=25)
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle=":", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.savefig("mandelbrot_comm_overhead.svg", bbox_inches="tight")
plt.close()