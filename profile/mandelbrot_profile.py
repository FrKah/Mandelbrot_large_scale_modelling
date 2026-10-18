import sys
import numpy as np
from mpi4py import MPI
from time import perf_counter as time

import matplotlib
matplotlib.use("Agg") # Since we are working inside the cluster.
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()

mode = "block"
chunk_size = 4
size = (2000, 2000) # Larger size, however still less that 10.000 x 10.000, because of the execution time.
xlim = (-2.2, 0.75)
ylim = (-1.3, 1.3)

argv = sys.argv[1:]
if argv:
    mode = argv.pop(0).lower()
if argv and mode == "cyclic":
    chunk_size = int(argv.pop(0))
if argv and "x" in argv[0]:
    size = tuple(map(int, argv.pop(0).split("x")))

size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]


if mode == "block":
    rows_per_proc = size[0] // num_procs # Standard division by number of ranks.
    remainder = size[0] % num_procs
    
    start = rank * rows_per_proc + min(rank, remainder)
    end = start + rows_per_proc + (1 if rank < remainder else 0)
    my_chunks = [(start, end)]
    all_chunks_info = []
    if rank == 0:
        for r in range(num_procs):
            s = r * rows_per_proc + min(r, remainder)
            e = s + rows_per_proc + (1 if r < remainder else 0)
            all_chunks_info.append((r, s, e))
else:
    chunks = [(s, min(s + chunk_size, size[0])) for s in range(0, size[0], chunk_size)] # division by a cyclic shared
    my_chunks = [c for i, c in enumerate(chunks) if (i % num_procs) == rank]
    all_chunks_info = [(i % num_procs, s, e) for i, (s, e) in enumerate(chunks)]

comm.Barrier()
t_start = time()

# compute local chunk
local_results = []
for start, end in my_chunks:
    chunk = np.zeros((end - start, size[1]))
    for x in range(start, end):
        cx = complex(xlim[0] + x * xconst, 0)
        for y in range(size[1]):
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z * z + c
                if np.abs(z) > 2:
                    chunk[x - start, y] = i
                    break
    local_results.append(chunk)

t_comp = time() - t_start
# Gather at rank 0
if rank == 0:
    image = np.zeros(size)
    local_idx = 0
    for owner, start, end in all_chunks_info:
        if owner == 0:
            image[start:end] = local_results[local_idx]
            local_idx += 1
        else:
            comm.Recv(image[start:end], source=owner, tag=start)
    
    comp_times = np.zeros(num_procs, dtype=np.float64)
    comp_times[0] = t_comp
    
    for r in range(1, num_procs):
        r_time = np.zeros(1, dtype=np.float64)
        comm.Recv(r_time, source=r, tag=9999)
        comp_times[r] = r_time[0]
else:
    for (start, end), chunk in zip(my_chunks, local_results):
        comm.Send(chunk, dest=0, tag=start)
    
    t_buf = np.array([t_comp], dtype=np.float64)
    comm.Send(t_buf, dest=0, tag=9999)

comm.Barrier()
# Compute the weight balancing
if rank == 0:
    t_slowest = np.max(comp_times)
    idle_times = t_slowest - comp_times
    total_core_time = num_procs * t_slowest
    idle_pct = (np.sum(idle_times) / total_core_time) * 100.0 if total_core_time > 0 else 0.0

    print(f"[{mode.upper()}] Slowest: {t_slowest:.4f}s | {idle_pct:.1f}%")

    csv_filename = f"profile_{mode}{f'_{chunk_size}' if mode == 'cyclic' else ''}.csv"
    with open(csv_filename, "w") as f:
        f.write("rank,compute_time,idle_time\n")
        for r in range(num_procs):
            f.write(f"{r},{comp_times[r]:.4f},{idle_times[r]:.4f}\n")

    plt.figure(figsize=(8, 6.5))
    ranks = np.arange(num_procs)

    plt.bar(ranks, comp_times, label="Computing", color="#1f77b4", width=0.8)
    
    plt.bar(
        ranks, idle_times, bottom=comp_times,
        label="Idle (waiting)", color="#e0e0e0",
        hatch="//", edgecolor="#707070", width=0.8
    )

    plt.axhline(t_slowest, color="black", linestyle="--", linewidth=1.8, label="Slowest rank")

    title_suffix = f"chunk_size={chunk_size}" if mode == "cyclic" else "contiguous blocks"
    plt.title(f"Load balance, {num_procs} processes — {idle_pct:.0f}% of core time idle\n({mode.capitalize()}: {title_suffix})", fontsize=12)
    plt.xlabel("Rank", fontsize=11)
    plt.ylabel("Compute time [s]", fontsize=11)
    plt.xticks(ranks)
    plt.ylim(0, max(t_slowest * 1.15, 0.1))
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    fig_filename = f"load_balance_{mode}{f'_{chunk_size}' if mode == 'cyclic' else ''}.png"
    plt.savefig(fig_filename, dpi=300, bbox_inches="tight")
    print(f"Saved graph: {fig_filename}")
