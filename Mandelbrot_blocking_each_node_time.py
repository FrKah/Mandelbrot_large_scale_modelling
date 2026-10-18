import sys
import numpy as np
from mpi4py import MPI
from time import perf_counter as time
# import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()

# Default parameters
chunk_size = 10
size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

argv = sys.argv[1:]
if argv: chunk_size = int(argv.pop(0))
if argv: size = tuple(map(int, argv.pop(0).split("x")))
if argv: xlim = tuple(map(float, argv.pop(0).split(":")))
if argv: ylim = tuple(map(float, argv.pop(0).split(":")))

size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

N = size[0] // num_procs
start_row = rank * N
end_row = (rank + 1) * N
local = np.zeros((N, size[1]))

# Synchronize processes before timing
comm.Barrier()
t0 = time()

# Compute local chunk
for x in range(start_row, end_row):
    cx = complex(xlim[0] + x * xconst, 0)
    for y in range(size[1]):
        c = cx + complex(0, ylim[0] + y * yconst)
        z = 0
        for i in range(100):
            z = z * z + c
            if np.abs(z) > 2:
                local[x - start_row, y] = i
                break

# This rank's own compute time, measured before any communication or waiting
t_compute = time() - t0

# Gather chunks at rank 0
if rank == 0:
    image = np.zeros(size)
    image[0:N] = local
    for r in range(1, num_procs):
        comm.Recv(image[r * N:(r + 1) * N], source=r)
else:
    comm.Send(local, dest=0)

# Synchronize processes and compute elapsed time
comm.Barrier()
elapsed = time() - t0

# Collect every rank's compute time on rank 0
all_times = comm.gather(t_compute, root=0)

# Rank 0 prints CSV rows
if rank == 0:
    print(f"{num_procs},{elapsed:.4f}", flush=True)
    for r, t in enumerate(all_times):
        print(f"RANK,{num_procs},{r},{t:.4f}", flush=True)

    # plt.rcParams.update({"font.size": 10})
    # plt.imshow(image.T, extent=np.concatenate([xlim, ylim]))
    # plt.xlabel(r"x / Re(p_0)")
    # plt.ylabel(r"y / Im(p_0)")
    # plt.margins(0, 0)
    # plt.savefig("Figure_1.png", bbox_inches="tight", pad_inches=0)