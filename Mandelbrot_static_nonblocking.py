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
my_rows = [x for x in range(size[0]) if (x // chunk_size) % num_procs == rank]
local = np.zeros((len(my_rows), size[1]))

# Synchronize processes before timing
comm.Barrier()
t0 = time()

# Compute local chunk
for k, x in enumerate(my_rows):
    cx = complex(xlim[0] + x * xconst, 0)
    for y in range(size[1]):
        c = cx + complex(0, ylim[0] + y * yconst)
        z = 0
        for i in range(100):
            z = z * z + c
            if np.abs(z) > 2:
                local[k, y] = i
                break

# Gather chunks at rank 0
if rank == 0:
    image = np.zeros(size)
    image[my_rows] = local
    requests = []
    all_rows = []
    all_bufs = []
    for r in range(1, num_procs):
        rows_r = [x for x in range(size[0]) if (x // chunk_size) % num_procs == r]
        buf = np.empty((len(rows_r), size[1]))
        requests.append(comm.Irecv(buf, source=r))
        all_rows.append(rows_r)
        all_bufs.append(buf)
    MPI.Request.Waitall(requests)
    for rows_r, buf in zip(all_rows, all_bufs):
        image[rows_r] = buf
else:
    req = comm.Isend(local, dest=0)
    req.Wait()


# Synchronize processes and compute elapsed time
comm.Barrier()
elapsed = time() - t0

# Rank 0 prints CSV row
if rank == 0:
    print(f"{num_procs},{elapsed:.4f}")
    print(image.sum())
    
    # plt.rcParams.update({"font.size": 10})
    # plt.imshow(image.T, extent=np.concatenate([xlim, ylim]))
    # plt.xlabel(r"x / Re(p_0)")
    # plt.ylabel(r"y / Im(p_0)")
    # plt.margins(0, 0)
    # plt.savefig("Figure_1.png", bbox_inches="tight", pad_inches=0)
