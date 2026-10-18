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

# Chunk k covers rows k*chunk_size ... (k+1)*chunk_size, given round-robin
num_chunks = -(-size[0] // chunk_size)
my_chunks = [k for k in range(num_chunks) if k % num_procs == rank]
my_rows = [x for x in range(size[0]) if (x // chunk_size) % num_procs == rank]

if rank == 0:
    image = np.zeros(size)
else:
    local = np.zeros((len(my_rows), size[1]))

# Synchronize processes before timing
comm.Barrier()
t0 = time()

# Compute each chunk and send it to rank 0 as soon as it is done
for j, k in enumerate(my_chunks):
    x0, x1 = k * chunk_size, min((k + 1) * chunk_size, size[0])
    buf = image[x0:x1] if rank == 0 else local[j * chunk_size:j * chunk_size + x1 - x0]
    for row, x in enumerate(range(x0, x1)):
        cx = complex(xlim[0] + x * xconst, 0)
        for y in range(size[1]):
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z * z + c
                if np.abs(z) > 2:
                    buf[row, y] = i
                    break
    if rank != 0:
        comm.Send(buf, dest=0, tag=k)

# Rank 0 receives the other chunks after computing its own
if rank == 0:
    for k in range(num_chunks):
        if k % num_procs != 0:
            x0, x1 = k * chunk_size, min((k + 1) * chunk_size, size[0])
            comm.Recv(image[x0:x1], source=k % num_procs, tag=k)

# Synchronize processes and compute elapsed time
comm.Barrier()
elapsed = time() - t0

# Rank 0 prints CSV row
if rank == 0:
    print(f"{num_procs},{elapsed:.4f}")
    print(image.sum())
