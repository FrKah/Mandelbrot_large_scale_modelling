import sys
import numpy as np
from mpi4py import MPI
from time import perf_counter as time
# import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()
host = MPI.Get_processor_name()

# num_procs check is done after parsing arguments

# Default parameters
chunk_size = 10
size = 2000, 2000
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

if num_procs == 1:
    t0 = time()
    for s in range(0, size[0], chunk_size):
        end = min(s + chunk_size, size[0])
        for x in range(s, end):
            cx = complex(xlim[0] + x * xconst, 0)
            for y in range(size[1]):
                c = cx + complex(0, ylim[0] + y * yconst)
                z = 0
                for i in range(100):
                    z = z * z + c
                    if np.abs(z) > 2:
                        break
    elapsed = time() - t0
    print(f"1,{chunk_size},{elapsed:.4f},{elapsed:.4f},{elapsed:.4f},{elapsed:.4f}")
    sys.exit(0)

num_workers = num_procs - 1

chunks = [(s, min(s + chunk_size, size[0])) for s in range(0, size[0], chunk_size)]
my_chunks = [c for i, c in enumerate(chunks) if (i % num_workers) + 1 == rank] # assigne value to each chunk

comm.Barrier()
t0 = time()
t_comp = 0.0;

local_chunks = []
if rank>0:
    t_start = time()
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
        t_comp = time()-t_start
        local_chunks.append(chunk)


if rank == 0:
    image = np.zeros(size)
    for i, (start, end) in enumerate(chunks):
        src = (i % num_workers) + 1
        comm.Recv(image[start:end], source=src)
else:
    for chunk in local_chunks:
        comm.Send(chunk, dest=0)


comm.Barrier()
elapsed = time() - t0

val_max = t_comp if rank > 0 else 0.0                                                                 
val_min = t_comp if rank > 0 else float("inf")                                                                
val_sum = t_comp if rank > 0 else 0.0                                                            
max_comp = comm.reduce(val_max, op=MPI.MAX, root=0)                                                            
min_comp = comm.reduce(val_min, op=MPI.MIN, root=0)                                                            
sum_comp = comm.reduce(val_sum, op=MPI.SUM, root=0)


if rank == 0 and sum_comp is not None and max_comp is not None and min_comp is not None:                                                                                                  
    avg_comp = sum_comp / num_workers                                                                          
    # Format: ranks,chunk_size,total_time,max_comp,min_comp,avg_comp                                           
    print(f"{num_procs},{chunk_size},{elapsed:.4f},{max_comp:.4f},{min_comp:.4f},{avg_comp:.4f}")

    # plt.rcParams.update({"font.size": 10})
    # plt.imshow(image.T, extent=np.concatenate([xlim, ylim]), cmap="inferno", origin="lower")
    # plt.xlabel(r"x / Re(p_0)")
    # plt.ylabel(r"y / Im(p_0)")
    # plt.margins(0, 0)
    # plt.savefig("Figure_1.png", bbox_inches="tight", pad_inches=0)
