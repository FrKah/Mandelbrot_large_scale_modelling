# Mandelbrot MPI (02616 Project 1)

An MPI (mpi4py) Mandelbrot implementation, benchmarked on the DTU HPC (`hpcintro` queue).

## Running

Every script takes the same positional arguments, all of which are optional:

```bash
mpirun -n <ranks> python3 <script>.py <chunk_size> <W>x<H> <xmin>:<xmax> <ymin>:<ymax>
mpirun -n 4 python3 Mandelbrot_blocking.py 10 1000x1000
```

Defaults: chunk 10, 1000x1000, x in [-2.2, 0.75], y in [-1.3, 1.3].

On the HPC, submit a job script with `bsub < <script>.sh` (or `.sub`). The scripts load these modules:

```bash
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
```

## Contents

| Folder / file | What it is |
|---|---|
| `Mandelbrot_blocking.py`, `Mandelbrot_blocking_bench.sh` | Baseline: rows distributed round-robin in chunks, blocking `Send`/`Recv`. |
| `Mandelbrot_blocking_each_node_time.py` | Baseline that also prints each rank's time (load balance). |
| `Mandelbrot_row_major.py`, `Mandelbrot_row_major_bench.sh` | Row-major version (no transpose needed for `imshow`). |
| `plot.py`, `plot_time_per_node.py`, `plot_row_major.py` | Plots for the baseline and row-major runs; output goes to `plots/`. |
| `run_results/` | Raw output and CSVs from the baseline runs. |
| `profile/` | Load balance of block vs cyclic distribution (2000x2000, 16 ranks). |
| `final_chunk_study/` | Chunk size sweep (1–64) × ranks (1–16), 1 node vs 4 nodes, 2000x2000. |
| `image-size/` | Image size (500²–10000²), 16 ranks, 1 node vs 4 nodes. |
| `project1/` | Blocking vs non-blocking comparison (see below). |
| `mandelbrot_mpi_dynamic.py`, `mandelbrot_mpi_common.py` | Dynamic scheduling: workers request new row chunks from rank 0. |
| `benchmark_dynamic.py`, `submit_dynamic_one_node.sh` | Benchmark and plots for the dynamic version (5000x5000, 1–16 ranks, 1 node). |

## Blocking vs non-blocking (`project1/`)

| File | Communication |
|---|---|
| `Mandelbrot_static_blocking.py` | One message per rank: `Send`/`Recv` |
| `Mandelbrot_static_nonblocking.py` | One message per rank: `Isend`/`Irecv` + `Waitall` |
| `Mandelbrot_chunk_blocking.py` | One message per chunk: `Send`/`Recv` (tag = chunk index) |
| `Mandelbrot_chunk_nonblocking.py` | One message per chunk: `Isend`; rank 0 posts every `Irecv` before computing |

Job scripts:
- `scaling_{blocking,nonblocking}.sub`: static versions, 1–8 ranks, 10000x10000. Output is `ranks,time,compute`, where `compute` is the slowest rank's compute time.
- `chunk.sub`: per-chunk versions, 1–16 ranks, 1000x1000, 3 repetitions. Output is `version,ranks,time`.

## Extracting baseline results

```bash
grep -E '^[0-9]+,[0-9.]+$' run_results/Mandelbrot_bench<jobid>.out > run_results/results.csv
grep '^RANK,' run_results/Mandelbrot_bench<jobid>.out > run_results/rank_times.csv
python plot.py
python plot_time_per_node.py
```
