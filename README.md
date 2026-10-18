# Mandelbrot MPI Benchmark

## Usage

### 1. Extract the results into a CSV

Collect total run times :
```bash
grep -E '^[0-9]+,[0-9.]+$' run_results/Mandelbrot_bench<jobid>.out > run_results/results.csv
```
Collect per node run time :
```bash
grep '^RANK,' run_results/Mandelbrot_bench<jobid>.out > run_results/rank_times.csv
```

### 2. Generate the plots

```bash
python plot.py
```
```bash
python plot_time_per_node.py
```

Save the graphs in `plot/`

## Row-major implementation

`Mandelbrot_row_major.py` uses rows for y and columns for x, so its image
array can be passed directly to `imshow` without transposing it. It accepts
the same arguments and prints the same `nprocs,t_total` CSV rows as
`Mandelbrot_blocking.py`:

```bash
mpirun -n 4 python3 -u Mandelbrot_row_major.py 10 1000x1000
```

Run the row-major benchmark and generate its separate figures:

```bash
bsub < Mandelbrot_row_major_bench.sh
grep -E '^[0-9]+,[0-9.]+$' run_results/Mandelbrot_row_major_benchmark%J.out \
    > run_results/row_major_results.csv
grep '^RANK,' run_results/Mandelbrot_row_major_benchmark%J.out \
    > run_results/row_major_rank_times.csv
python plot_row_major.py
```
