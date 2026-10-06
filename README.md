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