# Mandelbrot MPI Benchmark

## Usage

### 1. Extract the results into a CSV

```bash
grep -E '^[0-9]+,[0-9.]+$' run_results/Mandelbrot_bench<jobid>.out > run_results/results.csv
```

### 2. Generate the plots

```bash
python plot.py
```

Save the graphs in `plot/`