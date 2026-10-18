#!/bin/bash
#BSUB -J sizes_bench_mn
#BSUB -o sizes_%J.out
#BSUB -e sizes_%J.err
#BSUB -n 16
#BSUB -R "span[ptile=4]"
#BSUB -q hpcintro
#BSUB -W 00:30
#BSUB -R "rusage[mem=4GB]"

module load matplotlib/3.10.6-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

OUTPUT="benchmark_sizes_multinode_stats.csv"
PYTHON_SCRIPT="Mandelbrot_chunks.py"

echo "size,run,ranks,chunk_size,total_time,max_comp,min_comp,avg_comp" > "$OUTPUT"

FAST_SIZES=("500x500" "1000x1000" "2000x2000" "4000x4000")
SLOW_SIZES=("8000x8000" "10000x10000")

C=8
P=16

for S in "${FAST_SIZES[@]}"; do
    for RUN in {1..3}; do
        echo "=== [MULTINODE] size=$S, run=$RUN ==="
        mpirun -n "$P" python3 "$PYTHON_SCRIPT" "$C" "$S" | sed "s/^/${S},${RUN},/" >> "$OUTPUT"
    done
done

for S in "${SLOW_SIZES[@]}"; do
    echo "=== [MULTINODE] size=$S, run=1 ==="
    mpirun -n "$P" python3 "$PYTHON_SCRIPT" "$C" "$S" | sed "s/^/${S},1,/" >> "$OUTPUT"
done