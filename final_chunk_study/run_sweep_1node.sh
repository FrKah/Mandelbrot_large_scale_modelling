#!/bin/bash
#BSUB -J sweep_1node
#BSUB -o sweep_1node_%J.out
#BSUB -e sweep_1node_%J.err
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -q hpcintro
#BSUB -W 01:00
#BSUB -R "rusage[mem=2GB]"

module load matplotlib/3.10.6-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

OUTPUT="benchmark_sweep_1node_final.csv"
PYTHON_SCRIPT="Mandelbrot_chunks.py"
IMG_SIZE="2000x2000"

echo "ranks,chunk_size,total_time,max_comp,min_comp,avg_comp" > "$OUTPUT"

RANKS=(1 2 4 8 16)
CHUNKS=(1 2 4 8 16 32 64)

for P in "${RANKS[@]}"; do
    for C in "${CHUNKS[@]}"; do
        echo "=== [1-NODE] Executant ranks=$P, chunk_size=$C ==="
        mpirun -n "$P" python3 "$PYTHON_SCRIPT" "$C" "$IMG_SIZE" >> "$OUTPUT"
    done
done
