#!/bin/bash
#BSUB -J mandelbrot_profile
#BSUB -o profile_%J.out
#BSUB -e profile_%J.err
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -q hpcintro
#BSUB -W 00:15
#BSUB -R "rusage[mem=2GB]"

module load matplotlib/3.10.6-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

mpirun -n 16 python3 mandelbrot_profile.py block 2000x2000

mpirun -n 16 python3 mandelbrot_profile.py cyclic 4 2000x2000

ls -la load_balance_*.png
