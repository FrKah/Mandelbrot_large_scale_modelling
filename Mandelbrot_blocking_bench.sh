#BSUB -J Mandelbrot_bench
#BSUB -o run_results/Mandelbrot_blocking_bench%J.out
#BSUB -e run_results/Mandelbrot_blocking_bench%J.err
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -q hpcintro
#BSUB -W 00:30
#BSUB -R "rusage[mem=1GB]"

module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

for p in 1 2 4 8 16; do
    for rep in 1 2 3; do
        mpirun -n $p python3 -u Mandelbrot_blocking.py 10 1000x1000
    done
done