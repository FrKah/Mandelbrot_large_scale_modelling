#BSUB -J MandelbrotDynamic
#BSUB -o Outputs/MandelbrotDynamic_%J.out
#BSUB -e Outputs/MandelbrotDynamic_%J.err
#BSUB -n 16
#BSUB -R "span[ptile=16]"
#BSUB -q hpcintro
#BSUB -W 00:30
#BSUB -R "rusage[mem=2GB]"

module load mpi/5.0.8-gcc-13.4.0-binutils-2.44
python3 benchmark_dynamic.py \
  --mpiexec mpirun \
  --size 5000x5000 \
  --chunk-size 10 \
  --ranks 1,2,4,8,16 \
  --repetitions 3 \
  --label dynamic-one-node \
  --output-dir Outputs \
  --output-prefix benchmark_dynamic_one_node
