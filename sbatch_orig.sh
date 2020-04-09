#!/bin/bash
#
#SBATCH --job-nam=motionControl
#SBATCH --output=motionControl.log
#
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000
#SBATCH --ntasks=1
#SBATCH --time=900:00
#SBATCH --mail-user=kunj.parikh@sjsu.edu
#SBATCH --mail-type=END
#SBATCH --partition=gpu
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
module load python3
echo `date`
echo $PYTHONPATH
srun python3 train.py
echo `date`
