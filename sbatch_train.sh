#!/bin/bash
#
#SBATCH --job-nam=motionControl_train
#SBATCH --output=motionControl.log
#
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --ntasks=1
#SBATCH --time=900:00
#SBATCH --mail-user=kunj.parikh@sjsu.edu
#SBATCH --mail-type=END,FAIL,REQUEUE,STAGE_OUT
#SBATCH --partition=gpu
module load python3
module load cuda
echo `date`
echo $SLURM_NODELIST
python3 train.py
if [ $? -eq 0 ]
then
  echo "The script ran ok"
  cp -v /tmp/model_state.h5 /home/012532065/final/MotionControl/
  cp -v /tmp/model_error.h5 /home/012532065/final/MotionControl/
  echo `date`
  exit 0
else
  echo "The script failed" >&2
  echo `date`
  exit 1
fi
