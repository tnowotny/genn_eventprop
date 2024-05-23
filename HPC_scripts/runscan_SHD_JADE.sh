#! /bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=job1

# set number of jobs per node
#SBATCH --ntasks-per-node=4

# set number of GPUs
#SBATCH --gres=gpu:4

# set the partition to use
#SBATCH --partition=booster

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=t.nowotny@sussex.ac.uk

# run the application
. ~/test2/bin/activate
srun python runscan_SHD_JUWELS.py scan1/scan1_$[$SLURM_ARRAY_TASK_ID*4].json
srun python runscan_SHD_JUWELS.py scan1/scan1_$[$SLURM_ARRAY_TASK_ID*4+1].json
srun python runscan_SHD_JUWELS.py scan1/scan1_$[$SLURM_ARRAY_TASK_ID*4+2].json
srun python runscan_SHD_JUWELS.py scan1/scan1_$[$SLURM_ARRAY_TASK_ID*4+3].json
