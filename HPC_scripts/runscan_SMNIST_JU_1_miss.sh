#! /bin/bash

# set the account
#SBATCH --account=structuretofunction

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=job1

# set the number of nodes
#SBATCH --nodes=1

# set number of GPUs
#SBATCH --gres=gpu:4

# set the partition to use
#SBATCH --partition=booster

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=t.nowotny@sussex.ac.uk

# run the application
ID=`head -n $[$SLURM_ARRAY_TASK_ID*4+1] scan_SMNIST_JU_1/missing.txt | tail -n 1`
srun --exclusive -n 1 --gres=gpu:1 python runscan_SMNIST_JUWELS.py scan_SMNIST_JU_1/SMJ1_scan_$ID.json &
ID=`head -n $[$SLURM_ARRAY_TASK_ID*4+2] scan_SMNIST_JU_1/missing.txt | tail -n 1`
srun --exclusive -n 1 --gres=gpu:1 python runscan_SMNIST_JUWELS.py scan_SMNIST_JU_1/SMJ1_scan_$ID.json &
ID=`head -n $[$SLURM_ARRAY_TASK_ID*4+3] scan_SMNIST_JU_1/missing.txt | tail -n 1`
srun --exclusive -n 1 --gres=gpu:1 python runscan_SMNIST_JUWELS.py scan_SMNIST_JU_1/SMJ1_scan_$ID.json &
ID=`head -n $[$SLURM_ARRAY_TASK_ID*4+4] scan_SMNIST_JU_1/missing.txt | tail -n 1`
srun --exclusive -n 1 --gres=gpu:1 python runscan_SMNIST_JUWELS.py scan_SMNIST_JU_1/SMJ1_scan_$ID.json &

wait
