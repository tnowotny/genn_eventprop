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
name=fix\_$[$SLURM_ARRAY_TASK_ID]
for i in 0 1 2 3; do
    tid=$(head -n 1 $name)
    tail -n +2 $name > $name.tmp
    mv $name.tmp $name
    srun --exclusive -n 1 --gres=gpu:1 python runscan_SHD_JUWELS.py scan_JUWELS_30/J30_scan\_$tid.json &
done

wait
