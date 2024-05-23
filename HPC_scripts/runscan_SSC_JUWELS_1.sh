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
srun --exclusive -n 1 --gres=gpu:1 python runscan_SSC_JUWELS.py scan_SSC_JUWELS_1/JSSC1_scan_0.json &
srun --exclusive -n 1 --gres=gpu:1 python runscan_SSC_JUWELS.py scan_SSC_JUWELS_1/JSSC1_scan_1.json &
srun --exclusive -n 1 --gres=gpu:1 python runscan_SSC_JUWELS.py scan_SSC_JUWELS_1/JSSC1_scan_2.json &
srun --exclusive -n 1 --gres=gpu:1 python runscan_SSC_JUWELS.py scan_SSC_JUWELS_1/JSSC1_scan_3.json &

wait
