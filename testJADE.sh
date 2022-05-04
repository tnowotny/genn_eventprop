#! /bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=2:00:00

# set name of job
#SBATCH --job-name=job1

# set number of GPUs
#SBATCH --gres=gpu:1

# set the partition to use
#SBATCH --partition=small

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=t.nowotny@sussex.ac.uk

# run the application
. ~/test2/bin/activate
python train_SHD.py
