#!/bin/bash
#SBATCH --job-name=downsamp
#SBATCH --cpus-per-task 20
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=muddi004@odu.edu

enable_lmod
module load container_env pytorch-gpu/1.9.0
crun -p ~/envs/citationparser python csv-downsampler.py