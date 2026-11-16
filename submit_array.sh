#!/bin/bash
## SLURM array job: one replicate (seed) per task.
##   sbatch --array=1-300 submit_array.sh
## Optional arguments are passed to run_one.R, e.g.
##   sbatch --array=1-300 submit_array.sh design=reg methods=gated
#SBATCH --job-name=tpkfs
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tpkfs_%A_%a.out

mkdir -p logs results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
Rscript run_one.R seed=${SLURM_ARRAY_TASK_ID} out=results "$@"
