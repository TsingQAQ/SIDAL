#!/bin/bash
# =============================================================================
# SLURM wrapper for the SIDAL experiment scripts
# =============================================================================
#
# Submit from the repository root:
#   sbatch --array=0-359 exps/scripts/submit_slurm.sh -s run_benchmark_array.sh -c 2d_Branin/matern52 -dm c
#   sbatch --array=0-199 exps/scripts/submit_slurm.sh -s run_molecular_lambda_sweep.sh
#
# Options:
#   -s SCRIPT    Script in exps/scripts/ to run (required)
#   -t N         Threads per task (default: 4)
#   (all other arguments are passed through to SCRIPT)
#
# Each array task gets SLURM_ARRAY_TASK_ID, which the run scripts map to a
# (config, method, seed) combination; see the header of each script.
#
# Environment:
#   SIDAL_CONDA_ENV  Name or path of the conda env to activate (optional; if
#                    unset, the environment active at submission is used).
#   RESULTS_DIR      Where results are written (default: exps/results).
#
# Adjust the #SBATCH lines below (partition, time, memory) for your cluster.
# =============================================================================

#SBATCH -J sidal
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -o exps/logs/slurm_%A_%a.out
#SBATCH -e exps/logs/slurm_%A_%a.err

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p exps/logs

THREADS_PER_TASK=4
SCRIPT=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -t) THREADS_PER_TASK="$2"; shift 2 ;;
        -s) SCRIPT="$2"; shift 2 ;;
        *)  ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$SCRIPT" ]; then
    echo "Error: -s SCRIPT is required"
    echo "Usage: sbatch --array=0-N exps/scripts/submit_slurm.sh -s run_benchmark_array.sh [args]"
    exit 1
fi

export OMP_NUM_THREADS=$THREADS_PER_TASK
export MKL_NUM_THREADS=$THREADS_PER_TASK
export OPENBLAS_NUM_THREADS=$THREADS_PER_TASK

if [ -n "$SIDAL_CONDA_ENV" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$SIDAL_CONDA_ENV"
fi

echo "Job ID: $SLURM_JOB_ID  Array task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Script: $SCRIPT  Threads: $THREADS_PER_TASK  Args: ${ARGS[*]}"
echo "Python: $(which python)"
echo "Started: $(date)"

bash "exps/scripts/$SCRIPT" -n "$THREADS_PER_TASK" "${ARGS[@]}"
EXIT_CODE=$?

echo "Finished: $(date) (exit=$EXIT_CODE)"
exit $EXIT_CODE
