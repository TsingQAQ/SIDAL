#!/bin/bash
# =============================================================================
# Molecular Lambda Sweep: MED1 & MED2 with lambda = 25, 75 (paper Section 5.3)
# =============================================================================
#
# Runs pool-based active learning on molecular_med1 and molecular_med2
# with 5 acquisition functions (RS, US, IMSE, EPIG, AB-SID-iVAR_EWV)
# and 10 seeds, for each lambda value.
#
# Task grid: 2 lambdas x 2 configs x 5 acquisition functions x 10 seeds = 200 tasks
#
# Task ID mapping:
#   TASK_ID = seed + N_SEEDS * (acq + N_ACQS * (config + N_CONFIGS * lambda_idx))
#
# Usage (from the repository root):
#   # Smoke test
#   SLURM_ARRAY_TASK_ID=0 bash exps/scripts/run_molecular_lambda_sweep.sh --smoke
#
#   # All 200 tasks
#   sbatch --array=0-199 exps/scripts/submit_slurm.sh -s run_molecular_lambda_sweep.sh
#
#   # lambda=25 only (tasks 0-99) / lambda=75 only (tasks 100-199)
#   sbatch --array=0-99 exps/scripts/submit_slurm.sh -s run_molecular_lambda_sweep.sh
#   sbatch --array=100-199 exps/scripts/submit_slurm.sh -s run_molecular_lambda_sweep.sh
# =============================================================================

# Configs: only MED1 and MED2
CONFIGS=(
    "molecular_med1/tanimoto"
    "molecular_med2/tanimoto"
)

# Acquisition functions (indices 0-4 in molecular config)
ACQ_NAMES=(
    "RS"
    "US"
    "IMSE"
    "EPIG"
    "AB-SID-iVAR_EWV"
)

# Lambda values to sweep
LAMBDA_VALS=(25 75)

N_CONFIGS=${#CONFIGS[@]}
N_ACQS=${#ACQ_NAMES[@]}
N_LAMBDAS=${#LAMBDA_VALS[@]}
N_SEEDS=10

# Default settings
SMOKE_TEST=false
CPUS_PER_JOB=${SLURM_CPUS_PER_TASK:-4}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_TEST=true; shift ;;
        -n|--cpu_per_job) CPUS_PER_JOB="$2"; shift 2 ;;
        --n_seeds) N_SEEDS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Get task ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Parse task ID:
#   TASK_ID = seed + N_SEEDS * (acq + N_ACQS * (config + N_CONFIGS * lambda_idx))
SEED=$((TASK_ID % N_SEEDS))
REMAINDER=$((TASK_ID / N_SEEDS))
ACQ_IDX=$((REMAINDER % N_ACQS))
REMAINDER=$((REMAINDER / N_ACQS))
CONFIG_IDX=$((REMAINDER % N_CONFIGS))
LAMBDA_IDX=$((REMAINDER / N_CONFIGS))

MAX_TASK_ID=$(( N_LAMBDAS * N_CONFIGS * N_ACQS * N_SEEDS - 1 ))

# Validate
if [ "$LAMBDA_IDX" -ge "$N_LAMBDAS" ]; then
    echo "Error: TASK_ID=$TASK_ID out of range (max=$MAX_TASK_ID)"
    exit 1
fi

LAMBDA_VAL=${LAMBDA_VALS[$LAMBDA_IDX]}
CONFIG_NAME=${CONFIGS[$CONFIG_IDX]}
ACQ_NAME=${ACQ_NAMES[$ACQ_IDX]}

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$MODULE_DIR"

CONFIG_PATH="exps/configs/${CONFIG_NAME}/config.json"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

# Output directory: matches the layout read by fig_scripts/generate_lambda_sweep_composite.py
STORAGE_BASE="${RESULTS_DIR:-exps/results}/formal_benchmark"
PROBLEM_PART=$(dirname "$CONFIG_NAME")   # molecular_med1
KERNEL_PART=$(basename "$CONFIG_NAME")   # tanimoto
OUTPUT_DIR="${STORAGE_BASE}/pool/l${LAMBDA_VAL}/pool/${PROBLEM_PART}_/${KERNEL_PART}"

echo "=============================================================="
echo "Molecular Lambda Sweep: Task ${TASK_ID} / ${MAX_TASK_ID}"
echo "  Lambda: ${LAMBDA_VAL}"
echo "  Config: ${CONFIG_NAME} (idx ${CONFIG_IDX})"
echo "  Acquisition: ${ACQ_NAME} (idx ${ACQ_IDX})"
echo "  Seed: ${SEED}"
echo "  CPUs: ${CPUS_PER_JOB}"
echo "  Smoke test: ${SMOKE_TEST}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Node: ${SLURM_NODELIST:-localhost}"
echo "  Job ID: ${SLURM_JOB_ID:-N/A}, Array Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "=============================================================="

# Build override args
OVERRIDE_ARGS=""
if [ "$SMOKE_TEST" = true ]; then
    OVERRIDE_ARGS="--override optimization.n_iterations=2"
fi

# Run
python -u -m sidal.benchmarking.runner \
    --config "$CONFIG_PATH" \
    --acq_index "$ACQ_IDX" \
    --domain p \
    --seed "$SEED" \
    --n_cpus "$CPUS_PER_JOB" \
    --lambda_val "$LAMBDA_VAL" \
    --output_dir "$OUTPUT_DIR" \
    $OVERRIDE_ARGS

EXIT_CODE=$?

echo ""
echo "=============================================================="
echo "Done: l${LAMBDA_VAL} / ${CONFIG_NAME} / ${ACQ_NAME} / seed=${SEED} (exit=$EXIT_CODE)"
echo "=============================================================="

exit $EXIT_CODE
