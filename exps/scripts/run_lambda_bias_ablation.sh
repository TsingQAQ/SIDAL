#!/bin/bash
# Lambda-Bias Ablation Study: Single task runner (parallel mode)
# Studies how algorithms perform under different target distribution peakedness (λ)
# and with/without an independent GP bias term.
#
# Target distribution: P(x) ∝ exp(λ·y(x)) where:
#   - y(x) = f(x)         when bias=none (Row 1)
#   - y(x) = f(x) + b(x)  when bias=gp   (Row 2)
#   - f(x) and b(x) are independent GP samples
#
# Grid:
#   11 acq × 4 lambda × 2 bias × 10 seeds = 880 task IDs
#
# Lambda values (controls distribution peakedness):
#   idx 0: λ = -0.5  (T = 2.0)  - broad distribution
#   idx 1: λ = -1.0  (T = 1.0)  - standard
#   idx 2: λ = -2.0  (T = 0.5)  - peaked
#   idx 3: λ = -5.0  (T = 0.2)  - very peaked
#
# Bias modes:
#   idx 0: bias = none  (pure f)
#   idx 1: bias = gp    (f + b, where b is independent GP)
#
# Acquisition functions (11 total):
#   idx 0: RS, 1: US, 2: IMSE, 3: HAL, 4: AB-NoC, 5: AB-EWV,
#   6: TS-NoC, 7: TS-EWV, 8: mu-NoC, 9: mu-EWV, 10: mu-EPIG-NoC
#
# Task ID mapping: TASK_ID = seed + n_seeds * (acq + n_acqs * (bias + n_bias * lambda_idx))
#   seed changes fastest -> acq -> bias -> lambda changes slowest
#
# Usage:
#   # Smoke test single task (lambda=-0.5, bias=none, acq=RS, seed=0)
#   SLURM_ARRAY_TASK_ID=0 bash exps/scripts/run_lambda_bias_ablation.sh --smoke
#
#   # Run all 880 tasks (11 acq × 4 λ × 2 bias × 10 seeds)
#   sbatch --array=0-879 exps/scripts/submit_slurm.sh -s run_lambda_bias_ablation.sh
#
#   # Run only λ=-1.0 (idx=1): 11 acq × 2 bias × 10 seeds = 220 tasks, offset = 1*220 = 220
#   sbatch --array=220-439 exps/scripts/submit_slurm.sh -s run_lambda_bias_ablation.sh
#
#   # Run only mu-SID-EPIG_NoC (acq idx=10) across all settings (4 λ × 2 bias × 10 seeds = 80 tasks)
#   sbatch --array=100-109,210-219,320-329,430-439,540-549,650-659,760-769,870-879 \
#       exps/scripts/submit_slurm.sh -s run_lambda_bias_ablation.sh
#
#   # Smoke test all
#   sbatch --array=0-879 exps/scripts/submit_slurm.sh -s run_lambda_bias_ablation.sh --smoke

# Grid definition
LAMBDAS=(-0.5 -1.0 -2.0 -5.0)
TEMPS=(2.0 1.0 0.5 0.2)
BIAS_MODES=("none" "gp")
ACQ_NAMES=("RS" "US" "IMSE" "HAL" "AB-SID-iVAR_NoC" "AB-SID-iVAR_EWV" "TS-SID-iVAR_NoC" "TS-SID-iVAR_EWV" "mu-SID-iVAR_NoC" "mu-SID-iVAR_EWV" "mu-SID-EPIG_NoC")

N_LAMBDAS=${#LAMBDAS[@]}
N_BIAS=${#BIAS_MODES[@]}
N_ACQS=${#ACQ_NAMES[@]}
N_SEEDS=10

# Parse arguments
SMOKE_TEST=false
CPUS_PER_JOB=${SLURM_CPUS_PER_TASK:-4}
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_TEST=true; shift ;;
        -n|--cpu_per_job) CPUS_PER_JOB="$2"; shift 2 ;;
        -o|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Get task ID from SLURM or default to 0
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Parse task ID -> (seed, acq_idx, bias_idx, lambda_idx)
# Order: seed innermost -> acq -> bias -> lambda outermost
SEED=$((TASK_ID % N_SEEDS))
ACQ_IDX=$(((TASK_ID / N_SEEDS) % N_ACQS))
BIAS_IDX=$(((TASK_ID / (N_SEEDS * N_ACQS)) % N_BIAS))
LAMBDA_IDX=$((TASK_ID / (N_SEEDS * N_ACQS * N_BIAS)))

# Validate
MAX_TASK_ID=$((N_LAMBDAS * N_BIAS * N_ACQS * N_SEEDS - 1))
if [ "$LAMBDA_IDX" -ge "$N_LAMBDAS" ]; then
    echo "Error: TASK_ID=$TASK_ID out of range (max=$MAX_TASK_ID)"
    exit 1
fi

# Get values
LAMBDA=${LAMBDAS[$LAMBDA_IDX]}
TEMP=${TEMPS[$LAMBDA_IDX]}
BIAS_MODE=${BIAS_MODES[$BIAS_IDX]}
ACQ_NAME=${ACQ_NAMES[$ACQ_IDX]}

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$MODULE_DIR"

CONFIG_PATH="exps/configs/lambda_bias_ablation/gp_prior_2d_base.json"

# Verify config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

# Build output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${RESULTS_DIR:-exps/results}/lambda_bias_ablation"
fi
if [ "$SMOKE_TEST" = true ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_smoke"
fi

# Format lambda for folder name (replace - with m, . with p)
LAMBDA_FMT=$(echo "$LAMBDA" | sed 's/-/m/' | sed 's/\./p/')
FULL_OUTPUT_DIR="${OUTPUT_DIR}/lambda_${LAMBDA_FMT}_bias_${BIAS_MODE}"

# Determine use_bias value
if [ "$BIAS_MODE" = "gp" ]; then
    USE_BIAS="true"
else
    USE_BIAS="false"
fi

echo "=============================================================="
echo "Lambda-Bias Ablation: Task ${TASK_ID}"
echo "  Lambda: ${LAMBDA} (Temperature: ${TEMP})"
echo "  Bias: ${BIAS_MODE} (use_bias: ${USE_BIAS})"
echo "  Acquisition: ${ACQ_NAME} (idx: ${ACQ_IDX})"
echo "  Seed: ${SEED}"
echo "  Output: ${FULL_OUTPUT_DIR}"
echo "  CPUs: ${CPUS_PER_JOB}"
echo "  Smoke test: ${SMOKE_TEST}"
echo "  Node: ${SLURM_NODELIST:-localhost}"
echo "  Job ID: ${SLURM_JOB_ID:-N/A}, Array Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "=============================================================="

# Build overrides array.
# NOTE: --lambda_val below auto-syncs density_fn_type AND each SID/HAL acq's lambda
# (replaces the previous manual --override of density_fn_type and HAL temperature).
OVERRIDES=(
    --override "experiment_name=lambda_bias_ablation_l${LAMBDA_FMT}_b${BIAS_MODE}"
    --override "problem.kwargs.use_bias=${USE_BIAS}"
    --override "output.output_dir=${FULL_OUTPUT_DIR}"
)

if [ "$SMOKE_TEST" = true ]; then
    OVERRIDES+=(
        --override "optimization.n_iterations=2"
        --override "repetitions.n_trials=1"
    )
fi

# Run single acquisition function (--lambda_val auto-aligns eval and acq lambdas)
python -u -m sidal.benchmarking.runner \
    --config "$CONFIG_PATH" \
    --acq_index "$ACQ_IDX" \
    --domain c \
    --seed "$SEED" \
    --n_cpus "$CPUS_PER_JOB" \
    --lambda_val "$LAMBDA" \
    "${OVERRIDES[@]}"

EXIT_CODE=$?

echo ""
echo "=============================================================="
echo "Done: lambda=${LAMBDA} bias=${BIAS_MODE} acq=${ACQ_NAME} seed=${SEED} (exit=${EXIT_CODE})"
echo "=============================================================="

exit $EXIT_CODE
