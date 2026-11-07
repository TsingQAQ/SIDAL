#!/bin/bash
# =============================================================================
# Benchmark job-array runner: one (method, seed) of one config per task
# =============================================================================
#
# Task ID mapping (seed changes fastest):
#   TASK_ID = seed + N_SEEDS * method_index
# where method_index is the position in the config's `acquisition_functions`
# list, so the task count is  N_SEEDS x (number of methods in that config).
#
# Usage (from the repository root):
#   # Smoke test one task
#   SLURM_ARRAY_TASK_ID=0 bash exps/scripts/run_benchmark_array.sh -c 2d_Branin/matern52 -dm c --smoke
#
#   # All 12 methods x 30 seeds of a synthetic config, continuous and discrete
#   sbatch --array=0-359 exps/scripts/submit_slurm.sh -s run_benchmark_array.sh -c 2d_Branin/matern52 -dm c
#   sbatch --array=0-359 exps/scripts/submit_slurm.sh -s run_benchmark_array.sh -c 2d_Branin/matern52 -dm d
#
#   # A PES config (8 methods x 30 seeds)
#   sbatch --array=0-239 exps/scripts/submit_slurm.sh -s run_benchmark_array.sh -c 3d_SiSW/matern52 -dm c
#
#   # Print the task count for a config
#   bash exps/scripts/run_benchmark_array.sh -c 3d_SiSW/matern52 --count
#
# Options:
#   -c CONFIG          Config under exps/configs/, e.g. 2d_Branin/matern52 (required)
#   -dm d|c|p          Discrete grid, continuous, or pool (default: c)
#   --n_seeds N        Seeds per method (default: 30)
#   --acq_index I      Run only method I; the task grid is then just the seeds
#   --lambda_val L     Override the Boltzmann lambda
#   --n_iterations N   Override the number of iterations
#   -o DIR             Results root (default: ${RESULTS_DIR:-exps/results}/formal_benchmark)
#   -n N               CPUs per job (set by submit_slurm.sh)
#   --smoke            2 iterations only
#   --count            Print the number of tasks and exit
#
# Results go to  DIR/<continuous|discrete|pool>/<Problem_Nd>/<kernel>/,
# e.g. exps/results/formal_benchmark/continuous/Branin_2d/matern52/, which is
# the layout the scripts in fig_scripts/ read.
# =============================================================================

CONFIG_NAME=""
DOMAIN="c"
N_SEEDS=30
SINGLE_ACQ_IDX=""
LAMBDA_VAL=""
N_ITERATIONS=""
OUTPUT_ROOT="${RESULTS_DIR:-exps/results}/formal_benchmark"
CPUS_PER_JOB=${SLURM_CPUS_PER_TASK:-4}
SMOKE_TEST=false
COUNT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_NAME="$2"; shift 2 ;;
        -dm|--domain) DOMAIN="$2"; shift 2 ;;
        --n_seeds) N_SEEDS="$2"; shift 2 ;;
        --acq_index) SINGLE_ACQ_IDX="$2"; shift 2 ;;
        --lambda_val) LAMBDA_VAL="$2"; shift 2 ;;
        --n_iterations) N_ITERATIONS="$2"; shift 2 ;;
        -o|--output_dir) OUTPUT_ROOT="$2"; shift 2 ;;
        -n|--cpu_per_job) CPUS_PER_JOB="$2"; shift 2 ;;
        --smoke) SMOKE_TEST=true; shift ;;
        --count) COUNT_ONLY=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG_NAME" ]; then
    echo "Error: -c CONFIG is required (e.g. -c 2d_Branin/matern52)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG_PATH="exps/configs/${CONFIG_NAME}/config.json"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

N_ACQS=$(python -c "import json; print(len(json.load(open('$CONFIG_PATH'))['acquisition_functions']))")
if [ -n "$SINGLE_ACQ_IDX" ]; then
    N_TASKS=$N_SEEDS
else
    N_TASKS=$(( N_ACQS * N_SEEDS ))
fi
if [ "$COUNT_ONLY" = true ]; then
    echo "$N_TASKS tasks (use --array=0-$(( N_TASKS - 1 )))"
    exit 0
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TASK_ID" -ge "$N_TASKS" ]; then
    echo "Error: TASK_ID=$TASK_ID out of range (max=$(( N_TASKS - 1 )))"
    exit 1
fi
SEED=$(( TASK_ID % N_SEEDS ))
if [ -n "$SINGLE_ACQ_IDX" ]; then
    ACQ_IDX=$SINGLE_ACQ_IDX
else
    ACQ_IDX=$(( TASK_ID / N_SEEDS ))
fi
ACQ_NAME=$(python -c "import json; print(json.load(open('$CONFIG_PATH'))['acquisition_functions'][$ACQ_IDX]['name'])")

# "2d_Branin/matern52" -> "Branin_2d" and "matern52"
PROBLEM_PART=$(dirname "$CONFIG_NAME")
KERNEL_PART=$(basename "$CONFIG_NAME")
DIM=$(echo "$PROBLEM_PART" | grep -oE '^[0-9]+d')
if [ -n "$DIM" ]; then
    PROBLEM_NAME="$(echo "$PROBLEM_PART" | sed "s/^${DIM}_//")_${DIM}"
else
    PROBLEM_NAME="$PROBLEM_PART"
fi
case "$DOMAIN" in
    d) DOMAIN_FOLDER="discrete" ;;
    p) DOMAIN_FOLDER="pool" ;;
    *) DOMAIN_FOLDER="continuous" ;;
esac
OUTPUT_DIR="${OUTPUT_ROOT}/${DOMAIN_FOLDER}/${PROBLEM_NAME}/${KERNEL_PART}"

EXTRA_ARGS=()
[ "$SMOKE_TEST" = true ] && EXTRA_ARGS+=(--override optimization.n_iterations=2)
[ -n "$LAMBDA_VAL" ] && EXTRA_ARGS+=(--lambda_val "$LAMBDA_VAL")
[ -n "$N_ITERATIONS" ] && EXTRA_ARGS+=(--n_iterations "$N_ITERATIONS")

echo "=============================================================="
echo "Task ${TASK_ID}/$(( N_TASKS - 1 )): ${CONFIG_NAME} | ${ACQ_NAME} (idx ${ACQ_IDX}) | seed ${SEED} | domain ${DOMAIN}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Node: ${SLURM_NODELIST:-localhost}  Job: ${SLURM_JOB_ID:-N/A}"
echo "=============================================================="

python -u -m sidal.benchmarking.runner \
    --config "$CONFIG_PATH" \
    --acq_index "$ACQ_IDX" \
    --domain "$DOMAIN" \
    --seed "$SEED" \
    --n_cpus "$CPUS_PER_JOB" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
EXIT_CODE=$?

echo "Done: ${CONFIG_NAME} / ${ACQ_NAME} / seed=${SEED} (exit=${EXIT_CODE})"
exit $EXIT_CODE
