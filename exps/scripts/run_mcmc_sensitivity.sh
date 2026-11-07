#!/bin/bash
# MCMC Sensitivity V2: d=2,4,6,8, particle_mults=250,500,1000,1500, RWM only
#
# Task grid:
#   4 dims × 4 pmults × 2 acqs (AB,TS) × 10 seeds + 4 dims × 10 seeds (RS)
#   = 320 + 40 = 360 tasks
#
# Task ID mapping:
#   0-319:  AB/TS runs (seed + 10 * (acq + 2 * (pmult + 4 * dim)))
#   320-359: RS runs (seed + 10 * dim)
#
# Usage:
#   # All 360 tasks
#   sbatch --array=0-359 exps/scripts/submit_slurm.sh -s run_mcmc_sensitivity.sh
#
#   # Smoke test (1 per config)
#   sbatch --array=0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350 \
#       exps/scripts/submit_slurm.sh -s run_mcmc_sensitivity.sh --smoke

DIMS=(2 4 6 8)
PARTICLE_MULTS=(250 500 1000 1500)
N_DIMS=${#DIMS[@]}
N_PMULTS=${#PARTICLE_MULTS[@]}
N_SEEDS=10
N_AB_TS_TASKS=$((N_DIMS * N_PMULTS * 2 * N_SEEDS))  # 320

SMOKE_TEST=false
CPUS_PER_JOB=${SLURM_CPUS_PER_TASK:-4}

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_TEST=true; shift ;;
        -n|--cpu_per_job) CPUS_PER_JOB="$2"; shift 2 ;;
        *) shift ;;
    esac
done

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$MODULE_DIR"

CONFIG_PATH="exps/configs/mcmc_sensitivity/gp_prior_base.json"
OUTPUT_BASE="${RESULTS_DIR:-exps/results}/mcmc_sensitivity"

if [ "$TASK_ID" -lt "$N_AB_TS_TASKS" ]; then
    # AB or TS run
    SEED=$((TASK_ID % N_SEEDS))
    ACQ_IDX=$(((TASK_ID / N_SEEDS) % 2))  # 0=AB, 1=TS
    PMULT_IDX=$(((TASK_ID / (N_SEEDS * 2)) % N_PMULTS))
    DIM_IDX=$((TASK_ID / (N_SEEDS * 2 * N_PMULTS)))

    DIM=${DIMS[$DIM_IDX]}
    PMULT=${PARTICLE_MULTS[$PMULT_IDX]}
    N_PARTICLES=$((PMULT * DIM))

    if [ "$ACQ_IDX" -eq 0 ]; then
        ACQ_NAME="AB"
        ACQ_CONFIG_INDEX=0
    else
        ACQ_NAME="TS"
        ACQ_CONFIG_INDEX=2
    fi
    OUTPUT_DIR="${OUTPUT_BASE}/d${DIM}_p${N_PARTICLES}_rwm_${ACQ_NAME}"
else
    # RS run
    RS_ID=$((TASK_ID - N_AB_TS_TASKS))
    SEED=$((RS_ID % N_SEEDS))
    DIM_IDX=$((RS_ID / N_SEEDS))
    DIM=${DIMS[$DIM_IDX]}
    N_PARTICLES=0
    PMULT=0
    ACQ_NAME="RS"
    ACQ_CONFIG_INDEX=1
    OUTPUT_DIR="${OUTPUT_BASE}/d${DIM}_RS"
fi

# Compute scaled lengthscale: l(d) = 0.3 * sqrt(d/3)
SCALED_LS=$(awk "BEGIN {printf \"%.4f\", 0.3 * sqrt($DIM/3)}")
LENGTHSCALE="["
for ((i=0; i<DIM; i++)); do
    if [ $i -gt 0 ]; then LENGTHSCALE+=","; fi
    LENGTHSCALE+="${SCALED_LS}"
done
LENGTHSCALE+="]"

# Iterations: 100 * d / 3 (rounded up)
N_ITERATIONS=$(( (100 * DIM + 2) / 3 ))

echo "=============================================================="
echo "MCMC Sensitivity V2: Task ${TASK_ID}"
echo "  acq=${ACQ_NAME}, dim=${DIM}, particles=${N_PARTICLES} (${PMULT}*d), seed=${SEED}"
echo "  lengthscale=${SCALED_LS}, iterations=${N_ITERATIONS}"
echo "  Output: ${OUTPUT_DIR}"
echo "=============================================================="

# Build overrides
if [ "$ACQ_CONFIG_INDEX" -eq 1 ]; then
    OVERRIDES=(
        --override "experiment_name=mcmc_v2_d${DIM}_RS"
        --override "problem.kwargs.dim=${DIM}"
        --override "problem.kwargs.lengthscale=${LENGTHSCALE}"
        --override "model.hyperparameters.lengthscale=${LENGTHSCALE}"
        --override "optimization.n_iterations=${N_ITERATIONS}"
        --override "output.output_dir=${OUTPUT_DIR}"
    )
elif [ "$ACQ_CONFIG_INDEX" -eq 2 ]; then
    OVERRIDES=(
        --override "experiment_name=mcmc_v2_d${DIM}_p${N_PARTICLES}_rwm_TS"
        --override "problem.kwargs.dim=${DIM}"
        --override "problem.kwargs.lengthscale=${LENGTHSCALE}"
        --override "model.hyperparameters.lengthscale=${LENGTHSCALE}"
        --override "optimization.n_iterations=${N_ITERATIONS}"
        --override "acquisition_functions.2.hyperparameters.mutation_kernel=rwm"
        --override "acquisition_functions.2.hyperparameters.n_smc_particles=${N_PARTICLES}"
        --override "output.output_dir=${OUTPUT_DIR}"
    )
else
    OVERRIDES=(
        --override "experiment_name=mcmc_v2_d${DIM}_p${N_PARTICLES}_rwm_AB"
        --override "problem.kwargs.dim=${DIM}"
        --override "problem.kwargs.lengthscale=${LENGTHSCALE}"
        --override "model.hyperparameters.lengthscale=${LENGTHSCALE}"
        --override "optimization.n_iterations=${N_ITERATIONS}"
        --override "acquisition_functions.0.hyperparameters.mutation_kernel=rwm"
        --override "acquisition_functions.0.hyperparameters.n_smc_particles=${N_PARTICLES}"
        --override "output.output_dir=${OUTPUT_DIR}"
    )
fi

if [ "$SMOKE_TEST" = true ]; then
    OVERRIDES+=(--override "optimization.n_iterations=2")
fi

python -u -m sidal.benchmarking.runner \
    --config "$CONFIG_PATH" \
    --seed "$SEED" \
    --acq_index "$ACQ_CONFIG_INDEX" \
    --domain c \
    --n_cpus "$CPUS_PER_JOB" \
    "${OVERRIDES[@]}"

EXIT_CODE=$?
echo "Done: ${ACQ_NAME} d=${DIM} p=${N_PARTICLES} seed=${SEED} (exit=$EXIT_CODE)"
exit $EXIT_CODE
