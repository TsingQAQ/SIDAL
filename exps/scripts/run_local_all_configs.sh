#!/bin/bash
# Run all benchmark experiments
#
# Usage:
#   bash run_all_experiments.sh                    # Full run with defaults (discrete only)
#   bash run_all_experiments.sh -dm d             # Run discrete experiments only
#   bash run_all_experiments.sh -dm c             # Run continuous experiments only
#   bash run_all_experiments.sh -dm all           # Run both discrete and continuous
#   bash run_all_experiments.sh -p 8 -n 2          # 8 parallel jobs, 2 CPUs each
#   bash run_all_experiments.sh --smoke            # Smoke test (1 iteration only)
#   bash run_all_experiments.sh --smoke -p 4 -n 1  # Smoke test with custom parallelism
#   bash run_all_experiments.sh -v                 # Verbose output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")/configs"

# Default settings
PARALLEL_JOBS=4
CPUS_PER_JOB=1
SMOKE_TEST=false
VERBOSE=false
DOMAIN="d"  # Default to discrete only

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel_jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -n|--cpu_per_job)
            CPUS_PER_JOB="$2"
            shift 2
            ;;
        -dm|--domain)
            DOMAIN="$2"
            if [[ "$DOMAIN" != "d" && "$DOMAIN" != "c" && "$DOMAIN" != "all" ]]; then
                echo "Error: -dm/--domain must be 'd' (discrete), 'c' (continuous), or 'all'"
                exit 1
            fi
            shift 2
            ;;
        --smoke)
            SMOKE_TEST=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-p N] [-n N] [-dm d|c|all] [--smoke] [-v]"
            echo "  -p, --parallel_jobs  Max parallel jobs (default: 4)"
            echo "  -n, --cpu_per_job    CPUs per job (default: 1)"
            echo "  -dm, --domain        Domain type: 'd' (discrete), 'c' (continuous), 'all' (default: d)"
            echo "  --smoke              Smoke test mode (1 iteration, 1 seed)"
            echo "  -v, --verbose        Show full output (default: quiet)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Running all benchmark experiments"
echo "Domain: $DOMAIN"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "CPUs per job: $CPUS_PER_JOB"
echo "Smoke test: $SMOKE_TEST"
echo "Verbose: $VERBOSE"
echo "=========================================="

# Build common arguments
COMMON_ARGS="-p $PARALLEL_JOBS -n $CPUS_PER_JOB"
if [ "$SMOKE_TEST" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --smoke"
fi
if [ "$VERBOSE" = true ]; then
    COMMON_ARGS="$COMMON_ARGS -v"
fi

# Helper function to run experiment
run_exp() {
    local config_path="$1"
    local exp_name="$2"
    local exp_domain="$3"  # "d" for discrete, "c" for continuous

    # Check if we should run this experiment based on domain filter
    if [[ "$DOMAIN" == "all" || "$DOMAIN" == "$exp_domain" ]]; then
        if [ -f "$config_path" ]; then
            echo ""
            echo "Running $exp_name ($exp_domain)..."
            bash "$SCRIPT_DIR/run_local_single_config.sh" -c "$config_path" -dm "$exp_domain" $COMMON_ARGS
        else
            echo ""
            echo "Skipping $exp_name - config not found: $config_path"
        fi
    fi
}

# ============================================================
# DISCRETE EXPERIMENTS
# ============================================================

if [[ "$DOMAIN" == "d" || "$DOMAIN" == "all" ]]; then
    echo ""
    echo "============================================"
    echo "Running DISCRETE experiments"
    echo "============================================"
fi

# 1D discrete
run_exp "$CONFIG_DIR/1d_Forrester/matern52/config.json" "1d_Forrester" "d"
run_exp "$CONFIG_DIR/1d_Gramacy/matern52/config.json" "1d_Gramacy" "d"

# 2D discrete
run_exp "$CONFIG_DIR/2d_Branin/matern52/config.json" "2d_Branin" "d"
run_exp "$CONFIG_DIR/2d_Gramacy/matern52/config.json" "2d_Gramacy" "d"

# 3D discrete
run_exp "$CONFIG_DIR/3d_GP_prior/matern52/config.json" "3d_GP_prior" "d"
run_exp "$CONFIG_DIR/3d_Hartmann/matern52/config.json" "3d_Hartmann" "d"
run_exp "$CONFIG_DIR/3d_Ishigami/matern52/config.json" "3d_Ishigami" "d"

# 6D discrete (expensive)
# run_exp "$CONFIG_DIR/6d_Hartmann/matern52/config.json" "6d_Hartmann" "d"

# ============================================================
# CONTINUOUS EXPERIMENTS (same configs as discrete, different domain flag)
# ============================================================

if [[ "$DOMAIN" == "c" || "$DOMAIN" == "all" ]]; then
    echo ""
    echo "============================================"
    echo "Running CONTINUOUS experiments"
    echo "============================================"
fi

# 1D continuous
run_exp "$CONFIG_DIR/1d_Forrester/matern52/config.json" "1d_Forrester" "c"
run_exp "$CONFIG_DIR/1d_Gramacy/matern52/config.json" "1d_Gramacy" "c"

# 2D continuous
run_exp "$CONFIG_DIR/2d_Branin/matern52/config.json" "2d_Branin" "c"
run_exp "$CONFIG_DIR/2d_Gramacy/matern52/config.json" "2d_Gramacy" "c"

# 3D continuous
run_exp "$CONFIG_DIR/3d_GP_prior/matern52/config.json" "3d_GP_prior" "c"
run_exp "$CONFIG_DIR/3d_Hartmann/matern52/config.json" "3d_Hartmann" "c"
run_exp "$CONFIG_DIR/3d_Ishigami/matern52/config.json" "3d_Ishigami" "c"

# 6D continuous (expensive)
# run_exp "$CONFIG_DIR/6d_Hartmann/matern52/config.json" "6d_Hartmann" "c"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
