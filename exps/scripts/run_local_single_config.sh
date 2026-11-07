#!/bin/bash
# Run experiments in parallel from a JSON config file
#
# Usage:
#   bash run_parallel.sh -c config.json -p 12 -n 2 --python /path/to/python
#   bash run_parallel.sh -c config.json --smoke  # Smoke test (1 iteration, 1 seed)
#   bash run_parallel.sh -c config.json -v       # Verbose output
#   bash run_parallel.sh -c config.json -dm c    # Continuous domain
#
# Options:
#   -c, --config        Config file path (required)
#   -p, --parallel_jobs Max parallel jobs (default: 0 = unlimited)
#   -n, --cpu_per_job   CPUs per job (default: 1)
#   -dm, --domain       Domain type: 'd' (discrete) or 'c' (continuous), default: d
#   --python            Python path (default: python)
#   --smoke             Smoke test mode (1 iteration, 1 seed only)
#   -v, --verbose       Show full output (default: quiet, only show OK/FAILED)

CONFIG_PATH=""
MAX_JOBS=0
CPUS_PER_JOB=1
PYTHON_CMD="${PYTHON_CMD:-python}"  # Use env var or default to 'python' (assumes conda env is activated)
SMOKE_TEST=false
VERBOSE=false
DOMAIN="d"

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -p|--parallel_jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -n|--cpu_per_job)
            CPUS_PER_JOB="$2"
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        -dm|--domain)
            DOMAIN="$2"
            if [[ "$DOMAIN" != "d" && "$DOMAIN" != "c" ]]; then
                echo "Error: -dm/--domain must be 'd' (discrete) or 'c' (continuous)"
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
            echo "Usage: $0 -c config.json [-p N] [-n N] [-dm d|c] [--python PATH] [--smoke] [-v]"
            echo "  -c, --config        Config file path (required)"
            echo "  -p, --parallel_jobs Max parallel jobs (default: 0 = unlimited)"
            echo "  -n, --cpu_per_job   CPUs per job (default: 1)"
            echo "  -dm, --domain       Domain: 'd' (discrete) or 'c' (continuous), default: d"
            echo "  --python            Python path"
            echo "  --smoke             Smoke test mode (1 iteration, 1 seed)"
            echo "  -v, --verbose       Show full output (default: quiet)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: -c/--config is required"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

# Change to module directory (where this script lives, go up 2 levels from scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$MODULE_DIR"

echo "=============================================================="
echo "Working dir: $(pwd)"
echo "Config: $CONFIG_PATH"
echo "Domain: $DOMAIN (d=discrete, c=continuous)"
echo "Max jobs: $MAX_JOBS (0=unlimited)"
echo "CPUs/job: $CPUS_PER_JOB"
echo "Python: $PYTHON_CMD"
echo "Smoke test: $SMOKE_TEST"
echo "=============================================================="

# Get number of acquisition functions (use indices to support duplicates with different hyperparams)
N_ACQS=$($PYTHON_CMD -c "
import json
with open('$CONFIG_PATH') as f:
    config = json.load(f)
print(len(config['acquisition_functions']))
")

# Get acquisition names for display
ACQ_NAMES=$($PYTHON_CMD -c "
import json
with open('$CONFIG_PATH') as f:
    config = json.load(f)
for i, acq in enumerate(config['acquisition_functions']):
    hp = acq.get('hyperparameters', {})
    tag = ('_EWV' if hp['constraint_type'] else '_NoC') if 'constraint_type' in hp else ''
    print(f\"{i}:{acq['name']}{tag}\")
")

# Get seeds - always use all seeds from config (smoke test only changes n_iterations)
SEEDS=$($PYTHON_CMD -c "
import json
with open('$CONFIG_PATH') as f:
    config = json.load(f)
n = config['repetitions'].get('n_trials', 5)
print(' '.join(map(str, range(n))))
")

ACQ_DISPLAY_ARRAY=($ACQ_NAMES)
SEED_ARRAY=($SEEDS)

# Get output directory from config and insert domain folder
OUTPUT_DIR=$($PYTHON_CMD -c "
import json
from pathlib import Path
with open('$CONFIG_PATH') as f:
    config = json.load(f)
base_dir = config['output']['output_dir']
# Insert domain folder before last 2 components (problem/kernel)
# e.g., ./results/formal_benchmark/Branin_2d/matern52
#    -> ./results/formal_benchmark/discrete/Branin_2d/matern52
output_path = Path(base_dir)
parts = list(output_path.parts)
domain_folder = 'discrete' if '$DOMAIN' == 'd' else 'continuous'
if len(parts) >= 2:
    parts.insert(-2, domain_folder)
    print(str(Path(*parts)))
else:
    print(str(output_path / domain_folder))
")

# Append smoke_test suffix if in smoke mode
if [ "$SMOKE_TEST" = true ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_smoke"
fi

echo "Acquisitions ($N_ACQS total):"
for item in "${ACQ_DISPLAY_ARRAY[@]}"; do
    echo "  $item"
done
echo "Seeds: ${SEED_ARRAY[@]}"
echo "Total jobs: $((N_ACQS * ${#SEED_ARRAY[@]}))"
echo "Output: $OUTPUT_DIR"
if [ "$SMOKE_TEST" = true ]; then
    echo "NOTE: Smoke test mode - running only 1 iteration per experiment"
fi
echo ""

# Create log directory
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
echo "Logs (failed only): $LOG_DIR"
echo ""

# Function to run a single experiment and only keep log if it fails
run_exp() {
    local acq_index=$1
    local acq_display=$2
    local seed=$3
    local log_file="${LOG_DIR}/${acq_display}_seed${seed}.log"

    # Build extra args for smoke test
    local extra_args=""
    if [ "$SMOKE_TEST" = true ]; then
        extra_args="--n_iterations 1"
    fi

    if [ "$VERBOSE" = true ]; then
        # Verbose mode: show output with tee
        $PYTHON_CMD -u -m sidal.benchmarking.runner \
            --config "$CONFIG_PATH" \
            --acq_index "$acq_index" \
            --seed "$seed" \
            --domain "$DOMAIN" \
            --n_cpus "$CPUS_PER_JOB" \
            --output_dir "$OUTPUT_DIR" \
            $extra_args \
            2>&1 | tee "$log_file"
        local exit_code=${PIPESTATUS[0]}
    else
        # Quiet mode: output to log file only
        $PYTHON_CMD -u -m sidal.benchmarking.runner \
            --config "$CONFIG_PATH" \
            --acq_index "$acq_index" \
            --seed "$seed" \
            --domain "$DOMAIN" \
            --n_cpus "$CPUS_PER_JOB" \
            --output_dir "$OUTPUT_DIR" \
            $extra_args \
            > "$log_file" 2>&1
        local exit_code=$?
    fi

    # Delete log if successful, keep and report if failed
    if [ $exit_code -eq 0 ]; then
        rm -f "$log_file"
        if [ "$VERBOSE" = false ]; then
            echo "  OK: $acq_display seed=$seed"
        fi
    else
        # Log was empty (killed before any output), add failure info
        if [ ! -s "$log_file" ]; then
            echo "Process killed (exit code: $exit_code) - likely OOM" > "$log_file"
        fi
        echo "  FAILED: $acq_display seed=$seed (see $log_file)"
    fi
}

export -f run_exp
export PYTHON_CMD CONFIG_PATH CPUS_PER_JOB LOG_DIR OUTPUT_DIR SMOKE_TEST VERBOSE DOMAIN

RUNNING=0
for acq_idx in $(seq 0 $((N_ACQS - 1))); do
    acq_display="${ACQ_DISPLAY_ARRAY[$acq_idx]}"
    for seed in "${SEED_ARRAY[@]}"; do
        echo "Launching: $acq_display seed=$seed"
        run_exp "$acq_idx" "$acq_display" "$seed" &

        if [ "$MAX_JOBS" -gt 0 ]; then
            RUNNING=$((RUNNING + 1))
            if [ "$RUNNING" -ge "$MAX_JOBS" ]; then
                wait -n 2>/dev/null || wait
                RUNNING=$((RUNNING - 1))
            fi
        fi
    done
done

echo ""
echo "Waiting for jobs..."
wait

# Check for failures by looking at log files (logs are deleted on success)
echo ""
echo "=============================================================="
echo "Checking results..."
echo "=============================================================="
FAILED=0
for acq_idx in $(seq 0 $((N_ACQS - 1))); do
    acq_display="${ACQ_DISPLAY_ARRAY[$acq_idx]}"
    for seed in "${SEED_ARRAY[@]}"; do
        LOG_FILE="${LOG_DIR}/${acq_display}_seed${seed}.log"
        if [ -f "$LOG_FILE" ]; then
            echo "FAILED: $acq_display seed=$seed"
            echo "  Log: $LOG_FILE"
            echo "  Last 10 lines:"
            tail -10 "$LOG_FILE" | sed 's/^/    /'
            FAILED=$((FAILED + 1))
        fi
    done
done

# Remove log dir if empty
rmdir "$LOG_DIR" 2>/dev/null

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All experiments completed!"
else
    echo "WARNING: $FAILED experiment(s) failed. Check logs in $LOG_DIR"
fi
echo "=============================================================="
