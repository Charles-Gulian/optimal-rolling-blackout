#!/bin/bash
# ---------------------------------------------------------------------------
# run_simulation.sh -- SLURM job script
# Do not run directly. Submit via submit_simulation.sh:
#   bash submit_simulation.sh <run_name> [options]
# ---------------------------------------------------------------------------
#SBATCH --partition=savio2
#SBATCH --account=fc_power
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlesgulian@berkeley.edu

# RUN_NAME and EXTRA_ARGS are injected by submit_simulation.sh via --export
SCRATCH=/global/scratch/users/charlesgulian/optimal-rolling-blackout
RESULTS=$SCRATCH/results/$RUN_NAME

source ~/.bashrc
conda activate optimal-rolling-blackout
cd ~/optimal-rolling-blackout

# One Gurobi thread per worker. The driver sets Threads=1 per solve too; these
# belt-and-braces exports stop BLAS from oversubscribing on top of that.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Run name:    $RUN_NAME"
echo "Results dir: $RESULTS"
echo "Workers:     $SLURM_CPUS_PER_TASK"
echo "Start time:  $(date)"
echo "=========================================="

# Regenerate the derived configs on this machine rather than trusting whatever
# is in git -- they are generated artefacts, and RTS-ORB-HR depends on the
# retirement rung.
python data-scripts/make_system_config.py 6
python data-scripts/make_load_csv.py

python analysis/run_parallel.py \
    --configs    RTS-ORB RTS-ORB-HR \
    --seeds      0 1 2 3 4          \
    --years      $(seq -s' ' 2001 2020) \
    --n-workers  $SLURM_CPUS_PER_TASK   \
    --out-root   $RESULTS               \
    $EXTRA_ARGS
STATUS=$?

# Collect even on partial failure: the per-task tables that did land are still
# usable, and the summary names what is missing.
python analysis/collect_results.py --out-root $RESULTS

echo "=========================================="
echo "run_parallel exit status: $STATUS"
echo "End time: $(date)"
echo "=========================================="
exit $STATUS
