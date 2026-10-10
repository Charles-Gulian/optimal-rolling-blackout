#!/bin/bash
# ---------------------------------------------------------------------------
# run_experiments.sh -- SLURM job script
# Do not run directly. Submit via submit_experiments.sh:
#   bash submit_experiments.sh <run_name> [options]
# ---------------------------------------------------------------------------
#SBATCH --partition=savio2
#SBATCH --account=fc_power
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlesgulian@berkeley.edu

# RUN_NAME, CATALOG and EXTRA_ARGS are injected by submit_experiments.sh
SCRATCH=/global/scratch/users/charlesgulian/optimal-rolling-blackout
RESULTS=$SCRATCH/results/experiments/$RUN_NAME

source ~/.bashrc
conda activate optimal-rolling-blackout
cd ~/optimal-rolling-blackout

# One Gurobi thread per worker; the driver sets Threads=1 per solve too. These
# stop BLAS oversubscribing on top of that.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Run name:    $RUN_NAME"
echo "Catalog:     $CATALOG"
echo "Results dir: $RESULTS"
echo "Workers:     $SLURM_CPUS_PER_TASK"
echo "Start time:  $(date)"
echo "=========================================="

# NOTE: deliberately no make_system_config / make_load_csv here, unlike
# run_simulation.sh. The configs in git are already what we want (rung 6,
# T_set 21, rho 5000) and regeneration would be byte-identical -- so skipping
# it removes a dependency on the weather inputs and any chance of two jobs
# racing on shared CSVs. More importantly, the experiments MUST use the same
# configs that produced the catalog, and committed files are the stronger
# guarantee of that than regenerating on each node.

python analysis/run_experiments.py \
    --catalog     $CATALOG            \
    --configs     RTS-ORB RTS-ORB-HR  \
    --voll-models dynamic exogenous   \
    --n-workers   $SLURM_CPUS_PER_TASK \
    --out-root    $RESULTS            \
    $EXTRA_ARGS
STATUS=$?

echo "=========================================="
echo "run_experiments exit status: $STATUS"
echo "End time: $(date)"
echo "=========================================="
exit $STATUS
