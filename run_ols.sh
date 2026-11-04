#!/bin/bash
# ---------------------------------------------------------------------------
# run_ols.sh -- SLURM job script
# Do not run directly. Submit via submit_ols.sh:
#   bash submit_ols.sh <run_name> <config> [options]
# ---------------------------------------------------------------------------
#SBATCH --partition=savio2
#SBATCH --account=fc_power
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlesgulian@berkeley.edu

# RUN_NAME, CONFIG, SCREEN, VOLL, BETA, FAIR, LEX and EXTRA_ARGS are injected by submit_ols.sh
SCRATCH=/global/scratch/users/charlesgulian/optimal-rolling-blackout
RESULTS=$SCRATCH/results/ols/$RUN_NAME

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
echo "Config:      $CONFIG"
echo "Screen:      $SCREEN"
echo "VoLL models: $VOLL"
echo "Beta:        ${BETA:-<driver default>}"
echo "Fair:        ${FAIR:-<none>}"
echo "UE cap:      $([ "$LEX" = 0 ] && echo off || echo on)"
echo "Results dir: $RESULTS/$CONFIG"
echo "Workers:     $SLURM_CPUS_PER_TASK"
echo "Start time:  $(date)"
echo "=========================================="

# An empty value leaves the flag off entirely, so the driver default applies.
BETA_ARG=""; [ -n "$BETA" ] && BETA_ARG="--beta $BETA"
FAIR_ARG=""; [ -n "$FAIR" ] && FAIR_ARG="--fair $FAIR"
LEX_ARG="";  [ "$LEX" = 0 ] && LEX_ARG="--no-lex"

# No make_system_config / make_load_csv here, unlike run_simulation.sh:
# RTS-ORB-HR/gen.csv carries a hand retirement (123_STEAM_2) that regeneration
# would undo, and the re-solve must use the configs that produced the screen.

python scripts-gurobi/run_ols_parallel.py \
    --results    $SCREEN                \
    --configs    $CONFIG                \
    --voll       $VOLL                  \
    --n-workers  $SLURM_CPUS_PER_TASK   \
    --out-root   $RESULTS               \
    $BETA_ARG $FAIR_ARG $LEX_ARG        \
    $EXTRA_ARGS
STATUS=$?

echo "=========================================="
echo "run_ols_parallel exit status: $STATUS"
echo "End time: $(date)"
echo "=========================================="
exit $STATUS
