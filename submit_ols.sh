#!/bin/bash
# ---------------------------------------------------------------------------
# submit_ols.sh
# Usage: bash submit_ols.sh <run_name> <config> [options]
#
# Re-solves the loss-of-load days from a completed screen under each VoLL
# model. One job per config, so the two systems stay separate.
#
# Examples:
#   bash submit_ols.sh ols-100yr RTS-ORB
#   bash submit_ols.sh ols-100yr RTS-ORB-HR
#   bash submit_ols.sh ols-smoke RTS-ORB-HR --seeds 2 3 4
#   VOLL="static" bash submit_ols.sh ols-static RTS-ORB
#   SCREEN=results/RTS-ORB/simulation_results_20yrs.csv \
#       bash submit_ols.sh ols-20yr RTS-ORB
#
# VOLL defaults to "static dynamic".
# SCREEN defaults to results/<config>/simulation_results_100yrs.csv.
# Options after <config> are forwarded to run_ols_parallel.py.
#
# Resumable: resubmitting the same run_name skips any (config, voll, seed)
# that already has results.csv.
#
# Bring results back with:
#   rsync -avz --progress \
#     dtn.brc.berkeley.edu:/global/scratch/users/charlesgulian/optimal-rolling-blackout/results/ols/<run_name>/ \
#     results/ols/<run_name>/
# ---------------------------------------------------------------------------

set -e

if [ -z "$2" ]; then
    echo "Usage: bash submit_ols.sh <run_name> <config> [extra options]"
    echo "Example: bash submit_ols.sh ols-100yr RTS-ORB-HR"
    exit 1
fi

RUN=$1
CONFIG=$2
shift 2

SCRATCH=/global/scratch/users/charlesgulian/optimal-rolling-blackout
REPO=~/optimal-rolling-blackout
SCREEN=${SCREEN:-results/$CONFIG/simulation_results_100yrs.csv}
VOLL=${VOLL:-"static dynamic"}

# Fail now rather than after the job queues: without the screen there is
# nothing to re-solve.
if [ ! -f "$REPO/$SCREEN" ]; then
    echo "ERROR: missing $REPO/$SCREEN"
    echo "Run the screen first (run_simulation.py) or set SCREEN=<path>."
    exit 1
fi

mkdir -p $SCRATCH/logs
mkdir -p $SCRATCH/results/ols/$RUN

sbatch \
    --job-name=$RUN-$CONFIG \
    --output=$SCRATCH/logs/%j_$RUN-$CONFIG.out \
    --export=ALL,RUN_NAME=$RUN,CONFIG=$CONFIG,SCREEN=$SCREEN,VOLL="$VOLL",EXTRA_ARGS="$*" \
    run_ols.sh

echo "Submitted job '$RUN-$CONFIG'"
echo "Config:                   $CONFIG"
echo "Screen:                   $SCREEN"
echo "VoLL models:              $VOLL"
echo "Results will be saved to: $SCRATCH/results/ols/$RUN/$CONFIG"
echo "Log will be at:           $SCRATCH/logs/<jobid>_$RUN-$CONFIG.out"
echo ""
echo "Monitor with:  squeue -u charlesgulian"
