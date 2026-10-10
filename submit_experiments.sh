#!/bin/bash
# ---------------------------------------------------------------------------
# submit_experiments.sh
# Usage: bash submit_experiments.sh <run_name> [options]
#
# Re-solves the catalogued blackout days under each VOLL model. The catalog
# comes from a completed screen run (submit_simulation.sh), and defaults to
# orb-100yr; override with CATALOG_RUN.
#
# Examples:
#   bash submit_experiments.sh voll-ladder
#   bash submit_experiments.sh voll-smoke  --configs RTS-ORB --seeds 3 --years 2003
#   bash submit_experiments.sh voll-ladder --force
#   CATALOG_RUN=orb-100yr bash submit_experiments.sh voll-ladder
#
# All options after <run_name> are forwarded to analysis/run_experiments.py.
#
# Tasks are resumable: resubmitting the same run_name skips completed
# (config, model, seed, year) groups, so a job that hits the walltime can
# simply be resubmitted.
#
# Bring results back with:
#   rsync -avz --progress \
#     dtn.brc.berkeley.edu:/global/scratch/users/charlesgulian/optimal-rolling-blackout/results/experiments/<run_name>/ \
#     results/experiments/<run_name>/
# ---------------------------------------------------------------------------

set -e

if [ -z "$1" ]; then
    echo "Usage: bash submit_experiments.sh <run_name> [extra options]"
    echo "Example: bash submit_experiments.sh voll-ladder"
    exit 1
fi

RUN=$1
shift

SCRATCH=/global/scratch/users/charlesgulian/optimal-rolling-blackout
CATALOG_RUN=${CATALOG_RUN:-orb-100yr}
CATALOG=$SCRATCH/results/$CATALOG_RUN

# Fail now rather than after the job queues: without the catalog there is
# nothing to re-solve.
for cfg in RTS-ORB RTS-ORB-HR; do
    if [ ! -f "$CATALOG/$cfg/unserved_energy_days.csv" ]; then
        echo "ERROR: missing $CATALOG/$cfg/unserved_energy_days.csv"
        echo "Run the screen first (submit_simulation.sh) and let collect_results write it."
        exit 1
    fi
done

mkdir -p $SCRATCH/logs
mkdir -p $SCRATCH/results/experiments/$RUN

sbatch \
    --job-name=$RUN \
    --output=$SCRATCH/logs/%j_$RUN.out \
    --export=ALL,RUN_NAME=$RUN,CATALOG=$CATALOG,EXTRA_ARGS="$*" \
    run_experiments.sh

echo "Submitted job '$RUN'"
echo "Catalog:                  $CATALOG"
echo "Results will be saved to: $SCRATCH/results/experiments/$RUN"
echo "Log will be at:           $SCRATCH/logs/<jobid>_$RUN.out"
echo ""
echo "Monitor with:  squeue -u charlesgulian"
