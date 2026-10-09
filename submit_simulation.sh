#!/bin/bash
# ---------------------------------------------------------------------------
# submit_simulation.sh
# Usage: bash submit_simulation.sh <run_name> [options]
#
# Examples:
#   bash submit_simulation.sh orb-100yr
#   bash submit_simulation.sh orb-smoke   --configs RTS-ORB --seeds 0 --years 2003
#   bash submit_simulation.sh orb-100yr   --force
#
# All options after <run_name> are forwarded to analysis/run_parallel.py and
# override the defaults set in run_simulation.sh.
#
# Results land in $SCRATCH/results/<run_name>/<config>/seed<NN>/year<YYYY>/.
# Tasks are resumable: resubmitting the same run_name skips completed tasks,
# so a job that hits the walltime can simply be resubmitted.
#
# Bring results back from your laptop with:
#   rsync -avz --progress \
#     dtn.brc.berkeley.edu:/global/scratch/users/charlesgulian/optimal-rolling-blackout/results/<run_name>/ \
#     results/<run_name>/
# ---------------------------------------------------------------------------

set -e  # exit immediately if any command fails

if [ -z "$1" ]; then
    echo "Usage: bash submit_simulation.sh <run_name> [extra options]"
    echo "Example: bash submit_simulation.sh orb-100yr"
    exit 1
fi

RUN=$1
shift  # remaining args (if any) get forwarded to the job script

SCRATCH=/global/scratch/users/charlesgulian/optimal-rolling-blackout

mkdir -p $SCRATCH/logs
mkdir -p $SCRATCH/results/$RUN

sbatch \
    --job-name=$RUN \
    --output=$SCRATCH/logs/%j_$RUN.out \
    --export=ALL,RUN_NAME=$RUN,EXTRA_ARGS="$*" \
    run_simulation.sh

echo "Submitted job '$RUN'"
echo "Results will be saved to: $SCRATCH/results/$RUN"
echo "Log will be at:           $SCRATCH/logs/<jobid>_$RUN.out"
echo ""
echo "Monitor with:  squeue -u charlesgulian"
