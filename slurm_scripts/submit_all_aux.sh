#!/bin/bash
# Submit all aux EGOP rsvd sweeps. Run from repo root:
#   bash slurm_scripts/submit_all_aux.sh
# Or submit a subset:
#   bash slurm_scripts/submit_all_aux.sh 50 100 200

if [ $# -eq 0 ]; then
    RSVD_LIST=(50 100 200 400 800 1000 2000 4000 7940)
else
    RSVD_LIST=("$@")
fi

mkdir -p slurm_logs

for rsvd in "${RSVD_LIST[@]}"; do
    script="slurm_scripts/aux_egop0.01_rsvd${rsvd}.slurm"
    if [ ! -f "$script" ]; then
        echo "Skip: $script not found"
        continue
    fi
    echo "Submitting $script"
    sbatch "$script"
done
