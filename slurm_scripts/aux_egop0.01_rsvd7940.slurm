#!/bin/bash
#SBATCH --job-name=aux_egop_rsvd7940
#SBATCH --partition=gpu
#SBATCH --nodelist=g6
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/aux_egop0.01_rsvd7940_%j.out
#SBATCH --error=slurm_logs/aux_egop0.01_rsvd7940_%j.err

eval "$(/share/data/willett-group/rickliang/mc3/bin/conda 'shell.bash' 'hook')"
conda activate egop_optimizer

cd ~/Github/egop_optimizer/egop_optimizer

stdbuf -o0 -e0 python experiments/fashionMNIST_auxiliary_compare_v2.py \
    --rsvd_components 7940 \
    --egop_factor 0.01 \
    --num_iterations 50 \
    --epochs 100 \
    --no_show \
    --quiet
