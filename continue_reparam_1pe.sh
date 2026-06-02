#!/bin/bash
#SBATCH --job-name=reparam_1pe
#SBATCH --partition=willett-gpu
#SBATCH --gpus=nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8

SCRIPT_DIR=/share/data/willett-group/adepavia/resnet_results # change this to the path where you are saving your .files
sbatch --dependency=afterany:$SLURM_JOB_ID $SCRIPT_DIR/continue_reparam_1pe.sh
eval "$(/share/data/willett-group/adepavia/mc3/bin/conda 'shell.bash' 'hook')" # here change this to activate your conda environment as you usually do it
conda activate egop_optimizer # change this to your conda environment name
cd /home-nfs/adepavia/GitHub/egop_optimizer # change this to your repo path

python experiments/ImageNet_resnet34_reparam_1pe.py