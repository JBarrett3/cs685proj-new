#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=50GB
#SBATCH -p gpu,gpu-preempt
#SBATCH -G 1 # specifies for 24GB
#SBATCH -t 1:00:00
#SBATCH -o /home/jamesbarrett_umass_edu/cs685proj-new/conventional/slurm_logs/logs-%j.out
#SBATCH --constraint=vram40

module load conda/latest
CONDA_ENV=unsloth
conda activate $CONDA_ENV

# python /home/jamesbarrett_umass_edu/cs685proj-new/conventional/finetune.py
# python /home/jamesbarrett_umass_edu/cs685proj-new/conventional/finetune.py --training --lr 1e-1
# python /home/jamesbarrett_umass_edu/cs685proj-new/conventional/finetune.py --training --lr 1e-3
python /home/jamesbarrett_umass_edu/cs685proj-new/conventional/finetune.py --training