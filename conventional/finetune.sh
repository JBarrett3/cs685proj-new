#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=20GB
#SBATCH -p gpu
#SBATCH --gres=gpu:m40:1 # specifies for 24GB
#SBATCH -t 12:00:00
#SBATCH -o /home/jamesbarrett_umass_edu/cs685proj-new/conventional/slurm_logs/logs-%j.out

module load conda/latest
CONDA_ENV=unsloth
conda activate $CONDA_ENV

# python /home/jamesbarrett_umass_edu/cs685proj-new/conventional/finetune.py
# python /home/jamesbarrett_umass_edu/cs685proj-new/conventional/finetune.py --training --lr 1e-1
# python /home/jamesbarrett_umass_edu/cs685proj-new/conventional/finetune.py --training --lr 1e-3
python /home/jamesbarrett_umass_edu/cs685proj-new/conventional/finetune.py --training --lr 1e-5