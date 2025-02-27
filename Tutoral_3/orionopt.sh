#!/bin/bash                                          
#SBATCH --array=1-10
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --error=slurm/%A/serr/%a.err
#SBATCH --output=slurm/%A/sout/%a.out

module load miniconda/3
conda activate /home/mila/l/letournv/miniconda3/envs/photo

orion hunt -n fomopt python optest2.py \
-lr_fom~'uniform(1e16, 1e18)' \
-lr_ent~'uniform(0.001, 0.2)' \
-fom_phase~'uniform(10, 50, discrete=True)' \
-ent_phase~'uniform(10, 200, discrete=True)' \

