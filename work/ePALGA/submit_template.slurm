#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       32GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   1
#SBATCH --job-name  ePAL_GA0
#SBATCH --time      1:00:00
#SBATCH --partition debug

source /home/kjablonk/anaconda3/bin/activate
conda activate dispersant_screener

python run_ga_on_gbdt_gpr_surrogate.py  0
