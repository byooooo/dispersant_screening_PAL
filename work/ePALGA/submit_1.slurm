#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       32GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   1
#SBATCH --job-name  ePAL_GA1
#SBATCH --time      24:00:00

source /home/kjablonk/anaconda3/bin/activate
conda activate dispersant_screener

python run_ga_on_gbdt_gpr_surrogate.py  1
