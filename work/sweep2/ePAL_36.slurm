#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       32GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   1
#SBATCH --job-name  ePAL_36
#SBATCH --time      48:00:00
#SBATCH --partition serial

source /home/kjablonk/anaconda3/bin/activate
conda activate dispersant_screener

python run_pal_on_dispersant_repeats_cli.py 0.1 0.01 0.0625 1 . 10
