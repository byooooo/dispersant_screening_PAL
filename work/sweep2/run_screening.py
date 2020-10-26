# -*- coding: utf-8 -*-
import os
import subprocess
import time

import click
import pandas as pd

N_SAMPLES = [10, 20, 40, 60, 100, 200]
EPSILON = [0, 0.01, 0.05, 0.1]
BETA_SCALE = [1, 1 / 3, 1 / 9, 1 / 20]
DELTA = [0.01, 0.05]

SLURM_TEMPLATE = '''#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       32GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   1
#SBATCH --job-name  {name}
#SBATCH --time      24:00:00
#SBATCH --partition serial

source /home/kjablonk/anaconda3/bin/activate
conda activate dispersant_screener

python run_pal_on_dispersant_repeats_cli.py {epsilon} {delta} {beta_scale} 10 10 1 5 . {n_samples}
'''

THIS_DIR = os.path.dirname(__file__)


def write_submission_script(counter, epsilon, delta, beta_scale, n_samples):
    name = 'ePAL_{}'.format(counter)
    script = SLURM_TEMPLATE.format(**{
        'name': name,
        'epsilon': epsilon,
        'delta': delta,
        'beta_scale': beta_scale,
        'n_samples': n_samples
    })
    filename = name + '.slurm'
    with open(filename, 'w') as fh:
        fh.write(script)

    return filename


@click.command('cli')
@click.option('--submit', is_flag=True)
def main(submit):
    experiments = []
    counter = 0
    for n_samples in N_SAMPLES:
        for epsilon in EPSILON:
            for beta_scale in BETA_SCALE:
                for delta in DELTA:
                    experiment = {
                        'counter': counter,
                        'n_samples': n_samples,
                        'epsilon': epsilon,
                        'beta_scale': beta_scale,
                        'delta': delta
                    }

                    experiments.append(experiment)

                    fn = write_submission_script(counter, epsilon, delta, beta_scale, n_samples)

                    if submit:
                        subprocess.call('sbatch {}'.format(fn), shell=True)
                        time.sleep(5)
                    counter += 1

    df = pd.DataFrame(experiments)
    df.to_csv('all_experiments.csv', index=False)


if __name__ == '__main__':
    main()
