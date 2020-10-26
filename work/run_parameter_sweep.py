# -*- coding: utf-8 -*-
# Copyright 2020 PyPAL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import subprocess
import time

import click
import pandas as pd

N_SAMPLES = [60, 100]
EPSILON = [0.01, 0.05, 0.1]
BETA_SCALE = [1 / 9, 1 / 20]
DELTA = [0.05]

SLURM_TEMPLATE = '''#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       32GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   1
#SBATCH --job-name  {name}
#SBATCH --time      48:00:00
#SBATCH --partition serial

source /home/kjablonk/anaconda3/bin/activate
conda activate pypal

python run_pal_on_dispersant_repeats_cli.py {epsilon} {delta} {beta_scale} 1 . {n_samples}
'''

THIS_DIR = os.path.dirname(__file__)


def write_submission_script(counter, epsilon, delta, beta_scale, n_samples):
    name = 'ePALdispersant_{}'.format(counter)
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

                    SUBMISSIONSCRIPTNAME = write_submission_script(counter, epsilon, delta, beta_scale, n_samples)

                    if submit:
                        subprocess.call('sbatch {}'.format(SUBMISSIONSCRIPTNAME), shell=True, cwd='.')
                        time.sleep(10)
                    counter += 1

    df = pd.DataFrame(experiments)
    df.to_csv('all_experiments.csv', index=False)


if __name__ == '__main__':
    main()
