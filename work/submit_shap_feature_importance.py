# -*- coding: utf-8 -*-
import os
import subprocess
import time

import click

SLURM_TEMPLATE = '''#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       32GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   1
#SBATCH --job-name  {name}
#SBATCH --time      24:00:00
#SBATCH --partition serial

source /home/kjablonk/anaconda3/bin/activate
conda activate pypal

python get_shap_feature_importance.py {model} {i}
'''

THIS_DIR = os.path.dirname(__file__)


def write_submission_script(model, i):
    name = f'ePAL_SHAP_{model}_{i}'
    script = SLURM_TEMPLATE.format(**{'name': name, 'model': model, 'i': i})
    filename = name + '.slurm'
    with open(filename, 'w') as fh:
        fh.write(script)

    return filename


I = [0, 1, 2]
MODELS = [
    # "20201021-235959_dispersant_0.01_0.05_0.05_100", "20201022-000033_dispersant_0.1_0.05_0.1111111111111111_100",
    # "20201021-235927_dispersant_0.05_0.05_0.1111111111111111_60",
    # "20201021-235927_dispersant_0.1_0.05_0.1111111111111111_60", "20201021-235927_dispersant_0.01_0.05_0.05_60",
    # "20201022-000033_dispersant_0.05_0.05_0.05_100", '20201022-000105_dispersant_0.1_0.05_0.05_100',
    # '20201021-235927_dispersant_0.05_0.05_0.05_60'
    '20201021-235959_dispersant_0.1_0.05_0.05_60'
]


@click.command('cli')
@click.option('--submit', is_flag=True)
def main(submit):
    for model in MODELS:
        for i in I:
            submissionscript_name = write_submission_script(model, i)

            if submit:
                subprocess.call('sbatch {}'.format(submissionscript_name), shell=True, cwd='.')
                time.sleep(5)


if __name__ == '__main__':
    main()
