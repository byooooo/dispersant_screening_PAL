# De novo polymer design with multi objective active learning and molecular simulations

![Python package](https://github.com/byooooo/dispersant_screening_PAL/workflows/Python%20package/badge.svg)

This repository contains code that was used to generate the results discussed in

Jablonka, K. M.; Giriprasad, M. J.; Wang, S.; Smit, B.; Yoo, B. De Novo Polymer Design with Multi Objective Active Learning and Molecular Simulations. 2020.

[The active learning code, PyPAL, is maintained in its own repository.](https://github.com/kjappelbaum/pypal)

## `dispersant_screener` package

### Featurization

The code for the polymer featurization can be found in the `featurizer` module of the `dispersant_screener` package.

The `create_new_features.ipynb` illustrates the basic usage, i.e., how a dataframe with polymer SMILES can be featurized.

### Genetic algorithm and inversion of SMILES

The genetic algorithm is defined in the `ga` module of the `dispersant_screener` package. The inversion code is part of the `smiles2feat` module.

### Scripts for reproducing the main results discussed in the paper

We recommend that you [create a conda environment, based on the appropriate `.yml` file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```bash
conda env conda env create --file=environment_mac.yml
```

### Dispersants (Main text)

To reproduce the results, make sure that `../data/new_features_full_random.csv`, `../data/b1-b21_random_virial_large_new.csv`, and `../data/rg_results.csv` are in your working directory. Those relative paths are hardcoded in the command line interface.
Then use

```bash
python run_pal_on_dispersant_repeats_cli.py <epsilon> <delta> <beta_scale> <repeats> <outdir> <n_samples>
```

### Missing data

All [`PyPAL`](https://github.com/kjappelbaum/pypal) classes support missing data. For this work, we used coregionalized models that leverage correlations
between the different objectives. To reproduce the results, make sure that `../data/new_features_full_random.csv`, `../data/b1-b21_random_virial_large_new.csv`, and `../data/rg_results.csv` are in your working directory. Those relative paths are hardcoded in the command line interface.
Then use

```bash
python missing_data_test.py <epsilon> <delta> <beta_scale> <outdir>
```

### Genetic algorithm
 To reproduce the results, make sure that `X_train_GBDT.npy` and `y_train_GBDT.npy` are in your working directory. Those relative paths are hardcoded in the command line interface.
Then use

```bash
python run_ga_on_gbdt_gpr_surrogate.py
```


## Data

In folder `data`.

### Calculated Results based on the molecular simulations

- `b0_random_deltaG.csv` - data containing adsorption free energy (deltaGmin) for fractional factorial DOE random copolymers
- `b0_random_virial.csv` - data containing second virial coefficient (A2_normalized) for fractional factorial DOE random copolymers
- `b1-b21_random_deltaG.csv` - data containing adsorption free energy (deltaGmin) for full factorial DOE random copolymers (index reset to 0)
- `b1-b21_random_virial.csv` - data containing second virial coefficient (A2_normalized) for full factorial DOE random copolymers (index reset to 0)

### Feature Data

- `X_frac_random.csv` - Feature data for fractional factorial DOE random copolymers
- `X_full_random.csv` - Feature data for full factorial DOE random copolymers

### Data for the MOF case study

Reeused from [Moosavi, S. M.; Nandy, A.; Jablonka, K. M.; Ongari, D.; Janet, J. P.; Boyd, P. G.; Lee, Y.; Smit, B.; Kulik, H. J. Understanding the Diversity of the Metal-Organic Framework Ecosystem. Nature Communications 2020, 11 (1), 4068.](https://doi.org/10.1038/s41467-020-17755-8)

- `PMOF20K_traindata_7000_train.csv`
- `PMOF20K_traindata_7000_test.csv`
