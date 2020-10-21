# De novo polymer design with multi objective active learning and molecular simulations

![Python package](https://github.com/byooooo/dispersant_screening_PAL/workflows/Python%20package/badge.svg)

## Scripts for reproducing the results discussed in the paper

We recommend that you create a conda environment, based on the `.yml` files.

```bash
conda env conda env create --file=environment_mac.yml
```

### Dispersants (Main text)

### Missing data (Supplementary Information)

All [`PyPAL`](https://github.com/kjappelbaum/pypal) classes support missing data. For this work, we used coregionalized models that leverage correlations
between the different objectives. To reproduce the results, make sure that `data/new_features_full_random.csv`, `data/b1-b21_random_virial_large_new.csv`, and `data/rg_results.csv` are in your working directory. Those relative paths are hardcoded in the command line interface.
Then use

```bash
python missing_data_test.py <epsilon> <delta> <beta_scale> <outdir>
```

### MOF case study (Supplementary Information)

Make sure that `data/PMOF20K_traindata_7000_test.csv` and `data/PMOF20K_traindata_7000_train.csv` are in your working directory, those relative paths are hardcoded in the command line interface. Then use

```bash
python mof_case_study.py <epsilon> <delta> <beta_scale> <outdir>
```

If you do not specify any arguments, it uses the parameters used to produce the figure in the SI. We didn't tune the parameters extensively
and you can likely find better performance.

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
