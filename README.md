# dispersant_screening_PAL

![Python package](https://github.com/byooooo/dispersant_screening_PAL/workflows/Python%20package/badge.svg)

## Scripts for reproducing the results discussed in the paper

### Dispersants (Main text)

### Missing data (Supplementary Information)

### MOF case study (Supplementary Information)

Make sure that `PMOF20K_traindata_7000_test.csv` and `PMOF20K_traindata_7000_train.csv` are in your working directory. Then use

```bash
python mof_case_study.py <epsilon> <delta> <beta_scale> <outdir>
```

If you do not specify any arguments, it uses the parameters used to produce the figure in the SI. We didn't tune the parameters extensively
and you can likely find better performance.

## Data

In folder `data`.

### Calculated Results

- `b0_random_deltaG.csv` - data containing adsorption free energy (deltaGmin) for fractional factorial DOE random copolymers
- `b0_random_virial.csv` - data containing second virial coefficient (A2_normalized) for fractional factorial DOE random copolymers
- `b1-b21_random_deltaG.csv` - data containing adsorption free energy (deltaGmin) for full factorial DOE random copolymers (index reset to 0)
- `b1-b21_random_virial.csv` - data containing second virial coefficient (A2_normalized) for full factorial DOE random copolymers (index reset to 0)

### Feature Data

- `X_frac_random.csv` - Feature data for fractional factorial DOE random copolymers
- `X_full_random.csv` - Feature data for full factorial DOE random copolymers

## Theory

The adsorption free energies are estimated by taking the different in the minimum potential of mean force (W) from its bulk value.
In the NVT ensemble the PMF is equal to the Helmholtz free energy, which we then use to approximate the Gibbs free energy (assuming incompressibility).

The second virial coefficient is calculated using the following equation:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={A_2 = \dfrac{2\pi}{N^2} \int_0^\infty r^2[1-exp(-\beta W(r))]dr}">
</p>

where W(r) is the polymer-polymer potential of mean force in the radial direction and N is the number of beads in the polymer.
