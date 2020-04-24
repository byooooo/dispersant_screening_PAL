# dispersant_screening_PAL


## Data

### Calculated Results

- b0_random_deltaG.csv - data containing adsorption free energy (deltaGmin) for fractional factorial DOE random copolymers
- b0_random_virial.csv - data containing second virial coefficient (A2_normalized) for fractional factorial DOE random copolymers
- b1-b21_random_deltaG.csv - data containing adsorption free energy (deltaGmin) for full factorial DOE random copolymers (index reset to 0)
- b1-b21_random_virial.csv - data containing second virial coefficient (A2_normalized) for full factorial DOE random copolymers (index reset to 0)


### Feature Data

- X_frac_random.csv - Feature data for fractional factorial DOE random copolymers
- X_full_random.csv - Feature data for full factorial DOE random copolymers

## Theory

The adsorption free energies are estimated by taking the different in the minimum potential of mean force (W) from its bulk value.
In the NVT ensemble the PMF is equal to the Helmholtz free energy, which we then use to approximate the Gibbs free energy (assuming incompressibility).

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\Delta G_{ads} \approx W(z)_{min}-W(z)_{bulk}}">
</p>

The second virial coefficient is calculated using the following equation:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={A_2 = \dfrac{2\pi}{N^2} \int_0^\infty r^2[1-exp(-\beta W(r))]dr}">
</p>

where W(r) is the polymer-polymer potential of mean force in the radial direction and N is the number of beads in the polymer.



## fractional factorial

### adsorption free energies

<div>
  <img width = "600" src="./figures/batch0_ads_PMF.png">
</div>

### second virials coefficients

<div>
  <img width = "600" src="./figures/batch0_vir_PMF.png">
</div>

### full factorial

#### adsorption free energies

<div>
  <img width = "1000" src="./figures/batch1_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch2_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch3_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch4_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch5_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch6_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch7_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch8_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch9_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch10_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch11_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch12_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch13_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch14_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch15_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch16_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch17_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch18_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch19_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch20_ads_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch21_ads_PMF.png">
</div>

#### second virial coefficients

<div>
  <img width = "1000" src="./figures/batch1_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch2_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch3_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch4_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch5_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch6_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch7_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch8_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch9_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch10_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch11_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch12_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch13_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch14_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch15_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch16_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch17_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch18_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch19_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch20_vir_PMF.png">
</div>

<div>
  <img width = "1000" src="./figures/batch21_vir_PMF.png">
</div>

