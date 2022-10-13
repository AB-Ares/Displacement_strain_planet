[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4916799.svg)](http://doi.org/10.5281/zenodo.4916799)

# Displacement_strain_planet

Planetary gravity, crustal thickness, displacement, stress, and strain calculations in spherical harmonics.

## Description

**Displacement_strain_planet** provides several functions and example scripts for generating gravity, crustal thickness, displacement, lateral density variations, stress, and strain maps on a planet given a set of input constraints such as from observed gravity and topography data.

These functions solve the [Banerdt (1986)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB091iB01p00403) system of equations under different assumptions. Various improvements have been made to the model including the possibility to account for finite-amplitude correction and filtering [(Wieczorek & Phillips, 1998)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97JE03136), lateral density variations at any arbitrary depth and within the surface, flexure, or moho reliefs [(Wieczorek et al., 2013)](https://science.sciencemag.org/content/early/2012/12/04/science.1231530?versioned=true), and density difference between the surface topography and crust [(Broquet & Wieczorek, 2019)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JE005959). 

### Comments
#### Dependencies
Some of these functions rely heavily on the [pyshtools](https://shtools.github.io/SHTOOLS/) package of [Wieczorek & Meschede (2018)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GC007529) that is used to perform the spherical harmonic transforms, Legendre polynomial expansions, and finite-amplitude calculations.

#### Contribute
This code is still under development and benchmarking. If you find any bugs or errors in the code, please report them in GitHub or to adrien.broquet at oca.eu.

For this code, we work on the develop branch and merge it to the main branch (with a new version number) everytime significant addtions/improvements are made. If you plan on making contributions, please base everything on the develop branch.

### Benchmarks
Moho-relief calculations have been benchmarked to the [ctplanet](https://github.com/MarkWieczorek/ctplanet) package of Mark Wieczorek.  
Displacement calculations have been benchmarked to the analytical model of [Broquet & Wieczorek (2019)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JE005959).  
Strain calculations reproduce results published in the literature (e.g., [Banerdt & Golombek 2000](https://www.lpi.usra.edu/meetings/lpsc2000/pdf/2038.pdf)). 

## Methods
`Thin_shell_matrix` Solve the [Banerdt (1986)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB091iB01p00403) system of 5 equations under the mass-sheet approximation and assuming that potential internal density variations are contained within a spherical shell. The system links 8 parameters expressed in spherical harmonics: the topography, geoid at the surface, geoid at the moho depth, net acting load on the lithosphere, tangential load potential, flexure of the lithosphere, crustal root variations, and internal density variations. Minor corrections have been made in the geoid equations, and to the elastic shell transfer functions following [Beuthe (2008)](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1365-246X.2007.03671.x).

`Thin_shell_matrix_nmax` Solve the [Banerdt (1986)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB091iB01p00403) system of 5 equations with finite-amplitude correction and accounting for the potential presence of density variations within the surface or moho reliefs.

`DownContFilter` Compute the downward minimum-amplitude or -curvature filter of [Wieczorek & Phillips (1998)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97JE03136).

`corr_nmax_drho` Calculate the difference in gravitational potential exterior to relief referenced to a spherical interface (with or without laterally varying density) between the mass-sheet case and when using the finite-amplitude algorithm of [Wieczorek & Phillips (1998)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97JE03136).

`SH_deriv` Compute on the fly first and second-order spherical harmonic derivatives with respect to colatitude and longitude.

`SH_deriv_store` Compute and store first and second-order spherical harmonic derivatives with respect to colatitude and longitude.

`Displacement_strains` Calculate the [Banerdt (1986)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB091iB01p00403) equations to determine strains and stresses from displacements with some corrections following [Beuthe (2008)](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1365-246X.2007.03671.x).

`Principal_strainstress_angle` Calculate principal strains, stresses and their principal angles.

`Strainstress_from_principal` Calculate strains or stresses, from their principal values.

`Plt_tecto_Mars` Plot the [Knampeyer et al. (2006)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JE002708) dataset of extensional and compressional tectonic features on Mars.

## Example scripts
`Run_demo`  [<img src="misc/link1.svg" width="20">](https://ab-ares.github.io/Displacement_strain_planet/notebooks/Run_demo.html) A jupyter notebook that contains example scripts to determine flexure, moho-relief, and strains on Mars under different assumptions, including Airy and Pratt isostasy, or due to the sole presence of a mantle plume.

`Mars_crust_displacement` A script that demonstrates how to calculate the moho-relief and strains on Mars, as a function of the mean planetary crustal thickness and elastic thickness. The contributions from crustal root variations and displacement are shown assuming an elastic thickness of the lithosphere. We make use of the inferred displacement to predict the principal horizontal strains and principal angle, which are compared to extensional tectonic features mapped by [Knampeyer et al. (2006)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JE002708). 

`Mars_SouthPolarCap_displacement` A script that demonstrates how to calculate iteratively the flexure underneath the south polar cap of Mars as a function of elastic thickness and ice density. This computation is similar to that done in e.g., Broquet et al. (2021), accepted in JGR:Planets. 

`Venus_crust_displacement` A script that demonstrates how to calculate the moho-relief and strains on Venus, as a function of the mean planetary crustal thickness and elastic thickness. 

## How to install and run Displacement_strain_planet
If you would like to modify the source code, download the Displacement_strain_planet repository and install using pip (or pip3 depending on your installation).
```bash
    git clone https://github.com/AB-Ares/Displacement_strain_planet.git
    cd Displacement_strain_planet/
    pip install .
```
Alternatively, you can install Displacement-strain-planet via pip
```bash
   pip install Displacement-strain-planet
```

## To run the example scripts
```bash
    cd examples
    jupyter notebook Run_demo.ipynb
    python Mars_crust_displacement.py
```

## Author
[Adrien Broquet](https://www.lpl.arizona.edu/postdocs/adrien-broquet) (adrienbroquet@email.arizona.edu)

## Cite
You can cite the latest release of the package as:
Adrien Broquet. (2022). Displacement_strain_planet: 0.4.0 (Version 0.4.0). Zenodo. http://doi.org/10.5281/zenodo.4916799

```bash
@misc{Broquet2022,
    author = {Broquet, A.},
    title = {{Displacement{\_}strain{\_}planet}: Version 0.4.0},
    url = {https://github.com/AB-Ares/Displacement_strain_planet},
    doi = {10.5281/zenodo.4916799},
    year = {2022}}
```

## Acknowledgments
I would like to thank the SHTools developpers for putting together their great open-source code, and Mark Wieczorek & Jeff Andrews-Hanna for stimulating discussion. 
