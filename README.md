[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Displacement_strain_planet

Crustal thickness, displacement, stress and strain calculations on the sphere.

## Description

Displacement_strain_planet provides several functions and an example script for generating crustal thickness, displacement, gravity, lateral density variations, stress, and strain maps on a planet given a set of input constraints such as from observed gravity and topography data.

These functions solve the [Banerdt et al. (1986)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB091iB01p00403) system of equations under different assumptions. Various improvements have been made to the model including the possibility to account for finite-amplitude correction and filtering [(Wieczorek & Phillips, 1998)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97JE03136), lateral density variations at any arbitrary depth and within the surface or moho relief [(Wieczorek et al., 2013)](https://science.sciencemag.org/content/early/2012/12/04/science.1231530?versioned=true), and density difference between the surface topography and crust [(Broquet & Wieczorek, 2019)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JE005959). 

We note that some of these functions relies heavily on the [pyshtools](https://shtools.github.io/SHTOOLS/) package of [Wieczorek & Meschede (2018)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GC007529).

## Methods
`Thin_shell_matrix` Solve the [Banerdt et al. (1986)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB091iB01p00403) system of equations under the mass-sheet approximation and assuming that potential internal density variations are contained within a spherical shell. The system links 8 parameters expressed in spherical harmonics (degree $l$, order $m$): the topography ($H_{lm}$), geoid at the surface ($G_{lm}$), geoid at the moho depth ($Gc_{lm}$), net acting load on the lithosphere ($q_{lm}$), tangential load potential ($\omega_{lm}$), flexure of the lithosphere ($w_{lm}$), crustal thickness variations ($\delta c_{lm}$), and internal density variations ($\delta \rho_{lm}$). 

`Thin_shell_matrix_nmax` Solve the [Banerdt et al. (1986)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB091iB01p00403) system of equations with finite-amplitude correction and accounting for the potential presence of density variations within the surface or moho reliefs.

`DownContFilter` Compute the downward minimum-amplitude or -curvature filter of [Wieczorek & Phillips (1998)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97JE03136).

`corr_nmax_drho` Calculate the gravitational potential exterior to relief referenced to a spherical interface (with or without laterally varying density) difference between the mass-sheet case and when using the finite amplitude algorithm of [Wieczorek & Phillips (1998)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97JE03136).

`SH_deriv` Compute spherical harmonic derivatives (d'$\theta$, d'$\phi$, d'$\theta,\phi$, d''$\theta$, d''$\phi$, d''$\theta,\phi$, where $\theta$ is the colatitude and $\phi$ is the longitude) on the fly.

`SH_deriv_theta_phi` Compute and store spherical harmonic derivatives (d'$\theta$, d'$\phi$, d'$\theta,\phi$, d''$\theta$, d''$\phi$, d''$\theta,\phi$, where $\theta$ is the colatitude and $\phi$ is the longitude).

`Displacement_strains` Calculate the [Banerdt et al. (1986)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB091iB01p00403) equations to determine strains from displacements with a correction to the $\theta,\phi$ term.

`Principal_strain_angle` Calculate principal strains and angles.

`Plt_faults` Plot the [Knampeyer et al. (2006)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JE002708) dataset of extensional and compressional tectonic features on Mars.

## Example script
`Run_demo` A jupyter notebook that contains example scripts to determine flexure, moho relief, and strains under different assumptions, including Airy or Pratt isostasy.

## How to install and run ctplanet
Download the ctplanet repository and install using pip (or pip3 depending on your installation).
```bash
    git clone https://github.com/AB-Ares/Displacement_strain-planet.git
    pip install .
```

## To run the example script
```bash
    cd examples
    jupyter notebook Run_demo.ipynb
```

## Author
[Adrien Broquet](https://www.oca.eu/fr/adrien-broquet) (adrien.broquet@oca.eu)
