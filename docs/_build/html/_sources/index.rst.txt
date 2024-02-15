.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   installation.rst
   examples.rst
   references.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference Documentation

   source/Displacement_strain_planet.rst

DSP: Displacement_strain_planet
================================

Displacement_strain_planet (DSP) provides several functions and example scripts for generating, among others, gravity, crustal thickness, displacement, lateral density variations, stress, and strain maps on a planet given a set of input constraints such as from observed gravity and topography data.

These functions solve the `Banerdt (1986) <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JB091iB01p00403>`_ system of equations under different assumptions. The model links 8 parameters: the topography, geoid at the surface, geoid at the moho depth, net acting load on the lithosphere, tangential load potential, flexure of the lithosphere, crustal thickness variations, and internal density variations, through 5 equations. Minor corrections have been made to the geoid equations and displacement equations following `Beuthe (2008) <https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-246X.2007.03671.x>`_. All is required is that the user specifies 3 constraints and the model will solve for all other parameters. 

Various improvements have been made to the model, including the possibility to account for finite-amplitude correction and filtering `(Wieczorek & Phillips, 1998) <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97JE03136>`_, lateral density variations at any arbitrary depth and within the surface or moho-relief `(Wieczorek et al., 2013) <https://science.sciencemag.org/content/339/6120/671>`_, and density difference between the surface topography and crust `(Broquet & Wieczorek, 2019) <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JE005959>`_, or the addition of equations and constraints `(Broquet & Andrews-Hanna, 2024) <https://doi.org/10.1016/j.icarus.2023.115846>`_.

This routine has many applications and is highly versatile, and you can for example:

* Compute the relief along the crust-mantle interface based on the input constraints (3 constraints are required, e.g., 1. and 2. the model should match the observed gravity and topography of the planet 3. there are no lateral variations in density).

* Compute the geoid or displacement associated with a load and for a given elastic thickness.

* Compute lateral density variations to match the input constraints.

* Compute the associated strain and stresses and determine their principal horizontal components and directions.

* Compute Legendre polynomial first and second order derivatives.

In addition to these functions, I provide some example scripts to: 

* Solve for the moho-relief on Mars and Venus, and estimate the principal strains on these planets as a function of the input elastic thickness. 

* Compute the flexure beneath the south polar cap of Mars for an input elastic thickness. 

* and a jupyter notebook is also added with more information on estimating the moho-relief on Mars, assuming Airy or Pratt isostasy, the displacement due to a mantle plume underneath Tharsis or due to internal loading in phase with the surface topography. 
