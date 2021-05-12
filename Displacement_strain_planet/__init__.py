"""
Displacement_strain_planet
============================
Displacement_strain_planet provides several functions and example scripts
for generating crustal thickness, displacement, gravity, lateral density
variations, stress, and strain maps on a planet given a set of input
constraints such as from observed gravity and topography data.

These functions solve the Banerdt (1986) thin shell model under
different assumptions. Various improvements have been made to the model
including the possibility to account for finite-amplitude correction and
filtering (Wieczorek & Phillips, 1998), lateral density variations at any
arbitrary depth and within the surface or moho relief (Wieczorek et al.,
2013), and density difference between the surface topography and crust
(Broquet & Wieczorek, 2019).

We note that some of these functions relies heavily on the pyshtools package.

   Thin_shell_matrix
      Solve for the Banerdt et al. (1986) system of 5 equations with
      the possibility to account for finite-amplitude corrections
      and lateral density variations with the surface topography or
      moho relief.

   Thin_shell_matrix_nmax
      Solve the Banerdt (1986) system of 5 equations
      with finite-amplitude correction and accounting
      for the potential presence of density variations
      within the surface or moho reliefs.

    DownContFilter
      Compute the downward minimum-amplitude or
      -curvature filter of Wieczorek & Phillips,
      (1998).

    corr_nmax_drho
      Calculate the difference in gravitational exterior
      to relief referenced to a spherical interface
      (with or without laterally varying density)
      between the mass-sheet case and when using the
      finite amplitude algorithm of Wieczorek &
      Phillips (1998).

    SH_deriv
      Compute on the fly spherical harmonic derivatives
      (first and second order).

    SH_deriv_store
      Compute and store or load spherical harmonic derivatives
      (first and second order).

    Displacement_strains
      Computes the Banerdt (1986) equations to determine strains
      from displacements with a correction to the theta_phi term.

    Principal_strainstress_angle
      Calculate principal strains, stresses, and
      their principal angles.

    Plt_tecto_Mars
      Plot the Knampeyer et al. (2006) dataset of
      extensional and compressional tectonic features
      on Mars.
"""
from ._version import get_versions

from .B1986_nmax import Thin_shell_matrix
from .B1986_nmax import Thin_shell_matrix_nmax
from .B1986_nmax import DownContFilter
from .B1986_nmax import corr_nmax_drho

from .Displacement_strain import SH_deriv
from .Displacement_strain import SH_deriv_store
from .Displacement_strain import Displacement_strains
from .Displacement_strain import Principal_strainstress_angle
from .Displacement_strain import Plt_tecto_Mars

del B1986_nmax
del Displacement_strain

__version__ = get_versions()["version"]
del get_versions

__author__ = "Adrien Broquet"

__all__ = [
    "Thin_shell_matrix",
    "Thin_shell_matrix_nmax",
    "DownContFilter",
    "corr_nmax_drho",
    "SH_deriv",
    "SH_deriv_store",
    "Displacement_strains",
    "Principal_strainstress_angle",
    "Plt_tecto_Mars",
]
