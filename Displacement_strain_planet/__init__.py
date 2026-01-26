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

  thinshell class:
    invert_matrix
      Solve for the Banerdt et al. (1986) system of 5 equations with
      the possibility to account for finite-amplitude corrections
      and lateral density variations with the surface topography or
      moho relief.

    invert_matrix_nmax
      Solve the Banerdt (1986) system of 5 equations
      with finite-amplitude correction and accounting
      for the potential presence of density variations
      within the surface or moho reliefs.

    compute_strains
      Computes the Banerdt (1986) equations to determine strains
      from displacements with a correction to the theta_phi term.

  utils:
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

    SH_deriv_store
      Compute and store or load spherical harmonic derivatives
      (first and second order) over the whole sphere or 
      given a set of colatiudes/longitudes.

    SH_deriv
      Compute on the spherical harmonic derivatives
      (first and second order) at a given single colatiude/longitude 
      location.

    Plt_tecto_Mars
      Plot the Knampeyer et al. (2006) dataset of
      extensional and compressional tectonic features
      on Mars.
  
    Principal_strainstress_angle
      Calculate principal strains, stresses, and
      their principal angles.

    Strainstress_from_principal
      Calculate strains or stresses, from
      their principal values.
"""

from ._version import get_versions

from .thinshell import ThinShell

from .utils import spectral_degrad
from .utils import DownContFilter
from .utils import corr_nmax_drho
from .utils import SH_deriv
from .utils import SH_deriv_store
from .utils import Plt_tecto_Mars
from .utils import Principal_strainstress_angle
from .utils import Strainstress_from_principal

del utils
del thinshell

__version__ = get_versions()["version"]
del get_versions

__author__ = "Adrien Broquet"

__all__ = [
    "ThinShell",
    "spectral_degrad",
    "DownContFilter",
    "corr_nmax_drho",
    "SH_deriv",
    "SH_deriv_store",
    "Principal_strainstress_angle",
    "Strainstress_from_principal",
    "Plt_tecto_Mars",
]
