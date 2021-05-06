"""
Displacement_strain_planet
=======
Displacement_strain_planet provides several functions an an example script
for generating crustal thickness, displacement, gravity, lateral density
variation, stress, and strain maps on a planet given a set of input
constraints such as from observed gravity and topography data.

These functions solve the Banerdt et al. (1986) thin shell model under
different assumptions. Various improvements have been made to the model
including the possibility to account for finite-amplitude correction and
filtering (Wieczorek & Phillips, 1998), lateral density variations at any
arbitrary depth and within the surface or moho relief (Wieczorek et al.,
2013), and density difference between the surface topography and crust
(Broquet & Wieczorek, 2019).

We note that some of these functions relies heavily on the pyshtools package.

Notes
    Thin_shell_matrix            Solves the Banerdt et al. (1986) system of
                                 equations under the mass-sheet approximation
                                 and assuming that potential internal density
                                 variations are contained within a spherical
                                 shell.

    DownContFilter               Compute the downward minimum-amplitude or
                                 -curvature filter of Wieczorek & Phillips,
                                 (1998).

    Thin_shell_matrix_nmax       Solve the Banerdt et al. (1986) system of
                                 equations with finite-amplitude correction
                                 and accounting for the potential presence of
                                 density variations within the surface or
                                 moho reliefs.

    corr_nmax_drho               Calculate the gravitational potential exterior
                                 to relief referenced to a spherical interface
                                 (with or without laterally varying density)
                                 difference between the mass-sheet case and when
                                 using the finite amplitude algorithm of Wieczorek
                                 (2007).

    SH_deriv                     Compute spherical harmonic derivatives (first and
                                 second order) on the fly.

    SH_deriv_theta_phi           Compute and store spherical harmonic derivatives
                                 (first and second order).

    Displacement_strains         Calculate the Banerdt et al. (1986) equations to
                                 determine strains from displacements.

    Principal_strain_angle       Calculate principal strains and angles.

    Plt_tecto_Mars               Plot the Knampeyer et al. (2006) dataset of extensional
                                 and compressional tectonic features on Mars.
"""
from ._version import get_versions

from .B1986_nmax import Thin_shell_matrix
from .B1986_nmax import corr_nmax_drho
from .B1986_nmax import Thin_shell_matrix_nmax
from .B1986_nmax import DownContFilter

from .Displacement_strain import SH_deriv
from .Displacement_strain import SH_deriv_theta_phi
from .Displacement_strain import Displacement_strains
from .Displacement_strain import Principal_strain_angle
from .Displacement_strain import Plt_tecto_Mars

del B1986_nmax
del Displacement_strain

__version__ = get_versions()["version"]
del get_versions

__author__ = "Adrien Broquet"

__all__ = [
    "Thin_shell_matrix_nmax",
    "corr_nmax_drho",
    "Thin_shell_matrix",
    "DownContFilter",
    "SH_deriv",
    "SH_deriv_theta_phi",
    "Displacement_strains",
    "Principal_strain_angle",
    "Plt_tecto_Mars",
]
