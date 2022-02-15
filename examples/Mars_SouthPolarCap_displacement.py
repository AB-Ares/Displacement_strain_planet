import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from cmcrameri import cm
from Displacement_strain_planet import Thin_shell_matrix_nmax, Thin_shell_matrix

#################################################################
# In this example, we solve for the displacement of the surface at
# the south pole of Mars and for a given elastic thickness and ice
# cap density, as done in Broquet et al. (2021) e.g., Figure S6.
# All we assume is that the polar cap load is compensated by flexure
# of the surface and crust mantle interface, and that there are no
# crustal thickness and internation density variations.
#
# The function ouputs the following spherical harmonic coefficients:
# w_lm flexure,
# A_lm poloidal term of the tangential displacement,
# moho_relief_lm` moho relief,
# dc_lm isostatic crustal root variations,
# drhom_lm internal density variations,
# omega_lm tangential load potential,
# q_lm net load on the lithosphere,
# Gc_lm geoid at the moho depth,
# G_lm geoid at the surface, and
# H_lm topography.
#
# And the linear solution sols expressed as lambda functions
# of all components. Lambda functions can be used to re-calculate
# the same problem with different inputs very fast.
#################################################################

lmax = 90
# Constants
R = pysh.constants.Mars.r.value  # Mean planetary radius
G = pysh.constants.G.value  # Gravitational constant
gm = pysh.constants.Mars.gm.value  # GM given in the gravity
# model file
mass = gm / G  # Mass of the planet
g0 = gm / R**2  # Mean gravitational
# attraction of the planet
rhobar = mass * 3.0 / 4.0 / np.pi / R**3

# Parameters
c = 60e3  # Mean Crustal thickness
Te = 90e3  # Elastic thickness
rhom = 3500.0  # Mantle density
rhoc = 2900.0  # Crustal density
rhol = 1300.0  # Surface density
E = 100e9  # Young's modulus
v = 0.25  # Poisson's ratio

print("Elastic thickness is %.2f km" % (Te / 1e3))
print("Mean crustal thickness is %.2f km" % (c / 1e3))
print("Crustal density is %.2f kg m-3" % (rhoc))
print("Polar cap density is %.2f kg m-3" % (rhol))

args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, lmax, E, v, mass)
args_param_m2 = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax, E, v)
args_expand = dict(lmax=5 * lmax, lmax_calc=lmax)
args_fig = dict(figsize=(12, 10), dpi=100)
args_plot = dict(tick_interval=[45, 30], colorbar="bottom", cmap=cm.roma_r)

# grid_thickMOLA_lm is the spherical harmonic expansion of the grid_thickMOLA file found at
# https://zenodo.org/record/4682983
topo = pysh.SHCoeffs.from_file("data/grid_thickMOLA_lm.txt", lmax=lmax).coeffs
zeros = pysh.SHCoeffs.from_zeros(lmax=lmax).coeffs

iter = 0
residuals = 1e10
while residuals > 5:
    iter += 1
    if iter == 1:
        (w_deflec, sols,) = Thin_shell_matrix_nmax(
            *args_param_m,
            dc_lm=zeros.copy(),
            drhom_lm=zeros.copy(),
            H_lm=topo.copy(),
            nmax=1
        )[
            ::10
        ]  # get first and last arrays
        min1 = R - np.min(
            pysh.SHCoeffs.from_array(w_deflec).expand(lmax_calc=lmax).data
        )
    else:
        w_deflec = Thin_shell_matrix(
            *args_param_m2,
            first_inv=False,
            lambdify_func=sols,
            dc_lm=zeros.copy(),
            drhom_lm=zeros.copy(),
            H_lm=(topo - w_deflec).copy()
        )[0]
        min2 = -np.min(pysh.SHCoeffs.from_array(w_deflec).expand(lmax_calc=lmax).data)
        residuals = np.abs(min1 - min2)
        print(
            "Iteration %s, maximum flexure %.3f km, residuals %.3f km"
            % (iter, min2 / 1e3, residuals / 1e3)
        )
        min1 = min2

w_deflec[0, 0, 0] = 0
(pysh.SHCoeffs.from_array(w_deflec / 1e3).expand(**args_expand)).plot(
    cb_label="Flexure (km)", ticks="wSnE", ylabel=None, show=False, **args_plot
)
plt.show()
