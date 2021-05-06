import os
import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from cmcrameri import cm
from Displacement_strain_planet import *

#################################################################
# In this example, we solve for the displacement of the surface of 
# Mars by calling the function `Thin_shell_matrix_nmax`, assuming 
# that the gravity and topography of the planet are compensated by 
# a combination of crustal thickness variation and flexure. 
# 3 assumptions are required to solve the system, and we here assume 
# that the observed topography and geoid are known, and that there 
# are no density variations in the interior.

# More information can be found in the jupyter notebook Run_demo 

# We will also make use of a downward continuation minimum-amplitude 
# filter to damp unrealistic oscilations of the moho-relief 
# For this, we call the optional argument filter, set it to "Ma", and 
# set the degree at which the filter equals 0.5 to 50 with a call to 
# filter_half.

# The function ouputs the following spherical harmonic coefficients: \
# w_lm flexure, \
# A_lm poloidal term of the tangential displacement,  \
# moho_relief_lm` moho relief,  \
# dc_lm crustal thickness variation,  \
# drhom_lm internal density variations,  \
# omega_lm tangential load potential,  \
# q_lm net load on the lithosphere,  \
# Gc_lm geoid at the moho depth,  \
# G_lm geoid at the surface, and \
# H_lm topography.
   
# And the linear solution sols expressed as lambda functions 
# of all components. Lambda functions can be used to re-calculate 
# the same problem with different inputs very fast.
#################################################################

lmax_calc = 90  # Maximum spherical harmonic degree to perform all
# calculations
pot_clm = pysh.datasets.Mars.GMM3(lmax=lmax_calc)
topo_clm = pysh.datasets.Mars.MarsTopo2600(lmax=lmax_calc)

R = topo_clm.coeffs[0, 0, 0]  # Mean planetary radius
pot_clm = pot_clm.change_ref(r0=R)  # Downward continue to Mean
# planetary radius

# Compute the geoid as approximated in Banerdt's formulation
geoid_clm = pot_clm * R

# Constants
G = pysh.constants.G.value  # Gravitational constant
gm = pot_clm.gm  # GM given in the gravity
# model file
mass = gm / G  # Mass of the planet
g0 = gm / R ** 2  # Mean gravitational
# attraction of the planet
rhobar = mass * 3.0 / 4.0 / np.pi / R ** 3  # Mean density of the
# planet

# Remove 100% of C20
percent_C20 = 0.0
topo_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * topo_clm.coeffs[0, 2, 0]
geoid_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * geoid_clm.coeffs[0, 2, 0]

# Parameters
c = 50e3  # Mean Crustal thickness
Te = 100e3  # Elastic thickness
rhom = 3500.0  # Mantle density
rhoc = 2900.0  # Crustal density
rhol = 2900.0  # Surface density
E = 100e9  # Young's modulus
v = 0.25  # Poisson's ratio

args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax_calc, E, v, mass)
args_expand = dict(lmax=5 * lmax_calc, lmax_calc=lmax_calc)
args_fig = dict(figsize=(12, 10), dpi=100)

path = "%s/data" % (os.getcwd())
zeros = pysh.SHCoeffs.from_zeros(lmax=lmax_calc).coeffs
drhom_lm_plume = pysh.SHCoeffs.from_file(
    "%s/Example_plume_coeffs_clm.txt" % (path)
).coeffs

(
    w_lm,
    A_lm,
    moho_relief_lm,
    dc_lm,
    drhom_lm,
    omega_lm,
    q_lm,
    Gc_lm,
    G_lm,
    H_lm,
    sols,
) = Thin_shell_matrix_nmax(
    *args_param_m,
    G_lm=geoid_clm.coeffs,
    H_lm=topo_clm.coeffs,
    drhom_lm=zeros.copy(),
    filter="Ma",
    filter_half=50
)

# Plotting
args_plot = dict(tick_interval=[45, 30], colorbar="bottom", cmap=cm.roma_r)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, **args_fig)

grid_W = pysh.SHCoeffs.from_array(w_lm / 1e3).expand(**args_expand) - R / 1e3
grid_W.plot(ax=ax1, cb_label="Upward displacement (km)", **args_plot)

# Add zero displacement contour
ax1.contour(
    grid_W.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper"
)
pysh.SHCoeffs.from_array(dc_lm / 1e3).expand(**args_expand).plot(
    ax=ax2,
    cb_label="Crustal thickness" + " variation (km)",
    ticks="wSnE",
    ylabel=None,
    **args_plot
)

pysh.SHCoeffs.from_array(drhom_lm).expand(**args_expand).plot(
    ax=ax3, cb_label="Lateral density" + " variations (kg m$^{-3}$)", **args_plot
)

(pysh.SHCoeffs.from_array((H_lm - moho_relief_lm) / 1e3).expand(**args_expand)).plot(
    ax=ax4,
    cb_label="Moho depth (km)",
    ticks="wSnE",
    ylabel=None,
    show=False,
    **args_plot
)

plt.show()
