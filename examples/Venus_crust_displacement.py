import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from cmcrameri import cm
from Displacement_strain_planet import (
    Thin_shell_matrix_nmax,
    Displacement_strains_shtools,
    Principal_strainstress_angle,
)

#################################################################
# In this example, we solve for the displacement of the surface of
# Venus by calling the function `Thin_shell_matrix_nmax`, assuming
# that the gravity and topography of the planet are compensated by
# a combination of crustal root variations and flexure.
# 3 assumptions are required to solve the system, and we here assume
# that the observed topography and geoid are known, and that there
# are no density variations in the interior.
#
# Next, we will plot the associated principal horizontal strains,
# along with the principal angle, and show that these are
# consistent with the tectonic mapping of Knampeyer et al. (2006).
#
# More information can be found in the jupyter notebook Run_demo
#
# In the computation, we will make use of a downward continuation
# minimum-amplitude filter to damp unrealistic oscilations of the
# moho-relief. For this, we call the optional argument filter, set
# it to "Ma", and set the degree at which the filter equals 0.5 to
# 50 with a call to filter_half.
#
# The function ouputs the following spherical harmonic coefficients:
# w_lm flexure,
# A_lm poloidal term of the tangential displacement,
# moho_relief_lm` moho relief,
# dc_lm crustal root variations,
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

quiet = False
lmax = 40  # Maximum spherical harmonic degree to perform all
# calculations
pot_clm = pysh.datasets.Venus.MGNP180U(lmax=lmax)
topo_clm = pysh.datasets.Venus.VenusTopo719(lmax=lmax)

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
g0 = gm / R**2  # Mean gravitational
# attraction of the planet

# Remove 100% of C20
percent_C20 = 0.0
topo_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * topo_clm.coeffs[0, 2, 0]
geoid_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * geoid_clm.coeffs[0, 2, 0]

# Parameters
c = 20e3  # Mean Crustal thickness
Te = 100e3  # Elastic thickness
rhom = 3300.0  # Mantle density
rhoc = 2800.0  # Crustal density
rhol = 2800.0  # Surface density
E = 100e9  # Young's modulus
v = 0.25  # Poisson's ratio

print("Elastic thickness is %.2f km" % (Te / 1e3))
print("Mean crustal thickness is %.2f km" % (c / 1e3))
print("Crustal density is %.2f kg m-3" % (rhoc))

args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, lmax, E, v, mass)
args_expand = dict(lmax=5 * lmax, lmax_calc=lmax)
args_fig = dict(figsize=(12, 10), dpi=100)

zeros = pysh.SHCoeffs.from_zeros(lmax=lmax).coeffs

print("Computing displacements and crustal root variations")
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
    filter_half=30,
    quiet=quiet
)

# Plotting
args_plot = dict(tick_interval=[45, 30], colorbar="bottom", cmap=cm.roma_r)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, **args_fig)
ax3.set_visible(False)

grid_W = pysh.SHCoeffs.from_array(w_lm / 1e3).expand(**args_expand) - R / 1e3
grid_W.plot(ax=ax1, cb_label="Upward displacement (km)", **args_plot, ticks="WSne")

# Add zero displacement contour
ax1.contour(
    grid_W.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper"
)
pysh.SHCoeffs.from_array(dc_lm / 1e3).expand(**args_expand).plot(
    ax=ax2,
    cb_label="Crustal root variations (km)",
    ticks="wSnE",
    ylabel=None,
    **args_plot
)

(pysh.SHCoeffs.from_array((H_lm - moho_relief_lm) / 1e3).expand(**args_expand)).plot(
    ax=ax4, cb_label="Crustal thickness (km)", ticks="WSne", show=False, **args_plot
)

print("Computing strains")
args_param_s = (E, v, R, Te, lmax)

# Strain
(
    stress_theta,
    stress_phi,
    stress_theta_phi,
    eps_theta,
    eps_phi,
    omega,
    kappa_theta,
    kappa_phi,
    tau,
    tot_theta,
    tot_phi,
    tot_thetaphi,
) = Displacement_strains_shtools(A_lm, w_lm, *args_param_s, quiet=quiet)

# Principal strains
(
    min_strain,
    max_strain,
    sum_strain,
    principal_angle,
) = Principal_strainstress_angle(-tot_theta, -tot_phi, -tot_thetaphi)

print("Plotting")
args_plot = dict(
    tick_interval=[45, 30],
    colorbar="bottom",
    cmap=cm.vik,
)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, **args_fig)

pysh.SHGrid.from_array(min_strain * 1e3).plot(
    ax=ax1,
    ticks="WSne",
    cb_label="Minimum principal horizontal strain ($\\times 10^{-3}$)",
    cmap_limits=[-6, 20],
    **args_plot
)
pysh.SHGrid.from_array(max_strain * 1e3).plot(
    ax=ax2,
    cb_label="Maximum principal horizontal strain ($\\times 10^{-3}$)",
    cmap_limits=[-6, 20],
    ticks="wSnE",
    ylabel=None,
    **args_plot
)
pysh.SHGrid.from_array(sum_strain * 1e3).plot(
    ax=ax3,
    cb_label="Sum of principal horizontal strains ($\\times 10^{-3}$)",
    cmap_limits=[-6, 20],
    ticks="WSne",
    **args_plot
)
pysh.SHGrid.from_array(principal_angle).plot(
    ax=ax4,
    cb_label="Principal angle (°)",
    ticks="wSnE",
    ylabel=None,
    cmap_limits=[-90, 90],
    tick_interval=[45, 30],
    colorbar="bottom",
    cmap=cm.vikO,
)

# Plot strain direction
skip_i = int(lmax / 5)
skip = (slice(None, None, skip_i), slice(None, None, skip_i))
grid_long, grid_lat = np.meshgrid(
    pysh.SHGrid.from_array(principal_angle).lons(),
    pysh.SHGrid.from_array(principal_angle).lats(),
)
ones = np.ones(np.shape(principal_angle))
ax4.quiver(
    grid_long[skip],
    grid_lat[skip],
    ones[skip],
    ones[skip],
    scale=5e1,
    angles=principal_angle[skip],
    color="g",
)

plt.show()
