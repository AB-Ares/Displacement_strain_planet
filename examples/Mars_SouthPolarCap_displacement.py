import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from cmcrameri import cm
from Displacement_strain_planet import Thin_shell_matrix_nmax, Thin_shell_matrix
from pyshtools.expand import MakeGridDH

#################################################################
# In this example, we solve for the displacement of the surface at
# the south pole of Mars and for a given elastic thickness and ice
# cap density, as done in Broquet et al. (2021) e.g., Figure S6.
# We assume that the polar cap load is compensated by flexure
# of the surface and crust mantle interface, and that there are no
# crustal thickness and internation density variations.
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
# G_lm geoid at the surface,
# H_lm topography,
# and the linear solution sols expressed as lambda functions
# of all components. Lambda functions can be used to re-calculate
# the same problem with different inputs very fast.
#################################################################

lmax = 90
# Constants
R = pysh.constants.Mars.r.value  # Mean planetary radius
G = pysh.constants.G.value  # Gravitational constant
gm = pysh.constants.Mars.gm.value  # GM given in the gravity model file
mass = gm / G  # Mass of the planet
g0 = gm / R**2  # Mean gravitational attraction of the planet
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

# Slightly different inputs between Thin_shell_matrix_nmax and Thin_shell_matrix
args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, lmax, E, v, mass)
args_param_m2 = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax, E, v)
args_expand = dict(sampling=2, lmax=lmax, extend=False)
args_fig = dict(figsize=(12, 10), dpi=100)
args_plot = dict(tick_interval=[45, 30], colorbar="bottom", cmap=cm.roma_r)

# grid_thickMOLA_lm is the spherical harmonic expansion of the
# grid_thickMOLA file found at https://zenodo.org/record/4682983
# This file gives the thickness of the South Polar cap (without flexure)
# from an interpolation
thick = pysh.SHCoeffs.from_file("data/grid_thickMOLA_lm.txt", lmax=lmax).coeffs
zeros = pysh.SHCoeffs.from_zeros(lmax=lmax).coeffs

######################## Methods #######################
# Below, we present 2 inversion methods that give comparable
# results and use different options of the Displacement_strain_planet
# package.

######################## Iterative method #######################
# The South Polar cap thickness is given without flexure. We thus
# need to iterate to update the load as a function of flexure. Below,
# we interate until the maximum flexure difference between subsequent
# iteration becomes negligible.
iter = 0
iter_out = 200
residuals = 1e10
residuals_min = 5  # Minimum residual (m) to exit the iterative process
while (residuals > residuals_min) and (iter < iter_out):
    iter += 1
    if iter == 1:
        # Here we call Thin_shell_matrix_nmax which is going to build and output the inversion
        # matrix ('sols') together with the first flexure solution. The 'sols' will be then used
        # in the faster Thin_shell_matrix below to interate flexure until convergence
        out = Thin_shell_matrix_nmax(
            *args_param_m,
            dc_lm=zeros.copy(),  # No crustal root variations
            drhom_lm=zeros.copy(),  # No internal density variations
            H_lm=thick.copy(),
            nmax=1  # Mass-sheet approximation
        )
        w_deflec1 = out[0]  # w_lm
        sols = out[-1]  # matrix solutions

        # In Thin_shell_matrix_nmax, the flexure coefficients are referenced to R (mean pl. rad.), # so we remove R. MakeGridDH is a SHTOOLs routine that expands spherical harmonic
        # coefficients to a 2D grid.
        min1 = R - np.min(MakeGridDH(w_deflec1, **args_expand))
    else:
        # Here we set first_inv to false given that the inversion matrix has already been built
        # above and input the 'sols' obtained above
        out = Thin_shell_matrix(
            *args_param_m2,
            first_inv=False,
            lambdify_func=sols,
            dc_lm=zeros.copy(),
            drhom_lm=zeros.copy(),
            H_lm=(thick - w_deflec1).copy()  # Update thickness
        )
        w_deflec1 = out[0]  # w_lm

        # In Thin_shell_matrix, the flexure coefficients are referenced to 0
        min2 = -np.min(MakeGridDH(w_deflec1, **args_expand))

        # Criterion for residuals
        residuals = np.abs(min1 - min2)
        print(
            "Iteration %s, maximum flexure %.3f km, residuals %.3f km"
            % (iter, min2 / 1e3, residuals / 1e3)
        )
        min1 = min2
print("Iteration method, maximum flexure %.3f km" % (min2 / 1e3))

############## Single step method with add_equation #############
# The South Polar cap thickness is given without flexure. We thus
# use the add_equation option to tell the model that the 'thick'
# coefficients are equal to topography (H_lm) without flexure (w_lm),
# and redesign the load equation.
# The equation has to be written in symbolic math and respect the
# Displacement_strain_planet nomenclature.

# The 'thick' coeffs are input using the add_arrays option and are
# expressed as add_array1 in the package's convention.
# The input equation is thus thick = 'add_array1 = (H_lm + w_lm)'
# or 'add_array1 - (H_lm + w_lm)' in the package's left-handside only
# equation convention.
out = Thin_shell_matrix_nmax(
    *args_param_m,
    dc_lm=zeros.copy(),  # No crustal root variations
    drhom_lm=zeros.copy(),  # No internal density variations
    add_equation="add_array1 - (H_lm + w_lm)",
    add_arrays=thick.copy(),
    iterate=False,
    nmax=1  # Mass-sheet approximation
)
w_deflec2 = out[0]  # w_lm
min2 = R - np.min(MakeGridDH(w_deflec2, **args_expand))
print("Single step method, maximum flexure %.3f km" % (min2 / 1e3))

###################### Plotting the results #####################
# Set degree-0 flexure SHCoeffs to zero
w_deflec1[0, 0, 0] = 0
w_deflec2[0, 0, 0] = 0

f, (ax1, ax2) = plt.subplots(1, 2)
# Use SHTOOLs to plot
(pysh.SHCoeffs.from_array(w_deflec1 / 1e3).expand(lmax=2 * lmax, lmax_calc=lmax)).plot(
    ax=ax1, cb_label="Flexure (km)", ticks="wSnE", ylabel=None, show=False, **args_plot
)
(pysh.SHCoeffs.from_array(w_deflec2 / 1e3).expand(lmax=2 * lmax, lmax_calc=lmax)).plot(
    ax=ax2, cb_label="Flexure (km)", ticks="wSnE", ylabel=None, show=False, **args_plot
)
plt.show()
