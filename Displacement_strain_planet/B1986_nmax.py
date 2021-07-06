"""
Functions for calculating the Banerdt (1986) system of equations.
"""

import re
import numpy as np
import pyshtools as pysh
from sympy import linsolve, lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr

# ==== corr_nmax_drho ====


def corr_nmax_drho(
    dr_lm,
    drho_lm,
    shape_grid,
    rho_grid,
    lmax,
    mass,
    nmax,
    drho,
    R,
    c=0,
    density_var=False,
    filter_in=None,
    filter=None,
    filter_half=None,
):
    """
    Calculate the difference in gravitational exterior
    to relief referenced to a spherical interface
    (with or without laterally varying density)
    between the mass-sheet case and when using the
    finite amplitude algorithm of Wieczorek &
    Phillips (1998).

    Returns
    -------
    array, size of input dr_lm
        Array with the spherical harmonic coefficients of the
        difference between the mass-sheet and finite-ampltiude, with
        or without lateral density variations.

    Parameters
    ----------
    dr_lm : array, size (2,lmax+1,lmax+1)
        Array with spherical harmonic coefficients of the relief.
    drho_lm : array, size (2,lmax+1,lmax+1)
        Array with spherical harmonic coefficients of the lateral density contrast.
    shape_grid : array, size (2,2*(lmax+1),2*2(lmax+1))
        Array with a grid of the relief.
    rho_grid : array, size (2,2*(lmax+1),2*2(lmax+1))
        Array with a grid of the lateral density contrast.
    lmax : int
        Maximum spherical harmonic degree to compute for the derivatives.
    mass : float
        Mass of the planet.
    nmax : int
        Order of the finite-amplitude correction.
    drho : float
        Mean density contrast.
    R : float
        Mean radius of the planet.
    c : float, default = 0
        Maximum depth at which the density variations occurs.
    density_var : boolean, default = False
        If True, correct for density variations.
    filter_in : array, size(lmax+1), default = None.
        Array with the input filter to use.
    filter : string, default = None
        If 'Ma' or 'Mc', apply minimum-amplitude or minimum-curvature filtering.
    filter_half : int, default = None
        Spherical harmonic degree at which the filter equals 0.5.
    """
    # Finite-amplitude correction.
    MS_lm_nmax = pysh.SHCoeffs.from_zeros(lmax).coeffs
    # This is the computation in Thin_shell_matrix.
    for l in range(1, lmax + 1):
        MS_lm_nmax[:, l, : l + 1] = drho * dr_lm[:, l, : l + 1] / (2 * l + 1)
    MS_lm_nmax *= 4.0 * np.pi / mass

    if nmax != 1:
        # This is the correct calculation with finite-amplitude
        FA_lm_nmax, D = pysh.gravmag.CilmPlusRhoHDH(
            shape_grid, nmax, mass, rho_grid, lmax=lmax
        )
        MS_lm_nmax *= D ** 2
    else:
        FA_lm_nmax = MS_lm_nmax

    # Density contrast in the relief correction.
    if density_var:
        FA_lm_drho, D = pysh.gravmag.CilmPlusRhoHDH(
            shape_grid, 1, mass, rho_grid, lmax=lmax
        )
        MS_lm_drho = MS_lm_nmax
        if filter_in is not None:
            for l in range(1, lmax + 1):
                MS_lm_drho[:, l, : l + 1] /= filter_in[l]
                FA_lm_drho[:, l, : l + 1] /= filter_in[l]
        elif filter is not None:
            for l in range(1, lmax + 1):
                MS_lm_drho[:, l, : l + 1] /= DownContFilter(
                    l, filter_half, R, R - c, type=filter
                )
                FA_lm_drho[:, l, : l + 1] /= DownContFilter(
                    l, filter_half, R, R - c, type=filter
                )
        if nmax == 1:
            MS_lm_drho *= D ** 2

        # Divide because the thin-shell code multiplies by
        # density contrast, to correct for finite-amplitude.
        # Here we also correct for density variations, so the
        # correction is already scaled by the density contrast.
        delta_MS_FA = R * (FA_lm_drho - MS_lm_drho + FA_lm_nmax - MS_lm_nmax) / drho
    else:
        delta_MS_FA = R * (FA_lm_nmax - MS_lm_nmax)

    return delta_MS_FA


# ==== Thin_shell_matrix ====


def Thin_shell_matrix(
    g0,
    R,
    c,
    Te,
    rhom,
    rhoc,
    rhol,
    rhobar,
    lmax,
    E,
    v,
    base_drho=50e3,
    top_drho=0,
    filter_in=None,
    filter=None,
    filter_half=None,
    H_lm=None,
    drhom_lm=None,
    dc_lm=None,
    w_lm=None,
    omega_lm=None,
    q_lm=None,
    G_lm=None,
    Gc_lm=None,
    add_equation=None,
    add_array=None,
    quiet=False,
    remove_equation=None,
    w_corr=None,
    wdc_corr=None,
    H_corr=None,
    lambdify_func=None,
    first_inv=True,
    drho_corr=None,
):
    """
    Solve for the Banerdt et al. (1986) system of equations with
    the possibility to account for finite-amplitude corrections
    and lateral density variations with the surface topography or
    moho relief.

    Returns
    -------
    w_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        upward displacement.
    A_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        poloidal term of the tangential displacement.
    moho_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        moho-relief.
    dc_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        isostatic crustal root variations.
    drhom_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        lateral density variations.
    omega_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        tangential load potential.
    q_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        net load on the lithosphere.
    Gc_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        geoid at the moho depth.
    G_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        geoid at the surface.
    H_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        surface topography.
    lambdify_func : array, size(2,lmax+1,lmax+1)
        Array with the lambda functions (size lmax+1) of all components.
        Lambda functions can be used to re-calculate the same problem with different inputs very fast.

    Parameters
    ----------
    g0 : float
        Gravitational attraction at the surface.
    R : float
        Mean radius of the planet.
    c : float
        Average crustal thickness.
    Te : float
        Elastic thickness of the lithosphere.
    rhom : float
        Density of the mantle.
    rhoc : float
        Density of the crust.
    rhol : float
        Density of the surface topography.
    rhobar : float
        Mean density of the planet.
    lmax : int
        Maximum spherical harmonic degree of calculations.
    E : float
        Young's modulus.
    v : float
        Poisson's ratio.
    base_drho : float, default = 50e3
        Lower depth for the of the density contrast.
    top_drho : float, default = 0
        Upper depth for the of the density contrast.
    filter_in : array, size(lmax+1), default = None.
        Array with the input filter to use.
    filter : string, default = None
        If 'Ma' or 'Mc', apply minimum-amplitude or minimum-curvature filtering.
    filter_half : int, default = 50
        Spherical harmonic degree at which the filter equals 0.5.
    H_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        surface topography.
    drhom_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        lateral density variations.
    dc_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        isostatic crustal root variations.
    w_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        upward displacement.
    omega_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        tangential load potential.
    q_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        net load on the lithosphere.
    G_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        geoid at the surface.
    Gc_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        geoid at the moho depth.
    add_equation : string, default = None
        Equation to be added to the system. This must include at least
        one of the 8 parameters aboves.
    add_array : array size(2,lmax+1,lmax+1), default = None
        Array of spherical harmonics to be added in 'add_equation', which
        is written 'add_array'.
    quiet : boolean, default = False
        if True, print various outputs.
    remove_equation : string, default = None
        String of the equation to be removed. This must be either
        'G_lm', 'Gc_lm', 'w_lm', 'omega_lm', or 'q_lm'.
    w_corr : array size(2,lmax+1,lmax+1), default = None
        Array with spherical harmonic coefficients for finite-amplitude
        and or lateral density variations corrections of the w_lm relief.
    wdc_corr : array size(2,lmax+1,lmax+1), default = None
        Array with spherical harmonic coefficients for finite-amplitude
        and or lateral density variations corrections of the moho_lm relief.
    H_corr : array size(2,lmax+1,lmax+1), default = None
        Array with spherical harmonic coefficients for finite-amplitude
        and or lateral density variations corrections of the H_lm relief.
    lambdify_func : array size(lmax+1), default = None
        Reuse the lambidfy functions of the first run.
    first_inv : boolean, default = True
        If True, the code assumes that this is the first time doing
        the inversion in this setup, and will store the lambdify results
        in 'lambdify_func'
    drho_corr : array size(2,lmax+1,lmax+1), default = None
          Array with spherical harmonic coefficients for lateral
          lateral density variations corrections for omega_lm.
    """
    # Declare all possible input arrays.
    input_arrays = np.array(
        [w_lm, Gc_lm, q_lm, omega_lm, dc_lm, drhom_lm, G_lm, H_lm], dtype=object
    )
    input_constraints = np.array(
        ["w_lm", "Gc_lm", "q_lm", "omega_lm", "dc_lm", "drhom_lm", "G_lm", "H_lm"]
    )

    equation_order = np.array(["G_lm", "Gc_lm", "q_lm", "w_lm", "omega_lm"])

    # Perform some initial checks with input arrays to determine
    # what are the unknown and if a sufficient number of arrays
    # has been input.

    # Number of input arrays
    num_array_test = np.not_equal(
        np.array([type(arr) for arr in input_arrays]), type(None)
    )
    # Size of input arrays
    size_array_test = np.array([np.size(arr) for arr in input_arrays])
    # Input arrays
    constraint_test = input_constraints[num_array_test]
    # Other arrays
    not_constraint = input_constraints[~num_array_test]

    if lmax < 0:
        raise ValueError(
            "lmax must be greater or equal to 0. "
            + "Input value was {:s}.".format(repr(lmax))
        )

    if lmax > np.sqrt(np.max(size_array_test) / 2) - 1:
        raise ValueError(
            "lmax must be less or equal to %s. Input "
            % (np.sqrt(np.max(size_array_test) / 2) - 1)
            + "value was %s." % (lmax + 1)
        )

    if filter_in is not None and np.size(filter_in) != lmax + 1:
        raise ValueError(
            "Size of filter_in must be %s. Input " % (lmax + 1)
            + "size was %s." % (np.size(filter_in))
        )
    # The system is a total of 5 equations relating 8 unknowns.
    # If an additional equation is given, 2 arrays must be input
    # to find a solution.
    if add_equation is not None:
        if remove_equation is None:
            if np.sum(num_array_test) != 2:
                raise ValueError(
                    "Must input 2 arrays between %s." % (input_constraints)
                    + "\nNumber of input"
                    + " arrays was {:s}. ".format(repr(np.sum(num_array_test)))
                    + "Input arrays are %s." % (constraint_test)
                )
        else:
            if np.sum(num_array_test) != 3:
                raise ValueError(
                    "Must input 3 arrays between %s." % (input_constraints)
                    + "\nNumber of input"
                    + " arrays was {:s}. ".format(repr(np.sum(num_array_test)))
                    + "Input arrays are %s." % (constraint_test)
                )
        if "=" in add_equation:
            raise ValueError(
                "All terms of the added equation must be"
                + " on the same side, and there is no need to specify = 0"
                + ",the input equation is %s. " % (add_equation)
            )

        if all(sym not in add_equation for sym in input_constraints):
            raise ValueError(
                "The input equation must relate any of the 8"
                + " unknown arrays that are %s." % (input_constraints)
                + "\nThe input equation is %s." % (add_equation)
            )
    else:
        if remove_equation is None:
            # If no additional equation is given, 3 arrays must
            # be input to find a solution.
            if np.sum(num_array_test) != 3:
                raise ValueError(
                    "Must input 3 arrays between %s." % (input_constraints)
                    + "\nNumber of input arrays was {:s}. ".format(
                        repr(np.sum(num_array_test))
                    )
                    + "Input arrays are %s." % (constraint_test)
                )
        else:
            if np.sum(num_array_test) != 4:
                raise ValueError(
                    "Must input 4 arrays between %s." % (input_constraints)
                    + "\nNumber of input arrays was {:s}. ".format(
                        repr(np.sum(num_array_test))
                    )
                    + "Input arrays are %s." % (constraint_test)
                )

    if quiet is False:
        print("Input arrays are %s." % (constraint_test))
        print("Solving for %s." % (not_constraint))
        if filter_in is not None:
            print("Use input filter")
        elif filter is not None:
            print(
                "Minimum %s filter" % ("curvature" if filter == "Mc" else "amplitude")
            )
        if first_inv is True:
            print("First inversion, storing lambdify results")
        else:
            print("Using stored solutions with new inputs")

    # Allocate arrays to be used for outputs.
    shape = (2, lmax + 1, lmax + 1)
    if first_inv:
        lambdify_func = np.zeros((lmax + 1), dtype=object)
    if w_lm is None:
        w_lm = np.zeros(shape)
    if Gc_lm is None:
        Gc_lm = np.zeros(shape)
    if q_lm is None:
        q_lm = np.zeros(shape)
    if omega_lm is None:
        omega_lm = np.zeros(shape)
    if drhom_lm is None:
        drhom_lm = np.zeros(shape)
    if G_lm is None:
        G_lm = np.zeros(shape)
    if H_lm is None:
        H_lm = np.zeros(shape)
    if dc_lm is None:
        dc_lm = np.zeros(shape)
    if wdc_corr is None:
        wdc_corr = np.zeros(shape)
        H_corr = np.zeros(shape)
        w_corr = np.zeros(shape)
        drho_corr = np.zeros(shape)
    A_lm = np.zeros(shape)

    # For filtering drhom when sum_dc==0
    sum_dc = np.sum(dc_lm)

    if Te == 0:  # Avoid numerical problems with infinite values
        Te = 1
        print("Elastic thickness set to 1 to avoid numerical problems")

    # Precompute some constants.
    M = base_drho - top_drho  # Thickness of the density anomaly
    Re = R - 0.5 * Te  # Midpoint of the elastic shell.
    Re4 = Re ** 4
    drho = rhom - rhoc
    drhol = rhoc - rhol
    if Te == 0:  # Avoids error printing when dividing by zero.
        psi = np.inf
    else:
        psi = 12.0 * Re ** 2 / Te ** 2
    D = E * Te ** 3 / (12.0 * (1.0 - v ** 2))  # Shell's
    # rigidity.
    v1v = v / (1.0 - v)
    RCR = (R - c) / R

    gmoho = g0 * (1.0 + (((R - c) / R) ** 3 - 1) * rhoc / rhobar) / ((R - c) / R) ** 2
    if top_drho <= c:
        gdrho = (
            g0
            * (1.0 + (((R - top_drho) / R) ** 3 - 1) * rhoc / rhobar)
            / ((R - top_drho) / R) ** 2
        )
    else:
        gdrho = (
            g0
            * (1.0 + (((R - top_drho) / R) ** 3 - 1) * rhom / rhobar)
            / ((R - top_drho) / R) ** 2
        )

    # Store symbolized array names with sympy. Order is 
    # important.
    # These will be denoted e.g. 'H_lm1' for H_lm.
    add_constraints = ""
    if add_array is not None:
        add_constraints = " add_array1"
    add_constraints += " wdc_corr1 w_corr1 H_corr1 drho_corr1"

    a_symb_uknwn = symbols(" ".join([symb + "1 " for symb in not_constraint]))
    a_symb_knwn = symbols(
        " ".join([symb + "1 " for symb in constraint_test]) + add_constraints
    )
    args_symb = (constraint_test, not_constraint, a_symb_uknwn)

    if remove_equation is not None and quiet is False:
        print("Removing equation for %s." % (remove_equation))
    if add_equation is not None:
        if quiet is False:
            if add_array is None:
                print("Adding an additional equation where %s." % (add_equation))
            else:
                print(
                    "Adding an additional equation and array where %s." % (add_equation)
                )
        # Reformat added equation for sympy
        for string in input_constraints:
            add_equation = re.sub(
                r"(\b{}\b)".format(string), "%s" % (string) + "1", add_equation
            )
        if add_array is not None:
            add_equation = re.sub(
                r"(\b{}\b)".format("add_array"),
                "%s" % ("add_array") + "1",
                add_equation,
            )
        add_equation = parse_expr(add_equation)

    # Solve matrix over all degrees.
    for l in range(1, lmax + 1):  # Ignore degree 0 from
        # calculations
        Lapla = float(-l * (l + 1))  # Laplacian identity.

        # Degree-dependent from Banerdt correction after Beuthe
        # (2008).
        if l == 1:
            alpha = 1.0e-20
            gamma = 1.0e-20
            beta = 1.0e-20
            eta = 1.0e-20
            # No displacement for degree-1.
        else:
            alpha = (
                -Re4
                * (Lapla + 1.0 - v)
                / (
                    (D / (1.0 + 1.0 / psi))
                    * (Lapla ** 3 + 4.0 * Lapla ** 2 + 4.0 * Lapla)
                    + Re ** 2 * (E * Te) * (Lapla + 2.0)
                )
            )

            gamma = (Lapla * Re4 * ((1.0 / (1.0 + psi)) * (Lapla + 2.0) - 1 - v)) / (
                (D / (1.0 + 1.0 / psi)) * (Lapla ** 3 + 4.0 * Lapla ** 2 + 4.0 * Lapla)
                + Re ** 2 * (E * Te) * (Lapla + 2.0)
            )

            zeta = (
                (1.0 / (1.0 + psi))
                * (1.0 / (1.0 - v ** 2))
                * (Lapla + 1.0 + v)
                * (Lapla + 2.0)
            )
            beta = zeta * alpha + alpha + Re ** 2 / (E * Te)
            eta = (
                zeta * gamma
                + gamma
                - Re ** 2 / (E * Te * (1.0 + psi)) * (Lapla - psi * (1.0 + v))
            )

        if first_inv is True:
            # Degree & radius -dependent constants for potential
            # upward continuation
            Rl3 = R / float(l + 3)
            rhobconst = 3.0 / (rhobar * float(2 * l + 1))
            RCRl = RCR ** l
            RCRl1 = RCR ** (l + 1)
            RCRl2 = RCR ** (l + 2)

            DCfilter_mohoD = 1.0
            DCfilter_drhomt = 1.0
            DCfilter_drhomb = 1.0
            if filter_in is not None:
                DCfilter_mohoD = filter_in[l]
                if sum_dc == 0:
                    DCfilter_drhomt = filter_in[l]
                    DCfilter_drhomb = filter_in[l]
            elif filter is not None:
                DCfilter_mohoD = DownContFilter(l, filter_half, R, R - c, type=filter)
                if sum_dc == 0:
                    DCfilter_drhomt = DownContFilter(
                        l, filter_half, R, R - top_drho, type=filter
                    )
                    DCfilter_drhomb = DownContFilter(
                        l, filter_half, R, R - base_drho, type=filter
                    )

            if (R - top_drho) <= (R - c):
                RtRCl = ((R - top_drho) / (R - c)) ** l
            else:
                RtRCl = ((R - c) / (R - top_drho)) ** (l + 1)
            if (R - base_drho) <= (R - c):
                RbRCl = ((R - base_drho) / (R - c)) ** l
            else:
                RbRCl = ((R - c) / (R - base_drho)) ** (l + 1)

            RtRCl *= (R - top_drho) ** 3 / ((R - c) * R ** 2)
            RbRCl *= (R - base_drho) ** 3 / ((R - c) * R ** 2)

            RtRl3 = ((R - top_drho) / R) ** (l + 3)
            RbRl3 = ((R - base_drho) / R) ** (l + 3)

            # Symbolic definition.
            w_lm1, Gc_lm1, q_lm1, omega_lm1, dc_lm1, drhom_lm1, G_lm1, H_lm1 = symbols(
                " w_lm1 Gc_lm1 q_lm1" + " omega_lm1 dc_lm1 drhom_lm1 G_lm1 H_lm1 "
            )

            if add_array is not None:
                add_array1 = symbols(" add_array1 ")

            wdc_corr1, w_corr1, H_corr1, drho_corr1 = symbols(
                " wdc_corr1 w_corr1 H_corr1 drho_corr1"
            )

            # System of equations from Banerdt (1986).
            Eqns = [
                -G_lm1
                + rhobconst
                * (
                    rhol * H_lm1
                    + drhol * w_lm1
                    + drho * (w_lm1 - dc_lm1) * RCRl2 / DCfilter_mohoD
                    + drhom_lm1
                    * Rl3
                    * (RtRl3 / DCfilter_drhomt - RbRl3 / DCfilter_drhomb)
                )
                + rhol * H_corr1
                + drhol * w_corr1
                + drho * wdc_corr1 * RCRl,
                -Gc_lm1
                + rhobconst
                * (
                    (rhol * H_lm1 + drhol * w_lm1) * RCRl1
                    + drho * (w_lm1 - dc_lm1) * RCR ** 3
                    + drhom_lm1 * Rl3 * (RtRCl - RbRCl)
                )
                + (rhol * H_corr1 + drhol * w_corr1) * RCRl1
                + drho * wdc_corr1 * RCR ** 3,
                -q_lm1
                + g0 * (rhol * (H_lm1 - G_lm1) + drhol * w_lm1)
                + gmoho * drho * (w_lm1 - dc_lm1 - Gc_lm1)
                + gdrho * drhom_lm1 * M,
                -w_lm1 + alpha * q_lm1 + gamma * omega_lm1,
                -omega_lm1
                + v1v * rhol * g0 * Te * H_lm1 / R
                - (
                    drhol * g0 * v1v * Te
                    + rhoc * gmoho * (v1v * Te - c)
                    - rhom * gmoho * (Te - c)
                )
                * w_lm1
                / R
                - v1v * drho * gmoho * (Te - c) * dc_lm1 / R
                - 0.5
                * v1v
                * drhom_lm1
                * gdrho
                * (Te - top_drho)
                * np.min([M, Te - c])
                / R
                + drho_corr1,
            ]

            if add_equation is not None:
                Eqns.insert(len(Eqns), add_equation)

            if remove_equation is not None:
                for item in [remove_equation]:
                    Eqns.pop(int(np.where(equation_order == item)[0]))

            # Rearange system of equations using sympy.
            sol = linsolve(Eqns, a_symb_uknwn + a_symb_knwn)

            # Vectorize the linsolve function.
            linsolve_vector = lambdify(a_symb_uknwn + a_symb_knwn, list(sol))
            if first_inv:  # Store matrix solution for potential
                # Reutilisation later
                lambdify_func[l] = linsolve_vector
        else:
            linsolve_vector = lambdify_func[l]

        # Depending on the input arrays, pass a symbol or the
        # input values.
        H_lm1 = constraint_test_symb("H_lm", H_lm[:, l, : l + 1], *args_symb)
        G_lm1 = constraint_test_symb("G_lm", G_lm[:, l, : l + 1], *args_symb)
        Gc_lm1 = constraint_test_symb("Gc_lm", Gc_lm[:, l, : l + 1], *args_symb)
        q_lm1 = constraint_test_symb("q_lm", q_lm[:, l, : l + 1], *args_symb)
        omega_lm1 = constraint_test_symb(
            "omega_lm", omega_lm[:, l, : l + 1], *args_symb
        )
        dc_lm1 = constraint_test_symb("dc_lm", dc_lm[:, l, : l + 1], *args_symb)
        drhom_lm1 = constraint_test_symb(
            "drhom_lm", drhom_lm[:, l, : l + 1], *args_symb
        )
        w_lm1 = constraint_test_symb("w_lm", w_lm[:, l, : l + 1], *args_symb)

        # Results.
        if add_array is not None:
            outs = np.concatenate(
                np.array(
                    linsolve_vector(
                        w_lm1=w_lm1,
                        Gc_lm1=Gc_lm1,
                        G_lm1=G_lm1,
                        H_lm1=H_lm1,
                        q_lm1=q_lm1,
                        omega_lm1=omega_lm1,
                        dc_lm1=dc_lm1,
                        drhom_lm1=drhom_lm1,
                        add_array1=add_array[:, l, : l + 1],
                        wdc_corr1=wdc_corr[:, l, : l + 1],
                        H_corr1=H_corr[:, l, : l + 1],
                        w_corr1=w_corr[:, l, : l + 1],
                        drho_corr1=drho_corr[:, l, : l + 1],
                    ),
                    dtype=object,
                )
            )
        else:
            outs = np.concatenate(
                np.array(
                    linsolve_vector(
                        w_lm1=w_lm1,
                        Gc_lm1=Gc_lm1,
                        G_lm1=G_lm1,
                        H_lm1=H_lm1,
                        q_lm1=q_lm1,
                        omega_lm1=omega_lm1,
                        dc_lm1=dc_lm1,
                        drhom_lm1=drhom_lm1,
                        wdc_corr1=wdc_corr[:, l, : l + 1],
                        H_corr1=H_corr[:, l, : l + 1],
                        w_corr1=w_corr[:, l, : l + 1],
                        drho_corr1=drho_corr[:, l, : l + 1],
                    ),
                    dtype=object,
                )
            )

        # Determine how symbols are listed in the outputs because
        # solutions order depends on the input symbol order,
        # which depends on the user inputs.
        a_symbs = np.array(a_symb_uknwn + a_symb_knwn).astype("str")
        idx_w_lm = int(np.where(a_symbs == "w_lm1")[0])
        idx_G_lm = int(np.where(a_symbs == "G_lm1")[0])
        idx_Gc_lm = int(np.where(a_symbs == "Gc_lm1")[0])
        idx_H_lm = int(np.where(a_symbs == "H_lm1")[0])
        idx_omega_lm = int(np.where(a_symbs == "omega_lm1")[0])
        idx_drhom_lm = int(np.where(a_symbs == "drhom_lm1")[0])
        idx_dc_lm = int(np.where(a_symbs == "dc_lm1")[0])
        idx_q_lm = int(np.where(a_symbs == "q_lm1")[0])

        # Write solutions
        w_lm[:, l, : l + 1] = outs[idx_w_lm]
        Gc_lm[:, l, : l + 1] = outs[idx_Gc_lm]
        q_lm[:, l, : l + 1] = outs[idx_q_lm]
        omega_lm[:, l, : l + 1] = outs[idx_omega_lm]
        dc_lm[:, l, : l + 1] = outs[idx_dc_lm]
        drhom_lm[:, l, : l + 1] = outs[idx_drhom_lm]
        G_lm[:, l, : l + 1] = outs[idx_G_lm]
        H_lm[:, l, : l + 1] = outs[idx_H_lm]

        # Tangential displacement
        A_lm[:, l, : l + 1] = beta * q_lm[:, l, : l + 1] + eta * omega_lm[:, l, : l + 1]

    return (
        w_lm,
        A_lm,
        w_lm - dc_lm,
        dc_lm,
        drhom_lm,
        omega_lm,
        q_lm,
        Gc_lm,
        G_lm,
        H_lm,
        lambdify_func,
    )


# ==== constraint_test_symb ====


def constraint_test_symb(str_symb, arr, constraint_test, not_constraint, arr_symb):
    """
    This function return None or the input array depending on
    the input constraints in Thin_shell_matrix.

    Returns
    -------
      array, size of input arr or None
        Input array or None

    Parameters
    ----------
    str_symb : string
        String of the investigated symbol.
    arr : array, size (2,lmax+1,lmax+1)
        Array with spherical harmonic coefficients of the input array.
    constraint_test : list of strings, size variable
        List of input constraints (i.e., 'G_lm', 'drhom_lm'...).
    not_constraint : list of strings, size variable
        List of strings that are not input constraints (i.e., 'Gc_lm').
    arr_symb : list of sympy symbols
       List of all sympy symbols.
    """
    if str_symb in constraint_test:
        out = arr
    else:
        out = arr_symb[int(np.where(not_constraint == str_symb)[0])]

    return out


# ==== DownContFilter ====


def DownContFilter(l, half, R_ref, D_relief, type="Mc"):
    """
    Compute the downward minimum-amplitude or
    -curvature filter of Wieczorek & Phillips,
    (1998).

    Returns
    -------
    float
        Value of the filter at degree l

    Parameters
    ----------
    l : int
        The spherical harmonic degree.
    half : int
        The spherical harmonic degree where the filter is equal to 0.5.
    R_ref : float
        The reference radius of the gravitational field.
    D_relief : float
        The radius of the surface to downward continue to.
    type : string, default = "Mc"
        Filter type, minimum amplitude ("Ma") of curvature ("Mc")
    """
    if half == 0:
        DownContFilter = 1.0
    else:
        if type == "Mc":
            tmp = 1.0 / (
                float(half * half + half)
                * (float(2 * half + 1) * (R_ref / D_relief) ** half) ** 2
            )
            DownContFilter = (
                1.0
                + tmp
                * float(l * l + l)
                * (float(2 * l + 1) * (R_ref / D_relief) ** l) ** 2
            )
        elif type == "Ma":
            tmp = 1.0 / (float(2.0 * half + 1.0) * (R_ref / D_relief) ** half) ** 2
            DownContFilter = (
                1.0 + tmp * (float(2 * l + 1) * (R_ref / D_relief) ** l) ** 2
            )
        else:
            raise ValueError(
                "Error in DownContFilter, filter type must be either 'Ma' "
                + "or 'Mc' Input value was {:s}.".format(repr(type))
            )
    DownContFilter = 1.0 / DownContFilter

    return DownContFilter


# ==== Thin_shell_matrix_nmax ====


def Thin_shell_matrix_nmax(
    g0,
    R,
    c,
    Te,
    rhom,
    rhoc,
    rhol,
    lmax,
    E,
    v,
    mass,
    filter_in=None,
    filter=None,
    filter_half=50,
    nmax=5,
    H_lm=None,
    drhom_lm=None,
    dc_lm=None,
    w_lm=None,
    omega_lm=None,
    q_lm=None,
    G_lm=None,
    Gc_lm=None,
    C_lm=None,
    add_equation=None,
    add_array=None,
    quiet=True,
    remove_equation=None,
    base_drho=150e3,
    top_drho=50e3,
    delta_max=5,
    iter_max=250,
    delta_out=500e3,
):
    """
    Solve the Banerdt (1986) system of 5 equations
    with finite-amplitude correction and accounting
    for the potential presence of density variations
    within the surface or moho reliefs.

    Returns
    -------
    w_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        upward displacement.
    A_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        poloidal term of the tangential displacement.
    moho_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        moho-relief.
    dc_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        isostatic crustal root variations.
    drhom_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        lateral density variations.
    omega_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        tangential load potential.
    q_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        net load on the lithosphere.
    Gc_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        geoid at the moho depth.
    G_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        geoid at the surface.
    H_lm : array, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        surface topography.
    lambdify_func : array, size(2,lmax+1,lmax+1)
        Array with the lambda functions (size lmax+1) of all components.
        Lambda functions can be used to re-calculate the same problem with different inputs very fast.

    Parameters
    ----------
    g0 : float
        Gravitational attraction at the surface.
    R : float
        Mean radius of the planet.
    c : float
        Average crustal thickness.
    Te : float
        Elastic thickness of the lithosphere.
    rhom : float
        Density of the mantle.
    rhoc : float
        Density of the crust.
    rhol : float
        Density of the surface topography.
    lmax : int
        Maximum spherical harmonic degree of calculations.
    E : float
        Young's modulus.
    v : float
        Poisson's ratio.
    mass : float
        Mass of the planet.
    filter_in : array, size(lmax+1), default = None.
        Array with the input filter to use.
    filter : string, default = None
        If 'Ma' or 'Mc', apply minimum-amplitude or minimum-curvature filtering.
    filter_half : int, default = 50
        Spherical harmonic degree at which the filter equals 0.5.
    nmax : int, default = 5
        Maximum order of the finite-amplitude correction.
    H_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        surface topography.
    drhom_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        lateral density variations.
    dc_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        isostatic crustal root variations.
    w_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        upward displacement.
    omega_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        tangential load potential.
    q_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        net load on the lithosphere.
    G_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        geoid at the surface.
    Gc_lm : array, size(2,lmax+1,lmax+1), default = None
        Array with the spherical harmonic coefficients of the
        geoid at the moho depth.
    add_equation : string, default = None
        Equation to be added to the system. This must include at least
        one of the 8 parameters aboves.
    add_array : array size(2,lmax+1,lmax+1), default = None
        Array of spherical harmonics to be added in 'add_equation', which
        is written 'add_array'.
    quiet : boolean, default = False
        if True, print various outputs.
    remove_equation : string, default = None
        String of the equation to be removed. This must be either
        'G_lm', 'Gc_lm', 'w_lm', 'omega_lm', or 'q_lm'.
    base_drho : float, default = 150e3
        Lower depth for the of the density contrast.
    top_drho : float, default = 50e3
        Upper depth for the of the density contrast.
    delta_max : float, default = 5
        The algorithm will continue to iterate until the maximum difference in
        relief (or density contrast) between solutions is less than this
        value (in meters or kg m-3).
    iter_max : int, default = 250
        Maximum number of iterations before the algorithm stops.
    delta_out : float, default = 500e3
        If the delta is larger than this value, the algorithm stops
        and prints that it is not converging.
    """
    rhobar = mass * 3.0 / 4.0 / np.pi / R ** 3
    args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax, E, v)
    args_param_lm = dict(
        H_lm=H_lm,
        drhom_lm=drhom_lm,
        dc_lm=dc_lm,
        w_lm=w_lm,
        omega_lm=omega_lm,
        q_lm=q_lm,
        G_lm=G_lm,
        Gc_lm=Gc_lm,
        base_drho=base_drho,
        top_drho=top_drho,
        filter_in=filter_in,
        filter=filter,
        filter_half=filter_half,
        add_array=add_array,
        remove_equation=remove_equation,
        add_equation=add_equation,
        quiet=quiet,
    )

    lmaxgrid = 4 * lmax
    args_grid = dict(grid="DH2", lmax=lmaxgrid, extend=False, lmax_calc=lmax)

    # Precompute some sums that will be used later for checks
    sum_dc = np.sum(dc_lm)
    sum_w = np.sum(w_lm)
    sum_drho = np.sum(drhom_lm)

    # Density contrast not at topography or moho and no
    # finite-amplitude correctio, return
    if nmax == 1 and top_drho != 0 and base_drho != c:
        (
            w_lm_o,
            A_lm_o,
            moho_relief_lm_o,
            dc_lm_o,
            drhom_lm_o,
            omega_lm_o,
            q_lm_o,
            Gc_lm_o,
            G_lm_o,
            H_lm_o,
            lambdify_func_o,
        ) = Thin_shell_matrix(*args_param_m, **args_param_lm)

        if quiet is False:
            print("Returning without finite-amplitude corrections")
            print("Set the interfaces degree-0 coefficients")
        w_lm_o[0, 0, 0] = R
        dc_lm_o[0, 0, 0] = c
        moho_relief_lm_o[0, 0, 0] = R - c
        H_lm_o[0, 0, 0] = R

        return (
            w_lm_o,
            A_lm_o,
            moho_relief_lm_o,
            dc_lm_o,
            drhom_lm_o,
            omega_lm_o,
            q_lm_o,
            Gc_lm_o,
            G_lm_o,
            H_lm_o,
            lambdify_func_o,
        )

    else:
        # Correct for density contrast in surface or moho
        # relief, and/or finite-amplitude correction
        density_var_H, density_var_dc, density_var = False, False, False
        if drhom_lm is None or sum_drho != 0:
            density_var = True  # Variations in density
            if quiet is False:
                print(
                    "Iterate to account for density"
                    + " variations %s"
                    % (
                        "and finite-amplitude correction, nmax is %i" % (nmax)
                        if nmax > 1
                        else ""
                    )
                )
            if top_drho == 0:
                # Correct for density variations in the surface
                # relief
                density_var_H = True
            if base_drho == c:
                # Correct for density variations in the moho
                # relief
                density_var_dc = True
        else:
            if quiet is False:
                print("Iterate for finite-amplitude correction, nmax is %i" % (nmax))

        # If only finite-amplitude correction, density
        # contrast is multipled in the thin-shell code
        # we set it to 1. This will be changed later if required.
        ones = np.ones((2 * (lmaxgrid + 1), 2 * (2 * (lmaxgrid + 1))))
        H_drho_grid, w_drho_grid, wdc_drho_grid = ones, ones, ones
        drho_H, drho_wdc, drho_w = 1.0, 1.0, 1.0

        if drhom_lm is not None and sum_drho != 0:
            rho_grid = pysh.SHCoeffs.from_array(drhom_lm).expand(**args_grid).data
            rhoc = drhom_lm[0, 0, 0]
            rhol = drhom_lm[0, 0, 0]
            args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax, E, v)
            if quiet is False:
                print(
                    "rhol and rhoc are set to the mean input density variations %.2f kg m-3"
                    % (rhoc)
                )

        # Geoid correction due to density variations
        # and or finite-amplitude corrections.
        # Moho relief
        delta_wdc_geoid = pysh.SHCoeffs.from_zeros(lmax).coeffs
        # Deflected topography relief
        delta_w_geoid = pysh.SHCoeffs.from_zeros(lmax).coeffs
        # Surface topography relief
        delta_H_geoid = pysh.SHCoeffs.from_zeros(lmax).coeffs
        # Tangential load potential corrections due to density
        # variations at the reliefs
        drho_corr = pysh.SHCoeffs.from_zeros(lmax).coeffs

        precomp_H_grid, precomp_w_grid, precomp_dc_grid = False, False, False
        # Precompute grids
        if H_lm is not None:
            precomp_H_grid = True
            H_lm[0, 0, 0] = R
            H_grid = pysh.SHCoeffs.from_array(H_lm).expand(**args_grid).data
        if w_lm is not None and rhoc != rhol:
            precomp_w_grid = True
            if sum_w == 0:
                w_grid = ones * R
            else:
                w_lm[0, 0, 0] = R
                w_grid = pysh.SHCoeffs.from_array(w_lm).expand(**args_grid).data
        if w_lm is not None and dc_lm is not None:
            precomp_dc_grid = True
            wdc_lm = w_lm - dc_lm
            if sum_w == 0 and sum_dc == 0:
                wdc_grid = ones * R - c
            else:
                wdc_lm[0, 0, 0] = R - c
                wdc_grid = pysh.SHCoeffs.from_array(wdc_lm).expand(**args_grid).data

        # Error messages if iteration not converging
        var_unit = "km"
        var_relief = "Moho relief"
        if sum_dc == 0 and sum_w == 0:
            var_relief = "Grid density"
            var_unit = "kg m-3"
        elif sum_dc == 0:
            var_relief = "Flexure relief"

        lambdify_func_o = None
        first_inv = True
        delta = 1.0e9
        iter = 0
        # Iterate until convergence
        # First guess is using the mass-sheet case
        while delta > delta_max:
            iter += 1
            (
                w_lm_o,
                A_lm_o,
                moho_relief_lm_o,
                dc_lm_o,
                drhom_lm_o,
                omega_lm_o,
                q_lm_o,
                Gc_lm_o,
                G_lm_o,
                H_lm_o,
                lambdify_func_o,
            ) = Thin_shell_matrix(
                *args_param_m,
                **args_param_lm,
                wdc_corr=delta_wdc_geoid,
                w_corr=delta_w_geoid,
                H_corr=delta_H_geoid,
                first_inv=first_inv,
                lambdify_func=lambdify_func_o,
                drho_corr=drho_corr
            )
            first_inv = False

            # Scheme proposed in Wieczorek+(2013) SOM eq 21, 22
            # to speed up convergence delta(i+3) = (delta(i+2) +
            # delta(i+1))/2.
            if iter % 3 == 0:
                delta_wdc_geoid = (delta_wdc_geoid_2 + delta_wdc_geoid_1) / 2.0
                delta_H_geoid = (delta_H_geoid_2 + delta_H_geoid_1) / 2.0
                delta_w_geoid = (delta_w_geoid_2 + delta_w_geoid_1) / 2.0
                if quiet is False:
                    print(
                        "Skipping iteration %s, with convergence" % (iter) + " scheme"
                    )
                continue

            if density_var:
                rho_grid = pysh.SHCoeffs.from_array(drhom_lm_o).expand(**args_grid).data
                if drhom_lm is not None and sum_drho != 0:
                    rhoc = drhom_lm_o[0, 0, 0]
                    rhol = drhom_lm_o[0, 0, 0]
                    args_param_m = (
                        g0,
                        R,
                        c,
                        Te,
                        rhom,
                        rhoc,
                        rhol,
                        rhobar,
                        lmax,
                        E,
                        v,
                    )

                v1v = v / (1.0 - v)
                if density_var_H:
                    drho_corr += v1v * drhom_lm_o * g0 * Te * H_lm_o / R
                    drho_H = rhol
                    if drhom_lm is not None and sum_drho != 0:
                        H_drho_grid = rho_grid
                    else:
                        H_drho_grid = rho_grid + rhol
                if density_var_dc:
                    gmoho = (
                        g0
                        * (1.0 + (((R - c) / R) ** 3 - 1) * rhoc / rhobar)
                        / ((R - c) / R) ** 2
                    )
                    drho_corr += v1v * drhom_lm_o * gmoho * (Te - c) * dc_lm_o / R
                    drho_wdc = rhom - rhoc
                    if drhom_lm is not None and sum_drho != 0:
                        wdc_drho_grid = rhom - rho_grid
                    else:
                        wdc_drho_grid = rhom - (rhoc + rho_grid)

            H_lm_o[0, 0, 0] = R
            if not precomp_H_grid:
                H_grid = pysh.SHCoeffs.from_array(H_lm_o).expand(**args_grid).data
            delta_H_geoid = corr_nmax_drho(
                H_lm_o,
                drhom_lm_o,
                H_grid,
                H_drho_grid,
                lmax,
                mass,
                nmax,
                drho_H,
                R,
                density_var=density_var_H,
            )

            w_lm_o[0, 0, 0] = R
            if not precomp_w_grid:
                w_grid = pysh.SHCoeffs.from_array(w_lm_o).expand(**args_grid).data
            if rhoc != rhol:
                delta_w_geoid = corr_nmax_drho(
                    w_lm_o,
                    drhom_lm_o,
                    w_grid,
                    w_drho_grid,
                    lmax,
                    mass,
                    nmax,
                    drho_w,
                    R,
                )

            wdc_lm_o = w_lm_o - dc_lm_o
            wdc_lm_o[0, 0, 0] = R - c
            if not precomp_dc_grid:
                wdc_grid = pysh.SHCoeffs.from_array(wdc_lm_o).expand(**args_grid).data
            delta_wdc_geoid = corr_nmax_drho(
                wdc_lm_o,
                drhom_lm_o,
                wdc_grid,
                wdc_drho_grid,
                lmax,
                mass,
                nmax,
                drho_wdc,
                R,
                density_var=density_var_dc,
                filter=filter,
                filter_half=filter_half,
                c=c,
            )

            if iter != 1:
                if sum_dc == 0:
                    if sum_w != 0:
                        delta = abs(grid_prev - w_grid).max()
                        if quiet is False:
                            print(
                                "Iteration %i, Delta (km) = %.3f" % (iter, delta / 1e3)
                            )
                            print(
                                "Maximum displacement (km) = %.2f"
                                % (((w_grid - R) / 1e3).max())
                            )
                            print(
                                "Minimum displacement (km) = %.2f"
                                % (((w_grid - R) / 1e3).min())
                            )
                    else:
                        delta = abs(grid_prev - rho_grid).max()
                        if quiet is False:
                            print("Iteration %i, Delta (kg m-3) = %.3f" % (iter, delta))
                            print("Maximum density (kg m-3) = %.2f" % (rho_grid.max()))
                            print("Minimum density (kg m-3) = %.2f" % (rho_grid.min()))
                else:
                    delta = abs(grid_prev - (R - wdc_grid - c)).max()
                    if quiet is False:
                        print("Iteration %i, Delta (km) = %.3f" % (iter, delta / 1e3))
                        crust_thick = (H_grid - wdc_grid) / 1e3
                        print(
                            "Maximum Crustal thickness (km) = %.2f"
                            % (crust_thick.max())
                        )
                        print(
                            "Minimum Crustal thickness (km) = %.2f"
                            % (crust_thick.min())
                        )

            # Speed up convergence scheme
            if iter % 2 == 0:
                delta_wdc_geoid_2 = delta_wdc_geoid
                delta_H_geoid_2 = delta_H_geoid
                delta_w_geoid_2 = delta_w_geoid
            else:
                delta_wdc_geoid_1 = delta_wdc_geoid
                delta_H_geoid_1 = delta_H_geoid
                delta_w_geoid_1 = delta_w_geoid

            if sum_dc == 0:
                if sum_w != 0:
                    grid_prev = w_grid
                else:
                    grid_prev = rho_grid
            else:
                grid_prev = R - wdc_grid - c

            if iter > iter_max:
                raise ValueError(
                    "%s not converging, maximum iteration reached at %i, "
                    % (var_relief, iter)
                    + "delta was %s (%s) and delta_max is %s (%s)."
                    % (
                        "%.4f" % (delta / 1e3 if var_unit == "km" else delta),
                        var_unit,
                        "%.4f" % (delta_max / 1e3 if var_unit == "km" else delta_max),
                        var_unit,
                    )
                )
                exit(1)
            if delta > delta_out and iter != 1:
                raise ValueError(
                    "%s not converging, stopped at iteration %i, " % (var_relief, iter)
                    + "delta was %s (%s) and delta_out is %s (%s). Try modifying nmax%s"
                    % (
                        "%.4f" % (delta / 1e3 if var_unit == "km" else delta),
                        var_unit,
                        "%.4f" % (delta_out / 1e3 if var_unit == "km" else delta_out),
                        var_unit,
                        " or try filtering."
                        if (filter == 0 and filter_in is None)
                        else ".",
                    )
                )
                exit(1)

    if quiet is False:
        print("Set the interfaces degree-0 coefficients")
    w_lm_o[0, 0, 0] = R
    dc_lm_o[0, 0, 0] = c
    moho_relief_lm_o[0, 0, 0] = R - c
    H_lm_o[0, 0, 0] = R

    return (
        w_lm_o,
        A_lm_o,
        moho_relief_lm_o,
        dc_lm_o,
        drhom_lm_o,
        omega_lm_o,
        q_lm_o,
        Gc_lm_o,
        G_lm_o,
        H_lm_o,
        lambdify_func_o,
    )
