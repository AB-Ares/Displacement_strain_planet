"""
Functions for calculating the Banerdt (1986) system of equations.
"""

import re
import numpy as np
from sympy import linsolve, lambdify, symbols, Expr, expand, srepr
from sympy.parsing.sympy_parser import parse_expr
from pyshtools.backends.ducc0_wrapper import MakeGridDH
from pyshtools.gravmag import CilmPlusRhoHDH

# ==== corr_nmax_drho ====


def corr_nmax_drho(
    dr_lm,
    shape_grid,
    rho_grid,
    lmax,
    mass,
    nmax,
    drho,
    R,
    density_var=False,
):
    """
    Calculate the gravitational difference (with or
    without laterally varying density) between the
    mass-sheet case and when using the finite amplitude
    algorithm of Wieczorek & Phillips (1998).

    Returns
    -------
    array, size of input dr_lm
        Array with the spherical harmonic coefficients of the
        difference between the mass-sheet and finite-ampltiude
        geoid.

    Parameters
    ----------
    dr_lm : array, size (2,lmax+1,lmax+1)
        Array with spherical harmonic coefficients of the relief.
    shape_grid : array, size (2,2*(lmax+1),2*2(lmax+1))
        Array with a grid of the relief.
    rho_grid : array, size (2,2*(lmax+1),2*2(lmax+1))
        Array with a grid of the lateral density contrast.
    lmax : int
        Maximum spherical harmonic degree to compute for the
        derivatives.
    mass : float
        Mass of the planet.
    nmax : int
        Order of the finite-amplitude correction.
    drho : float
        Mean density contrast.
    R : float
        Mean radius of the planet.
    density_var : bool, optional, default = False
        If True, correct for density variations.
    """

    # Finite-amplitude correction.
    MS_lm_nmax = np.zeros((2, lmax + 1, lmax + 1))
    # This is the computation in Thin_shell_matrix.
    for l in range(1, lmax + 1):
        MS_lm_nmax[:, l, : l + 1] = drho * dr_lm[:, l, : l + 1] / (2 * l + 1)
    MS_lm_nmax *= 4.0 * np.pi / mass

    if nmax != 1:
        # This is the correct calculation with finite-amplitude
        FA_lm_nmax, D = CilmPlusRhoHDH(shape_grid, nmax, mass, rho_grid, lmax=lmax)
        MS_lm_nmax *= D**2
    else:
        FA_lm_nmax = MS_lm_nmax

    # Density contrast in the relief correction.
    if density_var and nmax == 1:
        MS_lm_drho, D = CilmPlusRhoHDH(shape_grid, nmax, mass, rho_grid, lmax=lmax)
        MS_lm_drho_cst = MS_lm_nmax.copy()
        MS_lm_drho_cst *= D**2

        # Divide because the thin-shell code multiplies by
        # density contrast, to correct for finite-amplitude.
        # Here we also correct for density variations, so the
        # correction is already scaled by the density contrast.
        delta_MS_FA = R * (MS_lm_drho - MS_lm_drho_cst) / drho
    else:
        if density_var and nmax != 1:
            delta_MS_FA = R * (FA_lm_nmax - MS_lm_nmax) / drho
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
    filter_half=50,
    H_lm=None,
    drhom_lm=None,
    dc_lm=None,
    w_lm=None,
    omega_lm=None,
    q_lm=None,
    G_lm=None,
    Gc_lm=None,
    add_equation=None,
    add_arrays=None,
    quiet=False,
    remove_equation=None,
    w_corr=None,
    wdc_corr=None,
    H_corr=None,
    drho_omega_corr=None,
    drho_q_corr=None,
    COM=True,
    lambdify_func=None,
    first_inv=True,
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
        crustal root variations.
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
        Array with the lambda functions (size lmax+1) of all
        components. Lambda functions can be used to
        re-calculate the same problem with different inputs
        very fast.

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
    base_drho : float, optional, default = 50e3
        Lower depth for the of the density contrast.
    top_drho : float, optional, default = 0
        Upper depth for the of the density contrast.
    filter_in : array, size(lmax+1), optional, default = None.
        Array with the input filter to use.
    filter : string, optional, default = None
        If 'Ma' or 'Mc', apply minimum-amplitude or
        minimum-curvature filtering.
    filter_half : int, default = 50
        Spherical harmonic degree at which the filter equals 0.5.
    H_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        surface topography.
    drhom_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        lateral density variations.
    dc_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        crustal root variations.
    w_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        upward displacement.
    omega_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        tangential load potential.
    q_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        net load on the lithosphere.
    G_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        geoid at the surface.
    Gc_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        geoid at the moho depth.
    add_equation : string, optional, default = None
        Equation to be added to the system. This must include at least
        one of the 8 parameters aboves.
    add_arrays : array size(N, 2,lmax+1,lmax+1), optional, default = None
        N arrays of spherical harmonics to be added in 'add_equation', which
        are written 'add_array1' 'add_array2' etc. Order is important.
    quiet : bool, optional, default = False
        if True, print various outputs.
    remove_equation : string, optional, default = None
        String of the equation to be removed. This must be either
        'G_lm', 'Gc_lm', 'w_lm', 'omega_lm', or 'q_lm'.
    w_corr : array size(2,lmax+1,lmax+1), optional, default = None
        Array with spherical harmonic coefficients for finite-amplitude
        and or lateral density variations corrections of the w_lm relief.
    wdc_corr : array size(2,lmax+1,lmax+1), optional, default = None
        Array with spherical harmonic coefficients for finite-amplitude
        and or lateral density variations corrections of the moho_lm relief.
    H_corr : array size(2,lmax+1,lmax+1), optional, default = None
        Array with spherical harmonic coefficients for finite-amplitude
        and or lateral density variations corrections of the H_lm relief.
    drho_omega_corr : array size(2,lmax+1,lmax+1), optional, default = None
        Array with spherical harmonic coefficients for lateral
        lateral density variations corrections for omega_lm.
    drho_q_corr : array size(2,lmax+1,lmax+1), optional, default = None
        Array with spherical harmonic coefficients for lateral
        lateral density variations corrections for q_lm.
    COM : bool, optional, default = True
        if True, force the model to be in a center-of-mass frame by setting
        the degree-1 geoid terms to zero.
    lambdify_func : array size(lmax+1), optional, default = None
        Reuse the lambidfy functions of the first run.
    first_inv : bool, optional, default = True
        If True, the code assumes that this is the first time doing
        the inversion in this setup, and will store the lambdify results
        in 'lambdify_func'
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
    sum_array_test = np.sum(num_array_test)
    # Input arrays
    constraint_test = input_constraints[num_array_test]
    # Other arrays
    not_constraint = input_constraints[~num_array_test]

    if lmax < 0:
        raise ValueError(
            "lmax must be greater or equal to 0. "
            + "Input value was {:s}.".format(repr(lmax))
        )

    for arr, csts in zip(input_arrays[num_array_test], constraint_test):
        if np.shape(arr) != (2, lmax + 1, lmax + 1):
            raise ValueError(
                "Input array should be dimensioned as (2, lmax+1, lmax+1),"
                + " where lmax = %s Input %s has shape of %s."
                % (lmax, csts, str(np.shape(arr)))
            )

    if filter_in is not None and np.size(filter_in) != lmax + 1:
        raise ValueError(
            "The size of filter_in must be %s. Input " % (lmax + 1)
            + "size was %s." % (np.size(filter_in))
        )

    if add_arrays is not None:
        if np.shape(add_arrays) == (2, lmax + 1, lmax + 1):
            single_add_arrays = True
        elif np.shape(add_arrays)[1:] == (2, lmax + 1, lmax + 1):
            single_add_arrays = False
        else:
            raise ValueError(
                "Add_arrays should be dimensioned as (N, 2, lmax+1, lmax+1),"
                + " where lmax is %s." % (lmax)
                + "\nInput array is dimensioned as %s." % (str(np.shape(add_arrays)))
            )

    # The system is a total of 5 equations relating 8 unknowns.
    # If an additional equation is given, 2 arrays must be input
    # to find a solution.

    if add_arrays is not None and "add_array" not in add_equation:
        raise ValueError(
            "add_arrays specified but not found in input equation"
            + "\nInput equation was %s" % (add_equation)
        )

    elif add_arrays is not None and add_equation is not None:
        add_muls = []
        add_muls_cnsts = []
        # Determine what parameters are input in add_equation and whether
        # a multiplication with add_array occurs (e.g. H_lm * add_array1).
        for cnsts in input_constraints:
            for i in [0] if single_add_arrays else range(np.shape(add_arrays)[0]):
                add_arr = "add_array%s" % (i + 1)
                if "%s*%s" % (cnsts, add_arr) in "%s" % (
                    expand(add_equation)
                ) or "%s*%s" % (add_arr, cnsts) in "%s" % (expand(add_equation)):
                    add_muls.append(i + 1)
                    add_muls_cnsts.append(cnsts)
                    if not quiet and first_inv:
                        print(
                            "! Warning:Thin_shell_matrix, we will use only the (0,lmax+1,0)"
                            + " coeffs of %s in the multiplication with %s !"
                            % (add_arr, cnsts)
                        )
                    # If finite-amplitude corrections and multiplications in
                    # add_equation, we will force the corrections to be zero
                    # where the associated parameter is zero, and also multiply
                    # the correction by the same coefficients. This takes care
                    # of some non-linearities.
                    # e.g., add_equation = H_lm * add_array1 -> H_lm * add_array1 = 0
                    # H_corr[H_lm==0] = 0
                    # H_corr *= add_array1
                    if not first_inv:
                        if cnsts == "drhom_lm":
                            drho_omega_corr[drhom_lm == 0] = 0.0
                            drho_omega_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]
                            drho_q_corr[drhom_lm == 0] = 0.0
                            drho_q_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]
                        elif cnsts == "omega_lm":
                            drho_omega_corr[omega_lm == 0] = 0.0
                            drho_omega_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]
                        elif cnsts == "q_lm":
                            drho_q_corr[q_lm == 0] = 0.0
                            drho_q_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]
                        elif cnsts == "H_lm":
                            H_corr[H_lm == 0] = 0.0
                            H_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]
                        elif cnsts == "dc_lm":
                            wdc_corr[dc_lm == 0] = 0.0
                            wdc_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]
                        elif cnsts == "w_lm":
                            w_corr[w_lm == 0] = 0.0
                            w_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]
                        elif cnsts == "G_lm":
                            H_corr[G_lm == 0] = 0.0
                            H_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]
                            w_corr[G_lm == 0] = 0.0
                            w_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]
                            wdc_corr[G_lm == 0] = 0.0
                            wdc_corr[:, : lmax + 1, : lmax + 1] *= add_arrays[
                                i, :, : lmax + 1, : lmax + 1
                            ]

    error_msg = "\nNumber of input arrays was {:s}. ".format(
        repr(sum_array_test)
    ) + "Input arrays are %s." % (constraint_test)
    if add_equation is not None:
        if "add_array" in add_equation and add_arrays is None:
            raise ValueError("Equation has input add_arrays, but add_arrays is None")
        if remove_equation is None:
            if sum_array_test != 2:
                raise ValueError(
                    "With add_equation, only 2 constraints are necessary. %s"
                    % (error_msg)
                )
        else:
            if sum_array_test != 3:
                raise ValueError(
                    "With remove_equation and add_equation, 3 constraints are "
                    + "necessary. %s" % (error_msg)
                )
        if "=" in add_equation:
            raise ValueError(
                "All terms of the added equation must be "
                + "on the same side, and there is no need to specify = 0, "
                + "the input equation is %s. " % (add_equation)
            )

        if all(sym not in add_equation for sym in input_constraints):
            raise ValueError(
                "The input equation must relate any of the 8 "
                + "unknown arrays that are %s." % (input_constraints)
                + "\nThe input equation is %s." % (add_equation)
            )
    else:
        if remove_equation is None:
            # If no additional equation is given, 3 arrays must
            # be input to find a solution.
            if sum_array_test != 3:
                raise ValueError("3 constraints are necessary. %s" % (error_msg))
        else:
            if sum_array_test != 4:
                raise ValueError(
                    "With remove_equation, only 4 constraints are necesasary."
                    + " %s" % (error_msg)
                )

    if not quiet and first_inv:
        print("Input arrays are %s." % (constraint_test))
        print("Solving for %s." % (not_constraint))
        add_eq_prev = ""
        if filter_in is not None:
            print("Use input filter")
        elif filter is not None:
            print(
                "Minimum %s filter" % ("curvature" if filter == "Mc" else "amplitude")
            )
        if first_inv:
            print("First inversion, storing lambdify results")
        else:
            print("Using stored solutions with new inputs")

    if dc_lm is not None:
        # Filtering drhom when there is no crustal root variations
        any_dc = np.sum(dc_lm[:, 1:, :]) != 0
    else:
        # No filtering for drhom
        any_dc = True

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
    if H_corr is None:
        H_corr = np.zeros(shape)
    if w_corr is None:
        w_corr = np.zeros(shape)
    if drho_omega_corr is None:
        drho_omega_corr = np.zeros(shape)
    if drho_q_corr is None:
        drho_q_corr = np.zeros(shape)
    A_lm = np.zeros(shape)

    if Te == 0:  # Avoid numerical problems with infinite values
        Te = 1
        if first_inv:
            print(
                "! Warning:Thin_shell_matrix, elastic thickness set to 1 m "
                + "to avoid numerical problems !"
            )

    # Precompute some constants.
    M = base_drho - top_drho  # Thickness of the density anomaly
    if M < 0:
        raise ValueError(
            "Thickness of the density anomaly (base_drho - top_drho) is negative. "
            + "base_drho and top_drho are probably inverted with values of "
            + "%.2f and %.2f (km), respectively" % (base_drho / 1e3, top_drho / 1e3)
        )

    Re = R - Te / 2.0  # Reference radius for displacement equations
    R_base_drho = R - base_drho
    R_top_drho = R - top_drho
    R_c = R - c
    Re4 = Re**4
    drho = rhom - rhoc
    drhol = rhoc - rhol
    eps = 12.0 * Re**2 / Te**2
    alph_B = 1.0 / (E * Te)
    # Avoids error printing when dividing by zero.
    D = E * Te**3 / (12.0 * (1.0 - v**2))  # Shell's
    # rigidity.
    v1v = v / (1.0 - v)
    RCR = R_c / R
    beta_B = 1.0 / (1.0 + eps)
    eta_B = eps / (1.0 + eps)
    mass_correc = (
        1.0
        / 3.0
        * (R_base_drho**3 - R_top_drho**3)
        / (R_top_drho**2 * (R_base_drho - R_top_drho))
    )
    mass_correc = 1.0
    # Mass correction for the mantle density anomaly to account for the
    # planet sphericity, work in progress.

    R_drho_mid = (R_top_drho + R_top_drho) / 2.0
    RTeR = (R - Te) / R
    gmoho = g0 * (1.0 + (RCR**3 - 1.0) * rhoc / rhobar) / RCR**2
    if Te <= c:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rhoc / rhobar) / RTeR**2
    else:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rhom / rhobar) / RTeR**2

    if top_drho <= c:
        gdrho = (
            g0
            * (1.0 + ((R_drho_mid / R) ** 3 - 1.0) * rhoc / rhobar)
            / (R_drho_mid / R) ** 2
        )
    else:
        gdrho = (
            g0
            * (1.0 + ((R_drho_mid / R) ** 3 - 1) * rhom / rhobar)
            / (R_drho_mid / R) ** 2
        )

    # If we have non-zero w_corr, this mean that geoid corrections
    # for density contrasts within the flexure should be accounted for
    w_corr_test = np.sum(w_corr[:, 1:, :]) != 0 and drhol != 0

    # Store symbolized array names with sympy. Order is 
    # important. These will be denoted e.g. 'H_lm1' for H_lm.
    add_constraints = ""

    if add_arrays is not None:
        for i in [0] if single_add_arrays else range(np.shape(add_arrays)[0]):
            if i + 1 not in add_muls:
                add_constraints += " add_array%s" % (i + 1)
    add_constraints += " wdc_corr1 w_corr1 H_corr1 drho_omega_corr1 drho_q_corr1"

    a_symb_uknwn = symbols(" ".join([symb + "1 " for symb in not_constraint]))
    a_symb_knwn = symbols(
        " ".join([symb + "1 " for symb in constraint_test]) + add_constraints
    )
    args_symb = (constraint_test, not_constraint, a_symb_uknwn)

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

    if remove_equation is not None and not quiet and first_inv:
        print("Removing equation for: %s." % (remove_equation))
    if add_equation is not None:
        if not quiet and first_inv:
            print("Adding an equation:\n%s." % (add_equation))
        # Reformat added equation for sympy
        for string in input_constraints:
            add_equation = re.sub(
                r"(\b{}\b)".format(string), "%s" % (string) + "1", add_equation
            )
        add_equation = parse_expr(add_equation)

    degrees = np.arange(lmax + 1, dtype=float)
    Lapla = -degrees * (degrees + 1)  # Laplacian identity.

    if first_inv:
        # Define some arrays
        # Filters

        DCfilter_mohoD = np.ones_like(degrees)
        DCfilter_mohoDc, DCfilter_drhomc, DCfilter_drhom = (
            DCfilter_mohoD,
            DCfilter_mohoD,
            DCfilter_mohoD,
        )
        if filter_in is not None:
            if any_dc:
                DCfilter_mohoD, DCfilter_mohoDc = filter_in, filter_in
            else:
                DCfilter_drhom, DCfilter_drhomc = filter_in, filter_in
        elif filter is not None:
            if any_dc:
                DCfilter_mohoD = DownContFilter(
                    degrees,
                    filter_half,
                    R,
                    R_c,
                    type=filter,
                    quiet=quiet,
                )
                DCfilter_mohoDc = DownContFilter(
                    degrees, filter_half, R_c, R_c, type=filter
                )
            else:
                DCfilter_drhom = DownContFilter(
                    degrees, filter_half, R, R_base_drho, type=filter
                )
                DCfilter_drhomc = DownContFilter(
                    degrees, filter_half, R_c, R_base_drho, type=filter
                )

        rhobconst = 3.0 / (rhobar * (2.0 * degrees + 1.0))

        # Continuation arrays
        Rl3 = R / (degrees + 3.0)
        RCRl1 = RCR ** (degrees + 1.0)
        RCRl2 = RCR ** (degrees + 2.0)

        if R_top_drho <= R_c:
            RtRCl = (R_top_drho / R_c) ** degrees
        else:
            RtRCl = (R_c / R_top_drho) ** (degrees + 1.0)
        if R_base_drho <= R_c:
            RbRCl = (R_base_drho / R_c) ** degrees
        else:
            RbRCl = (R_c / R_base_drho) ** (degrees + 1.0)

        RtRCl *= R_top_drho**3 / (R_c * R**2)
        RbRCl *= R_base_drho**3 / (R_c * R**2)
        RtbRl3 = (R_top_drho / R) ** (degrees + 3.0) - (R_base_drho / R) ** (
            degrees + 3.0
        )

    # Solve matrix over all degrees.
    for l in range(1, lmax + 1):  # Ignore degree 0 from calculations
        if first_inv:
            # Symbolic definition.
            w_lm1, Gc_lm1, q_lm1, omega_lm1, dc_lm1, drhom_lm1, G_lm1, H_lm1 = symbols(
                " w_lm1 Gc_lm1 q_lm1 omega_lm1 dc_lm1 drhom_lm1 G_lm1 H_lm1 "
            )

            wdc_corr1, w_corr1, H_corr1, drho_omega_corr1, drho_q_corr1 = symbols(
                " wdc_corr1 w_corr1 H_corr1 drho_omega_corr1 drho_q_corr1 "
            )

            # System of equations from Banerdt (1986).
            Eqns = [
                # eq (1) G_lm
                -G_lm1
                + (
                    rhobconst[l]
                    * (
                        rhol * H_lm1
                        + drhol * w_lm1
                        + drho * (w_lm1 - dc_lm1) * RCRl2[l] / DCfilter_mohoD[l]
                        + drhom_lm1 * Rl3[l] * RtbRl3[l] / DCfilter_drhom[l]
                    )
                    + rhol * H_corr1
                    + ((drhol * w_corr1) if not w_corr_test else w_corr1)
                    + drho * wdc_corr1 * RCRl2[l]  # / DCfilter_mohoD[l]
                    # Still unsure about that filtering part
                )
                * (
                    0.0 if "G_lm" in not_constraint and COM and l == 1 else 1.0
                ),  # Force the degree-1 geoid to zero
                # eq(2) Gc_lm
                -Gc_lm1
                + (
                    rhobconst[l]
                    * (
                        (rhol * H_lm1 + drhol * w_lm1) * RCRl1[l]
                        + drho * (w_lm1 - dc_lm1) * RCR**3 / DCfilter_mohoDc[l]
                        + drhom_lm1
                        * Rl3[l]
                        * (RtRCl[l] - RbRCl[l])
                        / DCfilter_drhomc[l]
                    )
                    + (
                        rhol * H_corr1
                        + ((drhol * w_corr1) if not w_corr_test else w_corr1)
                    )
                    * RCRl1[l]
                    + drho * wdc_corr1 * RCR**3  # / DCfilter_mohoDc[l]
                    # Still unsure about that filtering part
                )
                * (
                    0.0 if "Gc_lm" in not_constraint and COM and l == 1 else 1.0
                ),  # Force the degree-1 geoid to zero
                # eq (3) q_lm
                -q_lm1
                + g0 * (rhol * (H_lm1 - G_lm1) + drhol * w_lm1)
                + gmoho * drho * (w_lm1 - dc_lm1 - Gc_lm1)
                + gdrho * drhom_lm1 * M * mass_correc
                + drho_q_corr1,
                # eq (4) w_lm
                eta_B * D * Lapla[l] * (Lapla[l] + 2) ** 2 * w_lm1
                + Re**2 / alph_B * (Lapla[l] + 2) * w_lm1
                + Re4 * ((Lapla[l] + 2) - 1.0 - v) * q_lm1
                - Re4 * (beta_B * (Lapla[l] + 2) - 1.0 - v) * Lapla[l] * omega_lm1,
                # eq (5) omega_lm
                -omega_lm1
                + v1v * rhol * g0 * Te * H_lm1 / R
                - (
                    drhol * g0 * v1v * Te
                    - rhoc * gmoho * (c if c < Te else 0)
                    # If crust-mantle interface below Te, no tangential load associated
                    - rhom * gTe * np.max([Te - c, 0])
                    # If crust-mantle interface below Te, no tangential load associated
                )
                * w_lm1
                / R
                + v1v * drho * gmoho * np.max([Te - c, 0]) * (dc_lm1 - w_lm1) / R
                - 0.5
                * v1v
                * drhom_lm1
                * mass_correc
                * gdrho
                * (Te - top_drho)
                * (np.min([M, Te - top_drho]) if top_drho < Te else 0)
                # If mantle load below Te, no tangential load associated
                / R + drho_omega_corr1,
            ]

            if add_equation is not None:
                add_equation_subbed = add_equation.copy()
                if add_arrays is not None:
                    for i in (
                        [0] if single_add_arrays else range(np.shape(add_arrays)[0])
                    ):
                        if i + 1 in add_muls:
                            add_equation_subbed = add_equation_subbed.subs(
                                "add_array%s" % (i + 1),
                                add_arrays[0, l, 0]
                                if single_add_arrays
                                else add_arrays[i, 0, l, 0],
                            )

                if add_equation_subbed != parse_expr("0"):
                    Eqns.insert(len(Eqns), add_equation_subbed)
                else:
                    if np.size(input_constraints) - sum_array_test != 5:
                        raise ValueError(
                            "System cannot be determined at degree %s " % (l)
                            + "where add_equation becomes 0 = 0"
                        )

                if not quiet and add_equation_subbed != add_eq_prev and first_inv:
                    add_eq_prev = add_equation_subbed
                    print(
                        "Additional equation starting at degree %s is %s"
                        % (
                            l,
                            add_equation_subbed
                            if add_equation_subbed != parse_expr("0")
                            else "0 = 0",
                        )
                    )

            # At degree-1, w_lm vanishes from eq (4), and makes the eq only
            # relate q_lm and omega_lm. w_lm should be zero at degree-1 in
            # a center of mass-reference frame.
            # Thus, we replace the degree-1 equation for omega_lm by eq (4)
            # and eq (4) now becomes w_lm = 0 if:
            if (
                l == 1
                and COM  # 1) We are in a COM (default = True)
                and (  # 2) w_lm is not in add_equation,
                    # or not directly related to one other symbol
                    add_equation is not None
                    and (
                        "w_lm" not in str(add_equation_subbed)
                        or (
                            "w_lm" in str(add_equation_subbed)
                            and srepr(add_equation_subbed).count("Symbol") > 2
                        )
                    )
                    or add_equation is None
                )
                and (
                    "w_lm"
                    not in constraint_test  # 3) w_lm is not an input, or equal zero
                    or ("w_lm" in constraint_test and w_lm[0, 1, 0] == 0)
                )
            ):
                Eqns[4] = Eqns[3].copy()
                Eqns[3] = w_lm1

            if remove_equation is not None and l != 1:
                for item in [remove_equation]:
                    Eqns.pop(int(np.where(equation_order == item)[0]))

            # Rearange system of equations using sympy.
            sol = linsolve(Eqns, a_symb_uknwn + a_symb_knwn)

            # Vectorize the linsolve function.
            # Store matrix solution for potential later reutilisation
            lambdify_func[l] = lambdify(a_symb_uknwn + a_symb_knwn, list(sol))

        # Results.
        # Depending on the input arrays, pass a symbol or the
        # input values.
        args_linsolve = dict(
            w_lm1=test_symb("w_lm", w_lm[:, l, : l + 1], *args_symb),
            Gc_lm1=test_symb("Gc_lm", Gc_lm[:, l, : l + 1], *args_symb),
            G_lm1=test_symb("G_lm", G_lm[:, l, : l + 1], *args_symb),
            H_lm1=test_symb("H_lm", H_lm[:, l, : l + 1], *args_symb),
            q_lm1=test_symb("q_lm", q_lm[:, l, : l + 1], *args_symb),
            omega_lm1=test_symb("omega_lm", omega_lm[:, l, : l + 1], *args_symb),
            dc_lm1=test_symb("dc_lm", dc_lm[:, l, : l + 1], *args_symb),
            drhom_lm1=test_symb("drhom_lm", drhom_lm[:, l, : l + 1], *args_symb),
            wdc_corr1=wdc_corr[:, l, : l + 1],
            H_corr1=H_corr[:, l, : l + 1],
            w_corr1=w_corr[:, l, : l + 1],
            drho_omega_corr1=drho_omega_corr[:, l, : l + 1],
            drho_q_corr1=drho_q_corr[:, l, : l + 1],
        )

        if add_arrays is not None:
            add_arr = ""
            for i in [0] if single_add_arrays else range(np.shape(add_arrays)[0]):
                if i + 1 not in add_muls:
                    add_arr += "'add_array%s':add_arrays[%s:, l, : l + 1], " % (
                        i + 1,
                        "%s," % (i) if not single_add_arrays else "",
                    )
            args_linsolve = dict(args_linsolve, **dict(eval("{%s}" % (add_arr))))

        outs = np.concatenate(np.array(lambdify_func[l](**args_linsolve), dtype=object))

        if np.any([isinstance(arrs, Expr) for arrs in outs]):
            raise ValueError(
                "System is non-evenly determined at degree %i, cannot solve" % (l)
                + "\nSystem of equations: \n%s" % (Eqns)
                + "\nSolutions found:"
                + "\nw_lm = %s" % (outs[idx_w_lm])
                + "\nq_lm = %s" % (outs[idx_q_lm])
                + "\nomega_lm = %s" % (outs[idx_omega_lm])
                + "\ndc_lm = %s" % (outs[idx_dc_lm])
                + "\ndrhom_lm = %s" % (outs[idx_drhom_lm])
                + "\nG_lm = %s" % (outs[idx_G_lm])
                + "\nGc_lm = %s" % (outs[idx_Gc_lm])
                + "\nH_lm = %s" % (outs[idx_H_lm])
                + "%s"
                % (
                    "Make sure the add_equation doesn't involve w_lm, G_lm,"
                    + " or Gc_lm, which are specifically treated when COM is True (default)"
                    if l == 1 and COM
                    else ""
                )
            )

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
        A_lm[:, l, : l + 1] = (
            beta_B
            * (1.0 / (1.0 - v**2))
            * (Lapla[l] + 1.0 + v)
            * (Lapla[l] + 2)
            * w_lm[:, l, : l + 1]
            + w_lm[:, l, : l + 1]
            + Re**2 * alph_B * q_lm[:, l, : l + 1]
            - Re
            * alph_B
            / (1.0 + eps)
            * (Lapla[l] - eps * (1.0 + v))
            * Re
            * omega_lm[:, l, : l + 1]
        )

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


# ==== test_symb ====


def test_symb(str_symb, arr, constraint_test, not_constraint, arr_symb):
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
    out = (
        arr
        if str_symb in constraint_test
        else arr_symb[int(np.where(not_constraint == str_symb)[0])]
    )

    return out


# ==== DownContFilter ====


def DownContFilter(l, half, R_ref, D_relief, type="Mc", quiet=False):
    """
    Compute the downward minimum-amplitude or
    -curvature filter of Wieczorek & Phillips,
    (1998).

    Returns
    -------
    float
        Value of the filter at degrees l

    Parameters
    ----------
    l : array
        Array of spherical harmonic degrees.
    half : int
        The spherical harmonic degree where the filter is equal to 0.5.
    R_ref : float
        The reference radius of the gravitational field.
    D_relief : float
        The radius of the surface to downward continue to.
    type : string, optional, default = "Mc"
        Filter type, minimum amplitude ("Ma") of curvature ("Mc")
    quiet : bool, optional, default = True
        if True, prints a warning when D_relief > R_ref.
    """

    if D_relief > R_ref:
        if not quiet:
            print(
                "! Warning:DownContFilter, D_relief > R_ref, cannot "
                + "use a downward continuation filter. "
                + "Setting value to 1 ! [D_relief = %s, R_ref = %s km]"
                % (D_relief / 1e3, R_ref / 1e3)
            )
        return np.ones_like(l)

    if half == 0:
        DownContFilter = 1.0
    else:
        if type == "Mc":
            tmp = 1.0 / (
                (half * half + half)
                * ((2 * half + 1) * (R_ref / D_relief) ** half) ** 2
            )
            DownContFilter = (
                1.0 + tmp * (l * l + l) * ((2 * l + 1) * (R_ref / D_relief) ** l) ** 2
            )
        elif type == "Ma":
            tmp = 1.0 / ((2.0 * half + 1.0) * (R_ref / D_relief) ** half) ** 2
            DownContFilter = 1.0 + tmp * ((2 * l + 1) * (R_ref / D_relief) ** l) ** 2
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
    lmaxgrid=None,
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
    add_arrays=None,
    quiet=True,
    remove_equation=None,
    COM=True,
    base_drho=50e3,
    top_drho=0,
    delta_max=5,
    iter_max=250,
    delta_out=500e3,
    iterate=True,
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
        crustal root variations.
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
        Array with the lambda functions (size lmax+1) of all
        components. Lambda functions can be used to re-calculate
        the same problem with different inputs very fast.

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
    filter_in : array, size(lmax+1), optional, default = None.
        Array with the input filter to use.
    filter : string, optional, default = None
        If 'Ma' or 'Mc', apply minimum-amplitude or minimum-curvature
        filtering.
    filter_half : int, optional, default = 50
        Spherical harmonic degree at which the filter equals 0.5.
    nmax : int, optional, default = 5
        Maximum order of the finite-amplitude correction.
    lmaxgrid : int, optional, default = None
        If None, this parameter is set to 3*lmax.
        Resolution of the input grid for the finite-amplitude correction
        routines. For accurate results, this parameter should be about
        3 times lmax, though this should be verified for each application.
        Lowering this parameter significantly increases speed.
    H_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        surface topography.
    drhom_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        lateral density variations.
    dc_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        crustal root variations.
    w_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        upward displacement.
    omega_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        tangential load potential.
    q_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        net load on the lithosphere.
    G_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        geoid at the surface.
    Gc_lm : array, size(2,lmax+1,lmax+1), optional, default = None
        Array with the spherical harmonic coefficients of the
        geoid at the moho depth.
    add_equation : string, optional, default = None
        Equation to be added to the system. This must include at least
        one of the 8 parameters aboves.
    add_arrays : array size(N, 2,lmax+1,lmax+1), optional, default = None
        N arrays of spherical harmonics to be added in 'add_equation', which
        are written 'add_array1' 'add_array2' etc. Order is important.
    quiet : bool, optional, default = False
        if True, print various outputs.
    COM : bool, optional, default = True
        if True, force the model to be in a center-of-mass frame by setting
        the degree-1 geoid terms to zero.
    remove_equation : string, optional, default = None
        String of the equation to be removed. This must be either
        'G_lm', 'Gc_lm', 'w_lm', 'omega_lm', or 'q_lm'.
    base_drho : float, optional, default = 50e3
        Lower depth for the of the density contrast.
    top_drho : float, optional, default = 0
        Upper depth for the of the density contrast.
    delta_max : float, optional, default = 5
        The algorithm will continue to iterate until the maximum
        difference in relief (or density contrast) between solutions
        is less than this value (in meters or kg m-3).
    iter_max : int, optional, default = 250
        Maximum number of iterations before the algorithm stops.
    delta_out : float, optional, default = 500e3
        If the delta is larger than this value, the algorithm stops
        and prints that it is not converging.
    iterate : bool, optional, default = True
        if False, solve the system without any corrections.
    """
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    R_c = R - c
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
        add_arrays=add_arrays,
        remove_equation=remove_equation,
        add_equation=add_equation,
        quiet=quiet,
        COM=COM,
    )

    if lmaxgrid is None:
        # Increase grid resolution to avoid aliasing in the CilmPlus routines
        lmaxgrid = 3 * lmax
    elif lmaxgrid < lmax:
        raise ValueError(
            "Error in Thin_shell_matrix_nmax, lmaxgrid cannot be lower "
            + "than lmax. lmaxgrid is %s and lmax is %s" % (lmaxgrid, lmax)
        )
    args_grid = dict(sampling=2, lmax=lmaxgrid, extend=False, lmax_calc=lmax)

    # Precompute some sums that will be used later for checks
    any_dc = np.sum(dc_lm) != 0 if dc_lm is not None else None
    any_w = np.sum(w_lm) != 0 if w_lm is not None else None
    any_drho = np.sum(drhom_lm) != 0 if drhom_lm is not None else None

    # Density contrast not at topography or moho and no
    # finite-amplitude correctio, return
    if (
        nmax == 1
        and (not any_drho or (top_drho != 0 and base_drho != c))
        and any_drho is not None
        or not iterate
    ):
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

        if not quiet:
            print("Returning without corrections")
            print("Set the interfaces degree-0 coefficients")
        w_lm_o[0, 0, 0] = R
        dc_lm_o[0, 0, 0] = 0
        moho_relief_lm_o[0, 0, 0] = R_c
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
        density_var_H, density_var_dc, density_var_w = False, False, False
        # Precompute grids
        precomp_drho, precomp_H_grid, precomp_w_grid, precomp_dc_grid = (
            False,
            False,
            False,
            False,
        )
        if drhom_lm is None or any_drho:
            if top_drho == 0:
                # Correct for density variations in the surface
                # relief
                density_var_H = True
            if base_drho == c or top_drho == c:
                # Correct for density variations in the moho
                # relief
                density_var_dc = True
            if base_drho < c and top_drho == 0 and rhol == rhoc:
                # Correct for density variations in the flexure relief
                # within the crust
                density_var_w = True

        # If only finite-amplitude correction, density
        # contrast is multipled in the thin-shell code and
        # we set the density contrast to 1. This will be changed later if required.
        ones = np.ones((2 * (lmaxgrid + 1), 2 * (2 * (lmaxgrid + 1))))
        H_drho_grid, w_drho_grid, wdc_drho_grid = ones, ones, ones
        drho_H, drho_wdc, drho_w = 1.0, 1.0, 1.0

        if drhom_lm is not None and any_drho:
            rho_grid = MakeGridDH(drhom_lm, **args_grid)
            precomp_drho = True
            if drhom_lm[0, 0, 0] > 500:
                if base_drho <= c:
                    rhoc = drhom_lm[0, 0, 0]
                    rhol = drhom_lm[0, 0, 0]
                    if not quiet:
                        print(
                            "rhol and rhoc are set to the mean input density variations (%.2f kg m-3)"
                            % (rhoc)
                        )
                else:
                    rhom = drhom_lm[0, 0, 0]
                    if not quiet:
                        print(
                            "rhom is set to the mean input density variations (%.2f kg m-3)"
                            % (rhom)
                        )
                # Update parameters
                args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax, E, v)
            else:
                # Density variations is in the crust
                if base_drho <= c:
                    rho_grid += rhoc
                    if not quiet:
                        print(
                            "Add input rhoc (%.2f kg m-3) to density variations"
                            % (rhoc)
                        )
                # Density variations is in the mantle
                else:
                    rho_grid += rhom
                    if not quiet:
                        print(
                            "Add input rhom (%.2f kg m-3) to density variations"
                            % (rhom)
                        )

        # Geoid correction due to density variations
        # and or finite-amplitude corrections
        shape = (2, lmax + 1, lmax + 1)
        # Moho relief
        delta_wdc_geoid = np.zeros(shape)
        # Deflected topography relief
        delta_w_geoid = np.zeros(shape)
        # Surface topography relief
        delta_H_geoid = np.zeros(shape)
        # Tangential load potential corrections due to density
        # variations at the reliefs
        drho_omega_corr = np.zeros(shape)
        drho_q_corr = np.zeros(shape)

        # Precompute grids
        if H_lm is not None:
            precomp_H_grid = True
            H_lm[0, 0, 0] = R
            H_grid = MakeGridDH(H_lm, **args_grid)
        if w_lm is not None and rhoc != rhol:
            precomp_w_grid = True
            if any_w:
                w_lm[0, 0, 0] = R
                w_grid = MakeGridDH(w_lm, **args_grid)
            else:
                w_grid = ones * R
        if w_lm is not None and dc_lm is not None:
            precomp_dc_grid = True
            wdc_lm = w_lm - dc_lm
            if any_w and any_dc:
                wdc_lm[0, 0, 0] = R_c
                wdc_grid = MakeGridDH(wdc_lm, **args_grid)
            else:
                wdc_grid = ones * R_c

        lambdify_func_o = None
        first_inv, first_drhom, first_nmax = True, True, True
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
                drho_omega_corr=drho_omega_corr,
                drho_q_corr=drho_q_corr,
                first_inv=first_inv,
                lambdify_func=lambdify_func_o,
            )
            first_inv, comp_w_grid = False, False

            # Precompute some sums that will be used later for checks
            any_dc = np.sum(dc_lm_o[:, 1:, :]) != 0 if any_dc is None else any_dc
            any_w = np.sum(w_lm_o[:, 1:, :]) != 0 if any_w is None else any_w
            any_drho = (
                np.sum(drhom_lm_o[:, 1:, :]) != 0 if any_drho is None else any_drho
            )

            # Correct for density contrast in surface or moho
            # relief, and/or finite-amplitude correction
            if drhom_lm is None or any_drho:
                if not quiet and first_drhom:
                    first_drhom = False
                    print(
                        "Iterate to account for density"
                        + " variations %s"
                        % (
                            "and finite-amplitude correction, nmax is %i" % (nmax)
                            if nmax > 1
                            else ""
                        )
                    )
            else:
                if not quiet and first_nmax:
                    first_nmax = False
                    print(
                        "Iterate for finite-amplitude correction, nmax is %i" % (nmax)
                    )

            # Scheme proposed in Wieczorek+(2013) SOM eq 21, 22
            # to speed up convergence delta(i+3) = (delta(i+2) +
            # delta(i+1))/2.
            if iter % 3 == 0:
                delta_wdc_geoid = (delta_wdc_geoid_2 + delta_wdc_geoid_1) / 2.0
                delta_H_geoid = (delta_H_geoid_2 + delta_H_geoid_1) / 2.0
                delta_w_geoid = (delta_w_geoid_2 + delta_w_geoid_1) / 2.0
                drho_omega_corr = (delta_drho_omega_2 + delta_drho_omega_1) / 2.0
                drho_q_corr = (delta_drho_q_2 + delta_drho_q_1) / 2.0
                if not quiet:
                    print(
                        "Skipping iteration %s, with convergence" % (iter) + " scheme"
                    )
                continue

            if any_drho and not precomp_drho:
                rho_grid = MakeGridDH(drhom_lm_o, **args_grid)
                if drhom_lm_o[0, 0, 0] > 500:
                    if base_drho <= c:
                        rhoc = drhom_lm_o[0, 0, 0]
                        rhol = drhom_lm_o[0, 0, 0]
                    else:
                        rhom = drhom_lm_o[0, 0, 0]
                    args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax, E, v)
                else:
                    if base_drho <= c:
                        rho_grid += rhoc
                    else:
                        rho_grid += rhom

            v1v = v / (1.0 - v)
            gmoho = g0 * (1.0 + ((R_c / R) ** 3 - 1.0) * rhoc / rhobar) / (R_c / R) ** 2
            if density_var_H:
                drho_H = rhol
                H_drho_grid = rho_grid
                drho_omega_corr = v1v * drhom_lm_o * g0 * Te * H_lm_o / R
                drho_q_corr = drhom_lm_o * (H_lm_o - G_lm_o) * g0
            if density_var_dc:
                if density_var_H:
                    drho_omega_corr += v1v * drhom_lm_o * gmoho * (Te - c) * dc_lm_o / R
                    drho_q_corr += drhom_lm_o * dc_lm_o * gmoho
                else:
                    drho_omega_corr = v1v * drhom_lm_o * gmoho * (Te - c) * dc_lm_o / R
                    drho_q_corr = drhom_lm_o * dc_lm_o * gmoho
                drho_wdc = rhom - rhoc
                if base_drho <= c:
                    # Anomaly in the crust
                    wdc_drho_grid = rhom - rho_grid
                else:
                    # Anomaly in the mantle
                    wdc_drho_grid = rho_grid - rhoc
            if density_var_w:
                drho_w = (rhoc - rhol) if rhoc != rhol else 1
                w_drho_grid = rhoc - rho_grid
                if density_var_H or density_var_dc:
                    drho_omega_corr += v1v * drhom_lm_o * g0 * Te * w_lm_o / R
                    drho_q_corr += drhom_lm_o * w_lm_o * g0
                else:
                    drho_omega_corr = v1v * drhom_lm_o * g0 * Te * w_lm_o / R
                    drho_q_corr = drhom_lm_o * w_lm_o * g0

            H_lm_o[0, 0, 0] = R
            if any_drho or (iter == 1 and precomp_H_grid) or (not precomp_H_grid):
                if not precomp_H_grid:
                    H_grid = MakeGridDH(H_lm_o, **args_grid)
                delta_H_geoid = corr_nmax_drho(
                    H_lm_o,
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
            if rhoc != rhol or density_var_w:
                if not precomp_w_grid:
                    w_grid = MakeGridDH(w_lm_o, **args_grid)
                    comp_w_grid = True
                delta_w_geoid = corr_nmax_drho(
                    w_lm_o,
                    w_grid,
                    w_drho_grid if density_var_w else ones,
                    lmax,
                    mass,
                    nmax,
                    drho_w,
                    R,
                    density_var=density_var_w,
                )

            wdc_lm_o = w_lm_o - dc_lm_o
            wdc_lm_o[0, 0, 0] = R_c
            if any_dc or any_w:
                if not precomp_dc_grid:
                    wdc_grid = MakeGridDH(wdc_lm_o, **args_grid)
                delta_wdc_geoid = corr_nmax_drho(
                    wdc_lm_o,
                    wdc_grid,
                    wdc_drho_grid,
                    lmax,
                    mass,
                    nmax,
                    drho_wdc,
                    R,
                    density_var=density_var_dc,
                )

            if iter != 1:
                if not any_dc:
                    if any_w:
                        if not comp_w_grid:
                            w_grid = MakeGridDH(w_lm_o, **args_grid)
                            comp_w_grid = True
                        delta = abs(grid_prev - w_grid).max()
                        if not quiet:
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
                        if not quiet:
                            print("Iteration %i, Delta (kg m-3) = %.3f" % (iter, delta))
                            print("Maximum density (kg m-3) = %.2f" % (rho_grid.max()))
                            print("Minimum density (kg m-3) = %.2f" % (rho_grid.min()))
                else:
                    delta = abs(grid_prev - (R - wdc_grid - c)).max()
                    if not quiet:
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
                delta_drho_omega_2 = drho_omega_corr
                delta_drho_q_2 = drho_q_corr
            else:
                delta_wdc_geoid_1 = delta_wdc_geoid
                delta_H_geoid_1 = delta_H_geoid
                delta_w_geoid_1 = delta_w_geoid
                delta_drho_omega_1 = drho_omega_corr
                delta_drho_q_1 = drho_q_corr

            if any_dc:
                grid_prev = R - wdc_grid - c
            else:
                if not comp_w_grid:
                    w_grid = MakeGridDH(w_lm_o, **args_grid)
                grid_prev = w_grid if any_w else rho_grid

            # Error messages if iteration not converging
            var_unit = "km"
            var_relief = "Moho relief"
            if not any_dc and not any_w:
                var_relief = "Grid density"
                var_unit = "kg m-3"
            elif not any_dc:
                var_relief = "Flexure relief"

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
                        if (filter is None and filter_in is None)
                        else ".",
                    )
                )

    if not quiet:
        print("Set the interfaces degree-0 coefficients")
    w_lm_o[0, 0, 0] = R
    dc_lm_o[0, 0, 0] = 0
    moho_relief_lm_o[0, 0, 0] = R_c
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
