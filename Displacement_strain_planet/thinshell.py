"""
ThinShell class with functions for calculating the Banerdt (1986) system of equations and strains.
"""

import re
from copy import deepcopy
import numpy as np
from sympy import linsolve, lambdify, symbols, Expr, expand, srepr, var
from sympy.parsing.sympy_parser import parse_expr
from pyshtools.expand import MakeGridDH
from pyshtools.shclasses import SHCoeffs
from pyshtools.gravmag import CilmPlusRhoHDH
from .utils import DownContFilter, corr_nmax_drho, SH_Mul


class ThinShell:
    """
    ThinShell class

    """

    def __init__(
        self,
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
        filter_type=None,
        lmaxgrid=None,
        filter_half=50,
        quiet=True,
        delta_max=5,
        iter_max=250,
        delta_out=500e3,
        iterate=True,
        nmax=5,
    ):
        """
        ThinShell class which contains the inversion constants.

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
        filter_type : string, optional, default = None
            If 'Ma' or 'Mc', apply minimum-amplitude or minimum-curvature
            filtering. If None, no filtering.
        filter_half : int, optional, default = 50
            Spherical harmonic degree at which the filter equals 0.5.
        lmaxgrid : int, optional, default = None
            If None, this parameter is set to 3*lmax.
            Resolution of the input grid for the finite-amplitude correction
            routines. For accurate results, this parameter should be about
            3 times lmax, though this should be verified for each application.
            Lowering this parameter significantly increases speed.
        quiet : bool, optional, default = False
            If True, print various outputs.
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
        nmax : int, optional, default = 5
            Maximum order of the finite-amplitude correction.
        """

        self.g0 = g0
        self.R = R
        self.c = c
        self.Te = Te
        self.rhom = rhom
        self.rhoc = rhoc
        self.rhol = rhol
        self.rhobar = mass * 3.0 / 4.0 / np.pi / R**3
        self.lmax = lmax
        self.E = E
        self.v = v
        self.mass = mass

        if lmaxgrid is None:
            # Increase grid resolution to avoid aliasing in the CilmPlus routines
            self.lmaxgrid = 3 * lmax
        elif lmaxgrid < lmax:
            raise ValueError(
                "Error in ThinShell, lmaxgrid cannot be lower "
                + f"than lmax. lmaxgrid is {lmaxgrid} and lmax is {lmax}"
            )
        self.filter_in = filter_in
        self.filter_type = filter_type
        self.filter_half = filter_half
        self.quiet = quiet
        self.delta_max = delta_max
        self.iter_max = iter_max
        self.delta_out = delta_out
        self.iterate = iterate
        self.nmax = nmax

        # Attribute placeholders. Will be updated after the
        # inversion is ran. They are defined in invert_matrix
        # and invert_matrix_nmax
        self.w_lm = None
        self.A_lm = None
        self.moho_lm = None
        self.crust_lm = None
        self.dc_lm = None
        self.drhom_lm = None
        self.omega_lm = None
        self.q_lm = None
        self.Gc_lm = None
        self.G_lm = None
        self.H_lm = None
        self.sols = None

    def copy(self):
        """
        Return a deep copy of the ThinShell instance.
        """
        return deepcopy(self)

    def __repr__(self):
        return (
            f"  Planetary radius: R = {self.R!r}\n"
            f"  Planetary mass: mass = {self.mass!r}\n"
            f"  Planetary attraction: g0 = {self.g0!r}\n"
            f"  Planetary mean density: rhobar = {self.rhobar}\n"
            f"  Elastic thickness: Te = {self.Te!r}\n"
            f"  Crustal thickness: c = {self.c}\n"
            f"  Bulk densities, crust: rhoc = {self.rhoc!r}"
            f"  surface load: rhol = {self.rhol}"
            f"  mantle: rhom = {self.rhom}\n"
            f"  Young's modulus: E = {self.E!r}"
            f"  Poisson's ratio: v = {self.v!r}\n"
            f"  lmax = {self.lmax!r}"
            f"  lmaxgrid = {self.lmaxgrid!r}\n"
            f"  filter_in = {self.filter_in!r}"
            f"  filter_type = {self.filter_type!r}"
            f"  filter_half = {self.filter_half!r}\n"
            f"  delta_max = {self.delta_max!r}"
            f"  iter_max = {self.iter_max!r}"
            f"  delta_out = {self.delta_out!r}"
            f"  iterate = {self.iterate!r}\n"
            f"  nmax = {self.nmax!r}"
            f"  quiet = {self.quiet!r}"
        )

    def info(self):
        """
        Print a summary of the data stored in the ThinShell instance.
        """
        print(repr(self))

    def update_value(self, params, values):
        """
        Change or update a parameter in the class.
        """

        param_mapping = {
            "filter_in": "filter_in",
            "filter_type": "filter_type",
            "filter_half": "filter_half",
            "quiet": "quiet",
            "delta_max": "delta_max",
            "iter_max": "iter_max",
            "delta_out": "delta_out",
            "iterate": "iterate",
            "nmax": "nmax",
            "g0": "g0",
            "R": "R",
            "c": "c",
            "Te": "Te",
            "rhom": "rhom",
            "rhoc": "rhoc",
            "rhol": "rhol",
            "rhobar": "rhobar",
            "lmax": "lmax",
            "E": "E",
            "v": "v",
            "mass": "mass",
            "lmaxgrid": "lmaxgrid",
        }

        for param, value in zip(
            *(
                arg if isinstance(arg, (list, tuple, set)) else [arg]
                for arg in (params, values)
            )
        ):

            if param in param_mapping:
                setattr(self, param_mapping[param], value)
            else:
                raise ValueError(f"Unknown parameter: {param}")

    # ==== invert_matrix ====

    def invert_matrix(
        self,
        rho_lm_profile=None,
        rho_depth=None,
        base_drho=None,
        top_drho=None,
        H_lm=None,
        drhom_lm=None,
        drhom_crust=None,
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
        crust–mantle relief.

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
            moho or crust–mantle relief.
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
            geoid at the crust–mantle interface.
        G_lm : array, size(2,lmax+1,lmax+1)
            Array with the spherical harmonic coefficients of the
            geoid at the surface.
        H_lm : array, size(2,lmax+1,lmax+1)
            Array with the spherical harmonic coefficients of the
            planet's shape.
        lambdify_func : array, size(2,lmax+1,lmax+1)
            Array with the lambda functions (size lmax+1) of all
            components. Lambda functions can be used to
            re-calculate the same problem with different inputs
            very fast.

        Parameters
        ----------
        rho_lm_profile : array, optional, default = None
            Spherical harmonic expansion of the density variations at
            each of the depths specified in rho_depth.
            Size must be (2,lmax+1,lmax+1,len(rho_depth)).
        rho_depth : array, optional, default = None
            Depth array associated with the input interior density variations.
        base_drho : float, optional, default = None
            Lower depth for the of the density contrast. If None, set to c.
        top_drho : float, optional, default = None
            Upper depth for the of the density contrast. If None, set
            to 0 km (surface).
        H_lm : array, size(2,lmax+1,lmax+1), optional, default = None
            Array with the spherical harmonic coefficients of the
            planet's shape.
        drhom_lm : array, size(2,lmax+1,lmax+1), optional, default = None
            Array with the spherical harmonic coefficients of the
            lateral density variations.
        drhom_crust : array, size(2,lmax+1,lmax+1), optional, default = None
            Array with the spherical harmonic coefficients of the
            crustal lateral density variations. This parameter only works
            if rho_lm_profile is true and set for the mantle. To specify
            crustal density when rho_lm_profile is false please use drhom_lm.
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
            geoid at the crust–mantle interface.
        add_equation : string, optional, default = None
            Equation to be added to the system. This must include at least
            one of the 8 parameters aboves.
        add_arrays : array size(N, 2,lmax+1,lmax+1), optional, default = None
            N arrays of spherical harmonics to be added in 'add_equation', which
            are written 'add_array1' 'add_array2' etc. Order is important.
        remove_equation : string, optional, default = None
            String of the equation to be removed. This must be either
            'G_lm', 'Gc_lm', 'w_lm', 'omega_lm', or 'q_lm'.
        quiet : bool, default = False
            if False, print some information regarding the function
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
            If True, force the model to be in a center-of-mass frame by setting
            the degree-1 geoid terms to zero.
        lambdify_func : array size(lmax+1), optional, default = None
            Use the lambidfy functions (i.e. design of the inversion matrix,
            without the specific inputs) of another run.
        first_inv : bool, optional, default = True
            If True, the code assumes that this is the first time doing
            the inversion in this setup, and will store the lambdify results
            in 'lambdify_func'
        """

        g0 = self.g0
        R = self.R
        c = self.c
        Te = self.Te
        rhom = self.rhom
        rhoc = self.rhoc
        rhol = self.rhol
        lmax = self.lmax
        rhobar = self.rhobar

        filter_type = self.filter_type
        filter_in = self.filter_in
        quiet = self.quiet

        if base_drho is None:
            base_drho = c
        if top_drho is None:
            top_drho = 0

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
        _zero_expr = parse_expr("0")

        if lmax < 0:
            raise ValueError(
                "lmax must be greater or equal to 0. " + f"Input value was {lmax}."
            )

        for arr, csts in zip(input_arrays[num_array_test], constraint_test):
            if np.shape(arr) != (2, lmax + 1, lmax + 1):
                raise ValueError(
                    "Input array should be dimensioned as (2, lmax+1, lmax+1),"
                    + f" where lmax = {lmax} Input {csts} has shape of {str(np.shape(arr))}."
                )

        if filter_in is not None and np.size(filter_in) != lmax + 1:
            raise ValueError(
                f"The size of filter_in must be {lmax+1}. Input "
                + f"size was {np.size(filter_in)}."
            )

        if add_arrays is not None:
            if np.shape(add_arrays) == (2, lmax + 1, lmax + 1):
                single_add_arrays = True
            elif np.shape(add_arrays)[1:] == (2, lmax + 1, lmax + 1):
                single_add_arrays = False
            else:
                raise ValueError(
                    "Add_arrays should be dimensioned as (N, 2, lmax+1, lmax+1),"
                    + f" where lmax is {lmax}."
                    + f"\nInput array is dimensioned as {str(np.shape(add_arrays))}."
                )

        # The system is a total of 5 equations relating 8 unknowns.
        # If an additional equation is given, 2 arrays must be input
        # to find a solution.

        if add_arrays is not None and "add_array" not in add_equation:
            raise ValueError(
                "add_arrays specified but not found in input equation"
                + f"\nInput equation was {add_equation}"
            )

        if add_arrays is not None and add_equation is not None:
            add_muls = []
            # Determine what parameters are input in add_equation and whether
            # a multiplication with add_array occurs (e.g. H_lm * add_array1).
            for cnsts in input_constraints:
                for i in [0] if single_add_arrays else range(np.shape(add_arrays)[0]):
                    add_arr = f"add_array{i+1}"
                    if (
                        f"{cnsts}*{add_arr}" in f"{expand(add_equation)}"
                        or f"{add_arr}*{cnsts}" in f"{expand(add_equation)}"
                    ):
                        add_muls.append(i + 1)
                        if not quiet and first_inv:
                            print(
                                "! Warning:Thin_shell_matrix, we will use only the (0,lmax+1,0)"
                                + f" coeffs of {add_arr} in the multiplication with {cnsts} !"
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
                            mask_lmax = slice(None, lmax + 1)
                            add_array_m = add_arrays[i, :, mask_lmax, mask_lmax]
                            match cnsts:
                                case "drhom_lm":
                                    drho_omega_corr[drhom_lm == 0] = 0.0
                                    drho_omega_corr[
                                        :, mask_lmax, mask_lmax
                                    ] *= add_array_m
                                    drho_q_corr[drhom_lm == 0] = 0.0
                                    drho_q_corr[:, mask_lmax, mask_lmax] *= add_array_m
                                case "omega_lm":
                                    drho_omega_corr[omega_lm == 0] = 0.0
                                    drho_omega_corr[
                                        :, mask_lmax, mask_lmax
                                    ] *= add_array_m
                                case "q_lm":
                                    drho_q_corr[q_lm == 0] = 0.0
                                    drho_q_corr[:, mask_lmax, mask_lmax] *= add_array_m
                                case "H_lm":
                                    H_corr[H_lm == 0] = 0.0
                                    H_corr[:, mask_lmax, mask_lmax] *= add_array_m
                                case "dc_lm":
                                    wdc_corr[dc_lm == 0] = 0.0
                                    wdc_corr[:, mask_lmax, mask_lmax] *= add_array_m
                                case "w_lm":
                                    w_corr[w_lm == 0] = 0.0
                                    w_corr[:, mask_lmax, mask_lmax] *= add_array_m
                                case "G_lm":
                                    H_corr[G_lm == 0] = 0.0
                                    H_corr[:, mask_lmax, mask_lmax] *= add_array_m
                                    w_corr[G_lm == 0] = 0.0
                                    w_corr[:, mask_lmax, mask_lmax] *= add_array_m
                                    wdc_corr[G_lm == 0] = 0.0
                                    wdc_corr[:, mask_lmax, mask_lmax] *= add_array_m

        error_msg = (
            f"\nNumber of input arrays was {sum_array_test}. "
            + f"Input arrays are {constraint_test}."
        )
        if add_equation is not None:
            if "add_array" in add_equation and add_arrays is None:
                raise ValueError(
                    "Equation has input add_arrays, but add_arrays is None"
                )
            if remove_equation is None:
                if sum_array_test != 2:
                    raise ValueError(
                        f"With add_equation, only 2 constraints are necessary. {error_msg}"
                    )
            else:
                if sum_array_test != 3:
                    raise ValueError(
                        "With remove_equation and add_equation, 3 constraints are "
                        + f"necessary. {error_msg}"
                    )
            if "=" in add_equation:
                raise ValueError(
                    "All terms of the added equation must be "
                    + "on the same side, and there is no need to specify = 0, "
                    + f"the input equation is {add_equation}."
                )

            if all(sym not in add_equation for sym in input_constraints):
                raise ValueError(
                    "The input equation must relate any of the 8 "
                    + f"unknown arrays that are {input_constraints}."
                    + f"\nThe input equation is {add_equation}."
                )
        else:
            if remove_equation is None:
                # If no additional equation is given, 3 arrays must
                # be input to find a solution.
                if sum_array_test != 3:
                    raise ValueError(f"3 constraints are necessary. {error_msg}")
            else:
                if sum_array_test != 4:
                    raise ValueError(
                        "With remove_equation, only 4 constraints are necesasary."
                        + f" {error_msg}"
                    )

        if not quiet and first_inv:
            print(f"Input arrays are {constraint_test}.")
            print(f"Solving for {not_constraint}.")
            add_eq_prev = ""
            if filter_in is not None:
                print("Use input filter")
            elif filter_type is not None:
                print(
                    f"Minimum {'curvature' if filter_type == 'Mc' else 'amplitude'} filter"
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

        if first_inv:
            lambdify_func = np.zeros((lmax + 1), dtype=object)

        # Allocate arrays to be used for outputs.
        shape = (2, lmax + 1, lmax + 1)
        w_lm = np.zeros(shape) if w_lm is None else w_lm
        Gc_lm = np.zeros(shape) if Gc_lm is None else Gc_lm
        q_lm = np.zeros(shape) if q_lm is None else q_lm
        omega_lm = np.zeros(shape) if omega_lm is None else omega_lm
        drhom_lm = np.zeros(shape) if drhom_lm is None else drhom_lm
        G_lm = np.zeros(shape) if G_lm is None else G_lm
        H_lm = np.zeros(shape) if H_lm is None else H_lm
        dc_lm = np.zeros(shape) if dc_lm is None else dc_lm
        wdc_corr = np.zeros(shape) if wdc_corr is None else wdc_corr
        H_corr = np.zeros(shape) if H_corr is None else H_corr
        w_corr = np.zeros(shape) if w_corr is None else w_corr
        drhom_crust = np.zeros(shape) if drhom_crust is None else drhom_crust
        drho_omega_corr = (
            np.zeros(shape) if drho_omega_corr is None else drho_omega_corr
        )
        drho_q_corr = np.zeros(shape) if drho_q_corr is None else drho_q_corr
        A_lm = np.zeros(shape)

        if Te == 0:  # Avoid numerical problems with infinite values
            Te = 1
            if first_inv:
                print(
                    "! Warning:Thin_shell_matrix, elastic thickness set to 1 m "
                    + "to avoid numerical problems !"
                )

        # Precompute some constants.
        drholm_profile_check = False
        if rho_depth is not None and rho_lm_profile is not None:
            drholm_profile_check = True
            # Last interface is assumed to be thin
            M = np.diff(rho_depth, append=rho_depth[-1] + 0.1e-8)
            if np.any(M < 0):
                raise ValueError(
                    "Thickness of the density anomaly is negative. "
                    + "Change order of rho_depth"
                )
        else:
            M = base_drho - top_drho  # Thickness of the density anomaly
            if M < 0:
                raise ValueError(
                    "Thickness of the density anomaly (base_drho - top_drho) is negative. "
                    + "base_drho and top_drho are probably inverted with values of "
                    + f"{base_drho / 1e3:.2f} and {top_drho / 1e3:.2f} (km), respectively"
                )

        Re = R - Te / 2.0  # Reference radius for displacement equations
        R_base_drho = R - base_drho
        R_top_drho = R - top_drho
        R_c = R - c
        Re4 = Re**4

        drhol = rhoc - rhol
        eps = 12.0 * Re**2 / Te**2
        alph_B = 1.0 / (self.E * Te)
        # Avoids error printing when dividing by zero.
        D = self.E * Te**3 / (12.0 * (1.0 - self.v**2))  # Shell's
        # rigidity.
        v1v = self.v / (1.0 - self.v)
        RCR = R_c / R
        beta_B = 1.0 / (1.0 + eps)
        eta_B = eps / (1.0 + eps)
        # Mass correction for the mantle density anomaly to account for the
        # planet sphericity, work in progress.
        mass_correc = (
            1.0
            / 3.0
            * (R_base_drho**3 - R_top_drho**3)
            / (R_top_drho**2 * (R_base_drho - R_top_drho))
        )
        mass_correc = 1.0

        drho = rhom - rhoc
        gmoho = g0 * (1.0 + (RCR**3 - 1.0) * rhoc / rhobar) / RCR**2
        if rho_depth is not None and c in rho_depth:
            if np.all(rho_depth <= c):
                rho_c_mantle = rho_lm_profile[np.argmin(np.abs(c - rho_depth)), 0, 0, 0]
                gmoho = g0 * (1.0 + (RCR**3 - 1.0) * rho_c_mantle / rhobar) / RCR**2
                drho = rhom - rho_c_mantle
            elif np.all(rho_depth >= c):
                drho = rho_lm_profile[0, 0, 0, 0] - rhoc
            else:
                raise ValueError(
                    (
                        "This function cannot yet deal with density variations "
                        "in both the crust and mantle. Here rho_depth has values "
                        "lower and higher than crustal thickness."
                    )
                )

        RTeR = (R - Te) / R
        if Te <= c:
            gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rhoc / rhobar) / RTeR**2
        else:
            gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rhom / rhobar) / RTeR**2

        if drholm_profile_check:
            gdrho = np.zeros_like(rho_depth)
            for i, depth in enumerate(rho_depth):
                R_top_drho = R - depth
                R_base_drho = R - depth - M[i]
                R_drho_mid = (R_top_drho + R_base_drho) / 2.0
                gdrho[i] = (
                    g0
                    * (
                        1.0
                        + ((R_drho_mid / R) ** 3 - 1.0)
                        * rho_lm_profile[i, 0, 0, 0]
                        / rhobar
                    )
                    / (R_drho_mid / R) ** 2
                )

        else:
            R_drho_mid = (R_top_drho + R_base_drho) / 2.0
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

        # Store symbolized array names with sympy. Order is
        # important. These will be denoted e.g. 'H_lm1' for H_lm.
        add_constraints = ""

        if add_arrays is not None:
            for i in [0] if single_add_arrays else range(np.shape(add_arrays)[0]):
                if i + 1 not in add_muls:
                    add_constraints += f" add_array{i+1}"
        add_constraints += (
            " wdc_corr1 w_corr1 H_corr1 drho_omega_corr1 drho_q_corr1 drhom_crust1"
        )

        if drholm_profile_check:
            # Add the interior density profile as a known
            for i in range(len(rho_depth)):
                add_constraints += f" drho_lm1_{i} "
            # Remove drhom_lm1 from the constraints
            add_constraints = add_constraints.replace(" drhom_lm1", "")
            not_constraint = not_constraint[not_constraint != ["drhom_lm"]]
            constraint_test = constraint_test[constraint_test != ["drhom_lm"]]

        a_symb_uknwn = symbols(" ".join([symb + "1 " for symb in not_constraint]))
        a_symb_knwn = symbols(
            " ".join([symb + "1 " for symb in constraint_test]) + add_constraints
        )
        args_symb = (constraint_test, not_constraint, a_symb_uknwn)

        # Determine how symbols are listed in the outputs because
        # solutions order depends on the input symbol order,
        # which depends on the user inputs.
        a_symbs = tuple(str(x) for x in a_symb_uknwn + a_symb_knwn)
        idx_w_lm = a_symbs.index("w_lm1")
        idx_G_lm = a_symbs.index("G_lm1")
        idx_Gc_lm = a_symbs.index("Gc_lm1")
        idx_H_lm = a_symbs.index("H_lm1")
        idx_omega_lm = a_symbs.index("omega_lm1")

        if not drholm_profile_check:
            idx_drhom_lm = a_symbs.index("drhom_lm1")

        idx_dc_lm = a_symbs.index("dc_lm1")
        idx_q_lm = a_symbs.index("q_lm1")

        if remove_equation is not None and not quiet and first_inv:
            print(f"Removing equation for: {remove_equation}.")
        if add_equation is not None:
            if not quiet and first_inv:
                print(f"Adding an equation:\n{add_equation}.")
            # Reformat added equation for sympy
            for string in input_constraints:
                add_equation = re.sub(rf"(\b{string}\b)", f"{string}1", add_equation)
            add_equation = parse_expr(add_equation)

        degrees = np.arange(lmax + 1, dtype=float)
        Lapla = -degrees * (degrees + 1)  # Laplacian identity.

        param_filt = (degrees, self.filter_half)
        kw_filt = {"filter_type": filter_type, "quiet": quiet}

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
            elif filter_type is not None:
                if any_dc:
                    DCfilter_mohoD = DownContFilter(
                        *param_filt,
                        R,
                        R_c,
                        **kw_filt,
                    )
                    DCfilter_mohoDc = DownContFilter(*param_filt, R_c, R_c, **kw_filt)
                else:
                    DCfilter_drhom = DownContFilter(
                        *param_filt, R, R_base_drho, **kw_filt
                    )
                    DCfilter_drhomc = DownContFilter(
                        *param_filt, R_c, R_base_drho, **kw_filt
                    )

            rhobconst = 3.0 / (rhobar * (2.0 * degrees + 1.0))

            # Continuation arrays
            Rl3 = R / (degrees + 3.0)
            RCRl = RCR**degrees
            # RCRl1 = RCR ** (degrees + 1.0)
            RCRl2 = RCR ** (degrees + 2.0)

            if drholm_profile_check:
                DCfilter_drhom = np.ones((len(rho_depth), len(degrees)))
                DCfilter_drhomc = np.ones_like(DCfilter_drhom)
                RtRCl = np.zeros_like(DCfilter_drhom)
                RbRCl = np.zeros_like(DCfilter_drhom)
                RtbRl3_profile = np.zeros_like(DCfilter_drhom)

                for i, depth in enumerate(rho_depth):
                    R_top_drho = R - depth
                    R_base_drho = R - depth - M[i]

                    if filter_in is not None and not any_dc:
                        DCfilter_drhom[i], DCfilter_drhomc[i] = filter_in, filter_in
                    elif filter_type is not None and not any_dc:
                        DCfilter_drhom[i] = DownContFilter(
                            *param_filt,
                            R,
                            R_base_drho,
                            **kw_filt,
                        )
                        DCfilter_drhomc[i] = DownContFilter(
                            *param_filt,
                            R_c,
                            R_base_drho,
                            **kw_filt,
                        )

                    if R_top_drho <= R_c:
                        RtRCl[i] = (R_top_drho / R_c) ** degrees
                    else:
                        RtRCl[i] = (R_c / R_top_drho) ** (degrees + 1.0)
                    if R_base_drho <= R_c:
                        RbRCl[i] = (R_base_drho / R_c) ** degrees
                    else:
                        RbRCl[i] = (R_c / R_base_drho) ** (degrees + 1.0)

                    RtRCl[i] *= R_top_drho**3 / (R_c * R**2)
                    RbRCl[i] *= R_base_drho**3 / (R_c * R**2)
                    RtbRl3_profile[i] = (R_top_drho / R) ** (degrees + 3.0) - (
                        R_base_drho / R
                    ) ** (degrees + 3.0)
            else:
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

        # For drhom_crust
        RtRCl_c = RCR ** (degrees + 1.0)
        RtRCl_c *= R**3 / (R_c * R**2)
        RbRCl_c = 1**degrees
        RbRCl_c *= R_c**3 / (R_c * R**2)
        RtbRl3_c = 1 ** (degrees + 3.0) - RCR ** (degrees + 3.0)
        DCfilter_drhom_c = DownContFilter(*param_filt, R, R_c, **kw_filt)
        DCfilter_drhomc_c = DownContFilter(*param_filt, R_c, R_c, **kw_filt)
        gdrho_c = (
            g0
            * (1.0 + ((R_drho_mid / R) ** 3 - 1.0) * rhoc / rhobar)
            / (R_drho_mid / R) ** 2
        )
        M_c = c

        if first_inv and (
            (
                np.sum(DCfilter_drhomc) != lmax + 1
                and (
                    np.sum(DCfilter_drhomc) / len(rho_depth) != lmax + 1
                    if rho_depth is not None
                    else 1
                )
            )
            and np.sum(DCfilter_mohoD) != lmax + 1
        ):
            raise ValueError("Double filtering error")

        if not quiet and first_inv:
            if np.sum(DCfilter_drhomc) != lmax + 1 and (
                np.sum(DCfilter_drhomc) / len(rho_depth) != lmax + 1
                if rho_depth is not None
                else 1
            ):
                print("Filtering drhom (interior density)")
            elif np.sum(DCfilter_mohoD) != lmax + 1:
                print("Filtering dc_lm (crust–mantle relief)")

        # Solve matrix over all degrees.
        for l in range(1, lmax + 1):  # Ignore degree 0 from calculations
            if first_inv:
                # Symbolic definition.
                w_lm1, Gc_lm1, q_lm1, omega_lm1, dc_lm1, drhom_lm1, G_lm1, H_lm1 = (
                    symbols(
                        " w_lm1 Gc_lm1 q_lm1 omega_lm1 dc_lm1 drhom_lm1 G_lm1 H_lm1 "
                    )
                )

                # Add interior density anomalies
                if drholm_profile_check:
                    drho_lm1_ = var(
                        ", ".join([f"drho_lm1_{i}" for i in range(len(rho_depth))])
                    )

                (
                    wdc_corr1,
                    w_corr1,
                    H_corr1,
                    drho_omega_corr1,
                    drho_q_corr1,
                    drhom_crust1,
                ) = symbols(
                    " wdc_corr1 w_corr1 H_corr1 drho_omega_corr1 drho_q_corr1 drhom_crust1 "
                )
                # System of equations from Banerdt (1986).

                if drholm_profile_check:
                    Eqns = [
                        # eq (1) G_lm
                        -G_lm1
                        + (
                            rhobconst[l]
                            * (
                                rhol * H_lm1
                                + drhol * w_lm1
                                + drho * (w_lm1 - dc_lm1) * RCRl2[l] / DCfilter_mohoD[l]
                                + sum(
                                    drho_lm1_[i]
                                    * Rl3[l]
                                    * RtbRl3_profile[i, l]
                                    / DCfilter_drhom[i, l]
                                    for i in range(len(rho_depth))
                                )
                                + drhom_crust1
                                * Rl3[l]
                                * RtbRl3_c[l]
                                / DCfilter_drhom_c[l]
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
                            * (g0 / gmoho)
                            * (
                                (rhol * H_lm1 + drhol * w_lm1) * RCRl[l]  # RCRl1[l]
                                + drho
                                * (w_lm1 - dc_lm1)
                                * RCR  # **3
                                / DCfilter_mohoDc[l]
                                + sum(
                                    drho_lm1_[i]
                                    * Rl3[l]
                                    * (RtRCl[i, l] - RbRCl[i, l])
                                    / DCfilter_drhomc[i, l]
                                    for i in range(len(rho_depth))
                                )
                                + drhom_crust1
                                * Rl3[l]
                                * (RtRCl_c[l] - RbRCl_c[l])
                                / DCfilter_drhomc_c[l]
                            )
                            + (
                                rhol * H_corr1
                                + ((drhol * w_corr1) if not w_corr_test else w_corr1)
                            )
                            * RCRl[l]  # RCRl1[l]
                            + drho * wdc_corr1 * RCR  # **3  # / DCfilter_mohoDc[l]
                            # Still unsure about that filtering part
                        )
                        * (
                            0.0 if "Gc_lm" in not_constraint and COM and l == 1 else 1.0
                        ),  # Force the degree-1 geoid to zero
                        # eq (3) q_lm
                        -q_lm1
                        + g0 * (rhol * (H_lm1 - G_lm1) + drhol * w_lm1)
                        + gmoho * drho * (w_lm1 - dc_lm1 - Gc_lm1)
                        + sum(
                            gdrho[i] * drho_lm1_[i] * M[i]
                            for i in range(len(rho_depth))
                        )
                        + gdrho_c * drhom_crust1 * M_c
                        + drho_q_corr1,
                        # eq (4) w_lm
                        eta_B * D * Lapla[l] * (Lapla[l] + 2) ** 2 * w_lm1
                        + Re**2 / alph_B * (Lapla[l] + 2) * w_lm1
                        + Re4 * ((Lapla[l] + 2) - 1.0 - self.v) * q_lm1
                        - Re4
                        * (beta_B * (Lapla[l] + 2) - 1.0 - self.v)
                        * Lapla[l]
                        * omega_lm1,
                        # eq (5) omega_lm
                        -omega_lm1
                        + v1v * rhol * g0 * Te * H_lm1 / R
                        - (
                            drhol * g0 * v1v * Te
                            - rhoc * gmoho * (c if c < Te else 0)
                            # If crust-mantle interface below Te, no tangential load associated
                            - rhom * gTe * max(Te - c, 0)
                            # If crust-mantle interface below Te, no tangential load associated
                        )
                        * w_lm1
                        / R
                        + v1v
                        * drho
                        * gmoho
                        * max(Te - c, 0)
                        * (dc_lm1 - w_lm1)
                        / R
                        - 0.5
                        * v1v
                        * sum(
                            drho_lm1_[i]
                            * gdrho[i]
                            * (Te - top_drho)
                            * (min(M[i], Te - top_drho) if top_drho < Te else 0)
                            for i in range(len(rho_depth))
                        )
                        # If mantle load below Te, no tangential load associated
                        / R
                        - 0.5
                        * v1v
                        * drhom_crust1
                        * mass_correc
                        * gdrho_c
                        * Te
                        * min(M_c, Te)
                        / R
                        + drho_omega_corr1,
                    ]
                else:
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
                            * (g0 / gmoho)
                            * (
                                (rhol * H_lm1 + drhol * w_lm1) * RCRl[l]  # RCRl1[l]
                                + drho
                                * (w_lm1 - dc_lm1)
                                * RCR
                                / DCfilter_mohoDc[l]  # RCR**3 / DCfilter_mohoDc[l]
                                + drhom_lm1
                                * Rl3[l]
                                * (RtRCl[l] - RbRCl[l])
                                / DCfilter_drhomc[l]
                            )
                            + (
                                rhol * H_corr1
                                + ((drhol * w_corr1) if not w_corr_test else w_corr1)
                            )
                            * RCRl[l]  # RCRl1[l]
                            + drho * wdc_corr1 * RCR  # **3  # / DCfilter_mohoDc[l]
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
                        + Re4 * ((Lapla[l] + 2) - 1.0 - self.v) * q_lm1
                        - Re4
                        * (beta_B * (Lapla[l] + 2) - 1.0 - self.v)
                        * Lapla[l]
                        * omega_lm1,
                        # eq (5) omega_lm
                        -omega_lm1
                        + v1v * rhol * g0 * Te * H_lm1 / R
                        - (
                            drhol * g0 * v1v * Te
                            - rhoc * gmoho * (c if c < Te else 0)
                            # If crust-mantle interface below Te, no tangential load associated
                            - rhom * gTe * max(Te - c, 0)
                            # If crust-mantle interface below Te, no tangential load associated
                        )
                        * w_lm1
                        / R
                        + v1v
                        * drho
                        * gmoho
                        * max(Te - c, 0)
                        * (dc_lm1 - w_lm1)
                        / R
                        - 0.5
                        * v1v
                        * drhom_lm1
                        * mass_correc
                        * gdrho
                        * (Te - top_drho)
                        * (min(M, Te - top_drho) if top_drho < Te else 0)
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
                                    f"add_array{i+1}",
                                    (
                                        add_arrays[0, l, 0]
                                        if single_add_arrays
                                        else add_arrays[i, 0, l, 0]
                                    ),
                                )

                    if not quiet and add_equation_subbed != add_eq_prev and first_inv:
                        add_eq_prev = add_equation_subbed
                        print(
                            f"Additional equation starting at degree {l} is "
                            f"{add_equation_subbed if add_equation_subbed != parse_expr('0') else '0 = 0'}"
                        )
                    if add_equation_subbed != _zero_expr:
                        Eqns.insert(len(Eqns), add_equation_subbed)
                    else:
                        if np.size(input_constraints) - sum_array_test != 5:
                            raise ValueError(
                                f"System cannot be determined at degree {l} "
                                + "where add_equation becomes 0 = 0"
                            )

                # w_lm should be zero at degree-1 for a static body.
                # In order for this to be properly handled, we replace
                # the w_lm degree-1 equation eq(4) by w_lm = 0 if:
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
                    Eqns[4] = w_lm1

                if remove_equation is not None and l != 1:
                    for item in [remove_equation]:
                        Eqns.pop(int(np.where(equation_order == item)[0][0]))

                # Rearange system of equations using sympy.
                sol = linsolve(Eqns, a_symb_uknwn + a_symb_knwn)

                # Vectorize the linsolve function.
                # Store matrix solution for potential later reutilisation
                lambdify_func[l] = lambdify(a_symb_uknwn + a_symb_knwn, list(sol))

            # Results.
            # Depending on the input arrays, pass a symbol or the
            # input values.
            args_linsolve = {
                "w_lm1": test_symb("w_lm", w_lm[:, l, : l + 1], *args_symb),
                "Gc_lm1": test_symb("Gc_lm", Gc_lm[:, l, : l + 1], *args_symb),
                "G_lm1": test_symb("G_lm", G_lm[:, l, : l + 1], *args_symb),
                "H_lm1": test_symb("H_lm", H_lm[:, l, : l + 1], *args_symb),
                "q_lm1": test_symb("q_lm", q_lm[:, l, : l + 1], *args_symb),
                "omega_lm1": test_symb("omega_lm", omega_lm[:, l, : l + 1], *args_symb),
                "dc_lm1": test_symb("dc_lm", dc_lm[:, l, : l + 1], *args_symb),
                "wdc_corr1": wdc_corr[:, l, : l + 1],
                "H_corr1": H_corr[:, l, : l + 1],
                "drhom_crust1": drhom_crust[:, l, : l + 1],
                "w_corr1": w_corr[:, l, : l + 1],
                "drho_omega_corr1": drho_omega_corr[:, l, : l + 1],
                "drho_q_corr1": drho_q_corr[:, l, : l + 1],
            }

            if add_arrays is not None:
                add_arr = ""
                for i in [0] if single_add_arrays else range(np.shape(add_arrays)[0]):
                    if i + 1 not in add_muls:
                        add_arr += f"'add_array{i + 1}': add_arrays[{i + ',' if not single_add_arrays else ''} :, l, : l + 1], "
                args_linsolve = dict(args_linsolve, **dict(eval(f"{{{add_arr}}}")))

            if drholm_profile_check:
                add_arr = ""
                for i in range(len(rho_depth)):
                    add_arr += f"'drho_lm1_{i}':rho_lm_profile[{i}, :, l, : l + 1], "
                args_linsolve = dict(args_linsolve, **dict(eval(f"{{{add_arr}}}")))
            else:
                args_linsolve.update(
                    {
                        "drhom_lm1": test_symb(
                            "drhom_lm", drhom_lm[:, l, : l + 1], *args_symb
                        )
                    }
                )

            outs = np.concatenate(
                np.array(lambdify_func[l](**args_linsolve), dtype=object)
            )

            if np.any([isinstance(arrs, Expr) for arrs in outs]):
                raise ValueError(
                    f"System is non-evenly determined at degree {l}, cannot solve"
                    + f"\nSystem of equations: \n{Eqns}"
                    + "\nSolutions found:"
                    + f"\nw_lm = {outs[idx_w_lm]}"
                    + f"\nq_lm = {outs[idx_q_lm]}"
                    + f"\nomega_lm = {outs[idx_omega_lm]}"
                    + f"\ndc_lm = {outs[idx_dc_lm]}"
                    + (
                        f"\ndrhom_lm = {outs[idx_drhom_lm]}"
                        if not drholm_profile_check
                        else ""
                    )
                    + f"\nG_lm = {outs[idx_G_lm]}"
                    + f"\nGc_lm = {outs[idx_Gc_lm]}"
                    + f"\nH_lm = {outs[idx_H_lm]}"
                    + (
                        f"\nMake sure the add_equation doesn't involve w_lm, G_lm, or Gc_lm, "
                        f"which are specifically treated when COM is True (default)"
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
            if not drholm_profile_check:
                drhom_lm[:, l, : l + 1] = outs[idx_drhom_lm]
            G_lm[:, l, : l + 1] = outs[idx_G_lm]
            H_lm[:, l, : l + 1] = outs[idx_H_lm]

            # Tangential displacement (eq. 89 Beuthe 2008)
            # Note that omega (Beuthe) = Re * omega
            A_lm[:, l, : l + 1] = (
                beta_B
                * (1.0 / (1.0 - self.v**2))
                * (Lapla[l] + 1.0 + self.v)
                * (Lapla[l] + 2)
                * w_lm[:, l, : l + 1]
                + w_lm[:, l, : l + 1]
                + Re**2 * alph_B * q_lm[:, l, : l + 1]
                - Re
                * alph_B
                / (1.0 + eps)
                * (Lapla[l] - eps * (1.0 + self.v))
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

    # ==== invert_matrix_nmax ====

    def invert_matrix_nmax(
        self,
        H_lm=None,
        drhom_lm=None,
        drhom_crust=None,
        dc_lm=None,
        w_lm=None,
        omega_lm=None,
        q_lm=None,
        G_lm=None,
        Gc_lm=None,
        add_equation=None,
        add_arrays=None,
        remove_equation=None,
        COM=True,
        rho_lm_profile=None,
        rho_depth=None,
        rho_crust_parallel=False,
        base_drho=None,
        top_drho=None,
    ):
        """
        Solve the Banerdt (1986) system of 5 equations
        with finite-amplitude correction and accounting
        for the potential presence of density variations
        within the surface or crust–mantle reliefs.

        Returns
        -------
        lambdify_func : array, size(2,lmax+1,lmax+1)
            An array with the lambda functions
            (i.e., the design of the inversion matrix without the inputs)
            of all components. Lambda functions can be used to re-calculate
            the same problem with different inputs very fast.

        Parameters
        ----------
        H_lm : array, size(2,lmax+1,lmax+1), optional, default = None
            Array with the spherical harmonic coefficients of the
            planet's shape.
        drhom_lm : array, size(2,lmax+1,lmax+1), optional, default = None
            Array with the spherical harmonic coefficients of the
            lateral density variations.
        drhom_crust : array, size(2,lmax+1,lmax+1), optional, default = None
            Array with the spherical harmonic coefficients of the
            crustal lateral density variations. This parameter only works
            if rho_lm_profile is true and set for the mantle. To specify
            crustal density when rho_lm_profile is false please use drhom_lm.
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
            geoid at the crust–mantle interface.
        add_equation : string, optional, default = None
            Equation to be added to the system. This must include at least
            one of the 8 parameters aboves.
        add_arrays : array size(N, 2,lmax+1,lmax+1), optional, default = None
            N arrays of spherical harmonics to be added in 'add_equation', which
            are written 'add_array1' 'add_array2' etc. Order is important.
        COM : bool, optional, default = True
            If True, force the model to be in a center-of-mass frame by setting
            the degree-1 geoid terms to zero.
        remove_equation : string, optional, default = None
            String of the equation to be removed. This must be either
            'G_lm', 'Gc_lm', 'w_lm', 'omega_lm', or 'q_lm'.
        rho_lm_profile : array, optional, default = None
            Spherical harmonic expansion of the density variations at each of
            the depths specified in rho_depth.
            Size must be (len(rho_depth), 2,lmax+1,lmax+1).
        rho_depth : array, optional, default = None
            Depth array associated with the input interior density variations.
        rho_crust_parallel : bool, optional, default = False
            If true, assume that density variations in the crust have
            relief parallel to the surface topography.
        base_drho : float, optional, default = None
            Lower depth for the of the density contrast. If None, set to c.
        top_drho : float, optional, default = None
            Upper depth for the of the density contrast. If None, set
            to 0 km (surface).
        """

        R = self.R
        c = self.c
        rhom = self.rhom
        rhoc = self.rhoc
        rhol = self.rhol
        quiet = self.quiet

        if base_drho is None:
            base_drho = c
        if top_drho is None:
            top_drho = 0

        # Add crust-mantle interface in drhom_lm_profile if not present
        if rho_depth is not None and np.all(rho_depth <= c):
            if 0 not in rho_depth:
                raise ValueError("rho_depth should start at a depth of 0")
            if c not in rho_depth:
                if not quiet:
                    print(
                        (
                            "Add crust-mantle interface in drhom_lm_profile",
                            " and assume that the density contrast is the same",
                            " as in the last rho_lm_profile layer",
                        )
                    )
                rho_depth = np.insert(rho_depth, len(rho_depth), c)
                rho_lm_profile = np.insert(
                    rho_lm_profile, len(rho_depth), rho_lm_profile[-1], axis=0
                )

        if rho_lm_profile is not None and rho_depth is not None:
            if not quiet:
                print("Using input drho profile")
            if np.shape(rho_lm_profile)[0] != np.shape(rho_depth)[0]:
                raise ValueError(
                    "rho_lm_profile and rho_depth must have the same shape. "
                    + f"Shapes are {np.shape(rho_lm_profile)[0]} and"
                    + f" {np.shape(rho_depth)[0]}"
                )
            # Deals with unknowns in Thin_shell_matrix
            drhom_lm = rho_lm_profile[0, :, :, :]

            if np.any(rho_depth > c) and np.any(rho_depth < c):
                raise ValueError(
                    (
                        "rho_lm_profile cannot accept profiles encompassing",
                        " both the mantle and crust. Here rho_depth has values",
                        f" lower and higher than c, with c = {c/1e3:5d}",
                    )
                )

        elif drhom_crust is not None:
            raise ValueError(
                (
                    "The drhom_crust parameter only works together with",
                    " drholm_profile_check. To set the crustal density",
                    " please use drhom_lm with top_drho < c and base_drho <= c.",
                )
            )

        args_grid = {
            "sampling": 2,
            "lmax": self.lmaxgrid,
            "extend": False,
            "lmax_calc": self.lmax,
        }

        R_c = R - c
        args_param_lm = {
            "H_lm": H_lm,
            "drhom_lm": drhom_lm,
            "dc_lm": dc_lm,
            "w_lm": w_lm,
            "omega_lm": omega_lm,
            "q_lm": q_lm,
            "G_lm": G_lm,
            "drhom_crust": drhom_crust,
            "Gc_lm": Gc_lm,
            "base_drho": base_drho,
            "top_drho": top_drho,
            "rho_depth": rho_depth,
            "rho_lm_profile": rho_lm_profile,
            "add_arrays": add_arrays,
            "remove_equation": remove_equation,
            "add_equation": add_equation,
            "COM": COM,
        }

        # Precompute some sums that will be used later for checks
        any_dc = np.sum(dc_lm) != 0 if dc_lm is not None else None
        any_w = np.sum(w_lm) != 0 if w_lm is not None else None
        any_drho = np.sum(drhom_lm) != 0 if drhom_lm is not None else None

        # Density contrast not at topography or crust–mantle interface and no
        # finite-amplitude correctio, return
        if (
            self.nmax == 1
            and (not any_drho or (top_drho != 0 and base_drho != c))
            and any_drho is not None
            or not self.iterate
        ):
            (
                w_lm_o,
                A_lm_o,
                moho_lm_o,
                dc_lm_o,
                drhom_lm_o,
                omega_lm_o,
                q_lm_o,
                Gc_lm_o,
                G_lm_o,
                H_lm_o,
                lambdify_func_o,
            ) = self.invert_matrix(**args_param_lm)

            if not quiet:
                print("Returning without corrections")
                print("Set the interfaces degree-0 coefficients")
            w_lm_o[0, 0, 0] = R
            dc_lm_o[0, 0, 0] = 0
            moho_lm_o[0, 0, 0] = R_c
            H_lm_o[0, 0, 0] = R

            self.w_lm = SHCoeffs.from_array(w_lm_o)
            self.A_lm = SHCoeffs.from_array(A_lm_o)
            self.moho_lm = SHCoeffs.from_array(moho_lm_o)
            self.crust_lm = SHCoeffs.from_array(H_lm_o - moho_lm_o)
            self.dc_lm = SHCoeffs.from_array(dc_lm_o)
            self.drhom_lm = SHCoeffs.from_array(drhom_lm_o)
            self.omega_lm = SHCoeffs.from_array(omega_lm_o)
            self.q_lm = SHCoeffs.from_array(q_lm_o)
            self.Gc_lm = SHCoeffs.from_array(Gc_lm_o)
            self.G_lm = SHCoeffs.from_array(G_lm_o)
            self.H_lm = SHCoeffs.from_array(H_lm_o)
            self.sols = lambdify_func_o

            return

        # Correct for density contrast in surface or crust–mantle relief
        # relief, and/or finite-amplitude correction
        density_var_H = density_var_dc = density_var_w = False
        # Precompute grids
        precomp_drho = precomp_H_grid = precomp_w_grid = precomp_dc_grid = precomprho_grid_c = comp_rho_grid = False

        if drhom_lm is None or any_drho:
            if top_drho == 0 or (rho_depth is not None and 0 in rho_depth):
                # Correct for density variations in the surface
                # relief
                density_var_H = True
            if (c in (base_drho, top_drho)) or (
                rho_depth is not None and c in rho_depth
            ):
                # Correct for density variations in the crust–mantle
                # relief
                density_var_dc = True
            if base_drho < c and top_drho == 0 and rhol == rhoc:
                # Correct for density variations in the flexure relief
                # within the crust
                density_var_w = True

        # If only finite-amplitude correction, density
        # contrast is multipled in the thin-shell code and
        # we set the density contrast to 1. This will be changed later if required.
        ones = np.ones((2 * (self.lmaxgrid + 1), 2 * (2 * (self.lmaxgrid + 1))))
        H_drho_grid = w_drho_grid = wdc_drho_grid = ones
        drho_H = drho_wdc = drho_w = 1.0

        if drhom_lm is not None and any_drho:
            rho_grid = MakeGridDH(drhom_lm, **args_grid)
            comp_rho_grid = True
            precomp_drho = True
            if drhom_lm[0, 0, 0] > 1000:
                if base_drho <= c:
                    rhoc = drhom_lm[0, 0, 0]
                    rhol = drhom_lm[0, 0, 0]
                    if not quiet:
                        print(
                            "rhol and rhoc are set to the mean "
                            + f"input density variations ({rhoc:.2f} kg m-3)"
                        )
                else:
                    rhom = drhom_lm[0, 0, 0]
                    if not quiet:
                        print(
                            f"rhom is set to the mean input density variations ({rhom:.2f} kg m-3)"
                        )
                # Update parameters
                self.rhoc = rhoc
                self.rhol = rhol
                self.rhom = rhom
            else:
                # Density variations is in the crust
                if base_drho <= c:
                    rho_grid += rhoc
                    if not quiet:
                        print(
                            f"Add input rhoc ({rhoc:.2f} kg m-3) to crust density variations"
                        )
                # Density variations is in the mantle
                else:
                    rho_grid += rhom
                    if not quiet:
                        print(
                            f"Add input rhom ({rhom:.2f} kg m-3) to mantle density variations"
                        )

        precomp_drhocrust = False
        if drhom_crust is not None:
            rhocrust_grid = MakeGridDH(drhom_crust, **args_grid)
            precomp_drhocrust = True
            if drhom_lm[0, 0, 0] > 1000:
                rhoc = drhom_crust[0, 0, 0]
                rhol = drhom_crust[0, 0, 0]
                if not quiet:
                    print(
                        f"rhol and rhoc are set to the mean input density variations ({rhoc:.2f} kg m-3)"
                    )
                # Update parameters
                self.rhoc = rhoc
                self.rhol = rhol
            else:
                rhocrust_grid += rhoc
                if not quiet:
                    print(
                        f"Add input rhoc ({rhoc:.2f} kg m-3) to crust density variations"
                    )

        # Geoid correction due to density variations
        # and or finite-amplitude corrections
        shape = (2, self.lmax + 1, self.lmax + 1)
        # crust–mantle relief
        # Deflected topography relief
        # Surface topography relief
        # Tangential load potential corrections due to density
        # variations at the reliefs
        delta_wdc_geoid = delta_w_geoid = delta_H_geoid = drho_omega_corr = drho_q_corr = np.zeros(shape)

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

        if rho_depth is not None:
            if c in rho_depth:
                idx_c = np.argmin(np.abs(c - rho_depth))
                rho_c_mantle = rho_lm_profile[idx_c]
                if np.all(rho_depth <= c):
                    # Anomaly in the crust
                    wdc_drho_grid = rhom - MakeGridDH(rho_c_mantle, **args_grid)
                    drho_wdc = rhom - rho_c_mantle[0, 0, 0]
                if np.all(rho_depth >= c):
                    # Anomaly in the mantle
                    wdc_drho_grid = MakeGridDH(rho_c_mantle, **args_grid) - rhoc
                    drho_wdc = rho_c_mantle[0, 0, 0] - rhoc
            precomprho_grid_c = True

        lambdify_func_o = None
        first_inv, first_drhom, first_nmax = True, True, True
        delta = 1.0e9
        itera = 0
        degrees = np.arange(self.lmax + 1, dtype=float)
        args_c_nmax_d = (self.lmax, self.mass, self.nmax, R)

        # Iterate until convergence
        # First guess is using the mass-sheet case
        while delta > self.delta_max:
            itera += 1
            (
                w_lm_o,
                A_lm_o,
                moho_lm_o,
                dc_lm_o,
                drhom_lm_o,
                omega_lm_o,
                q_lm_o,
                Gc_lm_o,
                G_lm_o,
                H_lm_o,
                lambdify_func_o,
            ) = self.invert_matrix(
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

            if not precomp_drho:
                comp_rho_grid = False
            if not precomp_H_grid:
                comp_H_grid = False

            # Precompute some sums that will be used later for checks
            any_dc = np.sum(dc_lm_o[:, 1:, :]) != 0 if any_dc is None else any_dc
            any_w = np.sum(w_lm_o[:, 1:, :]) != 0 if any_w is None else any_w
            any_drho = (
                np.sum(drhom_lm_o[:, 1:, :]) != 0 if any_drho is None else any_drho
            )

            # Correct for density contrast in surface or crust–mantle
            # relief, and/or finite-amplitude correction
            if drhom_lm is None or any_drho:
                if not quiet and first_drhom:
                    first_drhom = False
                    print(
                        "Iterate to account for density variations "
                        f"{f'and finite-amplitude correction, nmax is {self.nmax}' if self.nmax > 1 else ''}"
                    )

            else:
                if not quiet and first_nmax:
                    first_nmax = False
                    print(
                        f"Iterate for finite-amplitude correction, nmax is {self.nmax}"
                    )

            # Scheme proposed in Wieczorek+(2013) SOM eq 21, 22
            # to speed up convergence delta(i+3) = (delta(i+2) +
            # delta(i+1))/2.
            if itera % 3 == 0:
                delta_wdc_geoid = (delta_wdc_geoid_2 + delta_wdc_geoid_1) / 2.0
                delta_H_geoid = (delta_H_geoid_2 + delta_H_geoid_1) / 2.0
                delta_w_geoid = (delta_w_geoid_2 + delta_w_geoid_1) / 2.0
                drho_omega_corr = (delta_drho_omega_2 + delta_drho_omega_1) / 2.0
                drho_q_corr = (delta_drho_q_2 + delta_drho_q_1) / 2.0
                if not quiet:
                    print(f"Skipping iteration {itera}, with convergence scheme")
                continue

            if (
                any_drho
                and not (precomp_drho or precomp_drhocrust)
                and rho_depth is None
            ):
                rho_grid = MakeGridDH(drhom_lm_o, **args_grid)
                rho_grid_var = rho_grid.copy() # Used in SH_mul for the corr factors
                comp_rho_grid = True

                if drhom_lm_o[0, 0, 0] > 1000:
                    if base_drho <= c:
                        rhoc = drhom_lm_o[0, 0, 0]
                        rhol = drhom_lm_o[0, 0, 0]
                    else:
                        rhom = drhom_lm_o[0, 0, 0]
                    self.rhoc = rhoc
                    self.rhol = rhol
                    self.rhom = rhom
                else:
                    if base_drho <= c:
                        rho_grid += rhoc
                    else:
                        rho_grid += rhom

            v1v = self.v / (1.0 - self.v)
            gmoho = (
                self.g0
                * (1.0 + (((R - c) / R) ** 3 - 1.0) * rhoc / self.rhobar)
                / ((R - c) / R) ** 2
            )

            # Correction for density variations in the surface topography relief
            if density_var_H:
                if not precomp_H_grid:
                    H_grid = MakeGridDH(H_lm_o, **args_grid)
                    comp_H_grid = True
                if not comp_rho_grid and not precomp_drho:
                    comp_rho_grid = True
                    rho_grid = MakeGridDH(drhom_lm_o, **args_grid) 

                mul_drho_H = SH_Mul(
                    drhom_lm_o, H_lm_o, grid1=rho_grid_var, grid2=H_grid, **args_grid
                ) # only take variations
                mul_drho_HG = SH_Mul(
                    drhom_lm_o, H_lm_o-G_lm_o, grid1=rho_grid_var, **args_grid
                ) # only take variations
                drho_H = rhol
                H_drho_grid = rho_grid
                drho_omega_corr = v1v * mul_drho_H * self.g0 * self.Te / R
                drho_q_corr = mul_drho_HG * self.g0

            # Correction for density variations in the crust–mantle relief
            if density_var_dc:
                if not comp_rho_grid and not precomp_drho:
                    comp_rho_grid = True
                    rho_grid = MakeGridDH(drhom_lm_o, **args_grid)

                # negative because density contrast (rhom-rhoc) matters here
                mul_drho_dc = SH_Mul(drhom_lm_o, dc_lm_o, grid1=-rho_grid_var if base_drho <= c else rho_grid_var, **args_grid)
                if density_var_H:
                    drho_omega_corr += v1v * mul_drho_dc * gmoho * max(self.Te - c, 0) / R
                    drho_q_corr += mul_drho_dc * gmoho
                else:
                    drho_omega_corr = v1v * mul_drho_dc * gmoho * max(self.Te - c, 0) / R
                    drho_q_corr = mul_drho_dc * gmoho

                if not precomprho_grid_c:
                    drho_wdc = rhom - rhoc
                    if base_drho <= c:
                        # Anomaly in the crust
                        wdc_drho_grid = rhom - rho_grid
                    else:
                        # Anomaly in the mantle
                        wdc_drho_grid = rho_grid - rhoc

            # Correction for density variations in the flexure relief
            if density_var_w:
                drho_w = (rhoc - rhol) if rhoc != rhol else 1
                w_drho_grid = rhoc - rho_grid

                mul_drho_w = SH_Mul(drhom_lm_o, w_lm_o, grid1=-rho_grid_var if top_drho == 0 else rho_grid_var, **args_grid)
                if density_var_H or density_var_dc:
                    drho_omega_corr += v1v * mul_drho_w * self.g0 * self.Te / R
                    drho_q_corr += mul_drho_w * self.g0
                else:
                    drho_omega_corr = v1v * mul_drho_w * self.g0 * self.Te / R
                    drho_q_corr = mul_drho_w * self.g0

            if drhom_crust is not None:
                if not precomp_H_grid and not comp_H_grid:
                    H_grid = MakeGridDH(H_lm_o, **args_grid)
                delta_H_geoid = corr_nmax_drho(
                    H_lm_o,
                    rhol,
                    H_grid,
                    rhocrust_grid,
                    *args_c_nmax_d,
                    degrees=degrees,
                    density_var=density_var_H,
                )
                wdc_lm_o = w_lm_o - dc_lm_o
                wdc_lm_o[0, 0, 0] = R_c
                if not precomp_dc_grid:
                    wdc_grid = MakeGridDH(wdc_lm_o, **args_grid)
                delta_wdc_geoid = corr_nmax_drho(
                    wdc_lm_o,
                    rhom - rhoc,
                    wdc_grid,
                    rhom - rhocrust_grid,
                    *args_c_nmax_d,
                    degrees=degrees,
                    density_var=density_var_dc,
                )

            H_lm_o[0, 0, 0] = R
            if (
                (any_drho and rho_depth is None and drhom_lm is None)
                or (itera == 1 and precomp_H_grid)
                or (not precomp_H_grid)
            ):
                # If density variations in surface relief or first iteration and H_lm
                # is an input or H_lm is not an input
                if not precomp_H_grid and not comp_H_grid:
                    H_grid = MakeGridDH(H_lm_o, **args_grid)

                delta_H_geoid = corr_nmax_drho(
                    H_lm_o,
                    drho_H,
                    H_grid,
                    H_drho_grid,
                    *args_c_nmax_d,
                    degrees=degrees,
                    density_var=density_var_H,
                )
                if (
                    rho_depth is not None
                    and np.all(rho_depth <= c)
                    and rho_crust_parallel
                ):
                    # Make sure we are in the crust
                    # Assume that density variations are // to surface
                    for idx_d, depth_a in enumerate(rho_depth):
                        # Density variations in the crust following the relief
                        if depth_a not in (0, c):
                            # depth_a = 0 done above
                            # depth_a = c done after
                            delta_H_geoid += (
                                CilmPlusRhoHDH(
                                    H_grid - depth_a,
                                    self.nmax,
                                    self.mass,
                                    MakeGridDH(
                                        rho_lm_profile[idx_d + 1]
                                        - rho_lm_profile[idx_d],
                                        **args_grid,
                                    ),
                                    lmax=self.lmax,
                                )[0]
                                * R
                                / rhol
                                * ((R - depth_a) / R) ** degrees.reshape(1, -1, 1)
                            )
                            # Pot -> geoid (*R)
                            # Divide by rhol as H_corr * rhol in ThinShell code
                            # Upward continue to R

            w_lm_o[0, 0, 0] = R
            if rhoc != rhol or density_var_w:
                if not precomp_w_grid:
                    w_grid = MakeGridDH(w_lm_o, **args_grid)
                    comp_w_grid = True  # Doesn't recompute at the end
                delta_w_geoid = corr_nmax_drho(
                    w_lm_o,
                    drho_w,
                    w_grid,
                    w_drho_grid if density_var_w else ones,
                    *args_c_nmax_d,
                    degrees=degrees,
                    density_var=density_var_w,
                )

            wdc_lm_o = w_lm_o - dc_lm_o
            wdc_lm_o[0, 0, 0] = R_c
            if any_dc or any_w:
                if not precomp_dc_grid:
                    wdc_grid = MakeGridDH(wdc_lm_o, **args_grid)
                delta_wdc_geoid = corr_nmax_drho(
                    wdc_lm_o,
                    drho_wdc,
                    wdc_grid,
                    wdc_drho_grid if density_var_dc else ones,
                    *args_c_nmax_d,
                    degrees=degrees,
                    density_var=density_var_dc,
                )

            if itera != 1:
                if not any_dc:
                    if any_w:
                        if not comp_w_grid:
                            w_grid = MakeGridDH(w_lm_o, **args_grid)
                            comp_w_grid = True
                        delta = abs(grid_prev - w_grid).max()
                        if not quiet:
                            print(f"Iteration {itera}, Delta (km) = {delta / 1e3:.3f}")
                            print(
                                f"Maximum displacement (km) = {((w_grid - R) / 1e3).max():.2f}"
                            )
                            print(
                                f"Minimum displacement (km) = {((w_grid - R) / 1e3).min():.2f}"
                            )
                    else:
                        delta = abs(grid_prev - rho_grid).max()
                        if not quiet:
                            print(f"Iteration {itera}, Delta (kg m-3) = {delta:.3f}")
                            print(f"Maximum density (kg m-3) = {rho_grid.max():.2f}")
                            print(f"Minimum density (kg m-3) = {rho_grid.min():.2f}")
                else:
                    delta = abs(grid_prev - (R - wdc_grid - c)).max()
                    if not quiet:
                        print(f"Iteration {itera}, Delta (km) = {delta / 1e3:.3f}")
                        crust_thick = (H_grid - wdc_grid) / 1e3
                        print(
                            f"Maximum Crustal thickness (km) = {crust_thick.max():.2f}"
                        )
                        print(
                            f"Minimum Crustal thickness (km) = {crust_thick.min():.2f}"
                        )

            # Speed up convergence scheme
            if itera % 2 == 0:
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
            var_relief = "Crust–mantle relief"
            if not any_dc and not any_w:
                var_relief = "Grid density"
                var_unit = "kg m-3"
            elif not any_dc:
                var_relief = "Flexure relief"

            if itera > self.iter_max:
                raise ValueError(
                    f"{var_relief} not converging, maximum iteration reached at {itera}, "
                    + f"delta was {delta / 1e3:.4f} ({var_unit}) and delta_max is {self.delta_max / 1e3:.4f} ({var_unit})."
                )
            if delta > self.delta_out and itera != 1:
                raise ValueError(
                    f"{var_relief} not converging, stopped at iteration {itera}, "
                    + f"delta was {delta / 1e3:.4f} ({var_unit}) and delta_out is {self.delta_out / 1e3:.4f} ({var_unit}). Try modifying nmax"
                    f"{' or try filtering.' if (self.filter_type is None and self.filter_in is None) else '.'}"
                )

        if not quiet:
            print("Set the interfaces degree-0 coefficients")
        w_lm_o[0, 0, 0] = R
        dc_lm_o[0, 0, 0] = 0
        moho_lm_o[0, 0, 0] = R_c
        H_lm_o[0, 0, 0] = R

        self.w_lm = SHCoeffs.from_array(w_lm_o)
        self.A_lm = SHCoeffs.from_array(A_lm_o)
        self.moho_lm = SHCoeffs.from_array(moho_lm_o)
        self.crust_lm = SHCoeffs.from_array(H_lm_o - moho_lm_o)
        self.dc_lm = SHCoeffs.from_array(dc_lm_o)
        self.drhom_lm = SHCoeffs.from_array(drhom_lm_o)
        self.omega_lm = SHCoeffs.from_array(omega_lm_o)
        self.q_lm = SHCoeffs.from_array(q_lm_o)
        self.Gc_lm = SHCoeffs.from_array(Gc_lm_o)
        self.G_lm = SHCoeffs.from_array(G_lm_o)
        self.H_lm = SHCoeffs.from_array(H_lm_o)
        self.sols = lambdify_func_o

    # ==== Displacement_strains_shtools ====

    def compute_strains(
        self,
        depth=0,
        lmax=None,
        lmaxgrid=None,
    ):
        """
        Computes the Banerdt (1986) equations to determine strains
        and stresses from the displacements. This function uses
        SHTOOLS to derive the spherical harmonic gradients. This
        does not support GLQ grids.

        Returns
        -------
        stress_theta : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the stress field in MPa with respect to colatitude.
            This is equation A12 from Banerdt (1986).
        stress_phi : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the stress field in MPa with respect to longitude.
            This is equation A13 from Banerdt (1986).
        stress_theta_phi : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the stress field in MPa with respect to
            colatitude and longitude.
            This is equation A14 from Banerdt (1986).
        eps_theta : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the elongation with respect to colatitude.
            This is equation A16 from Banerdt (1986).
        eps_phi : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the elongation with respect to longitude.
            This is equation A17 from Banerdt (1986).
        omega : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the shearing deformation.
            This is equation A18 from Banerdt (1986).
        kappa_theta : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the bending deformation with respect to colatitude.
            This is equation A19 from Banerdt (1986).
        kappa_phi : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the bending deformation with respect to longitude.
            This is equation A20 from Banerdt (1986).
        tau : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the twisting deformation.
            This is equation A21 from Banerdt (1986).
        tot_theta : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the total deformation with respect to colatitude.
        tot_phi : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the total deformation with respect to longitude.
        tot_thetaphi : array, size(lmaxgrid+1, 2*lmaxgrid+1)
            Array with the total deformation with respect to colatitude
            and longitude.

        Parameters
        ----------
        depth : float, optional, default = 0
            The depth at which stresses are estimated. Default is
            0 km (surface).
        lmax : int, optional, default = None
            Maximum spherical harmonic degree for computations.
            If None, lmax is inferred from the Class attribute.
        lmaxgrid : int, optional, default = None
            The maximum spherical harmonic degree resolvable by the grid.
            If None, this parameter is set to lmax.
            When grid=='GLQ', the gridshape is (lmaxgrid+1, 2*lmaxgrid+1) and
            (2*lmaxgrid+2, 2*(2*lmaxgrid+2)) when grid=='DH'.
            Quadrature grids following the convention of SHTOOLs.
            If None, the grid is set to 'GLQ'.
        """

        if lmax is None:
            lmax = self.lmax

        if lmaxgrid is None:
            lmaxgrid = self.lmaxgrid

        quiet = self.quiet

        if lmax != self.w_lm.lmax:
            if not quiet:
                print(f"Padding w_lm from lmax = {self.w_lm.lmax} to {lmax}")
            self.w_lm = self.w_lm.pad(lmax=lmax)

        if lmax != self.A_lm.lmax:
            if not quiet:
                print(f"Padding A_lm from lmax = {self.A_lm.lmax} to {lmax}")
            self.A_lm = self.A_lm.pad(lmax=lmax)

        if lmaxgrid is None:
            lmaxgrid = lmax
        elif lmaxgrid < lmax:
            raise ValueError(
                f"lmaxgrid should be higher or equal than lmax, input is {lmaxgrid}"
                + f" with lmax = {lmax}."
            )

        # Some constants for the elastic model.
        Te_half = self.Te / 2.0
        eps = (Te_half - depth) / (1 + (Te_half - depth) / self.R)
        psi = 12.0 * self.R**2 / self.Te**2
        D = (self.E * (self.Te**3)) / ((12.0 * (1.0 - self.v**2)))
        DpsiTeR = (D * psi) / (self.Te * self.R**2)
        R_m1 = 1.0 / self.R
        n_Rm2 = -(R_m1**2)

        # Remove reference radius
        self.A_lm.coeffs[0, 0, 0] = 0.0
        self.w_lm.coeffs[0, 0, 0] = 0.0

        nlat = 2 * lmaxgrid + 2
        nlon = 2 * nlat

        _, grid_colat = np.meshgrid(
            np.linspace(0, 2 * np.pi, nlon, endpoint=False),
            np.linspace(0, np.pi, nlat, endpoint=False),
        )

        sin_g_colat = np.sin(grid_colat)
        csc = np.divide(
            1.0, sin_g_colat, out=np.zeros_like(sin_g_colat), where=sin_g_colat != 0
        )
        cot = np.divide(
            1.0,
            np.tan(grid_colat),
            out=np.zeros_like(sin_g_colat),
            where=sin_g_colat != 0,
        )
        cotcsc = csc * cot

        kw_exp = {"extend": False, "lmax_calc": lmax, "lmax": lmaxgrid, "grid": "DH2"}
        w_deflec_ylm = R_m1 * self.w_lm.expand(**kw_exp).data

        w_lm_grad = self.w_lm.gradient(**kw_exp)
        A_lm_grad = self.A_lm.gradient(**kw_exp)

        # First order derivative
        A_lm_d1_t_cot = A_lm_grad.theta.data * cot  # cot pre-multiplication
        w_lm_d1_t_cot = w_lm_grad.theta.data * cot
        A_lm_d1_p = A_lm_grad.phi
        w_lm_d1_p = w_lm_grad.phi
        A_lm_d1_p.data *= sin_g_colat  # Remove the sin(theta) component of the gradient
        w_lm_d1_p.data *= sin_g_colat

        # Second order derivative
        A_lm_d1_grad = A_lm_d1_p.expand(lmax_calc=lmax).gradient(**kw_exp)
        w_lm_d1_grad = w_lm_d1_p.expand(lmax_calc=lmax).gradient(**kw_exp)
        A_lmd2_p_csc2 = A_lm_d1_grad.phi.data
        w_lmd2_p_csc2 = w_lm_d1_grad.phi.data
        A_lmd2_p_csc2 *= csc  # Remove the sin(theta) component of the gradient
        # and multiply by csc2 results in only * csc
        w_lmd2_p_csc2 *= csc
        A_lmd2_tp = A_lm_d1_grad.theta.data
        w_lmd2_tp = w_lm_d1_grad.theta.data

        # Laplacian identity for d2_theta
        lapla_a = SHCoeffs.from_zeros(lmax)
        for l in range(lmax + 1):
            lapla_a.coeffs[:, l, : l + 1] = l * (l + 1)

        A_lmd2_t = -(
            (self.A_lm * lapla_a).expand(**kw_exp).data + A_lm_d1_t_cot + A_lmd2_p_csc2
        )
        w_lmd2_t = -(
            (self.w_lm * lapla_a).expand(**kw_exp).data + w_lm_d1_t_cot + w_lmd2_p_csc2
        )

        # Beuthe 2008 formulas
        eps_theta = R_m1 * A_lmd2_t + w_deflec_ylm
        eps_phi = R_m1 * (A_lmd2_p_csc2 + A_lm_d1_t_cot) + w_deflec_ylm
        omega = R_m1 * (A_lmd2_tp * (1 + csc) - A_lm_d1_p.data * cotcsc)

        kappa_theta = n_Rm2 * w_lmd2_t + (-R_m1) * w_deflec_ylm
        kappa_phi = n_Rm2 * (w_lmd2_p_csc2 + w_lm_d1_t_cot) + (-R_m1) * w_deflec_ylm
        tau = 2.0 * n_Rm2 * (w_lmd2_tp * csc - w_lm_d1_p.data * cotcsc)

        stress_theta = (
            (eps_theta + self.v * eps_phi + eps * (kappa_theta + self.v * kappa_phi))
            * DpsiTeR
            / 1e6
        )  # MPa
        stress_phi = (
            (eps_phi + self.v * eps_theta + eps * (kappa_phi + self.v * kappa_theta))
            * DpsiTeR
            / 1e6
        )  # MPa
        stress_theta_phi = (
            (omega + eps * tau) * 0.5 * DpsiTeR * (1.0 - self.v) / 1e6
        )  # MPa

        tot_theta = eps_theta + kappa_theta * eps
        tot_phi = eps_phi + kappa_phi * eps
        tot_thetaphi = (omega + tau * eps) / 2.0

        return (
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
        else arr_symb[np.argwhere(str_symb == not_constraint)[0][0]]
    )

    return out
