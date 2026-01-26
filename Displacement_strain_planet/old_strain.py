"""
Functions for calculating Legendre polynomial derivatives, stresses
and strains and plotting the Knapmeyer et al. (2006) tectonic dataset.
"""

import numpy as np
import pyshtools as pysh
from .utils import SH_deriv_store

# ==== Displacement_strains ====


def Displacement_strains(
    A_lm,
    w_lm,
    E,
    v,
    R,
    Te,
    lmax,
    depth=0,
    colat_min=0,
    colat_max=180,
    lon_min=0,
    lon_max=360,
    grid="DH",
    lmaxgrid=None,
    Y_lm_d1_t=None,
    Y_lm_d1_p=None,
    Y_lm_d2_t=None,
    Y_lm_d2_p=None,
    Y_lm_d2_tp=None,
    y_lm=None,
    path=None,
    quiet=True,
):
    """
    Computes the Banerdt (1986) equations to determine strains
    and stresses from the displacements.

    Returns
    -------
    stress_theta : array, size(2*lmax+2,2*(2*lmax+2))
        Array with the stress field with respect to colatitude.
        This is equation A12 from Banerdt (1986).
    stress_phi : array, size(2,lmax+1,lmax+1)
        Array with the stress field with respect to longitude.
        This is equation A13 from Banerdt (1986).
    stress_theta_phi : array, size(2,lmax+1,lmax+1)
        Array with the stress field with respect to colatitude and longitude.
        This is equation A14 from Banerdt (1986).
    eps_theta : array, size(2,lmax+1,lmax+1)
        Array with the elongation with respect to colatitude.
        This is equation A16 from Banerdt (1986).
    eps_phi : array, size(2,lmax+1,lmax+1)
        Array with the elongation with respect to longitude.
        This is equation A17 from Banerdt (1986).
    omega : array, size(2,lmax+1,lmax+1)
        Array with the shearing deformation.
        This is equation A18 from Banerdt (1986).
    kappa_theta : array, size(2,lmax+1,lmax+1)
        Array with the bending deformation with respect to colatitude.
        This is equation A19 from Banerdt (1986).
    kappa_phi : array, size(2,lmax+1,lmax+1)
        Array with the bending deformation with respect to longitude.
        This is equation A20 from Banerdt (1986).
    tau : array, size(2,lmax+1,lmax+1)
        Array with the twisting deformation.
        This is equation A21 from Banerdt (1986).
    tot_theta : array, size(2,lmax+1,lmax+1)
        Array with the total deformation with respect to colatitude.
    tot_phi : array, size(2,lmax+1,lmax+1)
        Array with the total deformation with respect to longitude.
    tot_thetaphi : array, size(2,lmax+1,lmax+1)
        Array with the total deformation with respect to colatitude
        and longitude.

    Parameters
    ----------
    A_lm : array, float, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        poloidal term of the tangential displacement.
    w_lm : array, float, size(2,lmax+1,lmax+1)
        Array with the spherical harmonic coefficients of the
        upward displacement.
    E : float
        Young's modulus.
    v : float
        Poisson's ratio.
    R : float
        Mean radius of the planet.
    Te : float
        Elastic thickness of the lithosphere.
    lmax : int
        Maximum spherical harmonic degree for computations.
    depth : float, optional, default = 0
        The depth at which stresses are estimated.
    colat_min : float, optional, default = 0
        Minimum colatitude for grid computation of strains and stresses.
    colat_max : float, optional, default = 180
        Maximum colatitude for grid computation of strains and stresses.
    lon_min : float, optional, default = 0
        Minimum longitude for grid computation of strains and stresses.
    lon_max : float, optional, default = 360
        Maximum longitude for grid computation of strains and stresses.
    grid: string, optional, default = 'DH'
        Either 'DH' or 'GLQ' for Driscoll and Healy grids or Gauss-Legendre
        Quadrature grids following the convention of SHTOOLs.
    lmaxgrid : int, optional, default = None
        The maximum spherical harmonic degree resolvable by the grid.
        If None, this parameter is set to lmax.
        When grid=='GLQ', the gridshape is (lmaxgrid+1, 2*lmaxgrid+1) and
        (2*lmaxgrid+2, 2*(2*lmaxgrid+2)) when grid=='DH'.
    Y_lm_d1_t : array, float, size(2,lmax+1,lmax+1), optional, default = None
        Array with the first derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d1_p : array, float, size(2,lmax+1,lmax+1), optional, default = None
        Array with the first derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_t : array, float, size(2,lmax+1,lmax+1), optional, default = None
        Array with the second derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d2_p : array, float, size(2,lmax+1,lmax+1), optional, default = None
        Array with the second derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_tp : array, float, size(2,lmax+1,lmax+1), optional, default = None
        Array with the first derivative
        of Legendre polynomials with respect to colatitude and longitude.
    y_lm : array, float, size(2,lmax+1,lmax+1), optional, default = None
        Array of spherical harmonic functions.
    path : string, optional, default = None
        path where to find the stored Legendre polynomials.
    quiet : bool, optional, default = True
        If True, suppress printing output.
    """

    if lmax != np.shape(A_lm)[2] - 1:
        if quiet is False:
            print(
                "Padding A_lm and w_lm from lmax = %s to %s"
                % (np.shape(A_lm)[2] - 1, lmax)
            )
        A_lm = A_lm[:, : lmax + 1, : lmax + 1]
        w_lm = w_lm[:, : lmax + 1, : lmax + 1]

    if lmaxgrid is None:
        lmaxgrid = lmax
    elif lmaxgrid < lmax:
        raise ValueError(
            "lmaxgrid should be higher or equal than lmax, input is %s" % (lmaxgrid)
            + " with lmax = %s." % (lmax)
        )

    if grid == "GLQ":
        nlat = lmaxgrid + 1
        nlon = 2 * nlat - 1
    elif grid == "DH":
        nlat = 2 * lmaxgrid + 2
        nlon = 2 * nlat
    else:
        raise ValueError(
            "Grid format non recognized allowed inputs are 'DH' and 'GLQ', input was %s"
            % (grid)
        )

    if Y_lm_d1_p is not None:
        if quiet is False:
            print("Using input precomputed SH derivatives")
    else:
        if path is None:
            raise ValueError(
                "Need to speficify the path, here the path is {:s}.".format(repr(path))
            )
        (
            Y_lm_d1_t,
            Y_lm_d1_p,
            Y_lm_d2_t,
            Y_lm_d2_p,
            Y_lm_d2_tp,
            y_lm,
        ) = SH_deriv_store(lmax, path, lmaxgrid=lmaxgrid, grid=grid)

    # Some constants for the elastic model.
    Te_half = Te / 2.0
    eps = (Te_half - depth) / (1 + (Te_half - depth) / R)
    psi = 12.0 * R**2 / Te**2
    D = (E * (Te * Te * Te)) / ((12.0 * (1.0 - v**2)))
    DpsiTeR = (D * psi) / (Te * R**2)
    R_m1 = 1.0 / R
    n_Rm2 = -(R_m1**2)

    # Remove reference radius
    A_lm[0, 0, 0] = 0.0
    w_lm[0, 0, 0] = 0.0

    # Allocate arrays.
    shape = (nlat, nlon)
    omega = np.zeros(shape)
    kappa_theta = np.zeros(shape)
    kappa_phi = np.zeros(shape)
    tau = np.zeros(shape)
    eps_theta = np.zeros(shape)
    eps_phi = np.zeros(shape)

    deg2rad = np.pi / 180

    if grid == "GLQ":
        zeros, _ = pysh.expand.SHGLQ(lmax)
        grid_long, grid_colat = np.meshgrid(
            np.linspace(0, 2 * np.pi, nlon, endpoint=False),
            np.arccos(zeros),
        )
    else:
        grid_long, grid_colat = np.meshgrid(
            np.linspace(0, 2 * np.pi, nlon, endpoint=False),
            np.linspace(0, np.pi, nlat, endpoint=False),
        )

    mask = (
        (grid_colat > (colat_min - 1) * deg2rad)
        & (grid_colat < (colat_max + 1) * deg2rad)
        & (grid_long > (lon_min - 1) * deg2rad)
        & (grid_long < (lon_max + 1) * deg2rad)
    )
    sin_g_lat_m = np.sin(grid_colat[mask])
    csc = np.divide(
        1.0, sin_g_lat_m, out=np.zeros_like(sin_g_lat_m), where=sin_g_lat_m != 0
    )
    csc2 = np.divide(
        1.0, sin_g_lat_m**2, out=np.zeros_like(sin_g_lat_m), where=sin_g_lat_m != 0
    )
    cot = np.divide(
        1.0,
        np.tan(grid_colat[mask]),
        out=np.zeros_like(sin_g_lat_m),
        where=sin_g_lat_m != 0,
    )
    cotcsc = csc * cot

    # Convert 3-D of SH to 2-D indexed array
    w_lm = pysh.shio.SHCilmToCindex(w_lm, lmax)
    A_lm = pysh.shio.SHCilmToCindex(A_lm, lmax)

    y_lm = y_lm[mask]
    Y_lm_d2_t = Y_lm_d2_t[mask]
    Y_lm_d2_p = Y_lm_d2_p[mask]
    Y_lm_d1_t = Y_lm_d1_t[mask]
    Y_lm_d1_p = Y_lm_d1_p[mask]
    Y_lm_d2_tp = Y_lm_d2_tp[mask]

    ein_sum = "mij,ij->m"
    ein_sum_mul = "mik,ik,m->m"
    path_sum = ["einsum_path", (0, 1)]  # Generated from np.einsum_path
    path_mul = ["einsum_path", (0, 1), (0, 1)]  # Generated from np.einsum_path

    w_deflec_ylm = R_m1 * np.einsum(ein_sum, y_lm, w_lm, optimize=path_sum)
    eps_theta[mask] = (
        R_m1 * np.einsum(ein_sum, Y_lm_d2_t, A_lm, optimize=path_sum) + w_deflec_ylm
    )
    eps_phi[mask] = (
        R_m1
        * (
            np.einsum(ein_sum_mul, Y_lm_d2_p, A_lm, csc2, optimize=path_mul)
            + np.einsum(ein_sum_mul, Y_lm_d1_t, A_lm, cot, optimize=path_mul)
        )
        + w_deflec_ylm
    )
    omega[mask] = (
        2.0
        * R_m1
        * (
            np.einsum(ein_sum_mul, Y_lm_d2_tp, A_lm, csc, optimize=path_mul)
            - np.einsum(ein_sum_mul, Y_lm_d1_p, A_lm, cotcsc, optimize=path_mul)
        )
    )

    kappa_theta[mask] = (
        n_Rm2 * np.einsum(ein_sum, Y_lm_d2_t, w_lm, optimize=path_sum)
        + (-R_m1) * w_deflec_ylm
    )
    kappa_phi[mask] = (
        n_Rm2
        * (
            np.einsum(ein_sum_mul, Y_lm_d2_p, w_lm, csc2, optimize=path_mul)
            + np.einsum(ein_sum_mul, Y_lm_d1_t, w_lm, cot, optimize=path_mul)
        )
        + (-R_m1) * w_deflec_ylm
    )
    tau[mask] = (
        2.0
        * n_Rm2
        * (
            np.einsum(ein_sum_mul, Y_lm_d2_tp, w_lm, csc, optimize=path_mul)
            - np.einsum(ein_sum_mul, Y_lm_d1_p, w_lm, cotcsc, optimize=path_mul)
        )
    )

    stress_theta = (
        (eps_theta + v * eps_phi + eps * (kappa_theta + v * kappa_phi)) * DpsiTeR / 1e6
    )  # MPa
    stress_phi = (
        (eps_phi + v * eps_theta + eps * (kappa_phi + v * kappa_theta)) * DpsiTeR / 1e6
    )  # MPa
    stress_theta_phi = (omega + eps * tau) * 0.5 * DpsiTeR * (1.0 - v) / 1e6  # MPa

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
