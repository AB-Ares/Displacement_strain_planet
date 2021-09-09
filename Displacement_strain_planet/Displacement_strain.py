"""
Functions for calculating Legendre polynomial derivatives, stresses
and strains and plotting the Knapmeyer et al. (2006) tectonic dataset.
"""

import numpy as np
import pyshtools as pysh
from pathlib import Path

pi = np.pi

# ==== SH_deriv ====


def SH_deriv(theta, phi, lmax):
    """
    Compute spherical harmonic derivatives at a given
    location (first and second order).

    Returns
    -------
    Y_lm_d1_theta_a : array, size(2,lmax+1,lmax+1)
        Array with the first derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d1_phi_a : array, size(2,lmax+1,lmax+1)
        Array with the first derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_theta_a : array, size(2,lmax+1,lmax+1)
        Array with the second derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d2_phi_a : array, size(2,lmax+1,lmax+1)
        Array with the second derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_thetaphi_a : array, size(2,lmax+1,lmax+1)
        Array with the first derivative
        of Legendre polynomials with respect to colatitude and longitude.
    y_lm : array, size(2,lmax+1,lmax+1)
        Array of spherical harmonic functions.

    Parameters
    ----------
    theta : float
        Colatitude in radian.
    phi : float
        Longitude in radian.
    lmax : int
        Maximum spherical harmonic degree to compute for the derivatives.
    """
    shape = (2, lmax + 1, lmax + 1)
    Y_lm_d1_theta_a = np.zeros(shape)
    Y_lm_d1_phi_a = np.zeros(shape)
    Y_lm_d2_phi_a = np.zeros(shape)
    Y_lm_d2_thetaphi_a = np.zeros(shape)
    Y_lm_d2_theta_a = np.zeros(shape)
    y_lm = np.zeros(shape)

    cost = np.cos(theta)
    sint = np.sin(theta)
    if theta == 0 or theta == pi:
        dp_theta = np.zeros((int((lmax + 1) * (lmax + 2) / 2)))
        p_theta = np.zeros((int((lmax + 1) * (lmax + 2) / 2)))
        costsint = 0.0
        sintt = 0.0
    else:
        p_theta, dp_theta = pysh.legendre.PlmBar_d1(lmax, cost)
        dp_theta *= -sint  # Derivative with respect to
        # theta.
        costsint = cost / sint
        sintt = 1.0 / sint ** 2
    for l in range(lmax + 1):
        lapla = float(-l * (l + 1))
        for m in range(-l, l + 1):
            m_abs = np.abs(m)
            index = int(l * (l + 1) / 2 + m_abs)
            cosmphi = np.cos(m_abs * phi)
            sinmphi = np.sin(m_abs * phi)
            if m >= 0:
                msinmphi = -m * sinmphi  # First cos(m*phi)
                # derivative.
                m2cosphi = -(m ** 2) * cosmphi  # Second cos(m*phi)
                # derivative.
                Y_lm_d1_theta_a[0, l, m] = dp_theta[index] * cosmphi
                Y_lm_d1_phi_a[0, l, m] = p_theta[index] * msinmphi
                Y_lm_d2_phi_a[0, l, m] = p_theta[index] * m2cosphi
                Y_lm_d2_thetaphi_a[0, l, m] = dp_theta[index] * msinmphi
                y_lm[0, l, m] = p_theta[index] * cosmphi
            else:
                mcosmphi = m_abs * cosmphi
                m2sinphi = -(m_abs ** 2) * sinmphi
                Y_lm_d1_theta_a[1, l, m_abs] = dp_theta[index] * sinmphi
                Y_lm_d1_phi_a[1, l, m_abs] = p_theta[index] * mcosmphi
                Y_lm_d2_phi_a[1, l, m_abs] = p_theta[index] * m2sinphi
                Y_lm_d2_thetaphi_a[1, l, m_abs] = dp_theta[index] * mcosmphi
                y_lm[1, l, m_abs] = p_theta[index] * sinmphi

        if theta == 0 or theta == pi:
            Y_lm_d2_theta_a[:, l, : l + 1] = 0.0  # Not defined.
        else:
            # Make use of the Laplacian identity to estimate
            # last derivative.
            Y_lm_d2_theta_a[:, l, : l + 1] = (
                lapla * y_lm[:, l, : l + 1]
                - Y_lm_d1_theta_a[:, l, : l + 1] * costsint
                - sintt * Y_lm_d2_phi_a[:, l, : l + 1]
            )

    return (
        Y_lm_d1_theta_a,
        Y_lm_d1_phi_a,
        Y_lm_d2_theta_a,
        Y_lm_d2_phi_a,
        Y_lm_d2_thetaphi_a,
        y_lm,
    )


# ==== SH_deriv_store ====


def SH_deriv_store(lmax, path, lmaxgrid=None, save=True, compressed=False):
    """
    Compute and store or load spherical harmonic derivatives
    over the entire sphere (first and second order).

    Returns
    -------
    Y_lm_d1_theta_a : array, size(2,lmax+1,lmax+1)
        Array with the first derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d1_phi_a : array, size(2,lmax+1,lmax+1)
        Array with the first derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_theta_a : array, size(2,lmax+1,lmax+1)
        Array with the second derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d2_phi_a : array, size(2,lmax+1,lmax+1)
        Array with the second derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_thetaphi_a : array, size(2,lmax+1,lmax+1)
        Array with the first derivative
        of Legendre polynomials with respect to colatitude and longitude.
    y_lm_save : array, size(2,lmax+1,lmax+1)
        Array of spherical harmonic functions.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree to compute for the derivatives.
    path : string
        Path to store or load spherical harmonic derivatives.
    lmaxgrid : int, optional, default = None
        The maximum spherical harmonic degree resolvable by the grid,
        latitude(2 * lmaxgrid + 2) and longitude(2 * (2 * lmaxgrid + 2)).
        Should be higher or equal than lmax. If None, this parameter is set to lmax.
    save : boolean, optional, default = True
        If True, save the data at the given path location.
    compressed : boolean, optional, default = False
        If True, the data is saved in compressed .npz format instead of
        npy, which decreases the file size by about a factor 2. This is
        recommended when lmax > 75.
    """
    if lmaxgrid is None:
        lmaxgrid = lmax
    elif lmaxgrid < lmax:
        raise ValueError(
            "lmaxgrid should be higher or equal than lmax, input is %s" % (lmaxgrid)
            + " with lmax = %s." % (lmax)
        )
    n = 2 * lmaxgrid + 2
    poly_file = "%s/Y_lmsd1d2_lmax%s_lmaxgrid%s.%s" % (
        path,
        lmax,
        lmaxgrid,
        "npz" if compressed else "npy",
    )

    if Path(poly_file).exists() == 0:
        print(
            "Pre-compute SH derivatives, may take some"
            + " time depending on lmax and lmaxgrid, which are %s and %s."
            % (lmax, lmaxgrid)
        )
        index_size = int((lmax + 1) * (lmax + 2) / 2)
        shape_save = (n, 2 * n, 2, lmax + 1, lmax + 1)
        Y_lm_d1_theta_a = np.zeros(shape_save)
        Y_lm_d1_phi_a = np.zeros(shape_save)
        Y_lm_d2_phi_a = np.zeros(shape_save)
        Y_lm_d2_thetaphi_a = np.zeros(shape_save)
        Y_lm_d2_theta_a = np.zeros(shape_save)
        y_lm_save = np.zeros(shape_save)
        phi_ar = np.linspace(0, 360, 2 * n, endpoint=False) * pi / 180.0
        y_lm = np.zeros((len(phi_ar), 2, lmax + 1, lmax + 1))

        t_i = -1
        for theta in np.linspace(0, 180, n, endpoint=False) * pi / 180.0:
            print(" colatitude %s of 180" % (int(theta * 180 / pi)), end="\r")
            t_i += 1
            sint = np.sin(theta)
            cost = np.cos(theta)
            if theta == 0:
                dp_theta = np.zeros((index_size))
                p_theta = np.zeros((index_size))
            else:
                p_theta, dp_theta = pysh.legendre.PlmBar_d1(lmax, cost)
                dp_theta *= -sint  # Derivative with
                # respect to theta.
                costsint = cost / sint
                sintt = 1.0 / sint ** 2

            for l in range(lmax + 1):
                lapla = float(-l * (l + 1))
                for m in range(-l, l + 1):
                    m_abs = np.abs(m)
                    index = int(l * (l + 1) / 2 + m_abs)
                    cosmphi = np.cos(m_abs * phi_ar)
                    sinmphi = np.sin(m_abs * phi_ar)
                    if m >= 0:
                        msinmphi = -m * sinmphi  # First
                        # cos(m*phi)
                        # derivative
                        m2cosphi = -(m ** 2) * cosmphi  # Second
                        # cos(m*phi)
                        # derivative
                        Y_lm_d1_theta_a[t_i, :, 0, l, m] = dp_theta[index] * cosmphi
                        Y_lm_d1_phi_a[t_i, :, 0, l, m] = p_theta[index] * msinmphi
                        Y_lm_d2_phi_a[t_i, :, 0, l, m] = p_theta[index] * m2cosphi
                        Y_lm_d2_thetaphi_a[t_i, :, 0, l, m] = dp_theta[index] * msinmphi
                        y_lm[:, 0, l, m] = p_theta[index] * cosmphi
                    else:
                        mcosmphi = m_abs * cosmphi
                        m2sinphi = -(m_abs ** 2) * sinmphi
                        Y_lm_d1_theta_a[t_i, :, 1, l, m_abs] = dp_theta[index] * sinmphi
                        Y_lm_d1_phi_a[t_i, :, 1, l, m_abs] = p_theta[index] * mcosmphi
                        Y_lm_d2_phi_a[t_i, :, 1, l, m_abs] = p_theta[index] * m2sinphi
                        Y_lm_d2_thetaphi_a[t_i, :, 1, l, m_abs] = (
                            dp_theta[index] * mcosmphi
                        )
                        y_lm[:, 1, l, m_abs] = p_theta[index] * sinmphi

                y_lm_save[t_i, :, :, l, : l + 1] = y_lm[:, :, l, : l + 1]

                if theta == 0:
                    Y_lm_d2_theta_a[t_i, :, :, l, : l + 1] = 0.0
                    # Not defined.
                else:
                    # Make use of the Laplacian identity to
                    # estimate last derivative.
                    Y_lm_d2_theta_a[t_i, :, :, l, : l + 1] = (
                        lapla * y_lm[:, :, l, : l + 1]
                        - Y_lm_d1_theta_a[t_i, :, :, l, : l + 1] * costsint
                        - sintt * Y_lm_d2_phi_a[t_i, :, :, l, : l + 1]
                    )

        if save:
            if compressed:
                np.savez_compressed(
                    poly_file,
                    Y_lm_d1_t=Y_lm_d1_theta_a,
                    Y_lm_d1_p=Y_lm_d1_phi_a,
                    Y_lm_d2_t=Y_lm_d2_theta_a,
                    Y_lm_d2_p=Y_lm_d2_phi_a,
                    Y_lm_d2_tp=Y_lm_d2_thetaphi_a,
                    Y_lm=y_lm_save,
                )
            else:
                np.save(
                    poly_file,
                    [
                        Y_lm_d1_theta_a,
                        Y_lm_d1_phi_a,
                        Y_lm_d2_theta_a,
                        Y_lm_d2_phi_a,
                        Y_lm_d2_thetaphi_a,
                        y_lm_save,
                    ],
                )
    else:
        if compressed:
            print(
                "Loading precomputed compressed SH derivatives for strain calculations"
            )
            data = np.load(poly_file)
            Y_lm_d1_theta_a = data["Y_lm_d1_t"]
            Y_lm_d1_phi_a = data["Y_lm_d1_p"]
            Y_lm_d2_theta_a = data["Y_lm_d2_t"]
            Y_lm_d2_phi_a = data["Y_lm_d2_p"]
            Y_lm_d2_thetaphi_a = data["Y_lm_d2_tp"]
            y_lm_save = data["Y_lm"]
            print("Loading done")
        else:
            print("Loading precomputed SH derivatives for strain calculations")
            (
                Y_lm_d1_theta_a,
                Y_lm_d1_phi_a,
                Y_lm_d2_theta_a,
                Y_lm_d2_phi_a,
                Y_lm_d2_thetaphi_a,
                y_lm_save,
            ) = np.load(poly_file, allow_pickle=True)
            print("Loading done")

    return (
        Y_lm_d1_theta_a,
        Y_lm_d1_phi_a,
        Y_lm_d2_theta_a,
        Y_lm_d2_phi_a,
        Y_lm_d2_thetaphi_a,
        y_lm_save,
    )


# ==== Displacement_strains ====


def Displacement_strains(
    A_lm,
    w_lm,
    E,
    v,
    R,
    Te,
    lmax,
    colat_min=0,
    colat_max=180,
    lon_min=0,
    lon_max=360,
    lmaxgrid=None,
    only_deflec=False,
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
    from displacements with a correction to the theta_phi term.

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
        This is equation A18 from Banerdt (1986). Corrected for the prefactor 2.
    kappa_theta : array, size(2,lmax+1,lmax+1)
        Array with the bending deformation with respect to colatitude.
        This is equation A19 from Banerdt (1986).
    kappa_phi : array, size(2,lmax+1,lmax+1)
        Array with the bending deformation with respect to longitude.
        This is equation A20 from Banerdt (1986).
    tau : array, size(2,lmax+1,lmax+1)
        Array with the twisting deformation.
        This is equation A21 from Banerdt (1986). Corrected for the prefactor 2.

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
    colat_min : float, optional, default = 0
        Minimum colatitude for grid computation of strains and stresses.
    colat_max : float, optional, default = 180
        Maximum colatitude for grid computation of strains and stresses.
    lon_min : float, optional, default = 0
        Minimum longitude for grid computation of strains and stresses.
    lon_max : float, optional, default = 360
        Maximum longitude for grid computation of strains and stresses.
    lmaxgrid : int, optional, default = None
        The maximum spherical harmonic degree resolvable by the grid,
        latitude(2 * lmaxgrid + 2) and longitude(2 * (2 * lmaxgrid + 2)).
        If None, this parameter is set to lmax.
    only_deflec : bool, optional, default = False
        Output only the displacement grid for all latitude and longitudes.
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
        A_lm = pysh.SHCoeffs.from_array(A_lm).pad(lmax=lmax).coeffs
        w_lm = pysh.SHCoeffs.from_array(w_lm).pad(lmax=lmax).coeffs

    if lmaxgrid is None:
        lmaxgrid = lmax
    elif lmaxgrid < lmax:
        raise ValueError(
            "lmaxgrid should be higher or equal than lmax, input is %s" % (lmaxgrid)
            + " with lmax = %s." % (lmax)
        )
    n = 2 * lmaxgrid + 2

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
        ) = SH_deriv_store(lmax, path, lmaxgrid=lmaxgrid)

    # Some constants for the elastic model.
    Re = R - (0.5 * Te)
    psi = 12.0 * Re ** 2 / Te ** 2
    D = (E * (Te * Te * Te)) / ((12.0 * (1.0 - v ** 2)))
    DpsiTeR = (D * psi) / (Te * R ** 2)
    R_m1 = 1.0 / R
    n_Rm2 = -(R_m1 ** 2)
    Te_half = Te / 2.0

    # Remove reference radius
    A_lm[0, 0, 0] = 0.0
    w_lm[0, 0, 0] = 0.0

    # Allocate arrays.
    shape = (n, 2 * n)
    omega = np.zeros(shape)
    kappa_theta = np.zeros(shape)
    kappa_phi = np.zeros(shape)
    tau = np.zeros(shape)
    eps_theta = np.zeros(shape)
    eps_phi = np.zeros(shape)

    deg2rad = pi / 180.0
    grid_long, grid_lat = np.meshgrid(
        np.linspace(0, 360, 2 * n, endpoint=False) * deg2rad,
        np.linspace(0, 180, n, endpoint=False) * deg2rad,
    )
    mask = (
        (grid_lat > (colat_min - 1) * deg2rad)
        & (grid_lat < (colat_max + 1) * deg2rad)
        & (grid_long > (lon_min - 1) * deg2rad)
        & (grid_long < (lon_max + 1) * deg2rad)
    )
    sin_g_lat_m = np.sin(grid_lat[mask])
    csc = np.divide(
        1.0, sin_g_lat_m, out=np.zeros_like(sin_g_lat_m), where=sin_g_lat_m != 0
    )
    csc2 = np.divide(
        1.0, sin_g_lat_m ** 2, out=np.zeros_like(sin_g_lat_m), where=sin_g_lat_m != 0
    )
    cot = np.divide(
        1.0,
        np.tan(grid_lat[mask]),
        out=np.zeros_like(sin_g_lat_m),
        where=sin_g_lat_m != 0,
    )
    cotcsc = csc * cot

    y_lm = y_lm[mask]
    Y_lm_d2_t = Y_lm_d2_t[mask]
    Y_lm_d2_p = Y_lm_d2_p[mask]
    Y_lm_d1_t = Y_lm_d1_t[mask]
    Y_lm_d1_p = Y_lm_d1_p[mask]
    Y_lm_d2_tp = Y_lm_d2_tp[mask]

    ein_sum = "mijk,ijk->m"
    ein_sum_mul = "mijk,ijk,m->m"
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
    omega[mask] = R_m1 * (
        np.einsum(ein_sum_mul, Y_lm_d2_tp, A_lm, csc, optimize=path_mul)
        - np.einsum(ein_sum_mul, Y_lm_d1_p, A_lm, cotcsc, optimize=path_mul)
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
    tau[mask] = n_Rm2 * (
        np.einsum(ein_sum_mul, Y_lm_d2_tp, w_lm, csc, optimize=path_mul)
        - np.einsum(ein_sum_mul, Y_lm_d1_p, w_lm, cotcsc, optimize=path_mul)
    )

    stress_theta = (
        (eps_theta + v * eps_phi + Te_half * (kappa_theta + v * kappa_phi))
        * DpsiTeR
        / 1e6
    )  # MPa
    stress_phi = (
        (eps_phi + v * eps_theta + Te_half * (kappa_phi + v * kappa_theta))
        * DpsiTeR
        / 1e6
    )  # MPa
    stress_theta_phi = (omega + Te_half * tau) * 0.5 * DpsiTeR * (1.0 - v) / 1e6  # MPa

    kappa_theta[mask] *= Te_half  # Strain
    kappa_phi[mask] *= Te_half  # Strain
    tau[mask] *= Te_half  # Strain

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
    )


# ==== Principal_strainstress_angle ====


def Principal_strainstress_angle(s_theta, s_phi, s_theta_phi):
    """
    Calculate principal strains, stresses, and
    their principal angles.

    Returns
    -------
    min_strain : array, size same as input arrays
        Array with the minimum principal horizontal strain or stress.
    max_strain : array, size same as input arrays
        Array with the maximum principal horizontal strain or stress.
    sum_strain : array, size same as input arrays
        Array with the sum of the principal horizontal strain or stress.
    principal_angle : array, size same as input arrays
        Array with the principal strain or stress direction.

    Parameters
    ----------
    s_theta : array, float, size(n, 2 * n)
        Array of the colatitude component of the stress or strain field.
    s_phi : array, float, size(n, 2 * n)
        Array of the longitude component of the stress or strain field.
    s_theta_phi : array, float, size(n, 2 * n)
        Array of the colatitude and longitude component of the stress or strain field.
    """
    min_strain = 0.5 * (
        s_theta + s_phi - np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi ** 2)
    )
    max_strain = 0.5 * (
        s_theta + s_phi + np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi ** 2)
    )
    sum_strain = min_strain + max_strain

    principal_angle = 0.5 * np.arctan2(2 * s_theta_phi, s_theta - s_phi) * 180.0 / np.pi

    return min_strain, max_strain, sum_strain, principal_angle


# ==== Plt_tecto_Mars ====


def Plt_tecto_Mars(
    path,
    compression=False,
    extension=True,
    ax=None,
    compression_col="k",
    extension_col="purple",
    lw=1,
    legend_show=True,
    legend_loc="upper left",
):
    """
    Plot the Knampeyer et al. (2006) dataset of
    extensional and compressional tectonic features
    on Mars.

    Parameters
    ----------
    path : string
        path for the location of the Knameyer et al (2006) dataset.
    compression : bool, optional, default = False
        If True, plot compressive tectonic features.
    extension : bool, optional, default = True
        If True, plot extensive tectonic features.
    ax : object, optional, default = None
        Matplotlib axis.
    compression_col : string, optional, default = "k"
        Color of compressive tectonic features.
    extension_col : string, optional, default = "purple"
        Color of extensive tectonic features.
    lw : int, optional, default = 1
        Linewidth for the tectonic features
    legend_show : bool, optional, default = True
        If True, add a legend to the plot.
    legend_loc : string, optional, default = "upper left"
        Determine the legend position.
    """
    comp_fault_dat = np.loadtxt("%s/Knapmeyer_2006_compdata.txt" % (path))
    ext_fault_dat = np.loadtxt("%s/Knapmeyer_2006_extedata.txt" % (path))
    ind_comp_fault = np.isin(comp_fault_dat, np.arange(1, 5143 + 1, dtype=int))
    ind_comp_fault_2 = np.where(ind_comp_fault)[0]
    ind_ext_fault = np.isin(ext_fault_dat, np.arange(1, 9676 + 1, dtype=int))
    ind_ext_fault_2 = np.where(ind_ext_fault)[0]

    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)

    faults_inds = [ind_comp_fault_2, ind_ext_fault_2]
    faults_dats = [comp_fault_dat, ext_fault_dat]
    faults_cols = [compression_col, extension_col]
    labels = ["Compressional tectonic features", "Extensional tectonic features"]

    if compression and not extension:
        faults_inds = [faults_inds[0]]
        faults_dats = [faults_dats[0]]
        faults_cols = [faults_cols[0]]
        labels = [labels[0]]
    elif extension and not compression:
        faults_inds = [faults_inds[1]]
        faults_dats = [faults_dats[1]]
        faults_cols = [faults_cols[1]]
        labels = [labels[1]]

    for faults, dat, col, label in zip(faults_inds, faults_dats, faults_cols, labels):
        ax.plot(np.nan, np.nan, color=col, lw=lw, label=label)
        for indx in range(1, len(faults)):
            ind_fault_check = range(faults[indx - 1] + 1, faults[indx])
            fault_dat_lon = dat[ind_fault_check][::2]
            fault_dat_lat = dat[ind_fault_check][1::2]
            split = (
                np.argwhere((fault_dat_lon[:-1] * fault_dat_lon[1:] < 0)).ravel() + 1
            )
            if len(split) > 0:  # Make boundaries periodic by splitting positive
                # and negative lat lon
                fault_lon_split = np.split(fault_dat_lon, split)
                fault_dat_lat = np.split(fault_dat_lat, split)
                for fault_lon, fault_lat in zip(fault_lon_split, fault_dat_lat):
                    ax.plot((fault_lon + 360) % 360, fault_lat, color=col, lw=lw)
            else:
                ax.plot((fault_dat_lon + 360) % 360, fault_dat_lat, color=col, lw=lw)

    if legend_show:
        ax.legend(loc=legend_loc)
