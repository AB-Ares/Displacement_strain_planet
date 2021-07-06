"""
Functions for calculating Legendre polynomial derivatives, stresses
and strains and plotting the Knapmeyer et al. (2006) tectonic dataset.
"""

import numpy as np
import pyshtools as pysh
import matplotlib.pyplot as plt
from pathlib import Path
import time

pi = np.pi

# ==== SH_deriv ====


def SH_deriv(theta, phi, lmax):
    """
    Compute on the fly spherical harmonic derivatives
    (first and second order)

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
        Array of spherical harmonic functions
    Parameters
    ----------
    theta : float
        Colatitude in radian.
    phi : float
        Longitude in radian.
    lmax : int
        Maximum spherical harmonic degree to compute for the derivatives.
    """
    Y_lm_d1_theta_a = np.zeros((2, lmax + 1, lmax + 1))
    Y_lm_d1_phi_a = np.zeros((2, lmax + 1, lmax + 1))
    Y_lm_d2_phi_a = np.zeros((2, lmax + 1, lmax + 1))
    Y_lm_d2_thetaphi_a = np.zeros((2, lmax + 1, lmax + 1))
    Y_lm_d2_theta_a = np.zeros((2, lmax + 1, lmax + 1))
    y_lm = np.zeros((2, lmax + 1, lmax + 1))

    if theta == 0 or theta == pi:
        dp_theta = np.zeros((int((lmax + 1) * (lmax + 2) / 2)))
        p_theta = np.zeros((int((lmax + 1) * (lmax + 2) / 2)))
        costsint = 0.0
        sintt = 0.0
    else:
        p_theta, dp_theta = pysh.legendre.PlmBar_d1(lmax, np.cos(theta))
        dp_theta *= -np.sin(theta)  # Derivative with respect to
        # theta.
        p_theta /= np.sin(theta)
        costsint = np.cos(theta) / np.sin(theta)
        sintt = 1.0 / np.sin(theta) ** 2
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
                y_lm[0, l, m] = p_theta[index] * np.sin(theta) * cosmphi
            else:
                mcosmphi = m_abs * cosmphi
                m2sinphi = -(m_abs ** 2) * sinmphi
                Y_lm_d1_theta_a[1, l, m_abs] = dp_theta[index] * sinmphi
                Y_lm_d1_phi_a[1, l, m_abs] = p_theta[index] * mcosmphi
                Y_lm_d2_phi_a[1, l, m_abs] = p_theta[index] * m2sinphi
                Y_lm_d2_thetaphi_a[1, l, m_abs] = dp_theta[index] * mcosmphi
                y_lm[1, l, m_abs] = p_theta[index] * np.sin(theta) * sinmphi

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


def SH_deriv_store(lmax, path, save=True):
    """
    Compute and store or load spherical harmonic derivatives
    (first and second order).

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

    Parameters
    ----------
    path : string
        Path to store or load spherical harmonic derivatives.
    lmax : int
        Maximum spherical harmonic degree to compute for the derivatives.
    save : boolean
        Whether the data is saved at the given path location.
    """
    n = 2 * lmax + 2
    poly_file = "%s/Y_lmsd1d2_lmax%s.npy" % (path, lmax)

    if Path(poly_file).exists() == 0:
        print(
            "Pre-compute SH derivatives, may take some"
            + " time depending on lmax, input lmax is %s" % (lmax)
        )
        index_size = int((lmax + 1) * (lmax + 2) / 2)
        Y_lm_d1_theta_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        Y_lm_d1_phi_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        Y_lm_d2_phi_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        Y_lm_d2_thetaphi_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        Y_lm_d2_theta_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        phi_ar = np.linspace(0, 360, 2 * n, endpoint=False) * pi / 180.0
        y_lm = np.zeros((len(phi_ar), 2, lmax + 1, lmax + 1))

        t_i = -1
        for theta in np.linspace(0, 180, n, endpoint=False) * pi / 180.0:
            print(" colatitude %s of 180" % (int(theta * 180 / pi)), end="\r")
            t_i += 1
            if theta == 0 or theta == pi:
                dp_theta = np.zeros((index_size))
                p_theta = np.zeros((index_size))
            else:
                p_theta, dp_theta = pysh.legendre.PlmBar_d1(lmax, np.cos(theta))
                dp_theta *= -np.sin(theta)  # Derivative with
                # respect to theta.
                p_theta /= np.sin(theta)
                costsint = np.cos(theta) / np.sin(theta)
                sintt = 1.0 / np.sin(theta) ** 2

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

                y_lm[:, :, l, : l + 1] *= np.sin(theta)

                if theta == 0 or theta == pi:
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
            np.save(
                poly_file,
                [
                    Y_lm_d1_theta_a,
                    Y_lm_d1_phi_a,
                    Y_lm_d2_theta_a,
                    Y_lm_d2_phi_a,
                    Y_lm_d2_thetaphi_a,
                ],
            )
    else:
        print("Loading precomputed SH derivatives for strain calculations")
        (
            Y_lm_d1_theta_a,
            Y_lm_d1_phi_a,
            Y_lm_d2_theta_a,
            Y_lm_d2_phi_a,
            Y_lm_d2_thetaphi_a,
        ) = np.load(poly_file, allow_pickle=True)
        print("Loading done")

    return (
        Y_lm_d1_theta_a,
        Y_lm_d1_phi_a,
        Y_lm_d2_theta_a,
        Y_lm_d2_phi_a,
        Y_lm_d2_thetaphi_a,
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
    only_deflec=False,
    precomp=True,
    Y_lm_d1_t=None,
    Y_lm_d1_p=None,
    Y_lm_d2_t=None,
    Y_lm_d2_p=None,
    Y_lm_d2_tp=None,
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
        This is equation A12 from Banerdt (1986)
    stress_phi : array, size(2,lmax+1,lmax+1)
        Array with the stress field with respect to longitude.
        This is equation A13 from Banerdt (1986)
    stress_theta_phi : array, size(2,lmax+1,lmax+1)
        Array with the stress field with respect to colatitude and longitude.
        This is equation A14 from Banerdt (1986)
    eps_theta : array, size(2,lmax+1,lmax+1)
        Array with the elongation with respect to colatitude.
        This is equation A16 from Banerdt (1986)
    eps_phi : array, size(2,lmax+1,lmax+1)
        Array with the elongation with respect to longitude.
        This is equation A17 from Banerdt (1986).
    omega : array, size(2,lmax+1,lmax+1)
        Array with the shearing deformation.
        This is equation A18 from Banerdt (1986). Corrected for the prefactor 2
    kappa_theta : array, size(2,lmax+1,lmax+1)
        Array with the bending deformation with respect to colatitude.
        This is equation A19 from Banerdt (1986).
    kappa_phi : array, size(2,lmax+1,lmax+1)
        Array with the bending deformation with respect to longitude.
        This is equation A20 from Banerdt (1986).
    tau : array, size(2,lmax+1,lmax+1)
        Array with the twisting deformation.
        This is equation A21 from Banerdt (1986). Corrected for the prefactor 2

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
    only_deflec : int, optional, default = False
        Output only the displacement grid for all latitude and longitudes.
    precomp : int, optional, default = True
        Use precomputed the Legendre polynomials found at the 'path'.
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
    path : string, optional, default = None
        path where to find the store Legendre polynomials.
    quiet : int, optional, default = True
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

    n = 2 * lmax + 2

    if precomp:
        if Y_lm_d1_p is not None:
            print("Using loaded precomputed SH derivatives")
        else:
            if path is None:
                raise ValueError(
                    "Need to speficify the path, here the path is {:s}.".format(
                        repr(path)
                    )
                )
            Y_lm_d1_t, Y_lm_d1_p, Y_lm_d2_t, Y_lm_d2_p, Y_lm_d2_tp = SH_deriv_store(
                lmax, path
            )
    else:
        if quiet is False:
            print(
                "Compute Spherical Harmonic derivatives on the"
                + " fly, this may be long"
            )

    # Some constants for the elastic model.
    Re = R - float(0.5 * Te)
    psi = 12.0 * Re ** 2 / Te ** 2
    D = (E * (Te * Te * Te)) / (float(12.0 * (1.0 - v ** 2)))
    DpsiTeR = (D * psi) / (Te * R ** 2)
    R_m1 = 1.0 / R

    # Remove reference radius
    A_lm[0, 0, 0] = 0.0
    w_lm[0, 0, 0] = 0.0

    # Allocate arrays.
    stress_theta = np.zeros((n, 2 * n))
    stress_phi = np.zeros((n, 2 * n))
    stress_theta_phi = np.zeros((n, 2 * n))
    omega = np.zeros((n, 2 * n))
    kappa_theta = np.zeros((n, 2 * n))
    kappa_phi = np.zeros((n, 2 * n))
    tau = np.zeros((n, 2 * n))
    eps_theta = np.zeros((n, 2 * n))
    eps_phi = np.zeros((n, 2 * n))

    t_i = -1
    for theta in np.linspace(0, 180, n, endpoint=False) * pi / 180.0:
        t_i += 1
        p_i = -1
        if theta < ((colat_min - 1) * pi / 180.0) or theta > (
            (colat_max + 1) * pi / 180.0
        ):
            # Not in the lat / lon range we investigate.
            continue

        if theta == 0 or theta == pi:
            csc = 0.0
            csc2 = 0.0
            cot = 0.0
        else:
            csc = 1.0 / np.sin(theta)
            csc2 = 1.0 / np.sin(theta) ** 2
            cot = 1.0 / np.tan(theta)

        for phi in np.linspace(0, 360, 2 * n, endpoint=False) * pi / 180.0:
            p_i += 1
            if phi < ((lon_min - 1) * pi / 180.0) or phi > ((lon_max + 1) * pi / 180.0):
                # Not in the lat / lon range we investigate.
                continue

            if precomp:
                y_lm = pysh.expand.spharm(lmax, theta, phi, degrees=False)
                w_lm_ylm = np.sum(w_lm * y_lm)
                d2Atheta = np.sum(Y_lm_d2_t[t_i, p_i] * A_lm)
                d2Aphi = np.sum(Y_lm_d2_p[t_i, p_i] * A_lm)
                d1Atheta = np.sum(Y_lm_d1_t[t_i, p_i] * A_lm)
                d1Aphi = np.sum(Y_lm_d1_p[t_i, p_i] * A_lm)
                d2Athetaphi = np.sum(Y_lm_d2_tp[t_i, p_i] * A_lm)
                d2wtheta = np.sum(Y_lm_d2_t[t_i, p_i] * w_lm)
                d2wphi = np.sum(Y_lm_d2_p[t_i, p_i] * w_lm)
                d1wtheta = np.sum(Y_lm_d1_t[t_i, p_i] * w_lm)
                d1wphi = np.sum(Y_lm_d1_p[t_i, p_i] * w_lm)
                d2wthetaphi = np.sum(Y_lm_d2_tp[t_i, p_i] * w_lm)
            else:
                (
                    Y_lm_d1_tb,
                    Y_lm_d1_pb,
                    Y_lm_d2_tb,
                    Y_lm_d2_pb,
                    Y_lm_d2_tpb,
                    y_lm,
                ) = SH_deriv(theta, phi, lmax)
                w_lm_ylm = np.sum(w_lm * y_lm)
                d2Atheta = np.sum(Y_lm_d2_tb * A_lm)
                d2Aphi = np.sum(Y_lm_d2_pb * A_lm)
                d1Atheta = np.sum(Y_lm_d1_tb * A_lm)
                d1Aphi = np.sum(Y_lm_d1_pb * A_lm)
                d2Athetaphi = np.sum(Y_lm_d2_tpb * A_lm)
                d2wtheta = np.sum(Y_lm_d2_tb * w_lm)
                d2wphi = np.sum(Y_lm_d2_pb * w_lm)
                d1wtheta = np.sum(Y_lm_d1_tb * w_lm)
                d1wphi = np.sum(Y_lm_d1_pb * w_lm)
                d2wthetaphi = np.sum(Y_lm_d2_tpb * w_lm)

            # Sum over theta.
            eps_theta[t_i, p_i] = R_m1 * (d2Atheta + w_lm_ylm)
            eps_phi[t_i, p_i] = R_m1 * (d2Aphi * csc2 + d1Atheta * cot + w_lm_ylm)
            omega[t_i, p_i] = R_m1 * csc * (d2Athetaphi - cot * d1Aphi)

            kappa_theta[t_i, p_i] = -(R_m1 ** 2) * (d2wtheta + w_lm_ylm)
            kappa_phi[t_i, p_i] = -(R_m1 ** 2) * (
                d2wphi * csc2 + d1wtheta * cot + w_lm_ylm
            )
            tau[t_i, p_i] = -(R_m1 ** 2) * csc * (d2wthetaphi - cot * d1wphi)

            stress_theta[t_i, p_i] = (
                eps_theta[t_i, p_i]
                + v * eps_phi[t_i, p_i]
                + Te / 2.0 * (kappa_theta[t_i, p_i] + v * kappa_phi[t_i, p_i])
            )
            stress_phi[t_i, p_i] = (
                eps_phi[t_i, p_i]
                + v * eps_theta[t_i, p_i]
                + Te / 2.0 * (kappa_phi[t_i, p_i] + v * kappa_theta[t_i, p_i])
            )
            stress_theta_phi[t_i, p_i] = omega[t_i, p_i] + Te / 2.0 * tau[t_i, p_i]

    stress_theta *= DpsiTeR / 1e6  # MPa
    stress_phi *= DpsiTeR / 1e6  # MPa
    stress_theta_phi *= 0.5 * DpsiTeR * (1.0 - v) / 1e6  # MPa

    kappa_theta *= Te / 2.0
    kappa_phi *= Te / 2.0
    tau *= Te / 2.0

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
    compression : boolean, optional, default = False
        If True, plot compressive tectonic features.
    extension : boolean, optional, default = True
        If True, plot extensive tectonic features.
    ax : object
        Matplotlib axis.
    compression_col : string, default = "k"
        Color of compressive tectonic features.
    extension_col : string, default = "purple"
        Color of extensive tectonic features.
    lw : int, default = 1
        Linewidth for the tectonic features
    legend_show : boolean, default = True
        If True, add a legend to the plot.
    legend_loc : string, default = "upper left"
        Determine the legend position.
    """
    comp_fault_dat = np.loadtxt("%s/Knapmeyer_2006_compdata.txt" % (path))
    ext_fault_dat = np.loadtxt("%s/Knapmeyer_2006_extedata.txt" % (path))
    ind_comp_fault = np.isin(comp_fault_dat, np.arange(1, 5143, dtype=float))
    ind_comp_fault_2 = np.where(ind_comp_fault)[0]
    ind_ext_fault = np.isin(ext_fault_dat, np.arange(1, 9676, dtype=float))
    ind_ext_fault_2 = np.where(ind_ext_fault)[0]

    if ax is None:
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
        for indx in range(1, int((len(faults) - 1) / 2)):
            ind_fault_check = range(faults[indx - 1] + 1, faults[indx])
            fault_dat_lon = dat[ind_fault_check][::2]
            fault_dat_lat = dat[ind_fault_check][1::2]
            ax.plot((fault_dat_lon + 360) % 360, fault_dat_lat, color=col, lw=lw)

    if legend_show:
        ax.legend(loc=legend_loc)
