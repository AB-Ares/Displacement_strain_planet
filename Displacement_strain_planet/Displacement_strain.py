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
        sintt = 1.0 / sint**2
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
                m2cosphi = -(m**2) * cosmphi  # Second cos(m*phi)
                # derivative.
                Y_lm_d1_theta_a[0, l, m] = dp_theta[index] * cosmphi
                Y_lm_d1_phi_a[0, l, m] = p_theta[index] * msinmphi
                Y_lm_d2_phi_a[0, l, m] = p_theta[index] * m2cosphi
                Y_lm_d2_thetaphi_a[0, l, m] = dp_theta[index] * msinmphi
                y_lm[0, l, m] = p_theta[index] * cosmphi
            else:
                mcosmphi = m_abs * cosmphi
                m2sinphi = -(m_abs**2) * sinmphi
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


def SH_deriv_store(
    lmax,
    path,
    dtype=np.float64,
    lmaxgrid=None,
    grid=None,
    save=True,
    compressed=False,
    colat_min=0,
    colat_max=180,
    quiet=True,
):
    """
    Compute and store or load spherical harmonic derivatives
    over the entire sphere (first and second order). The spherical
    harmonic degree and order correspond to the index l*(l+1)/2+m

    Returns
    -------
    Y_lm_d1_theta_a : array, size(2,(lmax+1)*(lmax+2)/2)
        Array with the first derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d1_phi_a : array, size(2,(lmax+1)*(lmax+2)/2)
        Array with the first derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_theta_a : array, size(2,(lmax+1)*(lmax+2)/2)
        Array with the second derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d2_phi_a : array, size(2,(lmax+1)*(lmax+2)/2)
        Array with the second derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_thetaphi_a : array, size(2,(lmax+1)*(lmax+2)/2)
        Array with the first derivative
        of Legendre polynomials with respect to colatitude and longitude.
    y_lm_save : array, size(2,(lmax+1)*(lmax+2)/2)
        Array of spherical harmonic functions.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree to compute for the derivatives.
    path : string
        Path to store or load spherical harmonic derivatives.
    dtype : data-type, optional, default = numpy.float64
        The desired data-type for the arrays (default is that of numpy).
        This can help reducing the size of the stored array.
    lmaxgrid : int, optional, default = None
        The maximum spherical harmonic degree resolvable by the grid.
        If None, this parameter is set to lmax.
        When grid=='GLQ', the gridshape is (lmaxgrid+1, 2*lmaxgrid+1) and
        (2*lmaxgrid+2, 2*(2*lmaxgrid+2)) when grid=='DH'.
    grid: string, optional, default = None
        Either 'DH' or 'GLQ' for Driscoll and Healy grids or Gauss-Legendre
        Quadrature grids following the convention of SHTOOLs.
        If None, the grid is set to 'GLQ'.
    save : bool, optional, default = True
        If True, save the data at the given path location.
    compressed : bool, optional, default = False
        If True, the data is saved in compressed .npz format instead of
        npy, which decreases the file size by about a factor 2. This is
        recommended when lmax > 75.
    colat_min : float, optional, default = 0
        Minimum colatitude for grid computation of SH derivatives.
    colat_max : float, optional, default = 180
        Maximum colatitude for grid computation of SH derivatives.
    quiet : bool, optional, default = True
        If True, suppress printing output.
    """
    if lmaxgrid is None:
        lmaxgrid = lmax
    elif lmaxgrid < lmax:
        raise ValueError(
            "lmaxgrid should be higher or equal than lmax, input is %s" % (lmaxgrid)
            + " with lmax = %s." % (lmax)
        )

    if grid is None or grid == "GLQ":
        nlat = lmaxgrid + 1
        nlon = 2 * nlat - 1
    elif grid == "DH":
        nlat = 2 * lmaxgrid + 2
        nlon = 2 * nlat
    else:
        raise ValueError(
            "Grid format non recognized allowed are 'DH' and 'GLQ', input was %s"
            % (grid)
        )

    poly_file = "%s/Y_lmsd1d2_%slmax%s_lmaxgrid%s_f%s.%s" % (
        path,
        "GLQ" if grid is None else grid,
        lmax,
        lmaxgrid,
        str(dtype)[-4:-2],
        "npz" if compressed else "npy",
    )

    if Path(poly_file).exists() == 0:
        if quiet is False:
            print(
                "Pre-compute SH derivatives, may take some"
                + " time depending on lmax and lmaxgrid, which are %s and %s."
                % (lmax, lmaxgrid)
            )
            print("dtype is %s." % (dtype))
        index_size = int((lmax + 1) * (lmax + 2) / 2)
        shape_save = (nlat, nlon, 2, index_size)
        Y_lm_d1_theta_a = np.zeros(shape_save, dtype=dtype)
        Y_lm_d1_phi_a = np.zeros(shape_save, dtype=dtype)
        Y_lm_d2_phi_a = np.zeros(shape_save, dtype=dtype)
        Y_lm_d2_thetaphi_a = np.zeros(shape_save, dtype=dtype)
        Y_lm_d2_theta_a = np.zeros(shape_save, dtype=dtype)
        y_lm_save = np.zeros(shape_save, dtype=dtype)
        phi_ar = np.linspace(0, 360, nlon, endpoint=False, dtype=dtype) * pi / 180.0
        y_lm = np.zeros((len(phi_ar), 2, index_size), dtype=dtype)
        cosmphi_a = np.zeros((lmax + 1, len(phi_ar)), dtype=dtype)
        sinmphi_a = np.zeros((lmax + 1, len(phi_ar)), dtype=dtype)
        p_theta_a = np.zeros((index_size, nlat // 2 + 1), dtype=dtype)
        dp_theta_a = np.zeros((index_size, nlat // 2 + 1), dtype=dtype)
        theta_range = (
            np.linspace(0, 180, nlat, endpoint=False, dtype=dtype) * pi / 180.0
        )

        sint = np.sin(theta_range)
        cost = np.cos(theta_range)
        sintt = np.divide(1.0, sint**2, out=np.zeros_like(sint), where=sint != 0)
        costsint = np.divide(cost, sint, out=np.zeros_like(sint), where=sint != 0)

        sign_conversion = False
        for t_i, theta in enumerate(theta_range):
            theta_180 = int(theta * 180 / pi)
            if theta == 0:
                dp_theta = np.zeros((index_size))
                p_theta = np.zeros((index_size))
            if quiet is False:
                print(" colatitude %s of %s" % (theta_180, colat_max), end="\r")
            if theta_180 < colat_min or theta_180 > colat_max:
                continue
            elif theta != 0:
                if colat_min != 0 or colat_max != 180:
                    # Don't use the symmetry speedup, which requires a whole sphere computation
                    p_theta, dp_theta = pysh.legendre.PlmBar_d1(lmax, cost[t_i])
                    dp_theta *= -sint[t_i]
                elif cost[t_i] >= 0:
                    p_theta_a[:, t_i], dp_theta_a[:, t_i] = pysh.legendre.PlmBar_d1(
                        lmax, cost[t_i]
                    )
                    if not sign_conversion:
                        # Given the symmetry of PlmBar_d1 & 'cost', we here get the sign conversions for positive and negative 'cost'.
                        tmp1, tmp2 = pysh.legendre.PlmBar_d1(lmax, -cost[t_i])
                        # Degree-0 is always zero
                        signs_p_theta = np.insert(p_theta_a[1:, t_i] / tmp1[1:], 0, 1)
                        signs_dp_theta = np.insert(dp_theta_a[1:, t_i] / tmp2[1:], 0, 1)
                        sign_conversion = True
                    p_theta = p_theta_a[:, t_i]
                    # Derivative with respect to theta.
                    dp_theta = dp_theta_a[:, t_i] * -sint[t_i]
                else:
                    # Sign conversion when cost is negative
                    idx_sign = nlat // 2 - t_i if nlat % 2 != 0 else nlat // 2 - t_i - 1
                    p_theta = p_theta_a[:, idx_sign] * signs_p_theta
                    dp_theta = dp_theta_a[:, idx_sign] * signs_dp_theta * -sint[t_i]

            for l in range(lmax + 1):
                lapla = float(-l * (l + 1))
                if (theta == 0) or (theta_180 == colat_min):
                    cosmphi_a[l] = np.cos(l * phi_ar)
                    sinmphi_a[l] = np.sin(l * phi_ar)

                m = np.arange(-l, l + 1, dtype=int)
                m_abs = np.abs(m)
                index = np.array(l * (l + 1) / 2 + m_abs, dtype=int)

                ## Positive orders
                m_i = m >= 0
                m_abs_i = m_abs[m_i]
                index_i = index[m_i]
                dp_t_ind = np.array([dp_theta[index_i]]).T
                p_t_ind = np.array([p_theta[index_i]]).T

                # First cos(m*phi) derivative
                msinmphi = sinmphi_a[m_abs_i, :] * np.array([-m[m_i]]).T
                # Second cos(m*phi) derivative
                m2cosphi = cosmphi_a[m_abs_i, :] * np.array([-(m[m_i] ** 2)]).T
                Y_lm_d1_theta_a[t_i, :, 0, index_i] = cosmphi_a[m_abs_i, :] * dp_t_ind
                Y_lm_d1_phi_a[t_i, :, 0, index_i] = p_t_ind * msinmphi
                Y_lm_d2_phi_a[t_i, :, 0, index_i] = p_t_ind * m2cosphi
                Y_lm_d2_thetaphi_a[t_i, :, 0, index_i] = msinmphi * dp_t_ind
                y_lm_save[t_i, :, 0, index_i] = cosmphi_a[m_abs_i] * p_t_ind

                ## Negative orders
                m_abs_i = m_abs[~m_i]
                index_i = index[~m_i]
                dp_t_ind = np.array([dp_theta[index_i]]).T
                p_t_ind = np.array([p_theta[index_i]]).T

                mcosmphi = cosmphi_a[m_abs_i] * np.array([m_abs_i]).T
                m2sinphi = sinmphi_a[m_abs_i] * np.array([-(m_abs_i**2)]).T
                Y_lm_d1_theta_a[t_i, :, 1, index_i] = sinmphi_a[m_abs_i] * dp_t_ind
                Y_lm_d1_phi_a[t_i, :, 1, index_i] = mcosmphi * p_t_ind
                Y_lm_d2_phi_a[t_i, :, 1, index_i] = m2sinphi * p_t_ind
                Y_lm_d2_thetaphi_a[t_i, :, 1, index_i] = mcosmphi * dp_t_ind
                y_lm_save[t_i, :, 1, index_i] = sinmphi_a[m_abs_i] * p_t_ind

                if theta == 0:
                    # Not defined.
                    Y_lm_d2_theta_a[t_i, :, :, index] = 0.0
                else:
                    # Make use of the Laplacian identity to
                    # estimate the last derivative.
                    Y_lm_d2_theta_a[t_i, :, :, index] = (
                        lapla * y_lm_save[t_i, :, :, index]
                        - Y_lm_d1_theta_a[t_i, :, :, index] * costsint[t_i]
                        - sintt[t_i] * Y_lm_d2_phi_a[t_i, :, :, index]
                    )

        if save:
            if quiet is False:
                print("Saving SH derivatives at: %s" % (path))
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
            if quiet is False:
                print("Not saving SH derivatives")
    else:
        if compressed:
            if quiet is False:
                print(
                    "Loading precomputed compressed SH derivatives for strain calculations"
                )
            with np.load(poly_file) as data:
                Y_lm_d1_theta_a = data["Y_lm_d1_t"]
                Y_lm_d1_phi_a = data["Y_lm_d1_p"]
                Y_lm_d2_theta_a = data["Y_lm_d2_t"]
                Y_lm_d2_phi_a = data["Y_lm_d2_p"]
                Y_lm_d2_thetaphi_a = data["Y_lm_d2_tp"]
                y_lm_save = data["Y_lm"]
            if quiet is False:
                print("Loading done")
        else:
            if quiet is False:
                print("Loading precomputed SH derivatives for strain calculations")
            (
                Y_lm_d1_theta_a,
                Y_lm_d1_phi_a,
                Y_lm_d2_theta_a,
                Y_lm_d2_phi_a,
                Y_lm_d2_thetaphi_a,
                y_lm_save,
            ) = np.load(poly_file, allow_pickle=True)
            if quiet is False:
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
    grid=None,
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
    colat_min : float, optional, default = 0
        Minimum colatitude for grid computation of strains and stresses.
    colat_max : float, optional, default = 180
        Maximum colatitude for grid computation of strains and stresses.
    lon_min : float, optional, default = 0
        Minimum longitude for grid computation of strains and stresses.
    lon_max : float, optional, default = 360
        Maximum longitude for grid computation of strains and stresses.
    lmaxgrid : int, optional, default = None
        The maximum spherical harmonic degree resolvable by the grid.
        If None, this parameter is set to lmax.
        When grid=='GLQ', the gridshape is (lmaxgrid+1, 2*lmaxgrid+1) and
        (2*lmaxgrid+2, 2*(2*lmaxgrid+2)) when grid=='DH'.
    grid: string, optional, default = None
        Either 'DH' or 'GLQ' for Driscoll and Healy grids or Gauss-Legendre
        Quadrature grids following the convention of SHTOOLs.
        If None, the grid is set to 'GLQ'.
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

    if grid is None or grid == "GLQ":
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
    eps = Te_half / (1 + Te_half / R)
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

    deg2rad = pi / 180.0
    grid_long, grid_lat = np.meshgrid(
        np.linspace(0, 360, nlon, endpoint=False) * deg2rad,
        np.linspace(0, 180, nlat, endpoint=False) * deg2rad,
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
        1.0, sin_g_lat_m**2, out=np.zeros_like(sin_g_lat_m), where=sin_g_lat_m != 0
    )
    cot = np.divide(
        1.0,
        np.tan(grid_lat[mask]),
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
        Array with the principal strain or stress direction in degrees.

    Parameters
    ----------
    s_theta : array, float, size(nlat, nlon)
        Array of the colatitude component of the stress or strain field.
    s_phi : array, float, size(nlat, nlon)
        Array of the longitude component of the stress or strain field.
    s_theta_phi : array, float, size(nlat, nlon)
        Array of the colatitude and longitude component of the stress or strain field.
    """
    min_strain = 0.5 * (
        s_theta + s_phi - np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
    )
    max_strain = 0.5 * (
        s_theta + s_phi + np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
    )
    sum_strain = min_strain + max_strain

    principal_angle = 0.5 * np.arctan2(2 * s_theta_phi, s_theta - s_phi) * 180.0 / np.pi

    return min_strain, max_strain, sum_strain, principal_angle


# ==== Principal_strainstress_angle ====


def Strainstress_from_principal(min_strain, max_strain, sum_strain, principal_angle):
    """
    Calculate strains or stresses, from
    their principal values.

    Returns
    -------
    s_theta : array, float, size same as input arrays
        Array of the colatitude component of the stress or strain field.
    s_phi : array, float, size same as input arrays
        Array of the longitude component of the stress or strain field.
    s_theta_phi : array, float, size same as input arrays
        Array of the colatitude and longitude component of the stress or strain field.

    Parameters
    ----------
    min_strain : array, size(nlat, nlon)
        Array with the minimum principal horizontal strain or stress.
    max_strain : array, size(nlat, nlon)
        Array with the maximum principal horizontal strain or stress.
    sum_strain : array, size(nlat, nlon)
        Array with the sum of the principal horizontal strain or stress.
    principal_angle : array, size(nlat, nlon)
        Array with the principal strain or stress direction in degrees.
    """

    deg2rad = np.pi / 180
    s_theta = (max_strain + min_strain) / 2.0 + (
        (max_strain - min_strain) / 2.0
    ) * np.cos(2.0 * principal_angle * deg2rad)
    s_phi = (max_strain + min_strain) / 2.0 - (
        (max_strain - min_strain) / 2.0
    ) * np.cos(2.0 * principal_angle * deg2rad)
    s_theta_phi = (
        0.5 * (max_strain - min_strain) * np.sin(2.0 * principal_angle * deg2rad)
    )

    return s_theta, s_phi, s_theta_phi


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
    ax : array of object, optional, default = None
        Matplotlib axes.
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
    idx_ext = 9676
    idx_comp = 5143
    labels = ["Compressional tectonic features", "Extensional tectonic features"]
    max_idx = [idx_comp, idx_ext]
    faults_cols = [compression_col, extension_col]

    if compression:
        comp_fault_dat = np.loadtxt("%s/Knapmeyer_2006_compdata.txt" % (path))
        ind_comp_fault = np.isin(comp_fault_dat, np.arange(1, idx_comp + 1, dtype=int))
        ind_comp_fault_2 = np.where(ind_comp_fault)[0]
    if extension:
        ext_fault_dat = np.loadtxt("%s/Knapmeyer_2006_extedata.txt" % (path))
        ind_ext_fault = np.isin(ext_fault_dat, np.arange(1, idx_ext + 1, dtype=int))
        ind_ext_fault_2 = np.where(ind_ext_fault)[0]

    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)

    if compression and not extension:
        faults_inds = [ind_comp_fault_2]
        faults_dats = [comp_fault_dat]
        faults_cols = [faults_cols[0]]
        labels = [labels[0]]
        max_idx = [max_idx[0]]
    elif extension and not compression:
        faults_inds = [ind_ext_fault_2]
        faults_dats = [ext_fault_dat]
        faults_cols = [faults_cols[1]]
        labels = [labels[1]]
        max_idx = [max_idx[1]]
    else:
        faults_inds = [ind_comp_fault_2, ind_ext_fault_2]
        faults_dats = [comp_fault_dat, ext_fault_dat]

    for faults, dat, col, label, mx_ix in zip(
        faults_inds, faults_dats, faults_cols, labels, max_idx
    ):
        for axes in [ax] if np.size(ax) == 1 else ax:
            axes.plot(np.nan, np.nan, color=col, lw=lw, label=label)
        for indx in range(1, len(faults) + 1):
            if indx == mx_ix:  # Add last point
                fault_dat_lon = dat[faults[indx - 1] + 1 :][::2]
                fault_dat_lat = dat[faults[indx - 1] + 1 :][1::2]
            else:
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
                    for axes in [ax] if np.size(ax) == 1 else ax:
                        axes.plot(fault_lon % 360, fault_lat, color=col, lw=lw)
            else:
                for axes in [ax] if np.size(ax) == 1 else ax:
                    axes.plot(fault_dat_lon % 360, fault_dat_lat, color=col, lw=lw)

    if legend_show:
        for axes in [ax] if np.size(ax) == 1 else ax:
            axes.legend(loc=legend_loc)
