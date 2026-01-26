"""
Utility functions for calculating the Banerdt (1986) system of equations.
"""

from pathlib import Path
import numpy as np
from pyshtools.gravmag import CilmPlusRhoHDH
from pyshtools.legendre import PlmBar_d1
from pyshtools.expand import SHGLQ, MakeGridPoint
from pyshtools.shclasses.shgrid import SHGrid

# ==== spectral_degrad ====


def spectral_degrad(
    clm, deg_str_grd, lmax_calc=None, smoothing=False, smoothing_m=None, quiet=False
):
    """
    Perform a spectral degradation to the input spherical harmonic
    coefficient given an input degree-strength map.

    Returns
    -------
    array, size(2*lmax+2,2*(2*lmax+2))
        Grid of the spectrally degraded clm coefficients.

    Parameters
    ----------
    clm : array, size (2,lmax+1,lmax+1)
        Array with spherical harmonic coefficients to be degraded.
    deg_str_grd : array, size (2*lmax+2,2*(2*lmax+2))
        Grid with the degree-strength map.
    lmax_calc : int, optional, default = None
        Sets the maximum expansion degree. If None, this is defined by the
        degree-strength.
    smoothing : bool, optional, default = False
        If True, perform a smoothing operation using the smoothing matrix
    smoothing_m : array, optional, default = None
        Smoothing matrix. If None, the matrix is [[-2, -1, 0, 1, 2], [0.5, 1, 1.5, 1, 0.5]].
        This translates into averaging 5 values at each lat/lon cell taking
        degree-strengths of [d-2, d-1, d, d+1, d+2] with weights of
        [0.5, 1, 1.5, 1, 0.5].
    quiet : bool, optional, default = True
        If True, prints the function progress.
    """

    clm_deg0 = clm[0, 0, 0].copy()
    clm[0, 0, 0] = 0.0
    arr_deg_str = range(
        int(np.floor(np.min(deg_str_grd)) + 1),
        int(np.max(deg_str_grd)) + 2 if lmax_calc is None else lmax_calc - 1,
    )
    degraded_grd = SHGrid.from_array(deg_str_grd) * 0.0
    grid_lon, grid_lat = np.meshgrid(degraded_grd.lons(), degraded_grd.lats())

    if smoothing_m is None:
        smoothing_m = [[-2, -1, 0, 1, 2], [0.75, 1, 1.5, 1, 0.75]]
    elif not quiet and smoothing:
        print(f"Changing smoothing matrix to {smoothing_m}")

    if smoothing:
        weight_sum = np.sum(smoothing_m[1])

    lmax_array = np.shape(clm)[2] - 1
    lmax_degstr = arr_deg_str[-1] + np.max(smoothing_m[0])
    if lmax_degstr != lmax_array:
        if not quiet:
            print(f"Padding clm from lmax = {lmax_array} to {lmax_degstr}")
        if lmax_degstr < lmax_array:
            clm = clm[:, : lmax_degstr + 1, : lmax_degstr + 1]
        else:
            clm = np.pad(
                clm,
                ((0, 0), (0, lmax_degstr - lmax_array), (0, lmax_degstr - lmax_array)),
                "constant",
            )

    for d_strength in arr_deg_str:
        if not quiet:
            print(f"Degree {d_strength:5d} / {arr_deg_str[-1]:5d}", end="\r")

        # Get lat/lon mask where the degree-strength is a specific value
        if d_strength == arr_deg_str[0]:
            mask = deg_str_grd <= d_strength
        else:
            mask = (deg_str_grd > d_strength_prev) * (deg_str_grd <= d_strength)

        if np.sum(mask) != 0:
            if smoothing:
                for deg, weight in zip(smoothing_m[0], smoothing_m[1]):
                    degraded_grd.data[mask] += (
                        MakeGridPoint(
                            clm,
                            lat=grid_lat[mask],
                            lon=grid_lon[mask],
                            lmax=d_strength + deg,
                        )
                        * weight
                        / weight_sum
                    )
            else:
                degraded_grd.data[mask] = MakeGridPoint(
                    clm, lat=grid_lat[mask], lon=grid_lon[mask], lmax=d_strength
                )

        d_strength_prev = d_strength

    return degraded_grd.data + clm_deg0


# ==== corr_nmax_drho ====


def corr_nmax_drho(
    dr_lm,
    drho,
    shape_grid,
    rho_grid,
    lmax,
    mass,
    nmax,
    R,
    degrees=None,
    drho_Thinshell=None,
    density_var=False,
):
    """
    Calculate the gravitational difference (with or
    without laterally varying density) between the
    mass-sheet case in the ThinShell system of equations
    and when using the finite amplitude algorithm of Wieczorek &
    Phillips (1998).

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
    drho : float or array(2, lmax+1, lmax+1)
        Mean density contrast or spherical harmonic coefficients
        for the mean density contrast.
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
    R : float
        Mean radius of the planet.
    degrees : array, optional, default = None
        Array with spherical harmonic degrees. size (lmax+1)
    drho_Thinshell : float, optional, default = None
        Mean density contrast used in Thinshell. Should be left
        to None in most cases. If None drho_Thinshell = drho
    density_var : bool, optional, default = False
        If True, correct for density variations.
    """

    if degrees is None:
        degrees = np.arange(lmax + 1, dtype=float)

    if drho_Thinshell is None:
        drho_Thinshell = drho

    # Finite-amplitude correction.
    # This is the computation in Thin_shell_matrix.
    MS_lm_nmax = drho * dr_lm / (2 * degrees.reshape(1, -1, 1) + 1) * 4.0 * np.pi / mass

    if nmax != 1:
        # This is the correct calculation with finite-amplitude
        FA_lm_nmax, D = CilmPlusRhoHDH(shape_grid, nmax, mass, rho_grid, lmax=lmax)
        MS_lm_nmax *= D**2
    else:
        FA_lm_nmax = MS_lm_nmax

    # Density contrast in the relief correction.
    if density_var and nmax == 1:
        MS_lm_drho, D = CilmPlusRhoHDH(shape_grid, nmax, mass, rho_grid, lmax=lmax)
        # MS_lm_drho_cst = MS_lm_nmax.copy()
        # MS_lm_drho_cst *= D**2
        MS_lm_nmax *= D**2

        # Divide because the thin-shell code multiplies by
        # density contrast to correct for finite-amplitude.
        # Here we also correct for density variations, so the
        # correction is already scaled by the density contrast.
        delta_MS_FA = R * (MS_lm_drho - MS_lm_nmax) / drho_Thinshell
    else:
        if density_var and nmax != 1:
            delta_MS_FA = R * (FA_lm_nmax - MS_lm_nmax) / drho_Thinshell
        else:
            delta_MS_FA = R * (FA_lm_nmax - MS_lm_nmax)

    return delta_MS_FA


# ==== DownContFilter ====


def DownContFilter(l, half, R_ref, D_relief, filter_type="Mc", quiet=False):
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
    filter_type : string, optional, default = "Mc"
        Filter type, minimum amplitude ("Ma") of curvature ("Mc").
        If None, returns an array of ones
    quiet : bool, optional, default = True
        If True, prints a warning when D_relief > R_ref.
    """

    if filter_type is None:
        return np.ones_like(l)

    if D_relief > R_ref:
        if not quiet:
            print(
                "! Warning:DownContFilter, D_relief > R_ref, cannot "
                + "use a downward continuation filter. "
                + f"Setting value to 1 ! [D_relief = {int(D_relief / 1e3)}"
                + f", R_ref = {int(R_ref / 1e3)} km]"
            )
        return np.ones_like(l)

    if half == 0:
        DCFilter = 1.0
    else:
        if filter_type == "Mc":
            tmp = 1.0 / (
                (half * half + half)
                * ((2 * half + 1) * (R_ref / D_relief) ** half) ** 2
            )
            DCFilter = (
                1.0 + tmp * (l * l + l) * ((2 * l + 1) * (R_ref / D_relief) ** l) ** 2
            )
        elif filter_type == "Ma":
            tmp = 1.0 / ((2.0 * half + 1.0) * (R_ref / D_relief) ** half) ** 2
            DCFilter = 1.0 + tmp * ((2 * l + 1) * (R_ref / D_relief) ** l) ** 2
        else:
            raise ValueError(
                "Error in DownContFilter, filter_type must be either 'Ma' "
                + f", 'Mc', or None. Input value was {filter_type}."
            )
    DCFilter = 1.0 / DCFilter

    return DCFilter


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
    if theta in (0, np.pi):
        dp_theta = np.zeros((int((lmax + 1) * (lmax + 2) / 2)))
        p_theta = np.zeros((int((lmax + 1) * (lmax + 2) / 2)))
        costsint = 0.0
        sintt = 0.0
    else:
        p_theta, dp_theta = PlmBar_d1(lmax, cost)
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

        if theta in (0, np.pi):
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
    colat_min=0,
    colat_max=180,
    lon_min=0,
    lon_max=360,
    grid="DH",
    dtype=np.float64,
    lmaxgrid=None,
    save=True,
    compressed=False,
    quiet=True,
):
    """
    Compute and store or load spherical harmonic derivatives
    (first and second order) over the entire sphere or given
    a set of colatiudes/longitudes bounds. The spherical
    harmonic degree and order correspond to the index l*(l+1)/2+m.
    This routine supports both Driscoll and Healy (DH) and
    Gauss-Legendre Quadrature (GLQ) grids.

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
    colat_min : float, optional, default = 0
        Minimum colatitude for grid computation of SH derivatives.
    colat_max : float, optional, default = 180
        Maximum colatitude for grid computation of SH derivatives.
    lon_min : float, optional, default = 0
        Minimum longitude for grid computation of SH derivatives.
    lon_max : float, optional, default = 360
        Maximum longitude for grid computation of SH derivatives.
    grid: string, optional, default = 'DH'
        Either 'DH' or 'GLQ' for Driscoll and Healy grids or Gauss-Legendre
        Quadrature grids following the convention of SHTOOLs.
    dtype : data-type, optional, default = numpy.float64
        The desired data-type for the arrays (default is that of numpy).
        This can help reducing the size of the stored array.
    lmaxgrid : int, optional, default = None
        The maximum spherical harmonic degree resolvable by the grid.
        If None, this parameter is set to lmax.
        The gridshape is (2*lmaxgrid+2, 2*(2*lmaxgrid+2)), DH2 grid.
        If None, the grid is set to 'GLQ'.
    save : bool, optional, default = True
        If True, save the data at the given path location.
    compressed : bool, optional, default = False
        If True, the data is saved in compressed .npz format instead of
        npy, which decreases the file size by about a factor 2. This is
        recommended when lmax > 75.
    quiet : bool, optional, default = True
        If True, suppress printing output.
    """

    if lmaxgrid is None:
        lmaxgrid = lmax
    elif lmaxgrid < lmax:
        raise ValueError(
            f"lmaxgrid should be higher or equal than lmax, input is {lmaxgrid}"
            + f" with lmax = {lmax}."
        )

    if (
        (colat_min < 0)
        or (colat_max > 180)
        or (lon_min < 0)
        or (lon_max > 360)
        or (colat_max < colat_min)
        or (lon_max < lon_min)
    ):
        raise ValueError(
            "colat_min, colat_max, lon_min, lon_max are not correctly "
            + "defined the min/max colatitudes and longitudes should "
            + "range from 0–180 and 0–360. "
            + f"Inputs are {colat_min}, {colat_max}, {lon_min}, {lon_max}"
        )

    poly_file = (
        f"{path}/Y_lmsd1d2_{grid}lmax{lmax}"
        + f"_lmaxgrid{lmaxgrid}_f{str(dtype)[-4:-2]}.{ 'npz' if compressed else 'npy'}"
    )

    if grid == "GLQ":
        nlat = lmaxgrid + 1
        nlon = 2 * nlat - 1
    elif grid == "DH":
        nlat = 2 * lmaxgrid + 2
        nlon = 2 * nlat
    else:
        raise ValueError(
            f"Grid format non recognized allowed are 'DH' and 'GLQ', input was {grid}"
        )

    if Path(poly_file).exists() == 0:
        if quiet is False:
            print(
                "Pre-compute SH derivatives, may take some"
                + f" time depending on lmax and lmaxgrid, which are {lmax} and {lmaxgrid}."
            )
            print(f"dtype is {dtype}.")

        index_size = int((lmax + 1) * (lmax + 2) / 2)
        shape_save = (nlat, nlon, 2, index_size)
        Y_lm_d1_theta_a = np.zeros(shape_save, dtype=dtype)
        Y_lm_d1_phi_a = np.zeros(shape_save, dtype=dtype)
        Y_lm_d2_phi_a = np.zeros(shape_save, dtype=dtype)
        Y_lm_d2_thetaphi_a = np.zeros(shape_save, dtype=dtype)
        Y_lm_d2_theta_a = np.zeros(shape_save, dtype=dtype)
        y_lm_save = np.zeros(shape_save, dtype=dtype)

        phi_ar = np.linspace(0, 2.0 * np.pi, nlon, endpoint=False, dtype=dtype)
        msinmphi = np.zeros((lmax + 1, len(phi_ar)), dtype=dtype)
        m2cosphi = np.zeros((lmax + 1, len(phi_ar)), dtype=dtype)
        mcosmphi = np.zeros((lmax + 1, len(phi_ar)), dtype=dtype)
        m2sinphi = np.zeros((lmax + 1, len(phi_ar)), dtype=dtype)
        cosmphi_a = np.zeros((lmax + 1, len(phi_ar)), dtype=dtype)
        sinmphi_a = np.zeros((lmax + 1, len(phi_ar)), dtype=dtype)
        lapla_a = np.zeros((index_size), dtype=dtype)

        # Whole sphere computation, only supports DH grids.
        if (
            (colat_min == 0)
            and (colat_max == 180)
            and (lon_min == 0)
            and (lon_max == 360)
            and grid == "DH"
        ):

            nlat_half = nlat // 2
            theta_range = np.linspace(
                0, np.pi / 2.0, nlat_half, endpoint=False, dtype=dtype
            )
            sint = np.sin(theta_range)
            cost = np.cos(theta_range)
            sintt = np.divide(1.0, sint**2, out=np.zeros_like(sint), where=sint != 0)
            costsint = np.divide(cost, sint, out=np.zeros_like(sint), where=sint != 0)
            sign_conversion = False
            for t_i, theta in enumerate(theta_range):
                t_i_s = nlat - t_i
                if quiet is False:
                    print(f" colatitude {int(theta * 180 / np.pi)} of 90", end="\r")
                if theta == 0:
                    dp_theta = np.zeros((index_size))
                    p_theta = np.zeros((index_size))
                elif theta != 0:
                    p_theta, dp_theta = PlmBar_d1(lmax, cost[t_i])
                    if not sign_conversion:
                        # Given the symmetry of PlmBar_d1 & 'cost',
                        # we here get the sign conversions for positive and
                        # negative 'cost'.
                        tmp1, tmp2 = PlmBar_d1(lmax, -cost[t_i])
                        # Degree-0 is always zero
                        signs_p_theta = np.insert(p_theta[1:] / tmp1[1:], 0, 1)
                        signs_dp_theta = np.insert(dp_theta[1:] / tmp2[1:], 0, 1)
                        sign_conversion = True

                    # Derivative with respect to theta.
                    dp_theta *= -sint[t_i]

                for l in range(lmax + 1):
                    m = np.arange(-l, l + 1)
                    m_abs = np.abs(m)
                    index = np.array(l * (l + 1) / 2 + m_abs, dtype=int)

                    if theta == theta_range[0]:
                        cosmphi_a[l] = np.cos(l * phi_ar)
                        sinmphi_a[l] = np.sin(l * phi_ar)
                        lapla_a[index] = float(-l * (l + 1))
                        ## Positive orders
                        m_i = m >= 0
                        m_abs_i = m_abs[m_i]
                        # First cos(m*phi) derivative
                        msinmphi[m_abs_i] = sinmphi_a[m_abs_i] * np.transpose(
                            [-m_abs_i]
                        )
                        # Second cos(m*phi) derivative
                        m2cosphi[m_abs_i] = cosmphi_a[m_abs_i] * np.transpose(
                            [-(m[m_i] ** 2)]
                        )
                        ## Negative orders
                        m_abs_i = m_abs[~m_i]
                        mcosmphi[m_abs_i] = cosmphi_a[m_abs_i] * np.transpose([m_abs_i])
                        m2sinphi[m_abs_i] = sinmphi_a[m_abs_i] * np.transpose(
                            [-(m_abs_i**2)]
                        )

                    ## Positive orders
                    m_i = m >= 0
                    m_abs_i = m_abs[m_i]
                    index_i = index[m_i]
                    dp_t_ind = np.transpose([dp_theta[index_i]])
                    p_t_ind = np.transpose([p_theta[index_i]])
                    Y_lm_d1_theta_a[t_i, :, 0, index_i] = cosmphi_a[m_abs_i] * dp_t_ind
                    Y_lm_d1_phi_a[t_i, :, 0, index_i] = p_t_ind * msinmphi[m_abs_i]
                    Y_lm_d2_phi_a[t_i, :, 0, index_i] = p_t_ind * m2cosphi[m_abs_i]
                    Y_lm_d2_thetaphi_a[t_i, :, 0, index_i] = (
                        msinmphi[m_abs_i] * dp_t_ind
                    )
                    y_lm_save[t_i, :, 0, index_i] = cosmphi_a[m_abs_i] * p_t_ind

                    ## Negative orders
                    m_abs_i = m_abs[~m_i]
                    index_i = index[~m_i]
                    dp_t_ind = np.transpose([dp_theta[index_i]])
                    p_t_ind = np.transpose([p_theta[index_i]])
                    Y_lm_d1_theta_a[t_i, :, 1, index_i] = sinmphi_a[m_abs_i] * dp_t_ind
                    Y_lm_d1_phi_a[t_i, :, 1, index_i] = mcosmphi[m_abs_i] * p_t_ind
                    Y_lm_d2_phi_a[t_i, :, 1, index_i] = m2sinphi[m_abs_i] * p_t_ind
                    Y_lm_d2_thetaphi_a[t_i, :, 1, index_i] = (
                        mcosmphi[m_abs_i] * dp_t_ind
                    )
                    y_lm_save[t_i, :, 1, index_i] = sinmphi_a[m_abs_i] * p_t_ind

                # theta = 0 Not defined.
                if theta != 0:
                    # Make use of the Laplacian identity to
                    # estimate the last derivative.
                    Y_lm_d2_theta_a[t_i] = (
                        lapla_a * y_lm_save[t_i]
                        - Y_lm_d1_theta_a[t_i] * costsint[t_i]
                        - sintt[t_i] * Y_lm_d2_phi_a[t_i]
                    )
                    # Make south Hemisphere using the sign conversion from symmetry
                    # Sign change
                    Y_lm_d2_theta_a[t_i_s] = Y_lm_d2_theta_a[t_i] * -signs_dp_theta
                    Y_lm_d1_theta_a[t_i_s] = Y_lm_d1_theta_a[t_i] * -signs_p_theta
                    Y_lm_d2_thetaphi_a[t_i_s] = Y_lm_d2_thetaphi_a[t_i] * -signs_p_theta
                    #
                    Y_lm_d1_phi_a[t_i_s] = Y_lm_d1_phi_a[t_i] * signs_p_theta
                    Y_lm_d2_phi_a[t_i_s] = Y_lm_d2_phi_a[t_i] * signs_p_theta
                    y_lm_save[t_i_s] = y_lm_save[t_i] * signs_p_theta

        else:
            phi_ind_1 = int(nlon / (phi_ar.max() * 180 / np.pi) * lon_min)
            phi_ind_2 = int(nlon / (phi_ar.max() * 180 / np.pi) * lon_max)
            phi_ind = slice(phi_ind_1, phi_ind_2)
            phi_ar_s = phi_ar[phi_ind]

            # For symmetry speedup
            if colat_min == 0 and colat_max == 180:
                p_theta_a = np.zeros((index_size, nlat // 2 + 1), dtype=dtype)
                dp_theta_a = np.zeros((index_size, nlat // 2 + 1), dtype=dtype)

            if grid == "GLQ":
                zeros, _ = SHGLQ(lmax)
                theta_range = np.arccos(zeros)
                step_theta = theta_range[-1] - theta_range[-2]
            else:
                theta_range, step_theta = np.linspace(
                    0, np.pi, nlat, endpoint=False, dtype=dtype, retstep=True
                )

            sint = np.sin(theta_range)
            cost = np.cos(theta_range)
            sintt = np.divide(1.0, sint**2, out=np.zeros_like(sint), where=sint != 0)
            costsint = np.divide(cost, sint, out=np.zeros_like(sint), where=sint != 0)

            sign_conversion = False
            for t_i, theta in enumerate(theta_range):
                theta_180 = theta * 180.0 / np.pi
                if theta == 0:
                    dp_theta = np.zeros((index_size))
                    p_theta = np.zeros((index_size))
                if quiet is False:
                    print(
                        f" colatitude {int(theta_180)} of {colat_max}",
                        end="\r",
                    )
                if theta_180 < colat_min or theta_180 > colat_max:
                    continue
                if theta != 0:
                    if colat_min != 0 or colat_max != 180:
                        # Don't use the symmetry speedup, which requires a whole sphere computation
                        p_theta, dp_theta = PlmBar_d1(lmax, cost[t_i])
                        dp_theta *= -sint[t_i]
                    elif cost[t_i] >= 0:
                        (
                            p_theta_a[:, t_i],
                            dp_theta_a[:, t_i],
                        ) = PlmBar_d1(lmax, cost[t_i])
                        if not sign_conversion:
                            # Given the symmetry of PlmBar_d1 & 'cost',
                            # we here get the sign conversions for positive
                            # and negative 'cost'.
                            tmp1, tmp2 = PlmBar_d1(lmax, -cost[t_i])
                            # Degree-0 is always zero
                            signs_p_theta = np.insert(
                                p_theta_a[1:, t_i] / tmp1[1:], 0, 1
                            )
                            signs_dp_theta = np.insert(
                                dp_theta_a[1:, t_i] / tmp2[1:], 0, 1
                            )
                            sign_conversion = True
                        p_theta = p_theta_a[:, t_i]
                        # Derivative with respect to theta.
                        dp_theta = dp_theta_a[:, t_i] * -sint[t_i]
                    else:
                        # Sign conversion when cost is negative
                        idx_sign = (
                            nlat // 2 - t_i if nlat % 2 != 0 else nlat // 2 - t_i - 1
                        )
                        p_theta = p_theta_a[:, idx_sign] * signs_p_theta
                        dp_theta = dp_theta_a[:, idx_sign] * signs_dp_theta * -sint[t_i]

                for l in range(lmax + 1):
                    m = np.arange(-l, l + 1)
                    m_abs = np.abs(m)
                    index = np.array(l * (l + 1) / 2 + m_abs, dtype=int)
                    # Do only once for a given theta
                    if (theta == theta_range[0]) or (
                        theta_180 <= (colat_min + step_theta * 180.0 / np.pi)
                    ):
                        cosmphi_a[l, phi_ind] = np.cos(l * phi_ar_s)
                        sinmphi_a[l, phi_ind] = np.sin(l * phi_ar_s)
                        lapla_a[index] = float(-l * (l + 1))
                        ## Positive orders
                        m_i = m >= 0
                        m_abs_i = m_abs[m_i]
                        # First cos(m*phi) derivative
                        msinmphi[m_abs_i, phi_ind] = sinmphi_a[
                            m_abs_i, phi_ind
                        ] * np.transpose([-m_abs_i])
                        # Second cos(m*phi) derivative
                        m2cosphi[m_abs_i, phi_ind] = cosmphi_a[
                            m_abs_i, phi_ind
                        ] * np.transpose([-(m[m_i] ** 2)])
                        ## Negative orders
                        m_abs_i = m_abs[~m_i]
                        mcosmphi[m_abs_i, phi_ind] = cosmphi_a[
                            m_abs_i, phi_ind
                        ] * np.transpose([m_abs_i])
                        m2sinphi[m_abs_i, phi_ind] = sinmphi_a[
                            m_abs_i, phi_ind
                        ] * np.transpose([-(m_abs_i**2)])

                    ## Positive orders
                    m_i = m >= 0
                    m_abs_i = m_abs[m_i]
                    index_i = index[m_i]
                    dp_t_ind = np.transpose([dp_theta[index_i]])
                    p_t_ind = np.transpose([p_theta[index_i]])

                    Y_lm_d1_theta_a[t_i, phi_ind, 0, index_i] = (
                        cosmphi_a[m_abs_i, phi_ind] * dp_t_ind
                    )
                    Y_lm_d1_phi_a[t_i, phi_ind, 0, index_i] = (
                        p_t_ind * msinmphi[m_abs_i, phi_ind]
                    )
                    Y_lm_d2_phi_a[t_i, phi_ind, 0, index_i] = (
                        p_t_ind * m2cosphi[m_abs_i, phi_ind]
                    )
                    Y_lm_d2_thetaphi_a[t_i, phi_ind, 0, index_i] = (
                        msinmphi[m_abs_i, phi_ind] * dp_t_ind
                    )
                    y_lm_save[t_i, phi_ind, 0, index_i] = (
                        cosmphi_a[m_abs_i, phi_ind] * p_t_ind
                    )

                    ## Negative orders
                    m_abs_i = m_abs[~m_i]
                    index_i = index[~m_i]
                    dp_t_ind = np.transpose([dp_theta[index_i]])
                    p_t_ind = np.transpose([p_theta[index_i]])

                    Y_lm_d1_theta_a[t_i, phi_ind, 1, index_i] = (
                        sinmphi_a[m_abs_i, phi_ind] * dp_t_ind
                    )
                    Y_lm_d1_phi_a[t_i, phi_ind, 1, index_i] = (
                        mcosmphi[m_abs_i, phi_ind] * p_t_ind
                    )
                    Y_lm_d2_phi_a[t_i, phi_ind, 1, index_i] = (
                        m2sinphi[m_abs_i, phi_ind] * p_t_ind
                    )
                    Y_lm_d2_thetaphi_a[t_i, phi_ind, 1, index_i] = (
                        mcosmphi[m_abs_i, phi_ind] * dp_t_ind
                    )
                    y_lm_save[t_i, phi_ind, 1, index_i] = (
                        sinmphi_a[m_abs_i, phi_ind] * p_t_ind
                    )

                # theta = 0 Not defined.
                if theta != 0:
                    # Make use of the Laplacian identity to
                    # estimate the last derivative.
                    Y_lm_d2_theta_a[t_i, phi_ind] = (
                        lapla_a * y_lm_save[t_i, phi_ind]
                        - Y_lm_d1_theta_a[t_i, phi_ind] * costsint[t_i]
                        - sintt[t_i] * Y_lm_d2_phi_a[t_i, phi_ind]
                    )

        if save:
            if quiet is False:
                print(f"Saving SH derivatives at: {path}")
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

    # Number of idx in the file
    idx_ext = 9676
    idx_comp = 5143
    labels = ["Compressional tectonic features", "Extensional tectonic features"]
    max_idx = [idx_comp, idx_ext]
    faults_cols = [compression_col, extension_col]

    if compression:
        comp_fault_dat = np.loadtxt(f"{path}/Knapmeyer_2006_compdata.txt")
        ind_comp_fault = np.isin(comp_fault_dat, np.arange(1, idx_comp + 1, dtype=int))
        ind_comp_fault_2 = np.where(ind_comp_fault)[0]
    if extension:
        ext_fault_dat = np.loadtxt(f"{path}/Knapmeyer_2006_extedata.txt")
        ind_ext_fault = np.isin(ext_fault_dat, np.arange(1, idx_ext + 1, dtype=int))
        ind_ext_fault_2 = np.where(ind_ext_fault)[0]

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(1, 1)

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
        (s_theta + s_phi) - np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
    )
    max_strain = 0.5 * (
        (s_theta + s_phi) - np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
    )
    sum_strain = min_strain + max_strain
    principal_angle = 0.5 * np.arctan2(2 * s_theta_phi, s_theta - s_phi) * 180.0 / np.pi

    return min_strain, max_strain, sum_strain, principal_angle


# ==== Principal_strainstress_angle ====


def Strainstress_from_principal(min_strain, max_strain, principal_angle):
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
