import numpy as np
import pyshtools as pysh
from pathlib import Path

pi = np.pi


def SH_deriv(theta, phi, lmax):

    #############################################################

    # Compute spherical harmonic derivatives on the fly.

    #############################################################

    Y_lm_d1_theta_a = np.zeros((2, lmax + 1, lmax + 1))
    Y_lm_d1_phi_a = np.zeros((2, lmax + 1, lmax + 1))
    Y_lm_d2_phi_a = np.zeros((2, lmax + 1, lmax + 1))
    Y_lm_d2_thetaphi_a = np.zeros((2, lmax + 1, lmax + 1))
    Y_lm_d2_theta_a = np.zeros((2, lmax + 1, lmax + 1))
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

    y_lm = pysh.expand.spharm(lmax, theta, phi, degrees=False)
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
            else:
                mcosmphi = m_abs * cosmphi
                m2sinphi = -(m_abs ** 2) * sinmphi
                Y_lm_d1_theta_a[1, l, m_abs] = dp_theta[index] * sinmphi
                Y_lm_d1_phi_a[1, l, m_abs] = p_theta[index] * mcosmphi
                Y_lm_d2_phi_a[1, l, m_abs] = p_theta[index] * m2sinphi
                Y_lm_d2_thetaphi_a[1, l, m_abs] = dp_theta[index] * mcosmphi

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


def SH_deriv_store(lmax, path):

    #############################################################

    # Compute spherical harmonic derivatives and store them.

    #############################################################

    n = 2 * lmax + 2
    poly_file = "%s/Y_lmsd1d2_lmax%s.npy" % (path, lmax)

    if Path(poly_file).exists() == 0:
        print("Pre-compute SH derivatives")
        index_size = int((lmax + 1) * (lmax + 2) / 2)
        Y_lm_d1_theta_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        Y_lm_d1_phi_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        Y_lm_d2_phi_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        Y_lm_d2_thetaphi_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        Y_lm_d2_theta_a = np.zeros((n, 2 * n, 2, lmax + 1, lmax + 1))
        phi_ar = np.linspace(0, 360, 2 * n, endpoint=False) * pi / 180.0

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
                    else:
                        mcosmphi = m_abs * cosmphi
                        m2sinphi = -(m_abs ** 2) * sinmphi
                        Y_lm_d1_theta_a[t_i, :, 1, l, m_abs] = dp_theta[index] * sinmphi
                        Y_lm_d1_phi_a[t_i, :, 1, l, m_abs] = p_theta[index] * mcosmphi
                        Y_lm_d2_phi_a[t_i, :, 1, l, m_abs] = p_theta[index] * m2sinphi
                        Y_lm_d2_thetaphi_a[t_i, :, 1, l, m_abs] = (
                            dp_theta[index] * mcosmphi
                        )

                p_i = -1
                for phi in phi_ar:
                    p_i += 1
                    y_lm = pysh.expand.spharm(l, theta, phi, degrees=False)
                    if theta == 0 or theta == pi:
                        Y_lm_d2_theta_a[t_i, p_i, :, l, : l + 1] = 0.0
                        # Not defined.
                    else:
                        # Make use of the Laplacian identity to
                        # estimate last derivative.
                        Y_lm_d2_theta_a[t_i, p_i, :, l, : l + 1] = (
                            lapla * y_lm[:, l, : l + 1]
                            - Y_lm_d1_theta_a[t_i, p_i, :, l, : l + 1] * costsint
                            - sintt * Y_lm_d2_phi_a[t_i, p_i, :, l, : l + 1]
                        )

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


def Displacement_strains(
    A_lm,
    w_lm,
    E,
    v,
    R,
    Te,
    lmax_calc,
    colat_min=-1e20,
    colat_max=1e20,
    lon_min=-1e20,
    lon_max=1e20,
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

    #############################################################

    # Computes the Banerdt (1986) equations to determine strains
    # from displacements with a correction to the theta_phi term.

    #############################################################

    if lmax_calc != np.shape(A_lm)[2] - 1:
        if quiet is False:
            print(
                "Padding A_lm and w_lm from lmax = %s to %s"
                % (np.shape(A_lm)[2] - 1, lmax_calc)
            )
        A_lm = pysh.SHCoeffs.from_array(A_lm).pad(lmax=lmax_calc).coeffs
        w_lm = pysh.SHCoeffs.from_array(w_lm).pad(lmax=lmax_calc).coeffs

    n = 2 * lmax_calc + 2

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
                lmax_calc, path
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
    D = (E * (Te * Te * Te)) / (float(float(12) * (float(1) - v ** 2)))
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
                y_lm = pysh.expand.spharm(lmax_calc, theta, phi, degrees=False)
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
                ) = SH_deriv(theta, phi, lmax_calc)
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
            # omega[t_i,p_i] = 2. * R_m1 * csc * (d2Athetaphi -
            # cot * d1Aphi) Error in Banerdt

            kappa_theta[t_i, p_i] = -(R_m1 ** 2) * (d2wtheta + w_lm_ylm)
            kappa_phi[t_i, p_i] = -(R_m1 ** 2) * (
                d2wphi * csc2 + d1wtheta * cot + w_lm_ylm
            )
            tau[t_i, p_i] = -(R_m1 ** 2) * csc * (d2wthetaphi - cot * d1wphi)
            # tau[t_i,p_i] = - 2. * R_m1**2 * csc * (d2wthetaphi -
            # cot * d1wphi) Error in Banerdt

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
            stress_theta_phi[t_i, p_i] = tau[t_i, p_i]

    stress_theta *= DpsiTeR / 1e6  # MPa
    stress_phi *= DpsiTeR / 1e6  # MPa
    stress_theta_phi *= (0.5 * DpsiTeR * (1.0 - v) * Te / 2.0) / 1e6  # MPa

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


def Principal_strain_angle(
    eps_theta, eps_phi, eps_theta_phi, k_theta, k_phi, k_theta_phi
):

    #############################################################

    # A function that computes principal strains and angles

    #############################################################

    strain_theta = -eps_theta - k_theta
    strain_phi = -eps_phi - k_phi
    strain_theta_phi = -eps_theta_phi - k_theta_phi
    min_strain = 0.5 * (
        strain_theta
        + strain_phi
        - np.sqrt((strain_theta - strain_phi) ** 2 + 4 * strain_theta_phi ** 2)
    )
    max_strain = 0.5 * (
        strain_theta
        + strain_phi
        + np.sqrt((strain_theta - strain_phi) ** 2 + 4 * strain_theta_phi ** 2)
    )
    sum_strain = min_strain + max_strain

    principal_angle = (
        0.5
        * np.arctan2(2 * strain_theta_phi, strain_theta - strain_phi)
        * 180.0
        / np.pi
    )
    principal_angle2 = (
        0.5
        * np.arctan2(2 * strain_theta_phi, strain_phi - strain_theta)
        * 180.0
        / np.pi
    )

    return min_strain, max_strain, sum_strain, principal_angle, principal_angle2


def Plt_tecto_Mars(
    path,
    compression=False,
    extension=True,
    ax=None,
    compression_col="k",
    extension_col="purple",
    lw=1,
):

    #############################################################

    # Plot the Knapmeyer et al. (2006) tectonic dataset.

    #############################################################

    comp_fault_dat = np.loadtxt("%s/Knapmeyer_2006_compdata.txt" % (path))
    ext_fault_dat = np.loadtxt("%s/Knapmeyer_2006_extedata.txt" % (path))
    ind_comp_fault = np.isin(comp_fault_dat, np.arange(1, 5143, dtype=float))
    ind_comp_fault_2 = np.where(ind_comp_fault)[0]
    ind_ext_fault = np.isin(ext_fault_dat, np.arange(1, 9676, dtype=float))
    ind_ext_fault_2 = np.where(ind_ext_fault)[0]

    if compression and not extension:
        faults_inds = ind_comp_fault_2
        faults_dats = comp_fault_dat
        faults_cols = compression_col
    elif extension and not compression:
        faults_inds = ind_ext_fault_2
        faults_dats = ext_fault_dat
        faults_cols = extension_col
    else:
        faults_inds = [ind_comp_fault_2, ind_ext_fault_2]
        faults_dats = [comp_fault_dat, ext_fault_dat]
        faults_cols = [compression_col, extension_col]

    for faults, dat, col in zip([faults_inds], [faults_dats], [faults_cols]):
        for indx in range(1, int((len(faults) - 1) / 2)):
            ind_fault_check = range(faults[indx - 1] + 1, faults[indx])
            fault_dat_lon = dat[ind_fault_check][::2]
            fault_dat_lat = dat[ind_fault_check][1::2]
            ax.plot((fault_dat_lon + 360) % 360, fault_dat_lat, color=col, lw=lw)
