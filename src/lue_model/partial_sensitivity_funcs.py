#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this module calculates partial sensitivity functions for environmental variables
ref: https://doi.org/10.1016/j.agrformet.2021.108708, https://doi.org/10.1029/2022MS003464

author: rde
first created: Sat Dec 23 2023 17:20:29 CET
"""

import numpy as np
import numexpr as ne

def f_temp_horn(temp, t_opt, k_t, alpha_ft):
    """
    calculate partial sensitivity function for temperature
    based on Horn and Schulz (2011)
    ref: https://doi.org/10.5194/bg-8-999-2011

    parameters:
    temp (array): temperature timeseries (degC)
    t_opt (float): parameter to calculate fT
                   (Optimal temperature in degC)
    k_t (float): parameter to calculate fT
                    (Sensitivity to temperature changes; degC-1)
    alpha_ft (float): lag parameter to calculate fT (dimensionless)

    returns:
    ft_horn (array): partial sensitivity function values for temperature
    """

    lag_step = 1  # number of previous timestep to be considered for the lag function

    # initialize t_f array
    t_f = np.zeros_like(temp)
    # calculate t_f
    for idx, tair in enumerate(temp):
        if idx == 0:
            t_f[idx] = (1.0 - alpha_ft) * tair + alpha_ft * tair
        else:
            t_f[idx] = (1.0 - alpha_ft) * tair + alpha_ft * t_f[idx - lag_step]

    # calculate ft_Horn
    # the scalar in numerator has been changed from 4 to 2 and
    # in denominator e(x**2) has been changed to (e(x))**2 to make the fT values between 0 and 1
    ft_eval_exp_num = -(t_f - t_opt) / k_t  # pylint: disable=unused-variable
    ft_eval_exp_deno = -(t_f - t_opt) / k_t  # pylint: disable=unused-variable
    ft_horn = (2.0 * ne.evaluate("exp(ft_eval_exp_num)")) / (
        1.0 + (ne.evaluate("exp(ft_eval_exp_deno)")) ** 2.0
    )

    return ft_horn


def f_vpd_co2_preles(vpd, c_a, kappa, ca_0, c_kappa, c_m):
    """
    calculate partial sensitivity function for VPD and CO2
    based on PRELES model
    ref: http://urn.fi/URN:ISBN:978-951-40-2395-8 (ISBN 978-951-40-2395-8 (PDF))

    parameters:
    vpd (array): VPD timeseries (Pa)
    c_a (array): atmospheric CO2 concentration (ppm)
    kappa (float): parameter to calculate fVPD
                   (Sensitivity to VPD changes in Pa-1)
    ca_0 (float): parameter to calculate fVPD
                  (Minimum optimal atmospheric CO2 concentration in ppm)
    c_kappa (float): parameter to calculate fVPD
                     (Sensitivity to CO2 changes; dimensionless)
    c_m (float): parameter to calculate fVPD
                 (CO2 fertilization intensity indicator in ppm)

    returns:
    fvpd (array): partial sensitivity function values for VPD and CO2
    """

    # negative sign is added to kappa as the bounds and initial are made positive
    kappa = -kappa

    f_vpd_part_eval_exp = (  # pylint: disable=unused-variable
        kappa * ((ca_0 / c_a) ** c_kappa) * vpd
    )
    f_vpd_part = ne.evaluate("exp(f_vpd_part_eval_exp)")

    # keep the values of f_vpd_part between 0 and 1
    f_vpd_part = np.where(f_vpd_part > 1.0, 1.0, f_vpd_part)
    f_vpd_part = np.where(f_vpd_part < 0.0, 0.0, f_vpd_part)

    f_co2_part = 1.0 + ((c_a - ca_0) / (c_a - ca_0 + c_m))

    fvpd = f_vpd_part * f_co2_part

    return fvpd, f_vpd_part, f_co2_part


def f_water_horn(wai_nor, w_i, k_w, alpha):
    """
    calculate sensitivity function for moisture stress
    based on Horn and Schulz (2011)
    ref: https://doi.org/10.5194/bg-8-999-2011

    parameters:
    wai_nor (array): normalized WAI timeseries (dimensionless or mm/mm)
    w_i (float): parameter to calculate fW
                (Optimal soil moisture in mm/mm)
    k_w (float): parameter to calculate fW
                 (Sensitivity to soil moisture changes; dimensionless)
    alpha (float): lag parameter to calculate fW (dimensionless)

    returns:
    fw (array): partial sensitivity function values for moisture stress
    """

    k_w = (
        -k_w
    )  # negative sign is added to k_w as the bounds and initial are made positive
    lag_step = 1  # number of previous timestep to be considered for the lag function

    # initialize w_f array
    w_f = np.zeros_like(wai_nor)
    # calculate w_f
    for idx, wai_nor_val in enumerate(wai_nor):
        if idx == 0:
            w_f[idx] = (1.0 - alpha) * wai_nor_val + alpha * wai_nor_val
        else:
            w_f[idx] = (1.0 - alpha) * wai_nor_val + alpha * w_f[idx - lag_step]
    
    fw_eval_exp = k_w * (w_f - w_i)  # pylint: disable=unused-variable
    fw_horn = 1.0 / (1.0 + ne.evaluate("exp(fw_eval_exp)"))  # calculate fw_Horn

    return fw_horn


def f_light_scalar_tal(fpar, ppfd, gamma_fl):
    """
    calculate sensitivity function for light scalar
    based on Light use efficiency - temperature acclimation and light (LUE-TAL) model
    ref: https://doi.org/10.1111/j.1365-2486.2007.01463.x

    parameters:
    fpar (array): fraction of photosynthetically active radiation (dimensionless)
    ppfd (array): photosynthetic photon flux density (umol photons m-2 s-1)
    gamma_fl (float): parameter to calculate fL
                      (Light saturation curvature indicator; umol photons-1 m2s)

    returns:
    fl_tal (array): partial sensitivity function values for light scalar
    """

    fl_tal = 1.0 / (gamma_fl * (fpar * ppfd) + 1.0)

    return fl_tal


def f_cloud_index_exp(mu_fci, sw_in=None, sw_in_pot=None, ci=None):
    """
    calculate sensitivity function for cloudiness index
    based on Bao et al. (2022)
    ref: https://doi.org/10.1016/j.agrformet.2021.108708

    parameters:
    mu_fci (float): parameter to calculate fCI
                    (Sensitivity to cloudiness index changes; dimensionless)
    sw_in (array): incoming shortwave radiation (W m-2)
    sw_in_pot (array): potential incoming shortwave radiation (W m-2)
    ci (array): cloudiness index (dimensionless)
    
    either ci or sw_in and sw_in_pot should be provided to calculate fCI
    
    returns:
    fci (array): partial sensitivity function values for cloudiness index
    ci (array): cloudiness index (dimensionless)
    """
    # when ci is not supplied, calculate it from sw_in and sw_in_pot
    if (ci is None) and (sw_in is not None) and (sw_in_pot is not None):

        # ignore the runtime warning produced by ZeroDivisionError
        # and later deal with the inf and nan values
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = 1.0 - (sw_in / sw_in_pot)

        # set inf (1.0/0) and nan (0.0/0.0) values produced by
        # ZeroDivisionError (when sw_in_pot is 0.0) to 0.0
        sw_in_pot_zero_mask = sw_in_pot == 0.0
        ci = np.where(sw_in_pot_zero_mask, 0.0, ci)

        # physically implausible case (sw_in > sw_in_pot) is also set to 0
        high_sw_in_mask = sw_in > sw_in_pot
        ci = np.where(high_sw_in_mask, 0.0, ci)

    # when ci is supplied, it can be directly used to calculate fCI
    elif ci is not None:
        pass
    else: # when ci is not supplied and sw_in and sw_in_pot are also not supplied
        raise ValueError(
            "Either ci or sw_in and sw_in_pot should be provided to calculate fCI"
        )

    # calculate fCI
    fci = ci**mu_fci

    return fci, ci
