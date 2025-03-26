#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run P-Model without acclimation
ref: https://doi.org/10.1029/2021MS002767; https://github.com/GiuliaMengoli/P-model_subDaily

author: rde, skoirala
first created: 2023-11-07
"""

import numpy as np
import numexpr as ne


def pmodel_plus(input_data, params, co2_var_name):
    """
    Run P-Model without acclimation

    parameters:
    input data (dict): contains timeseries of forcing variables
    params (dict): contains global parameters and parameters to be optimized for each site

    returns:
    output_dict (dict): contains timeseries of P-Model output variables
    """

    # get required input data arrays from the dictionary
    tair_vals = input_data["TA_GF"]
    co2_vals = input_data[co2_var_name]
    vpd_vals = input_data["VPD_GF"]  # pylint: disable=unused-variable
    i_abs_vals = input_data["Iabs"]
    z_v = input_data["elev"]

    # Atmospheric pressure as a function of eleveation (Pa)
    p_pa = params["kPo"] * (1.0 - params["kL"] * (z_v / params["kTo"])) ** (
        (params["kG"] * params["kMa"]) / (params["kR"] * params["kL"])
    )

    # Partial pressure of oxygen (Pa)
    o2_pa = params["kco"] * (1e-6) * p_pa

    # Michaelis-Menten constant for carboxylation (Pa)
    kc_pa_eval_exp = (params["EaKc"] * (tair_vals - 25.0)) / ( # pylint: disable=unused-variable
        298.15 * params["kR"] * (tair_vals + 273.15)
    )
    kc_pa_vals = params["Kc25"] * ne.evaluate("exp(kc_pa_eval_exp)")

    # Michaelis-Menten constant for oxygenation (Pa)
    ko_pa_eval_exp = (params["EaKo"] * (tair_vals - 25.0)) / ( # pylint: disable=unused-variable
        298.15 * params["kR"] * (tair_vals + 273.15)
    )
    ko_pa_vals = params["Ko25"] * ne.evaluate("exp(ko_pa_eval_exp)")

    # Effective Michaelis-Menten coefficient for Rubisco kinetics) (Pa)
    km_pa_vals = kc_pa_vals * (1.0 + (o2_pa / ko_pa_vals))

    # Standard atmospheric pressure at 0 m elevation (Pa)
    p_pa_sea_level = params["kPo"] * (
        1.0 - params["kL"] * (params["sea_level_elev"] / params["kTo"])
    ) ** ((params["kG"] * params["kMa"]) / (params["kR"] * params["kL"]))
    p_ratio = p_pa / p_pa_sea_level

    # Adjust photorespiratory compensation point at 25°C
    # for the pressure at given elevation
    gammastar25 = (
        params["gamma25"] * p_ratio
    )

    # Photorespiratory compensation point (GammaStar) (Pa)
    gamma_star_eval_exp = (params["EaGamma"] * (tair_vals - 25.0)) / ( # pylint: disable=unused-variable
        params["kR"] * 298.15 * (tair_vals + 273.15)
    )
    gamma_star_vals = gammastar25 * ne.evaluate("exp(gamma_star_eval_exp)")

    # water viscosity (η*) relative at 25°C (unitless)
    viscosity_water_star_eval_exp = -3.719 + (580.0 / ((tair_vals + 273.15) - 138.0)) # pylint: disable=unused-variable
    viscosity_water_star_vals = (
        ne.evaluate("exp(viscosity_water_star_eval_exp)") / 0.911
    )

    # Sensitivity of χ (ci/ca) to vapor pressure deficit (Pa^(1/2))
    xi_pa_eval_exp = (params["beta"] * (km_pa_vals + gamma_star_vals)) / ( # pylint: disable=unused-variable
        1.6 * viscosity_water_star_vals
    )
    xi_pa_vals = ne.evaluate("sqrt(xi_pa_eval_exp)")

    # Ambient CO2 partial pressure (Pa)
    ca_vals = co2_vals * (1.0e-6) * p_pa

    # Leaf-internal CO2 partial pressure (Pa)
    ci_vals = (
        (xi_pa_vals * ca_vals) + (gamma_star_vals * ne.evaluate("sqrt(vpd_vals)"))
    ) / (xi_pa_vals + ne.evaluate("sqrt(vpd_vals)"))

    # Temperature dependence function of quantum efficiency (mol CO2/mol photons)
    phi0_vals = (
        params["phi0_coeff_a"]
        + (params["phi0_coeff_b"] * tair_vals)
        - (params["phi0_coeff_c"] * tair_vals**2.0)
    )

    # Maximum rate of carboxylation (µmol CO2 m−2 s−1)
    vc_max_eval_exp = 1.0 - ( # pylint: disable=unused-variable
        (params["c"] * (ci_vals + 2.0 * gamma_star_vals)) / (ci_vals - gamma_star_vals)
    ) ** (2.0 / 3.0)
    vc_max_p_model_vals = (
        phi0_vals
        * i_abs_vals
        * ((ci_vals + km_pa_vals) / (ci_vals + 2.0 * gamma_star_vals))
        * ne.evaluate("sqrt(vc_max_eval_exp)")
    )

    # Rubisco-limited assimilation rate (µmol CO2 m−2 s−1)
    ac_vals = vc_max_p_model_vals * (
        (ci_vals - gamma_star_vals) / (ci_vals + km_pa_vals)
    )

    # Maximum rate of electron transport (µmol electrons m−2 s−1)
    j_max_eval_exp = ( # pylint: disable=unused-variable
        1.0
        / (
            1.0
            - (
                (params["c"] * (ci_vals + 2.0 * gamma_star_vals))
                / (ci_vals - gamma_star_vals)
            )
            ** (2.0 / 3.0)
        )
    ) - 1.0
    # j_max_vals cab be 0.0, as Iabs is 0.0 during night time
    j_max_vals = (4.0 * phi0_vals * i_abs_vals) / ne.evaluate("sqrt(j_max_eval_exp)")

    # Rate of electron transport (µmol electrons m−2 s−1)
    # zero values of j_max_vals, will produce runtime warning and nan values in j_vals,
    # suppress runtime warning, then replace nan in j_vals with 0.0, when Iabs is 0.0 (during night)
    with np.errstate(divide="ignore", invalid="ignore"):
        j_eval_exp = 1.0 + ((4.0 * phi0_vals * i_abs_vals) / j_max_vals) ** 2.0 # pylint: disable=unused-variable
        j_vals = (4.0 * phi0_vals * i_abs_vals) / ne.evaluate("sqrt(j_eval_exp)")

    # if Iabs is zero, then j_vals is zero
    j_idx = i_abs_vals == 0.0
    j_vals[j_idx] = 0.0

    # if Iabs is not zero and j_vals is zero, then j_vals is nan
    j_idx = (i_abs_vals != 0.0) & (j_vals == 0.0)
    j_vals[j_idx] = np.nan

    # Electron-transport limited assimilation (µmol electrons m−2 s−1)
    aj_vals = (j_vals / 4.0) * (
        (ci_vals - gamma_star_vals) / (ci_vals + 2.0 * gamma_star_vals)
    )

    # calculate GPP as minimum of Ac and Aj (µmol electrons m−2 s−1)
    gpp_p_vals = np.fmin(ac_vals, aj_vals)

    # Append the PModel op and intermediate variables to the output dictionary
    output_dict = {}
    output_dict["kcPa"] = kc_pa_vals
    output_dict["koPa"] = ko_pa_vals
    output_dict["kmPa"] = km_pa_vals
    output_dict["GammaStarM"] = gamma_star_vals
    output_dict["viscosityH2oStar"] = viscosity_water_star_vals
    output_dict["xiPaM"] = xi_pa_vals
    output_dict["Ca"] = ca_vals
    output_dict["ciM"] = ci_vals
    output_dict["phi0"] = phi0_vals
    output_dict["vcmaxPmodelM1"] = vc_max_p_model_vals
    output_dict["Ac1"] = ac_vals
    output_dict["JmaxM1"] = aj_vals
    output_dict["Jp1"] = j_vals
    output_dict["AJp1"] = aj_vals
    output_dict["GPPp"] = gpp_p_vals

    return output_dict
