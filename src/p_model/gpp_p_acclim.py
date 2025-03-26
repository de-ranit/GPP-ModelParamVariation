#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this module calculates GPPp with acclimation using P-Model and scaled by soil mpisture stress
ref: https://doi.org/10.1029/2021MS002767; https://github.com/GiuliaMengoli/P-model_subDaily

author: rde, skoirala
first created: 2023-11-07
"""
import numpy as np
import numexpr as ne

from src.p_model.pmodel_plus import pmodel_plus
from src.common.wai import calc_wai


def rep_sd(a, nstepsday):
    """
    copy the average noon values to all the timesteps (nstepsday) in a day

    parameters:
    a (array): array of average noon values
    nstepsday (int): number of timesteps in a day

    returns:
    array of average noon values copied to all the timesteps (nstepsday) in a day
    """
    return np.tile(a, nstepsday).reshape(nstepsday, -1).flatten(order="F")


def make_sd_from_daily(daily_dict, nstepsday, key_add="_opt"):
    """
    take each variable from the daily dictionary and
    repeat it nstepsday times and add to subdaily dictionary

    parameters:
    daily_dict (dict): dictionary with daily data
    nstepsday (int): number of timesteps in a day
    key_add (str): string to be added to the key names

    returns:
    sd_dict (dict): dictionary with subdaily data (where each variable is repeated nstepsday times)
    """
    sd_dict = {}
    for key, value in daily_dict.items():
        sd_dict[key + key_add] = rep_sd(
            value, nstepsday
        )  # rename the keys as *_opt and add to the dictionary
    return sd_dict


def fw_horn(wai_nor, w_i, k_w, alpha):
    """
    calculate sensitivity function for moisture stress

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
    fw = 1.0 / (1.0 + ne.evaluate("exp(fw_eval_exp)"))  # calculate fw_Horn

    return fw


def gpp_p_acclim(
    ip_df_dict,
    ip_df_daily,
    ip_df_daily_wai,
    wai_results,
    model_op_no_acclim_sd,
    nstepsday,
    params,
    co2_var_name,
    param_group_to_vary,
):
    """
    Calculate GPPp with acclimation

    parameters:
    ip_df_dict (dict): dictionary with input forcing data
    ip_df_daily (dict): dictionary with daily input forcing data (from noon)
    ip_df_daily_wai (dict): dictionary with input data for daily WAI calculation
    wai_results (dict): dictionary to store WAI output
    model_op_no_acclim_sd (dict): dictionary with PModel output without acclimation
    nstepsday (int): number of timesteps in a day
    params (dict): dictionary with parameter values
    co2_var_name (str): name of co2 variable
    param_group_to_vary (str): parameter group to be optimized per year, while other
                               parameters are kept constant across years in a site

    returns:
    o_a (dict): dictionary with PModel output with acclimation
    """

    model_op_acclim_dd = pmodel_plus(
        ip_df_daily, params, co2_var_name
    )  # , Temp_res) #Pass Mid-day average of daily data to PModelPlus

    # copy the average noon values (*_opt values) to all the timesteps (nstepsday) in a day
    model_op_acclim_dd["TA_GF"] = ip_df_daily["TA_GF"]
    model_op_acclim_dd["Iabs"] = ip_df_daily["Iabs"]
    model_op_acclim_sd = make_sd_from_daily(model_op_acclim_dd, nstepsday)

    # rename outputs
    o_na = model_op_no_acclim_sd  # model output using original sub-daily forcing data
    o_a = model_op_acclim_sd  # model output using
    # optimum forcing data from noon and copied to all timesteps in a day

    # add additional variables to o_a (output acclim) dict required for calculating GPPpOpt
    o_a["phi0"] = o_na[
        "phi0"
    ]  # add phi0 calculated from original data to o_a (output acclim) dict
    o_a["TA_OptK"] = (
        o_a["TA_GF_opt"] + 273.15
    )  # opt temperature (avg. noon value and rolled mean) in K
    o_a["TA_K"] = ip_df_dict["TA_GF"] + 273.15  # original temperature in K
    o_a["GammaStarM"] = o_na[
        "GammaStarM"
    ]  # add GammaStarM calculated from original data to o_a (output acclim) dict
    o_a["kmPa"] = o_na[
        "kmPa"
    ]  # add kmPa calculated from original data to o_a (output acclim) dict
    o_a["Iabs"] = ip_df_dict[
        "Iabs"
    ]  # add original Iabs data to o_a (output acclim) dict

    # acclimated sensitivity of χ (ci/ca) to vapor pressure deficit (Pa^(1/2))
    xi_pa_eval_exp = (  # pylint: disable=unused-variable
        params["beta"]
        * (o_a["kmPa_opt"] + o_a["GammaStarM_opt"])
        / (1.6 * o_a["viscosityH2oStar_opt"])
    )
    xi_pa_vals = ne.evaluate("sqrt(xi_pa_eval_exp)")

    # acclimated leaf-internal CO2 partial pressure
    # (with acclimated χ, and adjusted with the actual VPD) (Pa)
    vpd_vals_original = ip_df_dict["VPD_GF"]  # pylint: disable=unused-variable
    ci_vals = (
        (xi_pa_vals * o_a["Ca_opt"])
        + (o_a["GammaStarM_opt"] * ne.evaluate("sqrt(vpd_vals_original)"))
    ) / (xi_pa_vals + ne.evaluate("sqrt(vpd_vals_original)"))

    # Optimal maximum rate of carboxylation (µmol CO2 m−2 s−1)
    vc_max_opt_eval_exp = 1.0 - (  # pylint: disable=unused-variable
        (params["c"] * (ci_vals + 2.0 * o_a["GammaStarM_opt"]))
        / (ci_vals - o_a["GammaStarM_opt"])
    ) ** (2.0 / 3.0)
    vc_max_opt_vals = (
        o_a["phi0"]
        * o_a["Iabs_opt"]
        * ((ci_vals + o_a["kmPa_opt"]) / (ci_vals + 2.0 * o_a["GammaStarM_opt"]))
        * ne.evaluate("sqrt(vc_max_opt_eval_exp)")
    )

    # Optimal maximum rate of electron transport (µmol electrons m−2 s−1)
    j_max_opt_eval_exp = (  # pylint: disable=unused-variable
        1.0
        / (
            1.0
            - (
                (params["c"] * (ci_vals + 2.0 * o_a["GammaStarM_opt"]))
                / (ci_vals - o_a["GammaStarM_opt"])
            )
            ** (2.0 / 3.0)
        )
    ) - 1.0
    j_max_opt_vals = (4.0 * o_a["phi0"] * o_a["Iabs_opt"]) / ne.evaluate(
        "sqrt(j_max_opt_eval_exp)"
    )

    # calculate instantaneous Vcmax and Jmax
    vc_max_adjusted_eval_exp = (  # pylint: disable=unused-variable
        params["Ha"] / params["Rgas"]
    ) * (1.0 / o_a["TA_OptK"] - 1.0 / o_a["TA_K"])
    vc_max_adjusted_vals = vc_max_opt_vals * ne.evaluate(
        "exp(vc_max_adjusted_eval_exp)"
    )

    # Maximum rate of electron transport (µmol electrons m−2 s−1)
    j_max_adjusted_eval_exp = (  # pylint: disable=unused-variable
        params["Haj"] / params["Rgas"]
    ) * (1.0 / o_a["TA_OptK"] - 1.0 / o_a["TA_K"])
    j_max_adjusted_vals = j_max_opt_vals * ne.evaluate("exp(j_max_adjusted_eval_exp)")

    # acclimated Rubisco-limited assimilation rate (µmol CO2 m−2 s−1)
    ac_vals = (
        vc_max_adjusted_vals * (ci_vals - o_a["GammaStarM"]) / (ci_vals + o_a["kmPa"])
    )

    # Rate of electron transport (µmol electrons m−2 s−1)
    # zero values of j_max_adjusted_vals, will produce runtime warning and nan values in j_vals,
    # suppress runtime warning, then replace nan in j_vals with 0.0, when Iabs is 0.0 (during night)
    with np.errstate(divide="ignore", invalid="ignore"):
        j_eval_exp = (  # pylint: disable=unused-variable
            1.0 + ((4.0 * o_a["phi0"] * o_a["Iabs"]) / j_max_adjusted_vals) ** 2.0
        )
        j_vals = (4.0 * o_a["phi0"] * o_a["Iabs"]) / ne.evaluate("sqrt(j_eval_exp)")

    # if Iabs is zero, then j_vals is zero
    j_idx = o_a["Iabs"] == 0.0
    j_vals[j_idx] = 0.0

    # if Iabs is not zero and j_max_adjusted_vals is zero, then j_vals is nan
    j_idx = (o_a["Iabs"] != 0.0) & (j_max_adjusted_vals == 0.0)
    j_vals[j_idx] = np.nan

    # Electron-transport limited assimilation (µmol electrons m−2 s−1)
    aj_vals = (j_vals / 4.0) * (
        (ci_vals - o_a["GammaStarM"]) / (ci_vals + 2.0 * o_a["GammaStarM"])
    )

    # calculate GPP as minimum of Ac and Aj (µmol electrons m−2 s−1)
    gpp_p_opt_vals = np.fmin(ac_vals, aj_vals)

    # add acclimated (final) model ouputs to dictionary
    o_a["xiPa"] = xi_pa_vals
    o_a["ci"] = ci_vals
    o_a["vcmaxOpt"] = vc_max_opt_vals
    o_a["JmaxOpt"] = j_max_opt_vals
    o_a["vcmaxAdjusted"] = vc_max_adjusted_vals
    o_a["JmaxAdjusted"] = j_max_adjusted_vals
    o_a["Ac1Opt"] = ac_vals
    o_a["J"] = j_vals
    o_a["AJp1Opt"] = aj_vals
    o_a["GPPp_opt"] = gpp_p_opt_vals

    if (param_group_to_vary == "Group3") or (param_group_to_vary == "Group4"):

        wai_op_dict_list = {}
        for var in wai_results:
            wai_op_dict_list[var] = []

        for yr in np.unique(ip_df_dict["year"]):
            yr_idx = np.where(ip_df_dict["year"] == yr)[0]

            params["AWC"] = params[f"AWC_{int(yr)}"]
            params["theta"] = params[f"theta_{int(yr)}"]
            params["alphaPT"] = params[f"alphaPT_{int(yr)}"]
            params["meltRate_temp"] = params[f"meltRate_temp_{int(yr)}"]
            params["meltRate_netrad"] = params[f"meltRate_netrad_{int(yr)}"]

            # calculate WAI
            # first do spinup for calculation of wai using daily data
            # (for faster spinup loops when actual simulations are sub daily)
            wai_results = calc_wai(
                ip_df_daily_wai,
                wai_results,
                params,
                wai0=params["AWC"],
                nloops=params["nloop_wai_spin"],
                spinup=True,
                do_snow=True,
                normalize_wai=False,
                do_sublimation=True,
                nstepsday=nstepsday,
            )

            # then when steady state is achieved (after 5 spinup loops),
            # do the actual calculation of wai using subdaily/daily data
            wai_results = calc_wai(
                ip_df_dict,
                wai_results,
                params,
                wai0=wai_results["wai"][-1],
                nloops=params["nloop_wai_act"],
                spinup=False,
                do_snow=True,
                normalize_wai=True,
                do_sublimation=True,
                nstepsday=nstepsday,
            )

            for var, var_empty_list in wai_op_dict_list.items():
                var_empty_list.append(wai_results[var][yr_idx])

        for var, var_list in wai_op_dict_list.items():
            wai_results[var] = np.concatenate(var_list)

    else:
        # first do spinup for calculation of wai using daily data (for faster spinup loops)
        wai_results = calc_wai(
            ip_df_daily_wai,
            wai_results,
            params,
            wai0=params["AWC"],
            nloops=params["nloop_wai_spin"],
            spinup=True,
            do_snow=True,
            normalize_wai=False,
            do_sublimation=True,
            nstepsday=nstepsday,
        )

        # then when steady state is achieved (after 5 spinup loops),
        # do the actual calculation of wai using subdaily data
        wai_results = calc_wai(
            ip_df_dict,
            wai_results,
            params,
            wai0=wai_results["wai"][-1],
            nloops=params["nloop_wai_act"],
            spinup=False,
            do_snow=True,
            normalize_wai=True,
            do_sublimation=True,
            nstepsday=nstepsday,
        )

    if (param_group_to_vary == "Group2") or (param_group_to_vary == "Group4"):
        f_water_list = []
        for yr in np.unique(ip_df_dict["year"]):
            yr_idx = np.where(ip_df_dict["year"] == yr)[0]
            wai_ts = wai_results["wai_nor"][yr_idx]

            if ip_df_dict["KG"][0] != "B":
                # calculate sensitivity function for soil moisture
                f_water = fw_horn(
                    wai_ts,
                    params[f"W_I_{int(yr)}"],
                    params[f"K_W_{int(yr)}"],
                    params["alpha"],
                )
            else:
                # calculate sensitivity function for soil moisture
                f_water = fw_horn(
                    wai_ts,
                    params[f"W_I_{int(yr)}"],
                    params[f"K_W_{int(yr)}"],
                    params[f"alpha_{int(yr)}"],
                )

            f_water_list.append(f_water)
        fw = np.concatenate(f_water_list)
    else:
        # calculate sensitivity function for moisture stress
        fw = fw_horn(
            wai_results["wai_nor"],
            w_i=params["W_I"],
            k_w=params["K_W"],
            alpha=params["alpha"],
        )

    wai_results["fW"] = fw

    # multiply soil moisture sensitivity function with GPPp_opt to scale GPP values
    o_a["GPPp_opt_fW"] = gpp_p_opt_vals * fw
    o_a["wai_results"] = wai_results

    return o_a
