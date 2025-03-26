#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module calculates the partial sensitivity functions and
simulates GPP using the LUE model

author: rde
first created: Tue Dec 26 2023 17:33:47 CET
"""

import numpy as np
from src.common.get_params import get_params
from src.common.wai import calc_wai
from src.lue_model.partial_sensitivity_funcs import (
    f_temp_horn,
    f_vpd_co2_preles,
    f_water_horn,
    f_light_scalar_tal,
    f_cloud_index_exp,
)


def run_lue_model(
    param_values_scalar,
    param_names,
    ip_df_dict,
    ip_df_daily_wai,
    wai_output,
    nstepsday,
    fpar_var_name,
    co2_var_name,
    param_group_to_vary,
    param_names_to_vary,
):
    """
    Calculate partial sensitivity functions and simulate GPP using the LUE model

    Parameters:
    param_values_scalar (array): array of scalar of parameter values
    param_names (list): list of parameter names, which are optimized
    ip_df_dict (dict): dictionary with input forcing data
    ip_df_daily_wai (dict): dictionary with input data for daily WAI calculation
    wai_output (dict): dictionary to store WAI output
    nstepsday (int): number of sub-daily timesteps in a day
    fpar_var_name (str): name of fAPAR variable
    co2_var_name (str): name of CO2 variable

    Returns:
    lue_model_op (dict): dictionary with LUE model output
    """
    # recalculate parameter values based on the scalar values given by optimizer
    updated_params = get_params(ip_df_dict, param_names, param_values_scalar)

    # for the parameters varying per year, set the value of 1st year to
    # the original parameter key (so that the functions can use it)
    # later on, the actual values for each year will be used
    for param_name in param_names_to_vary:
        if (param_name == "alpha_fT_Horn") and (
            ip_df_dict["KG"][0] not in ["C", "D", "E"]
        ):
            pass
        elif (param_name == "alpha") and (ip_df_dict["KG"][0] != "B"):
            pass
        else:
            updated_params[param_name] = updated_params[
                f"{param_name}_{int(np.unique(ip_df_dict['year'])[0])}"
            ]

    # if WAI parameters are varying per year, calculate WAI for each years
    if (param_group_to_vary == "Group7") or (param_group_to_vary == "Group8"):

        wai_op_dict_list = {}
        for var in wai_output:
            wai_op_dict_list[var] = []

        for yr in np.unique(ip_df_dict["year"]):
            yr_idx = np.where(ip_df_dict["year"] == yr)[0]

            updated_params["AWC"] = updated_params[f"AWC_{int(yr)}"]
            updated_params["theta"] = updated_params[f"theta_{int(yr)}"]
            updated_params["alphaPT"] = updated_params[f"alphaPT_{int(yr)}"]
            updated_params["meltRate_temp"] = updated_params[f"meltRate_temp_{int(yr)}"]
            updated_params["meltRate_netrad"] = updated_params[
                f"meltRate_netrad_{int(yr)}"
            ]

            # calculate WAI
            # first do spinup for calculation of wai using daily data
            # (for faster spinup loops when actual simulations are sub daily)
            wai_results = calc_wai(
                ip_df_daily_wai,
                wai_output,
                updated_params,
                wai0=updated_params["AWC"],
                nloops=updated_params["nloop_wai_spin"],
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
                updated_params,
                wai0=wai_results["wai"][-1],
                nloops=updated_params["nloop_wai_act"],
                spinup=False,
                do_snow=True,
                normalize_wai=True,
                do_sublimation=True,
                nstepsday=nstepsday,
            )

            for var, var_empty_list in wai_op_dict_list.items():
                var_empty_list.append(wai_results[var][yr_idx])

        for var, var_list in wai_op_dict_list.items():
            wai_output[var] = np.concatenate(var_list)

    else:
        # calculate WAI
        # first do spinup for calculation of wai using daily data
        # (for faster spinup loops when actual simulations are sub daily)
        wai_results = calc_wai(
            ip_df_daily_wai,
            wai_output,
            updated_params,
            wai0=updated_params["AWC"],
            nloops=updated_params["nloop_wai_spin"],
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
            updated_params,
            wai0=wai_results["wai"][-1],
            nloops=updated_params["nloop_wai_act"],
            spinup=False,
            do_snow=True,
            normalize_wai=True,
            do_sublimation=True,
            nstepsday=nstepsday,
        )

    # calculate sensitivity function for temperature
    f_tair = f_temp_horn(
        ip_df_dict["TA_GF"],
        updated_params["T_opt"],
        updated_params["K_T"],
        updated_params["alpha_fT_Horn"],
    )

    # calculate sensitivity function for VPD
    f_vpd_co2, f_vpd_part, f_co2_part = f_vpd_co2_preles(
        ip_df_dict["VPD_GF"],
        ip_df_dict[co2_var_name],
        updated_params["Kappa_VPD"],
        updated_params["Ca_0"],
        updated_params["C_Kappa"],
        updated_params["c_m"],
    )

    # calculate sensitivity function for soil moisture
    f_water = f_water_horn(
        wai_results["wai_nor"],
        updated_params["W_I"],
        updated_params["K_W"],
        updated_params["alpha"],
    )

    # calculate sensitivity function for light scalar
    f_light = f_light_scalar_tal(
        ip_df_dict[fpar_var_name],
        ip_df_dict["PPFD_IN_GF"],
        updated_params["gamma_fL_TAL"],
    )

    # calculate sensitivity function for cloudiness index
    f_cloud, ci = f_cloud_index_exp(
        mu_fci=updated_params["mu_fCI"],
        sw_in=ip_df_dict["SW_IN_GF"],
        sw_in_pot=ip_df_dict["SW_IN_POT_ONEFlux"],
    )

    # # calculate GPP
    # gpp_lue = (
    #     updated_params["LUE_max"] * f_tair * f_vpd_co2 * f_water * f_light * f_cloud
    # ) * (ip_df_dict["PPFD_IN_GF"] * ip_df_dict[fpar_var_name])

    # re-calculating GPP using the sensitivity functions for which
    # the parameters are varying per year
    if param_group_to_vary == "Group1":
        gpp_lue_list = []
        for yr in np.unique(ip_df_dict["year"]):
            yr_idx = np.where(ip_df_dict["year"] == yr)[0]

            gpp_lue_yr = (
                updated_params[f"LUE_max_{int(yr)}"]
                * f_tair[yr_idx]
                * f_vpd_co2[yr_idx]
                * f_water[yr_idx]
                * f_light[yr_idx]
                * f_cloud[yr_idx]
            ) * (ip_df_dict["PPFD_IN_GF"][yr_idx] * ip_df_dict[fpar_var_name][yr_idx])

            gpp_lue_list.append(gpp_lue_yr)
        gpp_lue = np.concatenate(gpp_lue_list)

    elif param_group_to_vary == "Group2":
        f_tair_list = []
        for yr in np.unique(ip_df_dict["year"]):
            yr_idx = np.where(ip_df_dict["year"] == yr)[0]
            ta_ts = ip_df_dict["TA_GF"][yr_idx]

            if ip_df_dict["KG"][0] not in ["C", "D", "E"]:
                # calculate sensitivity function for temperature
                f_tair_yr = f_temp_horn(
                    ta_ts,
                    updated_params[f"T_opt_{int(yr)}"],
                    updated_params[f"K_T_{int(yr)}"],
                    updated_params["alpha_fT_Horn"],
                )
            else:
                # calculate sensitivity function for temperature
                f_tair_yr = f_temp_horn(
                    ta_ts,
                    updated_params[f"T_opt_{int(yr)}"],
                    updated_params[f"K_T_{int(yr)}"],
                    updated_params[f"alpha_fT_Horn_{int(yr)}"],
                )

            f_tair_list.append(f_tair_yr)
        f_tair = np.concatenate(f_tair_list)

    elif param_group_to_vary == "Group3":
        f_vpd_co2_list = []
        f_vpd_part_list = []
        f_co2_part_list = []
        for yr in np.unique(ip_df_dict["year"]):
            yr_idx = np.where(ip_df_dict["year"] == yr)[0]
            vpd_ts = ip_df_dict["VPD_GF"][yr_idx]
            co2_ts = ip_df_dict[co2_var_name][yr_idx]

            # calculate sensitivity function for VPD
            f_vpd_co2, f_vpd_part, f_co2_part = f_vpd_co2_preles(
                vpd_ts,
                co2_ts,
                updated_params[f"Kappa_VPD_{int(yr)}"],
                updated_params[f"Ca_0_{int(yr)}"],
                updated_params[f"C_Kappa_{int(yr)}"],
                updated_params[f"c_m_{int(yr)}"],
            )

            f_vpd_co2_list.append(f_vpd_co2)
            f_vpd_part_list.append(f_vpd_part)
            f_co2_part_list.append(f_co2_part)
        f_vpd_co2 = np.concatenate(f_vpd_co2_list)
        f_vpd_part = np.concatenate(f_vpd_part_list)
        f_co2_part = np.concatenate(f_co2_part_list)

    elif param_group_to_vary == "Group4":
        f_light_list = []
        for yr in np.unique(ip_df_dict["year"]):
            yr_idx = np.where(ip_df_dict["year"] == yr)[0]
            fpar_ts = ip_df_dict[fpar_var_name][yr_idx]
            ppfd_ts = ip_df_dict["PPFD_IN_GF"][yr_idx]

            # calculate sensitivity function for light scalar
            f_light = f_light_scalar_tal(
                fpar_ts,
                ppfd_ts,
                updated_params[f"gamma_fL_TAL_{int(yr)}"],
            )

            f_light_list.append(f_light)
        f_light = np.concatenate(f_light_list)

    elif param_group_to_vary == "Group5":
        f_cloud_list = []
        ci_list = []
        for yr in np.unique(ip_df_dict["year"]):
            yr_idx = np.where(ip_df_dict["year"] == yr)[0]
            sw_in_ts = ip_df_dict["SW_IN_GF"][yr_idx]
            sw_in_pot_ts = ip_df_dict["SW_IN_POT_ONEFlux"][yr_idx]

            # calculate sensitivity function for cloudiness index
            f_cloud, ci = f_cloud_index_exp(
                mu_fci=updated_params[f"mu_fCI_{int(yr)}"],
                sw_in=sw_in_ts,
                sw_in_pot=sw_in_pot_ts,
            )

            f_cloud_list.append(f_cloud)
            ci_list.append(ci)
        f_cloud = np.concatenate(f_cloud_list)
        ci = np.concatenate(ci_list)

    elif (param_group_to_vary == "Group6") or (param_group_to_vary == "Group8"):
        f_water_list = []
        for yr in np.unique(ip_df_dict["year"]):
            yr_idx = np.where(ip_df_dict["year"] == yr)[0]
            wai_ts = wai_results["wai_nor"][yr_idx]

            if ip_df_dict["KG"][0] != "B":
                # calculate sensitivity function for soil moisture
                f_water = f_water_horn(
                    wai_ts,
                    updated_params[f"W_I_{int(yr)}"],
                    updated_params[f"K_W_{int(yr)}"],
                    updated_params["alpha"],
                )
            else:
                # calculate sensitivity function for soil moisture
                f_water = f_water_horn(
                    wai_ts,
                    updated_params[f"W_I_{int(yr)}"],
                    updated_params[f"K_W_{int(yr)}"],
                    updated_params[f"alpha_{int(yr)}"],
                )

            f_water_list.append(f_water)
        f_water = np.concatenate(f_water_list)

    if param_group_to_vary != "Group1":
        gpp_lue = (
            updated_params["LUE_max"] * f_tair * f_vpd_co2 * f_water * f_light * f_cloud
        ) * (ip_df_dict["PPFD_IN_GF"] * ip_df_dict[fpar_var_name])

    # store model outputs in dictionary
    wai_results["fW"] = f_water
    lue_model_op = {}
    lue_model_op["gpp_lue"] = gpp_lue
    lue_model_op["fT"] = f_tair
    lue_model_op["fVPD"] = f_vpd_co2
    lue_model_op["fVPD_part"] = f_vpd_part
    lue_model_op["fCO2_part"] = f_co2_part
    lue_model_op["fW"] = f_water
    lue_model_op["fL"] = f_light
    lue_model_op["fCI"] = f_cloud
    lue_model_op["ci"] = ci
    lue_model_op["wai_results"] = wai_results

    return lue_model_op
