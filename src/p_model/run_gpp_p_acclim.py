#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calculate rolling mean of noon data and run P-model with acclimation and moisture stress
ref: https://doi.org/10.1029/2021MS002767; https://github.com/GiuliaMengoli/P-model_subDaily

author: rde, skoirala
first created: 2023-11-07
"""
import logging
import bottleneck as bn
import numpy as np
import pandas as pd

from src.common.get_params import get_params
from src.p_model.gpp_p_acclim import gpp_p_acclim

logger = logging.getLogger(__name__)


def getrollingmean(data, time_info, acclim_window, yr_a_t=None, site_id=None):
    """
    calculate rolling mean of forcing variables for acclimation period

    parameters:
    data (array): forcing variable data
    time_info (dict): dictionary with time info
    acclim_window (int): acclimation window in days
    yr_a_t (int): year for which acclim window
    site_id (str): site ID

    returns:
    data_acclim (array): rolling mean of forcing variables for acclimation period
    """
    window_size = int(np.floor(acclim_window))  # make acclimation window an integer

    # happens in few cases
    if window_size == 0:
        window_size = 1  # set window size to 1 if acclimation window is less than 1 day
        if yr_a_t is not None:
            logger.warning(
                "%s (%s): acclimation window is less than 1 day, replacing with a value of 1",
                site_id,
                yr_a_t,
            )
        else:
            logger.warning(
                "%s: acclimation window is less than 1 day, replacing with a value of 1",
                site_id,
            )

    # select noon data based on timeinfo and take average
    data_day = bn.nanmean(
        data.reshape(-1, time_info["nstepsday"])[
            :, time_info["start_hour_ind"] : time_info["end_hour_ind"]
        ],
        axis=1,
    )
    data_acclim = bn.move_mean(
        data_day, window=window_size, min_count=1
    )  # calculate rolling mean of average noon data
    return data_acclim


def get_daily_acclim_data(
    ip_df_dict, time_info, fpar_var_name, co2_var_name, acclim_window, year=None
):
    """
    select required forcing variables and calculate rolling mean for acclimation period

    parameters:
    ip_df_dict (dict): dictionary with input forcing data
    time_info (dict): dictionary with time info
    fpar_var_name (str): name of fpar variable
    co2_var_name (str): name of co2 variable
    acclim_window (int): acclimation window in days
    year (int): acclim window for which year

    returns:
    data_sd (dict): dictionary with rolling mean of forcing variables for acclimation period
    """
    varibs = (
        "PPFD_IN_GF",
        fpar_var_name,
        "TA_GF",
        co2_var_name,
        "VPD_GF",
        "SW_IN_POT_ONEFlux",
        "Iabs",
    )  # forcing variables needed for PModelPlus
    data_dd = {}
    for _var in varibs:
        if year is not None:
            data_dd[_var] = getrollingmean(
                ip_df_dict[_var], time_info, acclim_window, year, ip_df_dict["SiteID"]
            )  # calculate rolling mean of each forcing variables for acclimation period
        else:
            data_dd[_var] = getrollingmean(
                ip_df_dict[_var], time_info, acclim_window, site_id=ip_df_dict["SiteID"]
            )  # calculate rolling mean of each forcing variables for acclimation period
    data_dd["elev"] = float(
        ip_df_dict["elev"]
    )  # Append elevation with daily (noon) data

    return data_dd


def run_p_model(
    p_values_scalar,
    p_names,
    ip_df_dict,
    model_op_no_acclim_sd,
    ip_df_daily_wai,
    wai_output,
    time_info,
    fpar_var_name,
    co2_var_name,
    param_group_to_vary,
    param_names_to_vary,
):
    """
    forward run PModel with acclimation and return GPP scaled by soil moisture stress

    parameters:
    p_values_scalar (array): array of scalar values for parameters
    p_names (list): list of parameter names
    ip_df_dict (dict): dictionary with input forcing data
    model_op_no_acclim_sd (dict): dictionary with PModel output without acclimation
    ip_df_daily_wai (dict): dictionary with input data for daily WAI calculation
    wai_output (dict): dictionary to store WAI output
    time_info (dict): dictionary with time info
    fpar_var_name (str): name of fpar variable
    co2_var_name (str): name of co2 variable

    returns:
    p_model_acclim_fw_op (dict): dictionary with PModel output with acclimation
    """

    # recalculate parameter values based on the scalar values given by optimizer
    updated_params = get_params(ip_df_dict, p_names, p_values_scalar)

    # for the parameters varying per year, set the value of 1st year to
    # the original parameter key (so that the functions can use it)
    # later on, the actual values for each year will be used
    for param_name in param_names_to_vary:
        if (param_name == "alpha") and (ip_df_dict["KG"][0] != "B"):
            pass
        else:
            updated_params[param_name] = updated_params[
                f"{param_name}_{int(np.unique(ip_df_dict['year'])[0])}"
            ]

    if param_group_to_vary == "Group1":

        ip_df_daily_dict_list = {
            "PPFD_IN_GF": [],
            "FPAR_FLUXNET_EO": [],
            "TA_GF": [],
            "CO2_MLO_NOAA": [],
            "VPD_GF": [],
            "SW_IN_POT_ONEFlux": [],
            "Iabs": [],
        }

        resampled_time_arr = np.array(
            pd.DatetimeIndex(ip_df_dict["Time"])
            .to_period("D")
            .to_timestamp()
            .unique()
            .year
        )

        for yr in np.unique(resampled_time_arr):
            yr_idx = np.where(resampled_time_arr == yr)[0]

            ip_df_daily_dict = get_daily_acclim_data(
                ip_df_dict,
                time_info,
                fpar_var_name,
                co2_var_name,
                updated_params[f"acclim_window_{int(yr)}"],
                int(yr),
            )

            for var, coll_list in ip_df_daily_dict_list.items():
                if not isinstance(ip_df_daily_dict[var], float):
                    coll_list.append(ip_df_daily_dict[var][yr_idx])

        for var, var_list in ip_df_daily_dict_list.items():
            ip_df_daily_dict[var] = np.concatenate(var_list)

        ip_df_daily_dict["elev"] = float(ip_df_dict["elev"])

    else:
        ip_df_daily_dict = get_daily_acclim_data(
            ip_df_dict,
            time_info,
            fpar_var_name,
            co2_var_name,
            updated_params["acclim_window"],
        )

    p_model_acclim_fw_op = gpp_p_acclim(
        ip_df_dict,
        ip_df_daily_dict,
        ip_df_daily_wai,
        wai_output,
        model_op_no_acclim_sd,
        time_info["nstepsday"],
        updated_params,
        co2_var_name,
        param_group_to_vary,
    )

    return p_model_acclim_fw_op
