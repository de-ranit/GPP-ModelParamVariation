#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
forward run model with optimized parameters,
collect and save relevant results,
plot diagnostic figures

author: rde
first created: 2023-11-10
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.p_model.run_gpp_p_acclim import run_p_model
from src.lue_model.run_lue_model import run_lue_model
from src.p_model.pmodel_plus import pmodel_plus
from src.common.get_params import get_params
from src.postprocess.prep_results import prep_results
from src.postprocess.plot_site_level_results import plot_site_level_results
from src.common.opti_per_site_or_site_year import add_keys_for_group


def scale_back_params(xbest, p_names, params):
    """
    if parameter scalar (actual value/initial value) was further scaled between 0 and 1,
    convert them back to actual scalar values

    parameters:
    xbest (list): array of scalars of optimized parameter values (between 0 and 1)
    p_names (list): list of parameter names which were optimized
    params (dict): dictionary with parameter initial values and bounds

    returns:
    xbest (list): list of actual scalar values of optimized parameters
    """

    # get the scaled value of parameter bounds (ub/initial or lb/initial)
    p_ubound_scaled = []
    p_lbound_scaled = []
    for p in p_names:
        p_ubound_scaled.append(params[p]["ub"] / params[p]["ini"])
        p_lbound_scaled.append(params[p]["lb"] / params[p]["ini"])

    # calculate the multipliers and zero and scale back the parameters to actual scalar values
    multipliers = np.array(
        [ub - lb for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)]
    )
    zero = np.array(
        [-lb / (ub - lb) for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)]
    )
    xbest_actual = list(multipliers * (np.array(xbest) - zero))

    # return the actual scalar values of parameters
    return xbest_actual


def forward_run_model(
    ip_df_dict,
    ip_df_daily_wai,
    wai_output,
    time_info,
    settings_dict,
    xbest,
    opti_param_names=None,
):
    """
    forward run the model with optimized parameters,
    collect and save relevant results, plot diagnostic figures

    parameters:
    ip_df_dict (dict): dictionary with input forcing data
    ip_df_daily_wai (dict): dictionary with input data for daily WAI calculation
    wai_output (dict): dictionary to store WAI output
    time_info (dict): dictionary with time info
    settings_dict (dict): dictionary with settings
    xbest (array): array of scalars of optimized parameter values
    opti_param_names (list): list of parameter names which were optimized
                             (mandatory in case of per PFT/ global optimization)

    returns:
    model_op (dict): dictionary with model output
    p_names (list): list of parameter names which were optimized
    """

    # get the parameters intial and bounds, and constant values
    params = get_params(ip_df_dict)

    # get an array of unique years in a site
    unique_years_arr = np.unique(ip_df_dict["year"])

    # remove bad site years from unique_years_arr
    bad_site_yr_df = pd.read_csv("./site_info/bad_site_yr.csv", header=None)
    bad_site_yr_list = bad_site_yr_df[0].tolist()
    bad_site_yr_list.sort()

    for yrs in unique_years_arr:
        if ip_df_dict["SiteID"] + "_" + str(int(yrs)) in bad_site_yr_list:
            unique_years_arr = [x for x in unique_years_arr if int(x) != int(yrs)]

    unique_years_arr = np.array(unique_years_arr)

    # add extra parameters which will vary for each year
    param_dict, p_names_to_vary, _ = add_keys_for_group(
        settings_dict,
        params,
        unique_years_arr,
        ip_df_dict["KG"],
        ip_df_dict["SiteID"],
    )

    if settings_dict["model_name"] == "P_model":
        # run P-Model without acclimation
        model_op_no_acclim_sd = pmodel_plus(
            ip_df_dict, params, settings_dict["CO2_var"]
        )

        if settings_dict["scale_coord"]:
            scale_back_params(xbest, opti_param_names, param_dict)

        # run P-Model with acclimation
        model_op = run_p_model(
            xbest,
            opti_param_names,
            ip_df_dict,
            model_op_no_acclim_sd,
            ip_df_daily_wai,
            wai_output,
            time_info,
            settings_dict["fPAR_var"],
            settings_dict["CO2_var"],
            settings_dict["param_group_to_vary"],
            p_names_to_vary,
            unique_yrs_arr=unique_years_arr,
        )

    elif settings_dict["model_name"] == "LUE_model":
        if settings_dict["scale_coord"]:
            xbest = scale_back_params(xbest, opti_param_names, param_dict)

        # run LUE model with parameters given by optimizer
        model_op = run_lue_model(
            xbest,
            opti_param_names,
            ip_df_dict,
            ip_df_daily_wai,
            wai_output,
            time_info["nstepsday"],
            settings_dict["fPAR_var"],
            settings_dict["CO2_var"],
            settings_dict["param_group_to_vary"],
            p_names_to_vary,
            unique_yrs_arr=unique_years_arr,
        )
    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {settings_dict['model_name']}"
            "is not implemented"
        )
    return model_op, opti_param_names, xbest


def save_n_plot_model_results(ip_df_dict, model_op, settings_dict, xbest, p_names):
    """
    Prepare results of forwrad run with optimized parameters,
    model evaluation results, plot and save diagnostic figures

    parameters:
    ip_df_dict (dict): dictionary with input forcing data
    model_op (dict): dictionary with model output
    settings_dict (dict): dictionary with experiment settings
    xbest (list or dict): array of scalars of optimized parameter values (in case of all year/
                           per PFT/ global optimization); dictionary with scalars of optimized
                           parameter values (in case of site year optimization)
    p_names (list): list of parameter names which were optimized

    returns:
    save forward run results, model evaluation results, plot and save diagnostic figures
    """
    # prepare the results for saving
    result_dict = prep_results(ip_df_dict, model_op, settings_dict, xbest, p_names)

    # save the results
    serialized_result_path = Path(
        "model_results",
        settings_dict["model_name"],
        settings_dict["exp_name"],
        "serialized_model_results",
    )
    os.makedirs(serialized_result_path, exist_ok=True)
    serialized_result_path_filename = os.path.join(
        serialized_result_path, f"{ip_df_dict['SiteID']}_result.npy"
    )
    np.save(serialized_result_path_filename, result_dict)  # type: ignore

    # plot and save site level timeseries results
    plot_site_level_results(result_dict, ip_df_dict, settings_dict)

    return result_dict
