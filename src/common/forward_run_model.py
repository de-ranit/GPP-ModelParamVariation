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

    # add extra parameters which will vary for each year
    param_dict, p_names_to_vary, _ = add_keys_for_group(
        settings_dict["param_group_to_vary"],
        params,
        unique_years_arr,
        settings_dict["model_name"],
        ip_df_dict["KG"],
    )

    if settings_dict["model_name"] == "P_model":
        # run P-Model without acclimation
        model_op_no_acclim_sd = pmodel_plus(
            ip_df_dict, params, settings_dict["CO2_var"]
        )

        # # list of parameters which were optimized in all cases
        # p_names = [
        #     "acclim_window",
        #     "W_I",
        #     "K_W",
        #     "AWC",
        #     "theta",
        #     "alphaPT",
        #     "meltRate_temp",
        #     "meltRate_netrad",
        #     "sn_a",
        # ]
        # # alpha was also optimized in fW_Horn for arid sites
        # if ip_df_dict["KG"][0] == "B":
        #     # insert alpha at 4th position;
        #     # to work with old optimization results
        #     p_names.insert(3, "alpha")
        #     # p_names.append("alpha")
        # else:
        #     pass

        # # in case of global, PFT based optimization, xbest can have optimized parameter value for
        # # alpha, but we may not need it for specific sites (where KG doesn't start with B)
        # if (settings_dict["opti_type"] == "global_opti") or (
        #     settings_dict["opti_type"] == "per_pft"
        # ):
        #     if opti_param_names is not None:
        #         keep_xbest_idx = [
        #             opti_param_names.index(p) for p in p_names if p in opti_param_names
        #         ]
        #         xbest = [xbest[i] for i in keep_xbest_idx]  # only keep the xbest values
        #         # which are in p_names for a specific site
        #     else:
        #         raise ValueError(
        #             (
        #                 "`opti_param_names` should be provided when calling `forward_run_model()`"
        #                 "in case of running model, using optimized parameters"
        #                 "from global or per PFT optimization"
        #             )
        #         )

        # # if parameters scalars were further scaled between 0 and 1,
        # # convert them back to actual scalar values
        # if settings_dict["scale_coord"]:
        #     xbest = scale_back_params(xbest, p_names, params)

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
        )

    elif settings_dict["model_name"] == "LUE_model":
        # # list of parameters which were optimized in all cases
        # p_names = [
        #     "LUE_max",
        #     "T_opt",
        #     "K_T",
        #     "Kappa_VPD",
        #     "Ca_0",
        #     "C_Kappa",
        #     "c_m",
        #     "gamma_fL_TAL",
        #     "mu_fCI",
        #     "W_I",
        #     "K_W",
        #     "AWC",
        #     "theta",
        #     "alphaPT",
        #     "meltRate_temp",
        #     "meltRate_netrad",
        #     "sn_a",
        # ]
        # # alpha was also optimized in fW_Horn for arid sites
        # if ip_df_dict["KG"][0] == "B":
        #     if settings_dict["opti_type"] == "global_opti":
        #         # fix to maintain paramter order in case of global optimization
        #         p_names.insert(11, "alpha")
        #     else:
        #         p_names.append("alpha")
        # # alpha_fT_Horn was also optimized in fT_Horn for temperate,
        # # continental and polar sites
        # elif ip_df_dict["KG"][0] in ["C", "D", "E"]:
        #     if settings_dict["opti_type"] == "global_opti":
        #         # fix to maintain paramter order in case of global optimization
        #         p_names.insert(3, "alpha_fT_Horn")
        #     else:
        #         p_names.append("alpha_fT_Horn")
        # else:
        #     pass

        # in case of global, PFT based optimization, xbest can have optimized parameter value for
        # alpha and/or alpha_fT_Horn, but we may not need it for a specific site
        # if (settings_dict["opti_type"] == "global_opti") or (
        #     settings_dict["opti_type"] == "per_pft"
        # ):
        #     if opti_param_names is not None:
        #         keep_xbest_idx = [
        #             opti_param_names.index(p) for p in p_names if p in opti_param_names
        #         ]
        #         xbest = [xbest[i] for i in keep_xbest_idx]  # only keep the xbest values
        #         # which are in p_names for a specific site
        #     else:
        #         raise ValueError(
        #             (
        #                 "`opti_param_names` should be provided when calling `forward_run_model()`"
        #                 "in case of global or per PFT optimization"
        #             )
        #         )

        # if parameters scalars were further scaled between 0 and 1,
        # convert them back to actual scalar values
        # if settings_dict["scale_coord"]:
        #     xbest = scale_back_params(xbest, p_names, params)
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

    # load the results
    # result_dict_load = np.load(serialized_result_path_filename, allow_pickle=True).item()

    # plot and save site level timeseries results
    plot_site_level_results(result_dict, ip_df_dict, settings_dict)

    return result_dict
