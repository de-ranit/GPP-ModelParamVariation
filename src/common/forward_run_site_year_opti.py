#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forward run model for each site year with optimized parameters
and save model results, plot diagnostic figures

author: rde
first created: 2023-12-18
"""

import os
from pathlib import Path
import glob
import json
import logging
import numpy as np

from src.common.forward_run_model import forward_run_model
from src.common.forward_run_model import save_n_plot_model_results
from src.postprocess.plot_param_site_year import plot_opti_param_site_year

logger = logging.getLogger(__name__)


def forward_run_site_year_opti(
    ip_df_dict, ip_df_daily_wai, wai_output, time_info, settings_dict, site_name
):
    """
    perform forward run with optimized parameters for each site year for site year optimization
    prepare and save forward run, model evaluation results
    plot and save diagnostic figures

    parameters:
    ip_df_dict (dict): dictionary with input forcing data
    ip_df_daily_wai (dict): dictionary with input data for daily WAI calculation
    wai_output (dict): dictionary to store WAI output
    time_info (dict): dictionary with time info
    settings_dict (dict): dictionary with experiment settings
    site_name (str): Site ID

    returns:
    save model evaluation results, plot and save diagnostic figures
    """

    if settings_dict["model_name"] == "P_model":
        # list of variables in P-Model output
        model_op_var = [
            "kcPa_opt",
            "koPa_opt",
            "kmPa_opt",
            "GammaStarM_opt",
            "viscosityH2oStar_opt",
            "xiPaM_opt",
            "Ca_opt",
            "ciM_opt",
            "phi0_opt",
            "vcmaxPmodelM1_opt",
            "Ac1_opt",
            "JmaxM1_opt",
            "Jp1_opt",
            "AJp1_opt",
            "GPPp_opt",
            "TA_GF_opt",
            "Iabs_opt",
            "phi0",
            "TA_OptK",
            "TA_K",
            "GammaStarM",
            "kmPa",
            "Iabs",
            "xiPa",
            "ci",
            "vcmaxOpt",
            "JmaxOpt",
            "vcmaxAdjusted",
            "JmaxAdjusted",
            "Ac1Opt",
            "J",
            "AJp1Opt",
            "GPPp_opt_fW",
        ]
    elif settings_dict["model_name"] == "LUE_model":
        # list of variables in LUE Model output
        model_op_var = [
            "gpp_lue",
            "fT",
            "fVPD",
            "fVPD_part",
            "fCO2_part",
            "fW",
            "fL",
            "fCI",
            "ci",
            "wai_results",
        ]
    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {settings_dict['model_name']}"
            "is not implemented"
        )

    # initialize dictionaries to store model outputs
    model_op = {}
    for var in model_op_var:
        model_op[var] = np.zeros(ip_df_dict["Time"].size)

    # list of variables in WAI output
    wai_var = [
        "snomelt",
        "etsub",
        "pu",
        "sno",
        "wai",
        "wai_nor",
        "et",
        "etsno",
        "etsub_pot",
        "snofall",
        "pl",
        "fW",
    ]

    # initialize dictionaries to store WAI outputs
    wai_op = {}
    for var in wai_var:
        wai_op[var] = np.zeros(ip_df_dict["Time"].size)

    # add WAI output to model output dictionary
    model_op["wai_results"] = wai_op

    # initialize dictionary to store optimized parameters
    x_best_dict = {}

    # path to optimization results
    opti_dict_path = Path(
        "opti_results",
        settings_dict["model_name"],
        settings_dict["exp_name"],
        "opti_dicts",
    )
    opti_dict_path_filename = os.path.join(
        opti_dict_path, f"{site_name}_*_opti_dict.json"
    )  # filename where optimization results are saved

    # get the list of optimization results (one file per year)
    opti_dict_filename_list = glob.glob(opti_dict_path_filename)
    opti_dict_filename_list = sorted(opti_dict_filename_list, key=str.lower)

    # if no optimization results are found, raise error
    if not opti_dict_filename_list:
        raise FileNotFoundError(
            (
                f"no optimization results found for {site_name} in {opti_dict_path}, "
                "Have you performed optimization for this site/ experiment?"
            )
        )

    # initialize p_names with None, if optimization was successful for at least one site year,
    # p_names will be updated. If optimization was not successful for any site year, p_names will
    # remain None
    p_names = None

    # loop over the optimization results for each site year
    for opti_site_years_filename in opti_dict_filename_list:
        with open(
            opti_site_years_filename, "r", encoding="utf-8"
        ) as opti_dict_json_file:  # read the optimization results
            opti_dict = json.load(opti_dict_json_file)

        xbest = opti_dict["xbest"]  # get the optimized parameters for the site year

        if (xbest is None) or (
            np.isnan(np.array(xbest)).any()
        ):  # if the optimization was not successful, skip the site year
            logger.warning(
                "%s (%s): optimization was not successful (skipping forward run)",
                site_name,
                opti_dict["site_year"],
            )

            # set model output to NaN for the site year
            site_year_mask = ip_df_dict["year"] == opti_dict["site_year"]

            for k, v in model_op.items():
                if k == "wai_results":
                    for k1 in v:
                        v[k1][site_year_mask] = np.nan
                else:
                    v[site_year_mask] = np.nan

            # set the optimized parameters to NaN for the site year
            x_best_dict[opti_dict["site_year"]] = [np.nan]
        else:
            # if the optimization was successful, run the model with optimized parameters
            model_op_site_yr, p_names, xbest_actual = forward_run_model(
                ip_df_dict,
                ip_df_daily_wai,
                wai_output,
                time_info,
                settings_dict,
                xbest,
            )

            # store the model outputs for the site year
            site_year_mask = ip_df_dict["year"] == opti_dict["site_year"]

            for k, v in model_op.items():
                if k == "wai_results":
                    for k1 in v:
                        v[k1][site_year_mask] = model_op_site_yr[k][k1][site_year_mask]
                else:
                    v[site_year_mask] = model_op_site_yr[k][site_year_mask]

            # store the optimized parameters for the site year
            x_best_dict[opti_dict["site_year"]] = xbest_actual

    # if optimization was successful for at least one site year
    if p_names is not None:
        # save and plot model results
        result_dict = save_n_plot_model_results(
            ip_df_dict, model_op, settings_dict, x_best_dict, p_names
        )
        # plot optimized parameters for each site year
        plot_opti_param_site_year(result_dict, settings_dict)
    else:
        pass
