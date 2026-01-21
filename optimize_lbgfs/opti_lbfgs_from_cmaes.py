#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
further optimize the calibrated parameters obtained from
CMA-ES optimization using L-BFGS-B algorithm

author: rde
first created: Thu Oct 30 2025 16:59:44 CET
"""
# disable possible multithreading from the
# OPENBLAS and MKL linear algebra backends
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import glob
from pathlib import Path
import json
import logging
import logging.config
import importlib
from functools import partial
import numpy as np
from optimparallel import minimize_parallel
from scipy.optimize import Bounds

# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

from src.common.get_data import get_data
from src.p_model.pmodel_plus import pmodel_plus
from src.common.get_params import get_params
from src.common.opti_per_site_or_site_year import add_keys_for_group
from src.common.forward_run_model import scale_back_params
from src.p_model.p_model_cost_function import p_model_cost_function
from src.lue_model.lue_model_cost_function import lue_model_cost_function


def opti_lbfgs(mod_res_path, site_idx, et_var_name="ET"):

    # directory to store jacobian and non-linear statistics results
    dir_to_store = str(mod_res_path.parent.relative_to(Path("..") / "model_results"))
    os.makedirs(os.path.join("./opti_lbfgs", dir_to_store), exist_ok=True)

    # configure the logger: to log various information to a file
    logging.basicConfig(
        filename=(f"./opti_lbfgs/{dir_to_store}/" "opti_lbfgs.log"),
        filemode="a",
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # get the list of result files
    mod_res_file_list = glob.glob(f"{mod_res_path}/*.npy")
    mod_res_file_list.sort()  # sort the files by site ID

    # filter out bad sites
    filtered_mod_res_file_list = [
        files
        for files in mod_res_file_list
        if not (
            "CG-Tch" in files
            or "MY-PSO" in files
            or "GH-Ank" in files
            or "US-LWW" in files
        )
    ]

    # get name of the result file for the site
    res_file = filtered_mod_res_file_list[site_idx]

    # open the result dict
    res_dict = np.load(res_file, allow_pickle=True).item()

    print(f"optimizing for site: {res_dict['SiteID']}")

    # get experiment name to open the settings dict
    exp_name = str(
        Path(res_file).parent.parent.relative_to(Path("..") / "model_results")
    )
    settings_dict_file_path = f"../opti_results/{exp_name}/settings_dict.json"
    
    with open(settings_dict_file_path, "r") as fh:
        settings_dict = json.load(fh)  # load settings dict

    # set the et variable name if not existing in settings dict
    if "et_var_name" not in settings_dict.keys():
        settings_dict["et_var_name"] = et_var_name

    # get forcing data for the site
    ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
        res_dict["SiteID"], settings_dict
    )

    # get the parameters
    # parameter bounds are needed to calculate scalar of bounds for optimization
    params = get_params(ip_df_dict)

    # get an array of unique years in a site
    unique_years_arr = np.unique(ip_df_dict["year"])

    # add extra parameters which will vary for each year, while other params remain
    # constant for all years
    param_dict, p_names_to_vary, _ = add_keys_for_group(
        settings_dict,
        params,
        unique_years_arr,
        ip_df_dict["KG"],
        ip_df_dict["SiteID"],
    )

    # get scalar of calibrated parameter values
    opti_dict_file_path = Path(
        "..",
        "opti_results",
        settings_dict["model_name"],
        settings_dict["exp_name"],
        "opti_dicts",
        f"{ip_df_dict['SiteID']}_opti_dict.json",
    )
    with open(opti_dict_file_path, "r", encoding="utf-8") as file:
        xbest_dict = json.load(file)

    xbest = xbest_dict["xbest"]
    p_names = list(res_dict["Opti_par_val"].keys())
    xbest_actual = scale_back_params(xbest, p_names, params)
    settings_dict["scale_coord"] = False

    # get the scaled value of parameter bounds (ub/initial or lb/initial)
    p_ubound_scaled = []
    p_lbound_scaled = []
    for p in p_names:
        p_ubound_scaled.append(param_dict[p]["ub"] / param_dict[p]["ini"])
        p_lbound_scaled.append(param_dict[p]["lb"] / param_dict[p]["ini"])

    ##############
    # Define costhand and parameters in case of P Model
    if settings_dict["model_name"] == "P_model":
        # run pmodelPlus without acclimation
        model_op_no_acclim_sd = pmodel_plus(
            ip_df_dict, params, settings_dict["CO2_var"]
        )

        costhand = partial(
            p_model_cost_function,
            p_names=p_names,
            ip_df_dict=ip_df_dict,
            model_op_no_acclim_sd=model_op_no_acclim_sd,
            ip_df_daily_wai=ip_df_daily_wai,
            wai_output=wai_output,
            time_info=time_info,
            fpar_var_name=settings_dict["fPAR_var"],
            co2_var_name=settings_dict["CO2_var"],
            data_filtering=settings_dict["data_filtering"],
            cost_func=settings_dict["cost_func"],
            consider_yearly_cost=settings_dict["cost_iav"],
            param_group_to_vary=settings_dict["param_group_to_vary"],
            param_names_to_vary=p_names_to_vary,
            et_var_name=settings_dict["et_var_name"],
        )

    # Define costhand and parameters in case of LUE Model
    elif settings_dict["model_name"] == "LUE_model":
        # generate synthetic data to calculate 3rd and 4th
        # component of LUE model cost function
        synthetic_data = {
            "TA": np.linspace(-5.0, 40.0, time_info["nstepsday"] * 365),  # deg C
            "VPD": np.linspace(4500, 0.0, time_info["nstepsday"] * 365),  # Pa
            "CO2": np.linspace(400.0, 400.0, time_info["nstepsday"] * 365),  # PPM
            "wai_nor": np.linspace(
                0.0, 1.0, time_info["nstepsday"] * 365
            ),  # - (or mm/mm)
            "fPAR": np.linspace(0.0, 1.0, time_info["nstepsday"] * 365),  # -
            "PPFD": np.linspace(
                0.0, 600.0, time_info["nstepsday"] * 365
            ),  # umol photons m-2s-1
        }

        costhand = partial(
            lue_model_cost_function,
            param_names=p_names,
            ip_df_dict=ip_df_dict,
            ip_df_daily_wai=ip_df_daily_wai,
            wai_output=wai_output,
            nstepsday=time_info["nstepsday"],
            fpar_var_name=settings_dict["fPAR_var"],
            co2_var_name=settings_dict["CO2_var"],
            data_filtering=settings_dict["data_filtering"],
            cost_func=settings_dict["cost_func"],
            synthetic_data=synthetic_data,
            consider_yearly_cost=settings_dict["cost_iav"],
            param_group_to_vary=settings_dict["param_group_to_vary"],
            param_names_to_vary=p_names_to_vary,
            et_var_name=settings_dict["et_var_name"],
        )

    ini_costval = costhand(xbest_actual)

    bounds = Bounds(np.array(p_lbound_scaled), np.array(p_ubound_scaled))
    
    lbfgs_b_op = minimize_parallel(
        fun=costhand,
        x0=xbest_actual,
        bounds=bounds,
        options={"maxfun": 20000, "ftol": 1e-8},
        parallel={
            "loginfo": True,
            "time": True,
            "verbose": True,
            "max_workers": int(sys.argv[2]),
        },
    )

    percentage_reduction_in_cost = (
        (ini_costval - lbfgs_b_op.fun) / ini_costval * 100.0
    )

    lbfgs_b_op_dict = {
        "initial_cost": ini_costval,
        "fun": lbfgs_b_op.fun,
        "percentage_reduction_in_cost": percentage_reduction_in_cost,
        "jac": lbfgs_b_op.jac,
        "nfev": lbfgs_b_op.nfev,
        "njev": lbfgs_b_op.njev,
        "nit": lbfgs_b_op.nit,
        "status": lbfgs_b_op.status,
        "message": lbfgs_b_op.message,
        "x": lbfgs_b_op.x,
        "success": lbfgs_b_op.success,
        "loginfo": lbfgs_b_op.loginfo,
        "time": lbfgs_b_op.time,
    }

    logger.info(
        (
            "%s:"
            "Starting cost value: %.4f, Final cost value: %.4f, Percentage cost reduction: %4f, nfev: %d, njev: %d, nit: %d, status: %d, "
            "message: %s, success: %s, time elapsed: %.2f seconds" 
        ),
        ip_df_dict["SiteID"],
        ini_costval,
        lbfgs_b_op.fun,
        percentage_reduction_in_cost,
        lbfgs_b_op.nfev,
        lbfgs_b_op.njev,
        lbfgs_b_op.nit,
        lbfgs_b_op.status,
        lbfgs_b_op.message,
        str(lbfgs_b_op.success),
        lbfgs_b_op.time["elapsed"],
    )

    # save the jacobian and non-linear statistics results
    dir_to_save_results = os.path.join(f"./opti_lbfgs/{dir_to_store}", "lbgfgs_b_dicts")
    os.makedirs(dir_to_save_results, exist_ok=True)
    np.save(
        f"{dir_to_save_results}/{ip_df_dict['SiteID']}_lbfgs_b_results.npy",
        lbfgs_b_op_dict,
    )


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")

    mod_res_path_coll = {
        "g01_vary_lue_model_res_path": result_paths.g01_vary_lue_model_res_path,
        "g02_vary_lue_model_res_path": result_paths.g02_vary_lue_model_res_path,
        "g03_vary_lue_model_res_path": result_paths.g03_vary_lue_model_res_path,
        "g04_vary_lue_model_res_path": result_paths.g04_vary_lue_model_res_path,
        "g05_vary_lue_model_res_path": result_paths.g05_vary_lue_model_res_path,
        "g06_vary_lue_model_res_path": result_paths.g06_vary_lue_model_res_path,
        "g07_vary_lue_model_res_path": result_paths.g07_vary_lue_model_res_path,
        "g08_vary_lue_model_res_path": result_paths.g08_vary_lue_model_res_path,
        "g01_vary_p_model_res_path": result_paths.g01_vary_p_model_res_path,
        "g02_vary_p_model_res_path": result_paths.g02_vary_p_model_res_path,
        "g03_vary_p_model_res_path": result_paths.g03_vary_p_model_res_path,
        "g04_vary_p_model_res_path": result_paths.g04_vary_p_model_res_path,  
    }

    # run as `python opti_lbfgs_from_cmaes.py <site_index> <num_workers> <mod_res_path_key>`

    mod_res_path = mod_res_path_coll[sys.argv[3]]

    get_site_idx = int(sys.argv[1]) - 1
    opti_lbfgs(mod_res_path, site_idx=get_site_idx)
