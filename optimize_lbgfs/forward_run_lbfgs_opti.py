#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
forward run the model with parameters optimized using L-BFGS-B algorithm

author: rde
first created: Thu Nov 06 2025 13:09:08 CET
"""

import os
import sys
import glob
from pathlib import Path
import json
import importlib
import numpy as np

# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

from src.common.get_data import get_data
from src.common.forward_run_model import forward_run_model
from src.postprocess.prep_results import prep_results


def forward_run_lbfgs(mod_res_path, site_idx, et_var_name="ET"):

    # directory to store jacobian and non-linear statistics results
    dir_to_store = str(mod_res_path.parent.relative_to(Path("..") / "model_results"))
    os.makedirs(os.path.join("./lbfgs_b_mod_res", dir_to_store), exist_ok=True)

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

    print(f"forward run for site: {res_dict['SiteID']}")

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

    opti_param_names = list(res_dict["Opti_par_val"].keys())
    settings_dict["scale_coord"] = False

    # get the dict with lbfgs optimized parameters
    try:
        lbfgs_opti_dict = np.load(
            f"./opti_lbfgs/{dir_to_store}/lbgfgs_b_dicts/{res_dict['SiteID']}_lbfgs_b_results.npy",
            allow_pickle=True,
        ).item()
        xbest_lbfgs_b = lbfgs_opti_dict["x"].tolist()

        # forward run model with optimized parameters
        model_op, p_names, _ = forward_run_model(
            ip_df_dict,
            ip_df_daily_wai,
            wai_output,
            time_info,
            settings_dict,
            xbest_lbfgs_b,
            opti_param_names,
        )

        # prepare the results for saving
        result_dict = prep_results(
            ip_df_dict, model_op, settings_dict, xbest_lbfgs_b, p_names
        )

        # save the results
        serialized_result_path = Path(
            "lbfgs_b_mod_res",
            dir_to_store,
            "serialized_model_results",
        )
        os.makedirs(serialized_result_path, exist_ok=True)
        serialized_result_path_filename = os.path.join(
            serialized_result_path, f"{ip_df_dict['SiteID']}_result.npy"
        )
        np.save(serialized_result_path_filename, result_dict)  # type: ignore
    except FileNotFoundError:
        print(f"LBFGS-B results not found for site: {res_dict['SiteID']}")
        pass


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

    # run as `python forward_run_lbfgs_opti.py <mod_res_path_key>`

    mod_res_path = mod_res_path_coll[sys.argv[1]]

    for get_site_idx in range(0,198):
        forward_run_lbfgs(mod_res_path, site_idx=get_site_idx)
