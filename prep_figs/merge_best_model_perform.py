#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
select best results based on lowest cost value among different optimization experiments:
1. CMAES with big population size
2. CMAES with big population size + L-BFGS-B
3. CMAES with default population size

author: rde
first created: Sun Nov 30 2025 15:14:58 CET
"""
import sys
import os
import glob
import json
import shutil
import numpy as np
import pandas as pd
import logging


def collect_best_results(exp_names_dict, new_exp_name):

    site_info_df = pd.read_csv("../site_info/SiteInfo_BRKsite_list.csv")
    site_list = site_info_df["SiteID"].tolist()

    for site_name in site_list:
        cmaes_big_pop_opti_dict_path = f"../opti_results/{exp_names_dict['cmaes_big_pop']}/opti_dicts/{site_name}_opti_dict.json"

        try:
            with open(cmaes_big_pop_opti_dict_path, "r") as f:
                cmaes_big_pop_opti_dict = json.load(f)
        except FileNotFoundError:
            cmaes_big_pop_opti_dict = {"fbest": np.nan}

        cmaes_big_pop_and_lbfgs_b_path = f"../optimize_lbgfs/opti_lbfgs/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/lbgfgs_b_dicts/{site_name}_lbfgs_b_results.npy"
        try:
            cmaes_big_pop_and_lbfgs_b_opti_dict = np.load(
                cmaes_big_pop_and_lbfgs_b_path, allow_pickle=True
            ).item()
        except FileNotFoundError:
            cmaes_big_pop_and_lbfgs_b_opti_dict = {"fun": np.nan}

        cmaes_default_pop_opti_dict_path = f"../opti_results/{exp_names_dict['cmaes_default_pop']}/opti_dicts/{site_name}_opti_dict.json"
        try:
            with open(cmaes_default_pop_opti_dict_path, "r") as f:
                cmaes_default_pop_opti_dict = json.load(f)
        except FileNotFoundError:
            cmaes_default_pop_opti_dict = {"fbest": np.nan}

        cost_arr = np.array(
            [
                cmaes_big_pop_opti_dict["fbest"],
                cmaes_big_pop_and_lbfgs_b_opti_dict["fun"],
                cmaes_default_pop_opti_dict["fbest"],
            ]
        )

        save_path = f"../model_results/{new_exp_name}/serialized_model_results/"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        logging.basicConfig(
            filename=(f"../model_results/{new_exp_name}/selected_exp.log"),
            filemode="a",
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d,%H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        if any(isinstance(x, float) and np.isnan(x) for x in cost_arr.flat):
            logger.info(f"{site_name} No optimization result found.")
        else:
            idx = int(np.argmin(cost_arr))

            exp_name_dict = {
                0: "cmaes_big_pop",
                1: "cmaes_big_pop_and_lbfgs_b",
                2: "cmaes_default_pop",
            }
            print(
                f"Best opti for site {site_name} is idx {exp_name_dict[idx]} with cost {cost_arr[idx]}"
            )

            logger.info(
                f"Best opti for site {site_name} is idx {exp_name_dict[idx]} with cost {cost_arr[idx]}"
            )

            if idx == 0:
                res_dict_path = f"../model_results/{exp_names_dict['cmaes_big_pop']}/serialized_model_results/{site_name}_result.npy"
                shutil.copy2(res_dict_path, save_path)
            elif idx == 1:
                res_dict_path = f"../optimize_lbgfs/lbfgs_b_mod_res/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/serialized_model_results/{site_name}_result.npy"
                shutil.copy2(res_dict_path, save_path)
            elif idx == 2:
                res_dict_path = f"../model_results/{exp_names_dict['cmaes_default_pop']}/serialized_model_results/{site_name}_result.npy"
                shutil.copy2(res_dict_path, save_path)

if __name__ == "__main__":
    lue_g01 = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group1_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group1_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group1_cmaes_default_pop",
    }

    lue_g02 = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group2_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group2_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group2_cmaes_default_pop",
    }

    lue_g03 = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group3_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group3_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group3_cmaes_default_pop",
    }

    lue_g04 = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group4_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group4_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group4_cmaes_default_pop",
    }

    lue_g05 = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group5_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group5_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group5_cmaes_default_pop",
    }

    lue_g06 = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group6_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group6_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group6_cmaes_default_pop",
    }

    lue_g07 = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group7_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group7_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group7_cmaes_default_pop",
    }

    lue_g08 = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group8_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group8_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group8_cmaes_default_pop",
    }

    p_g01 = {
        "cmaes_big_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group1_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group1_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group1_correct_cmaes_default_pop",
    }

    p_g02 = {
        "cmaes_big_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group2_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group2_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group2_correct_cmaes_default_pop",
    }

    p_g03 = {
        "cmaes_big_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group3_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group3_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group3_correct_cmaes_default_pop",
    }

    p_g04 = {
        "cmaes_big_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group4_cmaes_big_pop",
        "cmaes_big_pop_and_lbfgs_b": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group4_cmaes_big_pop_and_lbfgs_b",
        "cmaes_default_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group4_correct_cmaes_default_pop",
    }

    collect_best_results(
        lue_g01,
        new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group1_best_opti_ever",
    )
    # collect_best_results(
    #     lue_g02,
    #     new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group2_best_opti_ever",
    # )
    # collect_best_results(
    #     lue_g03,
    #     new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group3_best_opti_ever",
    # )
    # collect_best_results(
    #     lue_g04,
    #     new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group4_best_opti_ever",
    # )
    # collect_best_results(
    #     lue_g05,
    #     new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group5_best_opti_ever",
    # )
    # collect_best_results(
    #     lue_g06,
    #     new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group6_best_opti_ever",
    # )
    # collect_best_results(
    #     lue_g07,
    #     new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group7_best_opti_ever",
    # )
    # collect_best_results(
    #     lue_g08,
    #     new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group8_best_opti_ever",
    # )

    # collect_best_results(
    #     p_g01,
    #     new_exp_name="P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group1_best_opti_ever",
    # )
    # collect_best_results(
    #     p_g02,
    #     new_exp_name="P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group2_best_opti_ever",
    # )
    # collect_best_results(
    #     p_g03,
    #     new_exp_name="P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group3_best_opti_ever",
    # )
    # collect_best_results(
    #     p_g04,
    #     new_exp_name="P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group4_best_opti_ever",
    # )
