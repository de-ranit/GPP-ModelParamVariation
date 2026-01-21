#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
If model calibration was stopped after 30 days. Choose the best parameters
for which the cost was the lowest during the CMA-ES run so far
and save them as calibrated parameters in an optimization dictionary.

author: rde
first created: Mon Dec 08 2025 14:58:13 CET
"""
from pathlib import Path
import sys
import json
import logging
import pandas as pd
import numpy as np
import cma

from src.common.get_data import get_data
from src.common.get_params import get_params
from src.common.opti_per_site_or_site_year import add_keys_for_group


def create_opti_dict(outcmaes_dir, site_list):

    parts = Path(outcmaes_dir).parts
    exp_name = "/".join(parts[1:])

    settings_dict_path = f"./opti_results/{exp_name}/settings_dict.json"

    with open(settings_dict_path, "r") as f:
        settings_dict = json.load(f)

    logfile = f"./opti_results/{exp_name}/model_optimize.log"

    logging.basicConfig(
        filename=logfile,
        filemode="a",
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    pylogger = logging.getLogger(__name__)

    for site in site_list:

        try:
            opti_dict_path = (
                f"./opti_results/{exp_name}/opti_dicts/{site}_opti_dict.json"
            )
            with open(opti_dict_path, "r", encoding="utf-8") as opti_dict_json_file:
                _ = json.load(opti_dict_json_file)

        except FileNotFoundError:

            print(f"Processing site: {site}, experiment: {exp_name}")

            xbest_file = f"{outcmaes_dir}/{site}_"

            logger = cma.CMADataLogger(xbest_file).load()

            cost_vals = logger.data["xrecent"][:, 4]
            best_cost_idx = cost_vals.argmin()

            xbest = logger.data["xrecent"][best_cost_idx][5:]
            fbest = cost_vals[best_cost_idx]
            evals_best = int(logger.data["xrecent"][best_cost_idx][0])
            evaluations = int(logger.data["xrecent"][-1][0]) * int(
                logger.data["xrecent"][0][1]
            )
            stop = {
                "custom": "job cancelled by user (xbest are parameters with best cost found so far)"
            }

            ip_df_dict, _, _, _ = get_data(site, settings_dict)
            params = get_params(ip_df_dict)

            # get an array of unique years in a site
            unique_years_arr = np.unique(ip_df_dict["year"])

            # remove bad site years from unique_years_arr
            bad_site_yr_df = pd.read_csv("./site_info/bad_site_yr.csv", header=None)
            bad_site_yr_list = bad_site_yr_df[0].tolist()
            bad_site_yr_list.sort()

            for yrs in unique_years_arr:
                if ip_df_dict["SiteID"] + "_" + str(int(yrs)) in bad_site_yr_list:
                    unique_years_arr = [
                        x for x in unique_years_arr if int(x) != int(yrs)
                    ]

            if len(unique_years_arr) == 0:
                logger.info(
                    "%s, All site years are marked as bad site years. Aborting optimization.",
                    ip_df_dict["SiteID"],
                )
                sys.exit(0)

            unique_years_arr = np.array(unique_years_arr)

            param_dict, p_names_to_vary, yr_specific_p_names = add_keys_for_group(
                settings_dict,
                params,
                unique_years_arr,
                ip_df_dict["KG"],
                ip_df_dict["SiteID"],
            )

            if settings_dict["model_name"] == "P_model":
                # list of parameters to be optimized in all cases
                p_names = [
                    "acclim_window",
                    "W_I",
                    "K_W",
                    "AWC",
                    "theta",
                    "alphaPT",
                    "meltRate_temp",
                    "meltRate_netrad",
                    "sn_a",
                ]
                # also optimize alpha in fW_Horn for arid sites
                if ip_df_dict["KG"][0] == "B":
                    # insert alpha at 4th position;
                    # to remain compatible with old optimization results
                    p_names.insert(3, "alpha")
                    # p_names.append("alpha")
                else:
                    pass

                # remove the parameters from p_names which will vary per year
                p_names = [item for item in p_names if item not in p_names_to_vary]
                p_names.extend(
                    yr_specific_p_names
                )  # extend the list with new param names (param_yr)

            elif settings_dict["model_name"] == "LUE_model":
                # list of parameters to be optimized in all cases
                p_names = [
                    "LUE_max",
                    "T_opt",
                    "K_T",
                    "Kappa_VPD",
                    "Ca_0",
                    "C_Kappa",
                    "c_m",
                    "gamma_fL_TAL",
                    "mu_fCI",
                    "W_I",
                    "K_W",
                    "AWC",
                    "theta",
                    "alphaPT",
                    "meltRate_temp",
                    "meltRate_netrad",
                    "sn_a",
                ]
                # also optimize alpha in fW_Horn for arid sites
                if ip_df_dict["KG"][0] == "B":
                    p_names.append("alpha")
                # also optimize alpha_fT_Horn in fT_Horn for temperate,
                # continental and polar sites
                elif ip_df_dict["KG"][0] in ["C", "D", "E"]:
                    p_names.append("alpha_fT_Horn")
                else:
                    pass

                # remove the parameters from p_names which will vary per year
                p_names = [item for item in p_names if item not in p_names_to_vary]
                p_names.extend(
                    yr_specific_p_names
                )  # extend the list with new param names (param_yr)

            op_opti = {}
            op_opti["site_year"] = None
            op_opti["xbest"] = xbest.tolist()
            op_opti["fbest"] = float(fbest)
            op_opti["iter_best"] = evals_best
            op_opti["evaluations"] = evaluations
            op_opti["stop"] = stop
            op_opti["opti_param_names"] = p_names

            # opti_dict_path = f"./opti_results/{exp_name}_copy_unfinished/opti_dicts/{site}_opti_dict.json"
            opti_dict_path = (
                f"./opti_results/{exp_name}/opti_dicts/{site}_opti_dict.json"
            )

            with open(
                opti_dict_path, "w", encoding="utf-8"
            ) as opti_dict_json_file:  # save the optimization results
                json.dump(
                    op_opti, opti_dict_json_file, indent=4, separators=(", ", ": ")
                )

            pylogger.info(
                "%s: Optimization dict created for site from incomplete CMA-ES run",
                site,
            )


if __name__ == "__main__":

    site_info = pd.read_csv("./site_info/SiteInfo_BRKsite_list.csv")
    site_list = site_info["SiteID"].tolist()
    site_list.remove("US-LWW")

    outcmaes_dir = [
        "outcmaes/LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group7_cmaes_default_pop",
        "outcmaes/LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_vary_group8_cmaes_default_pop",
        "outcmaes/P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group3_cmaes_default_pop",
        "outcmaes/P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_vary_group4_cmaes_default_pop",
    ]

    # for paths in outcmaes_dir:
    create_opti_dict(outcmaes_dir[3], site_list)
