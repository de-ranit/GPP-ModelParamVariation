#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is main script to optimize model for each site
or forward run model using the optimized parameters

author: rde
first created: 2023-11-07
"""

import json
from pathlib import Path
import sys
import logging
import logging.config
import glob
from multiprocessing.pool import Pool
from functools import partial
import os
import numpy as np
import pandas as pd

ROOT_PATH = Path(__file__).parent  # Get path of the current script
sys.path.append(str(ROOT_PATH / "src"))  # type: ignore # Add path for other scipts

# don't format the following lines
# fmt: off
from src.common.get_data import get_data  # pylint: disable=C0413
from src.common.opti_per_site_or_site_year import optimize_model  # pylint: disable=C0413
from src.common.opti_per_site_or_site_year import get_cmaes_options  # pylint: disable=C0413
from src.common.forward_run_model import forward_run_model  # pylint: disable=C0413
from src.common.forward_run_model import save_n_plot_model_results # pylint: disable=C0413
from src.postprocess.plot_exp_result import plot_exp_result # pylint: disable=C0413
# fmt: on


def calc_idx_for_jobarray(job_array_no, ini_idx, n_sites, max_n):
    """
    calculate the start and end index of site list for each job array

    Parameters:
    job_array_no (int): job array index == $SLURM_ARRAY_TASK_ID
    ini_idx (int): initial index of site list (filename_list)
    n_sites (int): no. of sites per job
    max_n (int): total no. of sites

    Returns:
    start_site_idx (int): start index of site list for each job array
    end_site_idx (int): end index of site list for each job array
    """

    if job_array_no == 1:  # for first job array
        start_site_idx = ini_idx
        end_site_idx = ini_idx + n_sites
    else:  # for other job arrays
        start_site_idx = ((job_array_no - 1) * n_sites) + ini_idx
        end_site_idx = (job_array_no * n_sites) + ini_idx
        if end_site_idx > max_n:
            end_site_idx = max_n
        else:
            pass
    return start_site_idx, end_site_idx


def filter_logs(ip_log_filename, clean_log_filename, msg_start, msg_end=None):
    """
    multiple log lines are written for each site (from each iteration of the otimizer),
    this function removes the duplicate lines and keeps only the last log per site

    Parameters:
    ip_log_filename (str): input log filename
    clean_log_filename (str): output log filename
    msg_start (int): index at which the actual log message starts (after timestamp and log level)
    msg_end (int): index at which the actual log message ends

    Returns:
    Create and save a .log file with unique log lines
    """
    seen_logs = set()  # create an empty set to store the unique log lines

    # read the original logfile
    with open(ip_log_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # write the unique log lines to a new file
    with open(clean_log_filename, "w", encoding="utf-8") as f:
        for line in reversed(lines):
            subset_line = line[
                msg_start:msg_end
            ]  # get only the log message (excluding the timestamp and log level)
            if (
                subset_line not in seen_logs
            ):  # if the log message is not already in the set,
                # write it to the new file and add it to the set
                seen_logs.add(subset_line)
                f.write(line)


def main_script_optimize(idx, site_list, settings_dict):
    """
    main script to optimize p-model for each site

    Parameters:
    idx (int): index of the site in site list (filename_list)
               for which optimizaton is being performed
    site_list (list): list of site ids
    settings_dict (dict): dictionary with model settings

    Returns:
    Create and save a .json file with optimization
    results (optimized model parameters, final cost function value, etc.)
    """

    # configure the logger: to log various information to a file
    logging.basicConfig(
        filename=(
            f"./opti_results/{settings_dict['model_name']}/"
            f"{settings_dict['exp_name']}/model_optimize.log"
        ),
        filemode="a",
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # get the site id
    if settings_dict["opti_type"] == "all_year":
        site_name = site_list[idx]
    elif settings_dict["opti_type"] == "site_year":
        site_name_year = site_list[idx]
        site_name = site_name_year.split("_")[0]
    else:
        raise ValueError("opti_type must be one of: all_year, site_year")

    # get the forcing data and other information for the site to optimize the Model
    ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
        site_name, settings_dict
    )

    # optimize model for each site
    ###########################################################################################
    # In case job was failed/ cancelled in middle, then don't optimize all the sites
    # during restart. Only optimize the remaining sites/ incomplete sites
    ###########################################################################################
    # check if the optimization results already exists for a certain site
    opti_dict_path = Path(
        "opti_results",
        settings_dict["model_name"],
        settings_dict["exp_name"],
        "opti_dicts",
    )  # Path where optimization results were saved
    opti_dict_path_filename = os.path.join(
        opti_dict_path, f"{ip_df_dict['SiteID']}_opti_dict.json"
    )
    if os.path.exists(opti_dict_path_filename):  # if optimization results already exist
        logger.info(
            "%s : Optimization already done",
            ip_df_dict["SiteID"],
        )
    else:
        # check if the CMAES output file exists for a certain site,
        # but optimization was incomplete (as optimization results file does not exist)
        outcmaes_path = Path(
            "outcmaes", settings_dict["model_name"], settings_dict["exp_name"]
        )
        outcmaes_filename = glob.glob(
            os.path.join(outcmaes_path, f"{ip_df_dict['SiteID']}_*.dat")
        )

        # remove incomplete CMAES output files if they exist
        if len(outcmaes_filename) > 0:
            for filename in outcmaes_filename:
                logger.info(
                    "%s : Deleting incomplete CMAES output file: %s",
                    ip_df_dict["SiteID"],
                    filename,
                )
                os.remove(filename)
        ##################################################################

        # perform site optimization
        optimize_model(
            ip_df_dict, ip_df_daily_wai, wai_output, time_info, settings_dict
        )


def main_script_forward(idx, site_list, settings_dict):
    """
    main script to forward run model for each site
    using the optimized parameters and plot GPP and ET time series per site

    Parameters:
    idx (int): index of the site in site list (filename_list)
               for which forward run is being performed
    site_list (list): list of site ids
    settings_dict (dict): dictionary with model settings

    Returns:
    Create and save a .npy file with forward run results (GPP and ET time series)
    Create and save a .png file with GPP and ET time series
    """

    # configure the logger: to log various information to a file
    os.makedirs(
        Path("model_results", settings_dict["model_name"], settings_dict["exp_name"]),
        exist_ok=True,
    )
    logging.basicConfig(
        filename=(
            f"./model_results/{settings_dict['model_name']}/"
            f"{settings_dict['exp_name']}/Model_forward_run.log"
        ),
        filemode="a",
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # get the site id
    site_name = site_list[idx]

    opti_dict_path = Path(
        "opti_results",
        settings_dict["model_name"],
        settings_dict["exp_name"],
        "opti_dicts",
    )
    opti_dict_path_filename = os.path.join(
        opti_dict_path, f"{site_name}_opti_dict.json"
    )  # filename where optimization results are saved
    try:
        with open(
            opti_dict_path_filename, "r", encoding="utf-8"
        ) as opti_dict_json_file:  # read the optimization results
            opti_dict = json.load(opti_dict_json_file)

        xbest = opti_dict["xbest"]  # get the optimized parameters for the site

        if (xbest is None) or (
            np.isnan(np.array(xbest)).any()
        ):  # if the optimization was not successful, skip the site
            logger.warning(
                "%s : optimization was not successful (skipping forward run)",
                site_name,
            )
        else:
            opti_param_names = opti_dict["opti_param_names"]

            # get the forcing data and other information for the site to forward run p-model
            ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
                site_name, settings_dict
            )

            # forward run model with optimized parameters
            model_op, p_names, xbest_actual = forward_run_model(
                ip_df_dict,
                ip_df_daily_wai,
                wai_output,
                time_info,
                settings_dict,
                xbest,
                opti_param_names,
            )

            # save and plot model results
            save_n_plot_model_results(
                ip_df_dict, model_op, settings_dict, xbest_actual, p_names
            )

    # if the optimization results file does not exist, skip the site
    except FileNotFoundError:
        logger.warning(
            "%s : optimization was not successful (skipping forward run)", site_name
        )


################################################################################
def main():
    """
    Mian script for model parameter optimization and forward runs
    """
    # read the model settings from json file
    # with open(ROOT_PATH / "p_model_settings.json", "r", encoding="utf-8") as json_file:
    #     model_settings = json.load(json_file)

    # read the model settings from excel file
    model_settings_df = pd.read_excel("model_settings.xlsx", header=None)
    # generate a dictionary with model settings
    model_settings = pd.Series(
        model_settings_df[1].values, index=model_settings_df[0]
    ).to_dict()

    # define experiment name
    model_settings["exp_name"] = (
        f"{model_settings['opti_type']}_"
        f"{model_settings['data_source']}_"
        f"{model_settings['fPAR_var']}_"
        f"{model_settings['CO2_var']}_"
        f"{model_settings['data_filtering']}_"
        f"{model_settings['cost_func']}"
        f"_{model_settings['append']}"
    )

    # get site list for which p-model will be optimized/ run forward
    site_info = pd.read_csv(ROOT_PATH / "site_info/SiteInfo_BRKsite_list.csv")
    site_id_list = site_info["SiteID"].tolist()

    ################################################################################
    # optimize/ forward run p-model
    if model_settings["run_mode"] == "optim":
        # dump the model settings to specific experiment folder for future refernce
        settings_dict_path = Path(
            "opti_results", model_settings["model_name"], model_settings["exp_name"]
        )
        os.makedirs(
            settings_dict_path, exist_ok=True
        )  # create the directory if it does not exist
        settings_dict_path_filename = os.path.join(
            settings_dict_path, "settings_dict.json"
        )  # filename to save the settings dictionary
        with open(
            settings_dict_path_filename, "w", encoding="utf-8"
        ) as settings_dict_json_file:  # save the settings dictionary
            json.dump(
                model_settings,
                settings_dict_json_file,
                indent=4,
                separators=(", ", ": "),
            )

        # save the CMAES options to specific experiment folder for future refernce
        cmaes_options, sigma0 = get_cmaes_options()
        cmaes_options["sigma_zero"] = sigma0
        cmaes_options_path_filename = os.path.join(
            Path(
                "opti_results", model_settings["model_name"], model_settings["exp_name"]
            ),
            "cmaes_options.json",
        )
        with open(
            cmaes_options_path_filename, "w", encoding="utf-8"
        ) as cmaes_options_json_file:  # save the settings dictionary
            json.dump(
                cmaes_options,
                cmaes_options_json_file,
                indent=4,
                separators=(", ", ": "),
            )

        #########################################################################
        # Optimizing parameters for all years of each site/ site year in parallel
        if model_settings["eval_mode"] == "parallel":
            # #########################################################################
            # This is suitable for using job array index as an argument
            # generate range of site idx per job array
            j_array_no = int(sys.argv[1])  # job array index == $SLURM_ARRAY_TASK_ID
            ini_idx = 0  # initial index of site list (filename_list)
            n_sites = int(sys.argv[2])  # no. of sites per job
            max_n = len(site_id_list)  # total no. of sites/ site year

            # calculate the start and end index of site list for each job array
            start_site_idx, end_site_idx = calc_idx_for_jobarray(
                j_array_no, ini_idx, n_sites, max_n
            )

            #########################################################################
            # use partial to make the function take only one argument
            main_script_optimize_partial = partial(
                main_script_optimize,
                site_list=site_id_list,
                settings_dict=model_settings,
            )

            # optimize each site in parallel
            # create a process pool with many workers
            with Pool(n_sites) as pool_obj:
                pool_obj.map(
                    main_script_optimize_partial,
                    range(start_site_idx, end_site_idx, 1),
                )
                pool_obj.close()

            # clean the log file
            input_filename = os.path.join(
                "opti_results",
                model_settings["model_name"],
                model_settings["exp_name"],
                "model_optimize.log",
            )
            output_filename = os.path.join(
                "opti_results",
                model_settings["model_name"],
                model_settings["exp_name"],
                "model_optimize_clean.log",
            )
            filter_logs(
                input_filename, output_filename, 21
            )  # 21 is index at which "[%(asctime)s]" (which is only
            # difference in each line) ends in each log line

        #########################################################################
        # Sequential optimization for each site/ site year
        elif model_settings["eval_mode"] == "sequence":
            if isinstance(model_settings["site_list"], str):
                # get the list of sites of interest
                sites_to_use = model_settings["site_list"].split(", ")
                # optimize sequentially for each site/ site year
                for s in sites_to_use:
                    # testing the code with single site/
                    # use a for loop for sequential model optimization
                    site_idx_ = site_id_list.index(
                        s
                    )  # give the site name or site name_site year here
                    print("running for site/ site year: ", site_id_list[site_idx_])
                    main_script_optimize(site_idx_, site_id_list, model_settings)
            else:
                raise ValueError(
                    (
                        "It is mandatory to fill the `site_list` field in"
                        "the model_settings.xlsx file when `eval_mode` is `sequence`"
                    )
                )
        else:
            raise ValueError(
                "eval_mode must be one of: parallel, sequence, when run_mode is optim"
            )

    ###########################################################################################
    # use the optimized parameters to forward run the model and calculate the model performance
    elif model_settings["run_mode"] == "forward":
        #########################################################################
        if model_settings["eval_mode"] == "parallel":
            # This is suitable for using job array index [1- ] as an argument
            # generate range of site idx per job array
            j_array_no = int(sys.argv[1])  # job array index == $SLURM_ARRAY_TASK_ID
            ini_idx = 0  # initial index of site list (filename_list)
            n_sites = int(sys.argv[2])  # no. of sites per job
            max_n = len(site_id_list)  # total no. of sites 202 in BRK15

            # calculate the start and end index of site list for each job array
            start_site_idx, end_site_idx = calc_idx_for_jobarray(
                j_array_no, ini_idx, n_sites, max_n
            )

            ################################################################################
            # use partial to make the function take only one argument
            main_script_forward_partial = partial(
                main_script_forward,
                site_list=site_id_list,
                settings_dict=model_settings,
            )

            # forward run each site/ site year in parallel
            # create a process pool with many workers
            with Pool(n_sites) as pool_obj:
                pool_obj.map(
                    main_script_forward_partial,
                    range(start_site_idx, end_site_idx, 1),
                )
                pool_obj.close()

            # clean the log file
            input_filename = os.path.join(
                "model_results",
                model_settings["model_name"],
                model_settings["exp_name"],
                "Model_forward_run.log",
            )
            output_filename = os.path.join(
                "model_results",
                model_settings["model_name"],
                model_settings["exp_name"],
                "Model_forward_run_clean.log",
            )
            filter_logs(
                input_filename, output_filename, 21
            )  # 21 is index at which "[%(asctime)s]" (which is only
            # difference

        #########################################################################
        elif (
            model_settings["eval_mode"] == "sequence"
        ):  # testing the code with single site/ use a for loop for sequential forward model run
            # get the list of sites of interest
            if isinstance(model_settings["site_list"], str):
                sites_to_use = model_settings["site_list"].split(", ")
                # optimize sequentially for each site/ site year
                for s in sites_to_use:
                    site_idx_ = site_id_list.index(s)  # give the site name here
                    print("running for site: ", site_id_list[site_idx_])
                    main_script_forward(site_idx_, site_id_list, model_settings)

                # clean the log file
                input_filename = os.path.join(
                    "model_results",
                    model_settings["model_name"],
                    model_settings["exp_name"],
                    "Model_forward_run.log",
                )
                output_filename = os.path.join(
                    "model_results",
                    model_settings["model_name"],
                    model_settings["exp_name"],
                    "Model_forward_run_clean.log",
                )
                filter_logs(
                    input_filename, output_filename, 21
                )  # 21 is index at which "[%(asctime)s]" (which is only
                # difference
            else:
                raise ValueError(
                    (
                        "It is mandatory to fill the `site_list` field in"
                        "the model_settings.xlsx file when `eval_mode` is `sequence`"
                    )
                )
        #########################################################################
        # plot histogram of optimized parameters, model performance metrices, etc.
        elif model_settings["eval_mode"] == "summarize_exp_results":
            plot_exp_result(model_settings)


################################################################################
if __name__ == "__main__":
    main()
