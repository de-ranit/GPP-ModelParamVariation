#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
contains path to all the model experiment results
This module will be imported as module in other scripts to access the
paths to the model results.

author: rde
first created: Mon Feb 05 2024 15:19:02 CET
"""

from pathlib import Path

################################################################################################
# P-model (P_W_hr or P_hr model results)
################################################################################################
per_site_yr_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_site_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_site_p_model_res_path_iav = Path(
    (
        "../model_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_pft_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)
glob_opti_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "global_opti_BRK15_FPAR_FLUXNET_EO_CO2_"
        "MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)

################################################################################################
# Bao model (Bao_hr model results)
################################################################################################
per_site_yr_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_site_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_site_lue_model_res_path_iav = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_pft_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever/"
        "serialized_model_results/"
    )
)
glob_opti_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA"
        "_nominal_cost_lue_best_opti_ever/serialized_model_results/"
    )
)

################################################################################################
# Experiments in which a group of parameters varied per year, while all other parameters
# remained constant for a site
# P-model (P_W_hr model results)
################################################################################################
g01_vary_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "nnse_unc_vary_group1_best_opti_ever/serialized_model_results/"
    )
)

g02_vary_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "nnse_unc_vary_group2_best_opti_ever/serialized_model_results/"
    )
)

g03_vary_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "nnse_unc_vary_group3_best_opti_ever/serialized_model_results/"
    )
)

g04_vary_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "nnse_unc_vary_group4_best_opti_ever/serialized_model_results/"
    )
)

################################################################################################
# Experiments in which a group of parameters varied per year, while all other parameters
# remained constant for a site
# Bao model (Bao_hr model results)
################################################################################################
g01_vary_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "lue_vary_group1_best_opti_ever/serialized_model_results/"
    )
)

g02_vary_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "lue_vary_group2_best_opti_ever/serialized_model_results/"
    )
)

g03_vary_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "lue_vary_group3_best_opti_ever/serialized_model_results/"
    )
)

g04_vary_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "lue_vary_group4_best_opti_ever/serialized_model_results/"
    )
)

g05_vary_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "lue_vary_group5_best_opti_ever/serialized_model_results/"
    )
)

g06_vary_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "lue_vary_group6_best_opti_ever/serialized_model_results/"
    )
)

g07_vary_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "lue_vary_group7_best_opti_ever/serialized_model_results/"
    )
)

g08_vary_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_"
        "lue_vary_group8_best_opti_ever/serialized_model_results/"
    )
)

################################################################################################
# Forcing data path
################################################################################################
hr_ip_data_path = Path(
    "/path/to/hourly/forcing/data/in/nc/format"
)
