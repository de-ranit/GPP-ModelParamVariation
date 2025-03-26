#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# disable too many lines in module warning
# pylint: disable=C0302
"""
Plot histogram of model performance metrices and optimized parameter values

author: rde
first created: 2023-11-10
"""

from pathlib import Path
import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt


def create_ax(ax, var_name, var, fig_title, breaks, breaks_label):
    """
    create a histogram of the model performance metrices at certain temporal resolutions

    parameters:
    ax (object): matplotlib axis object
    var_name (str): name of the variable
    var (array): array of values of the variable
    fig_title (str): title of the figure
    breaks (list): list of break values for the histogram bins
    breaks_label (list): list of break labels for the histogram bins

    returns:
    add histogram of the model performance metrices at
    certain temporal resolutions to the axis object
    """
    # remove nan values (in case of yearly/monthly metrices)
    var = var[~np.isnan(var)]

    # calculate the percentage of sites in each bin
    weights = (
        np.ones_like(var) / len(var)
    ) * 100.0

    ax.hist(
        var,
        bins=breaks,
        weights=weights,
        facecolor="#FF8081",
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xticks(breaks)
    ax.set_xticklabels(breaks_label)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels(np.arange(0, 101, 20))
    ax.tick_params(axis="both", which="major", labelsize=20.0)
    ax.set_xlabel(var_name, fontsize=22)
    ax.set_ylabel("Percentage of sites", fontsize=22)
    ax.set_title(f"{fig_title} (Total number of sites: {len(var)})", size=24)

def create_fig(var_list, temp_res, var_name, breaks, breaks_label):
    """
    create a figure with 5 subplots showing histogram of
    the model perfromance metrices at different temporal resolutions

    parameters:
    var_list (list): list of arrays of values of the variable
    temp_res (str): temporal resolution of the input data
    var_name (str): name of the variable
    breaks (list): list of break values for the histogram bins
    breaks_label (list): list of break labels for the histogram bins

    returns:
    fig (object): matplotlib figure object with histogram of model
    performance metrices at different temporal resolutions
    """
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        nrows=3, ncols=2, figsize=(30, 20)
    )

    ax_list = [ax1, ax2, ax3, ax4, ax5]
    fig_title_list = [
        f"{var_name}_{temp_res}",
        f"{var_name}_daily",
        f"{var_name}_weekly",
        f"{var_name}_monthly",
        f"{var_name}_yearly",
    ]

    for idx, item in enumerate(ax_list):
        if (temp_res == "Daily") and (idx == 0):
            # create an empty subplot for subdaily GPP when using daily data
            ax1.text(
                0.5,
                0.5,
                "No sub-daily model performance plot in case of model optim/run \nusing daily data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax1.transAxes,
                fontsize=22,
            )
            ax1.set_xticks([])
            ax1.set_yticks([])
        else:
            create_ax(
                item, var_name, var_list[idx], fig_title_list[idx], breaks, breaks_label
            )

    fig.delaxes(ax6) # delete the empty subplot
    fig.tight_layout()

    return fig


def plot_exp_result(model_settings):
    """
    plot the results (histogram of model performance
    metrices and optimized parameter values) of an experiment

    parameters:
    model_settings (dict): dictionary with experiment settings

    returns:
    save the figures in the experiment folder
    """

    # path where the results are stored
    in_path = Path(
        "model_results", model_settings["model_name"], model_settings["exp_name"]
    )

    # path to the serialized results
    serialize_res_path = Path(in_path, "serialized_model_results")

    # load the results
    mod_res_filelist = glob.glob(f"{serialize_res_path}/*.npy")
    mod_res_filelist = sorted(mod_res_filelist)

    # remove the results of site CG-Tch, MY-PSO, GH-Ank and US-LWW as
    # fPAR data (MODIS NDVI based) is probably incorrect for these sites
    mod_res_filelist = [
        files
        for files in mod_res_filelist
        if not (
            "CG-Tch" in files
            or "MY-PSO" in files
            or "GH-Ank" in files
            or "US-LWW" in files
        )
    ]

    try:
        # load the results of the first site to get the temporal resolution
        result_dict = np.load(mod_res_filelist[0], allow_pickle=True).item()
    except IndexError as ind_err:
        raise FileNotFoundError(
            (
                'No results found for the experiment, please perform'
                'model optimization and forward run first'
            )
        ) from ind_err

    # get the temporal resolution of the input data in the experiment
    temp_res = result_dict["Temp_res"]

    # in case of global optimization, get the global optimized parameters
    if model_settings["opti_type"] == "global_opti":
        opti_par_dict = result_dict["Opti_par_val"]
        param_name_arr = np.array(list(opti_par_dict.keys()))
        param_val_arr = np.array(list(opti_par_dict.values()))

        # add the alpha_fT_Horn parameter for LUE model from another site
        if model_settings["model_name"] == "LUE_model":
            result_dict_2 = np.load(mod_res_filelist[2], allow_pickle=True).item()
            param_name_arr = np.insert(param_name_arr, 3, "alpha_fT_Horn")
            param_val_arr = np.insert(
                param_val_arr, 3, result_dict_2["Opti_par_val"]["alpha_fT_Horn"]
            )

    # create empty arrays to store the model performance metrices and
    # optimized parameters for each site/PFT optimization
    nse_sd = np.zeros(len(mod_res_filelist))
    nse_d = np.zeros(len(mod_res_filelist))
    nse_w = np.zeros(len(mod_res_filelist))
    nse_m = np.zeros(len(mod_res_filelist))
    nse_y = np.zeros(len(mod_res_filelist))

    cod_sd = np.zeros(len(mod_res_filelist))
    cod_d = np.zeros(len(mod_res_filelist))
    cod_w = np.zeros(len(mod_res_filelist))
    cod_m = np.zeros(len(mod_res_filelist))
    cod_y = np.zeros(len(mod_res_filelist))

    r2_sd = np.zeros(len(mod_res_filelist))
    r2_d = np.zeros(len(mod_res_filelist))
    r2_w = np.zeros(len(mod_res_filelist))
    r2_m = np.zeros(len(mod_res_filelist))
    r2_y = np.zeros(len(mod_res_filelist))

    kge_sd = np.zeros(len(mod_res_filelist))
    kge_d = np.zeros(len(mod_res_filelist))
    kge_w = np.zeros(len(mod_res_filelist))
    kge_m = np.zeros(len(mod_res_filelist))
    kge_y = np.zeros(len(mod_res_filelist))

    rmse_sd = np.zeros(len(mod_res_filelist))
    rmse_d = np.zeros(len(mod_res_filelist))
    rmse_w = np.zeros(len(mod_res_filelist))
    rmse_m = np.zeros(len(mod_res_filelist))
    rmse_y = np.zeros(len(mod_res_filelist))

    # collect model performance metrices for P model w/o moisture stress, acclimation window
    if model_settings["model_name"] == "P_model":
        nse_sd_gpp_no_stress = np.zeros(len(mod_res_filelist))
        nse_d_gpp_no_stress = np.zeros(len(mod_res_filelist))
        nse_w_gpp_no_stress = np.zeros(len(mod_res_filelist))
        nse_m_gpp_no_stress = np.zeros(len(mod_res_filelist))
        nse_y_gpp_no_stress = np.zeros(len(mod_res_filelist))

        cod_sd_gpp_no_stress = np.zeros(len(mod_res_filelist))
        cod_d_gpp_no_stress = np.zeros(len(mod_res_filelist))
        cod_w_gpp_no_stress = np.zeros(len(mod_res_filelist))
        cod_m_gpp_no_stress = np.zeros(len(mod_res_filelist))
        cod_y_gpp_no_stress = np.zeros(len(mod_res_filelist))

        r2_sd_gpp_no_stress = np.zeros(len(mod_res_filelist))
        r2_d_gpp_no_stress = np.zeros(len(mod_res_filelist))
        r2_w_gpp_no_stress = np.zeros(len(mod_res_filelist))
        r2_m_gpp_no_stress = np.zeros(len(mod_res_filelist))
        r2_y_gpp_no_stress = np.zeros(len(mod_res_filelist))

        kge_sd_gpp_no_stress = np.zeros(len(mod_res_filelist))
        kge_d_gpp_no_stress = np.zeros(len(mod_res_filelist))
        kge_w_gpp_no_stress = np.zeros(len(mod_res_filelist))
        kge_m_gpp_no_stress = np.zeros(len(mod_res_filelist))
        kge_y_gpp_no_stress = np.zeros(len(mod_res_filelist))

        rmse_sd_gpp_no_stress = np.zeros(len(mod_res_filelist))
        rmse_d_gpp_no_stress = np.zeros(len(mod_res_filelist))
        rmse_w_gpp_no_stress = np.zeros(len(mod_res_filelist))
        rmse_m_gpp_no_stress = np.zeros(len(mod_res_filelist))
        rmse_y_gpp_no_stress = np.zeros(len(mod_res_filelist))

        acclim_window = np.zeros(len(mod_res_filelist))

    # collect LUE model specific optimized parameters
    elif model_settings["model_name"] == "LUE_model":
        lue_max = np.zeros(len(mod_res_filelist))
        t_opt = np.zeros(len(mod_res_filelist))
        k_t = np.zeros(len(mod_res_filelist))
        alpha_ft_horn = np.zeros(len(mod_res_filelist))
        kappa_vpd = np.zeros(len(mod_res_filelist))
        ca_zero = np.zeros(len(mod_res_filelist))
        c_kappa = np.zeros(len(mod_res_filelist))
        c_m = np.zeros(len(mod_res_filelist))
        gamma_fl_tal = np.zeros(len(mod_res_filelist))
        mu_fci = np.zeros(len(mod_res_filelist))
    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {model_settings['model_name']}"
            "is not implemented"
        )

    # collect common optimized parameters
    w_i = np.zeros(len(mod_res_filelist))
    k_w = np.zeros(len(mod_res_filelist))
    alpha = np.zeros(len(mod_res_filelist))
    awc = np.zeros(len(mod_res_filelist))
    theta = np.zeros(len(mod_res_filelist))
    alpha_pt = np.zeros(len(mod_res_filelist))
    melt_rate_temp = np.zeros(len(mod_res_filelist))
    melt_rate_netrad = np.zeros(len(mod_res_filelist))
    sn_a = np.zeros(len(mod_res_filelist))

    pft_arr = []  # collect PFTs for per PFT optimization

    # loop over the sites and store the model performance metrices and parameters for each site
    for idx, files in enumerate(mod_res_filelist):
        result_dict = np.load(files, allow_pickle=True).item()

        nse_sd[idx] = result_dict["NSE"][f"NSE_{result_dict['Temp_res']}"]
        nse_d[idx] = result_dict["NSE"]["NSE_d"]
        nse_w[idx] = result_dict["NSE"]["NSE_w"]
        nse_m[idx] = result_dict["NSE"]["NSE_m"]
        nse_y[idx] = result_dict["NSE"]["NSE_y"]

        cod_sd[idx] = result_dict["COD"][f"COD_{result_dict['Temp_res']}"]
        cod_d[idx] = result_dict["COD"]["COD_d"]
        cod_w[idx] = result_dict["COD"]["COD_w"]
        cod_m[idx] = result_dict["COD"]["COD_m"]
        cod_y[idx] = result_dict["COD"]["COD_y"]

        r2_sd[idx] = result_dict["R2"][f"R2_{result_dict['Temp_res']}"]
        r2_d[idx] = result_dict["R2"]["R2_d"]
        r2_w[idx] = result_dict["R2"]["R2_w"]
        r2_m[idx] = result_dict["R2"]["R2_m"]
        r2_y[idx] = result_dict["R2"]["R2_y"]

        kge_sd[idx] = result_dict["KGE"][f"KGE_{result_dict['Temp_res']}"]
        kge_d[idx] = result_dict["KGE"]["KGE_d"]
        kge_w[idx] = result_dict["KGE"]["KGE_w"]
        kge_m[idx] = result_dict["KGE"]["KGE_m"]
        kge_y[idx] = result_dict["KGE"]["KGE_y"]

        rmse_sd[idx] = result_dict["RMSE"][f"RMSE_{result_dict['Temp_res']}"]
        rmse_d[idx] = result_dict["RMSE"]["RMSE_d"]
        rmse_w[idx] = result_dict["RMSE"]["RMSE_w"]
        rmse_m[idx] = result_dict["RMSE"]["RMSE_m"]
        rmse_y[idx] = result_dict["RMSE"]["RMSE_y"]

        if (
            model_settings["model_name"] == "P_model"
        ):  # P Model specific parameters & metrices
            nse_sd_gpp_no_stress[idx] = result_dict["NSE_no_moisture_Stress"][  # type: ignore
                f"NSE_{result_dict['Temp_res']}"
            ]
            nse_d_gpp_no_stress[idx] = result_dict[  # type: ignore
                "NSE_no_moisture_Stress"
            ]["NSE_d"]
            nse_w_gpp_no_stress[idx] = result_dict[  # type: ignore
                "NSE_no_moisture_Stress"
            ]["NSE_w"]
            nse_m_gpp_no_stress[idx] = result_dict[  # type: ignore
                "NSE_no_moisture_Stress"
            ]["NSE_m"]
            nse_y_gpp_no_stress[idx] = result_dict[  # type: ignore
                "NSE_no_moisture_Stress"
            ]["NSE_y"]

            cod_sd_gpp_no_stress[idx] = result_dict["COD_no_moisture_Stress"][  # type: ignore
                f"COD_{result_dict['Temp_res']}"
            ]
            cod_d_gpp_no_stress[idx] = result_dict[  # type: ignore
                "COD_no_moisture_Stress"
            ]["COD_d"]
            cod_w_gpp_no_stress[idx] = result_dict[  # type: ignore
                "COD_no_moisture_Stress"
            ]["COD_w"]
            cod_m_gpp_no_stress[idx] = result_dict[  # type: ignore
                "COD_no_moisture_Stress"
            ]["COD_m"]
            cod_y_gpp_no_stress[idx] = result_dict[  # type: ignore
                "COD_no_moisture_Stress"
            ]["COD_y"]

            r2_sd_gpp_no_stress[idx] = result_dict["R2_no_moisture_Stress"][  # type: ignore
                f"R2_{result_dict['Temp_res']}"
            ]
            r2_d_gpp_no_stress[idx] = result_dict["R2_no_moisture_Stress"]["R2_d"]  # type: ignore
            r2_w_gpp_no_stress[idx] = result_dict["R2_no_moisture_Stress"]["R2_w"]  # type: ignore
            r2_m_gpp_no_stress[idx] = result_dict["R2_no_moisture_Stress"]["R2_m"]  # type: ignore
            r2_y_gpp_no_stress[idx] = result_dict["R2_no_moisture_Stress"]["R2_y"]  # type: ignore

            kge_sd_gpp_no_stress[idx] = result_dict["KGE_no_moisture_Stress"][  # type: ignore
                f"KGE_{result_dict['Temp_res']}"
            ]
            kge_d_gpp_no_stress[idx] = result_dict[  # type: ignore
                "KGE_no_moisture_Stress"
            ]["KGE_d"]
            kge_w_gpp_no_stress[idx] = result_dict[  # type: ignore
                "KGE_no_moisture_Stress"
            ]["KGE_w"]
            kge_m_gpp_no_stress[idx] = result_dict[  # type: ignore
                "KGE_no_moisture_Stress"
            ]["KGE_m"]
            kge_y_gpp_no_stress[idx] = result_dict[  # type: ignore
                "KGE_no_moisture_Stress"
            ]["KGE_y"]

            rmse_sd_gpp_no_stress[idx] = result_dict["RMSE_no_moisture_Stress"][  # type: ignore
                f"RMSE_{result_dict['Temp_res']}"
            ]
            rmse_d_gpp_no_stress[idx] = result_dict[  # type: ignore
                "RMSE_no_moisture_Stress"
            ]["RMSE_d"]
            rmse_w_gpp_no_stress[idx] = result_dict[  # type: ignore
                "RMSE_no_moisture_Stress"
            ]["RMSE_w"]
            rmse_m_gpp_no_stress[idx] = result_dict[  # type: ignore
                "RMSE_no_moisture_Stress"
            ]["RMSE_m"]
            rmse_y_gpp_no_stress[idx] = result_dict[  # type: ignore
                "RMSE_no_moisture_Stress"
            ]["RMSE_y"]

        # collect optimized parameters in case of all year/ PFT optimization
        if model_settings["opti_type"] == "all_year":
            if model_settings["model_name"] == "P_model":
                acclim_window[idx] = result_dict["Opti_par_val"]["acclim_window"]  # type: ignore
            elif model_settings["model_name"] == "LUE_model":
                lue_max[idx] = result_dict["Opti_par_val"]["LUE_max"]  # type: ignore
                t_opt[idx] = result_dict["Opti_par_val"]["T_opt"]  # type: ignore
                k_t[idx] = result_dict["Opti_par_val"]["K_T"]  # type: ignore
                try:  # as alpha is not always optimized
                    alpha_ft_horn_val = result_dict["Opti_par_val"]["alpha_fT_Horn"]  # type: ignore
                    alpha_ft_horn[idx] = alpha_ft_horn_val  # type: ignore
                except KeyError:
                    alpha_ft_horn[idx] = np.nan  # type: ignore
                kappa_vpd[idx] = result_dict["Opti_par_val"]["Kappa_VPD"]  # type: ignore
                ca_zero[idx] = result_dict["Opti_par_val"]["Ca_0"]  # type: ignore
                c_kappa[idx] = result_dict["Opti_par_val"]["C_Kappa"]  # type: ignore
                c_m[idx] = result_dict["Opti_par_val"]["c_m"]  # type: ignore
                gamma_fl_tal[idx] = result_dict["Opti_par_val"]["gamma_fL_TAL"]  # type: ignore
                mu_fci[idx] = result_dict["Opti_par_val"]["mu_fCI"]  # type: ignore

            w_i[idx] = result_dict["Opti_par_val"]["W_I"]
            k_w[idx] = result_dict["Opti_par_val"]["K_W"]
            try:  # as alpha is not always optimized
                alpha[idx] = result_dict["Opti_par_val"]["alpha"]
            except KeyError:
                alpha[idx] = np.nan
            awc[idx] = result_dict["Opti_par_val"]["AWC"]
            theta[idx] = result_dict["Opti_par_val"]["theta"]
            alpha_pt[idx] = result_dict["Opti_par_val"]["alphaPT"]
            melt_rate_temp[idx] = result_dict["Opti_par_val"]["meltRate_temp"]
            melt_rate_netrad[idx] = result_dict["Opti_par_val"]["meltRate_netrad"]
            sn_a[idx] = result_dict["Opti_par_val"]["sn_a"]
        elif model_settings["opti_type"] == "per_pft":
            pft_arr.append(result_dict["PFT"])
            if model_settings["model_name"] == "P_model":
                acclim_window[idx] = result_dict["Opti_par_val"]["acclim_window"]  # type: ignore
            elif model_settings["model_name"] == "LUE_model":
                lue_max[idx] = result_dict["Opti_par_val"]["LUE_max"]  # type: ignore
                t_opt[idx] = result_dict["Opti_par_val"]["T_opt"]  # type: ignore
                k_t[idx] = result_dict["Opti_par_val"]["K_T"]  # type: ignore
                try:  # as alpha is not always optimized
                    alpha_ft_horn_val = result_dict["Opti_par_val"]["alpha_fT_Horn"]  # type: ignore
                    alpha_ft_horn[idx] = alpha_ft_horn_val  # type: ignore
                except KeyError:
                    alpha_ft_horn[idx] = np.nan  # type: ignore
                kappa_vpd[idx] = result_dict["Opti_par_val"]["Kappa_VPD"]  # type: ignore
                ca_zero[idx] = result_dict["Opti_par_val"]["Ca_0"]  # type: ignore
                c_kappa[idx] = result_dict["Opti_par_val"]["C_Kappa"]  # type: ignore
                c_m[idx] = result_dict["Opti_par_val"]["c_m"]  # type: ignore
                gamma_fl_tal[idx] = result_dict["Opti_par_val"]["gamma_fL_TAL"]  # type: ignore
                mu_fci[idx] = result_dict["Opti_par_val"]["mu_fCI"]  # type: ignore

            w_i[idx] = result_dict["Opti_par_val"]["W_I"]
            k_w[idx] = result_dict["Opti_par_val"]["K_W"]
            try:  # as alpha is not always optimized
                alpha[idx] = result_dict["Opti_par_val"]["alpha"]
            except KeyError:
                alpha[idx] = np.nan
            awc[idx] = result_dict["Opti_par_val"]["AWC"]
            theta[idx] = result_dict["Opti_par_val"]["theta"]
            alpha_pt[idx] = result_dict["Opti_par_val"]["alphaPT"]
            melt_rate_temp[idx] = result_dict["Opti_par_val"]["meltRate_temp"]
            melt_rate_netrad[idx] = result_dict["Opti_par_val"]["meltRate_netrad"]
            sn_a[idx] = result_dict["Opti_par_val"]["sn_a"]

    # get array of optimized parameter values for each PFT
    if model_settings["opti_type"] == "per_pft":
        pft_arr = np.array(pft_arr)
        _, ix = np.unique(pft_arr, return_index=True)
        pft_arr = pft_arr[ix]

        if model_settings["model_name"] == "P_model":
            acclim_window = acclim_window[ix]  # type: ignore
        elif model_settings["model_name"] == "LUE_model":
            lue_max = lue_max[ix]  # type: ignore
            t_opt = t_opt[ix]  # type: ignore
            k_t = k_t[ix]  # type: ignore
            alpha_ft_horn = alpha_ft_horn[ix]  # type: ignore
            kappa_vpd = kappa_vpd[ix]  # type: ignore
            ca_zero = ca_zero[ix]  # type: ignore
            c_kappa = c_kappa[ix]  # type: ignore
            c_m = c_m[ix]  # type: ignore
            gamma_fl_tal = gamma_fl_tal[ix]  # type: ignore
            mu_fci = mu_fci[ix]  # type: ignore
        w_i = w_i[ix]
        k_w = k_w[ix]
        alpha = alpha[ix]
        awc = awc[ix]
        theta = theta[ix]
        alpha_pt = alpha_pt[ix]
        melt_rate_temp = melt_rate_temp[ix]
        melt_rate_netrad = melt_rate_netrad[ix]
        sn_a = sn_a[ix]

    # replace the values less than -1.0 (very low negatives)
    # with -1.1 to plot them in -inf to -1.0 bin
    nse_sd = np.where(nse_sd < -1.0, -1.1, nse_sd)
    nse_d = np.where(nse_d < -1.0, -1.1, nse_d)
    nse_w = np.where(nse_w < -1.0, -1.1, nse_w)
    nse_m = np.where(nse_m < -1.0, -1.1, nse_m)
    nse_y = np.where(nse_y < -1.0, -1.1, nse_y)

    cod_sd = np.where(cod_sd < -1.0, -1.1, cod_sd)
    cod_d = np.where(cod_d < -1.0, -1.1, cod_d)
    cod_w = np.where(cod_w < -1.0, -1.1, cod_w)
    cod_m = np.where(cod_m < -1.0, -1.1, cod_m)
    cod_y = np.where(cod_y < -1.0, -1.1, cod_y)

    kge_sd = np.where(kge_sd < -1.0, -1.1, kge_sd)
    kge_d = np.where(kge_d < -1.0, -1.1, kge_d)
    kge_w = np.where(kge_w < -1.0, -1.1, kge_w)
    kge_m = np.where(kge_m < -1.0, -1.1, kge_m)
    kge_y = np.where(kge_y < -1.0, -1.1, kge_y)

    if model_settings["model_name"] == "P_model":
        nse_sd_gpp_no_stress = np.where(
            nse_sd_gpp_no_stress < -1.0, -1.1, nse_sd_gpp_no_stress  # type: ignore
        )
        nse_d_gpp_no_stress = np.where(
            nse_d_gpp_no_stress < -1.0, -1.1, nse_d_gpp_no_stress  # type: ignore
        )
        nse_w_gpp_no_stress = np.where(
            nse_w_gpp_no_stress < -1.0, -1.1, nse_w_gpp_no_stress  # type: ignore
        )
        nse_m_gpp_no_stress = np.where(
            nse_m_gpp_no_stress < -1.0, -1.1, nse_m_gpp_no_stress  # type: ignore
        )
        nse_y_gpp_no_stress = np.where(
            nse_y_gpp_no_stress < -1.0, -1.1, nse_y_gpp_no_stress  # type: ignore
        )

        cod_sd_gpp_no_stress = np.where(
            cod_sd_gpp_no_stress < -1.0, -1.1, cod_sd_gpp_no_stress  # type: ignore
        )
        cod_d_gpp_no_stress = np.where(
            cod_d_gpp_no_stress < -1.0, -1.1, cod_d_gpp_no_stress  # type: ignore
        )
        cod_w_gpp_no_stress = np.where(
            cod_w_gpp_no_stress < -1.0, -1.1, cod_w_gpp_no_stress  # type: ignore
        )
        cod_m_gpp_no_stress = np.where(
            cod_m_gpp_no_stress < -1.0, -1.1, cod_m_gpp_no_stress  # type: ignore
        )
        cod_y_gpp_no_stress = np.where(
            cod_y_gpp_no_stress < -1.0, -1.1, cod_y_gpp_no_stress  # type: ignore
        )

        kge_sd_gpp_no_stress = np.where(
            kge_sd_gpp_no_stress < -1.0, -1.1, kge_sd_gpp_no_stress  # type: ignore
        )
        kge_d_gpp_no_stress = np.where(
            kge_d_gpp_no_stress < -1.0, -1.1, kge_d_gpp_no_stress  # type: ignore
        )
        kge_w_gpp_no_stress = np.where(
            kge_w_gpp_no_stress < -1.0, -1.1, kge_w_gpp_no_stress  # type: ignore
        )
        kge_m_gpp_no_stress = np.where(
            kge_m_gpp_no_stress < -1.0, -1.1, kge_m_gpp_no_stress  # type: ignore
        )
        kge_y_gpp_no_stress = np.where(
            kge_y_gpp_no_stress < -1.0, -1.1, kge_y_gpp_no_stress  # type: ignore
        )

    ######################## PLOT MODEL PERFORMANCE METRICES ########################
    # create the break intervals and labels for the histograms
    breaks_nse = np.linspace(-1.2, 1.0, 12).tolist()
    breaks_nse = [round(x, 1) for x in breaks_nse]
    breaks_label_nse = [str(i) for i in breaks_nse]
    breaks_label_nse[0] = "-inf"

    breaks_r2 = np.linspace(0.0, 1.0, 11).tolist()
    breaks_r2 = [round(x, 1) for x in breaks_r2]
    breaks_label_r2 = [str(i) for i in breaks_r2]

    breaks_cod = breaks_nse
    breaks_label_cod = breaks_label_nse

    breaks_rmse = [*range(0, 22, 2)]
    breaks_label_rmse = [str(i) for i in breaks_rmse]

    # create the figures for each model performance metrices
    nse_hist = create_fig(
        [nse_sd, nse_d, nse_w, nse_m, nse_y],
        temp_res,
        "NSE",
        breaks_nse,
        breaks_label_nse,
    )
    cod_hist = create_fig(
        [cod_sd, cod_d, cod_w, cod_m, cod_y],
        temp_res,
        "COD",
        breaks_cod,
        breaks_label_cod,
    )
    r2_hist = create_fig(
        [r2_sd, r2_d, r2_w, r2_m, r2_y],
        temp_res,
        "$R^2$",
        breaks_r2,
        breaks_label_r2,
    )
    kge_hist = create_fig(
        [kge_sd, kge_d, kge_w, kge_m, kge_y],
        temp_res,
        "KGE",
        breaks_nse,
        breaks_label_nse,
    )
    rmse_hist = create_fig(
        [rmse_sd, rmse_d, rmse_w, rmse_m, rmse_y],
        temp_res,
        "RMSE",
        breaks_rmse,
        breaks_label_rmse,
    )

    # save the figures
    fig_path = Path(in_path, "exp_figures")
    os.makedirs(fig_path, exist_ok=True)

    nse_hist.savefig(os.path.join(fig_path, "nse_hist.png"))
    cod_hist.savefig(os.path.join(fig_path, "cod_hist.png"))
    r2_hist.savefig(os.path.join(fig_path, "r2_hist.png"))
    kge_hist.savefig(os.path.join(fig_path, "kge_hist.png"))
    rmse_hist.savefig(os.path.join(fig_path, "rmse_hist.png"))

    plt.close("all")
    ###############################################################################

    ########## PLOT MODEL PERFORMANCE METRICES (P MODEL SIM W/O MOISTURE STRESS) ####
    if model_settings["model_name"] == "P_model":
        # create the histogram for each model performance metrices
        # for simulated GPP without moisture stress
        nse_hist = create_fig(
            [
                nse_sd_gpp_no_stress,  # type: ignore
                nse_d_gpp_no_stress,  # type: ignore
                nse_w_gpp_no_stress,  # type: ignore
                nse_m_gpp_no_stress,  # type: ignore
                nse_y_gpp_no_stress,  # type: ignore
            ],
            temp_res,
            "NSE",
            breaks_nse,
            breaks_label_nse,
        )
        cod_hist = create_fig(
            [
                cod_sd_gpp_no_stress,  # type: ignore
                cod_d_gpp_no_stress,  # type: ignore
                cod_w_gpp_no_stress,  # type: ignore
                cod_m_gpp_no_stress,  # type: ignore
                cod_y_gpp_no_stress,  # type: ignore
            ],
            temp_res,
            "COD",
            breaks_cod,
            breaks_label_cod,
        )
        r2_hist = create_fig(
            [
                r2_sd_gpp_no_stress,  # type: ignore
                r2_d_gpp_no_stress,  # type: ignore
                r2_w_gpp_no_stress,  # type: ignore
                r2_m_gpp_no_stress,  # type: ignore
                r2_y_gpp_no_stress,  # type: ignore
            ],
            temp_res,
            "$R^2$",
            breaks_r2,
            breaks_label_r2,
        )
        kge_hist = create_fig(
            [
                kge_sd_gpp_no_stress,  # type: ignore
                kge_d_gpp_no_stress,  # type: ignore
                kge_w_gpp_no_stress,  # type: ignore
                kge_m_gpp_no_stress,  # type: ignore
                kge_y_gpp_no_stress,  # type: ignore
            ],
            temp_res,
            "KGE",
            breaks_nse,
            breaks_label_nse,
        )
        rmse_hist = create_fig(
            [
                rmse_sd_gpp_no_stress,  # type: ignore
                rmse_d_gpp_no_stress,  # type: ignore
                rmse_w_gpp_no_stress,  # type: ignore
                rmse_m_gpp_no_stress,  # type: ignore
                rmse_y_gpp_no_stress,  # type: ignore
            ],
            temp_res,
            "RMSE",
            breaks_rmse,
            breaks_label_rmse,
        )

        # save the figures
        fig_path = Path(in_path, "exp_figures")
        os.makedirs(fig_path, exist_ok=True)

        nse_hist.savefig(os.path.join(fig_path, "nse_hist_no_moisture_stress.png"))
        cod_hist.savefig(os.path.join(fig_path, "cod_hist_no_moisture_stress.png"))
        r2_hist.savefig(os.path.join(fig_path, "r2_hist_no_moisture_stress.png"))
        kge_hist.savefig(os.path.join(fig_path, "kge_hist_no_moisture_stress.png"))
        rmse_hist.savefig(os.path.join(fig_path, "rmse_hist_no_moisture_stress.png"))

        plt.close("all")
    ###############################################################################

    ########## PLOT OPTIMIZED PARAMETERS (ALL YEAR OPTIMIZATION) ##################
    if model_settings["opti_type"] == "all_year":
        ############ P MODEL ############################
        if model_settings["model_name"] == "P_model":
            # plot the histogram of optimized parameters
            fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(40, 30))

            weights = (np.ones_like(acclim_window) / len(acclim_window)) * 100.0  # type: ignore

            axs[0, 0].hist(
                acclim_window,  # type: ignore
                weights=weights,
                bins=[*range(0, 110, 10)],
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[0, 0].set_xticks([*range(0, 110, 10)])
            axs[0, 0].tick_params(axis="both", which="major", labelsize=20.0)
            axs[0, 0].set_xlabel("Acclimation window (days)", fontsize=22)
            axs[0, 0].set_ylabel("Percentage of sites", fontsize=22)
            axs[0, 0].set_title("Frequency of acclimation window", size=24)

            axs[0, 1].hist(
                w_i,
                weights=weights,
                bins=[round(x, 1) for x in np.arange(0.0, 1.1, 0.1).tolist()],
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[0, 1].set_xticks(
                [round(x, 1) for x in np.arange(0.0, 1.1, 0.1).tolist()]
            )
            axs[0, 1].tick_params(axis="both", which="major", labelsize=20.0)
            axs[0, 1].set_xlabel("$W_I$ (mm/mm)", fontsize=22)
            axs[0, 1].set_ylabel("Percentage of sites", fontsize=22)
            axs[0, 1].set_title("Frequency of $W_I$", size=24)

            axs[0, 2].hist(
                k_w,
                weights=weights,
                bins=np.arange(-30, -2.5, 2.5).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[0, 2].set_xticks(np.arange(-30, -2.5, 2.5).tolist())
            axs[0, 2].tick_params(axis="both", which="major", labelsize=20.0)
            axs[0, 2].set_xlabel("$K_W$ (-)", fontsize=22)
            axs[0, 2].set_ylabel("Percentage of sites", fontsize=22)
            axs[0, 2].set_title("Frequency of $K_W$", size=24)

            axs[1, 0].hist(
                alpha,
                weights=weights,
                bins=[round(x, 1) for x in np.arange(0, 1.1, 0.1).tolist()],
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[1, 0].set_xticks([round(x, 1) for x in np.arange(0, 1.1, 0.1).tolist()])
            axs[1, 0].tick_params(axis="both", which="major", labelsize=20.0)
            axs[1, 0].set_xlabel("alpha (-)", fontsize=22)
            axs[1, 0].set_ylabel("Percentage of sites", fontsize=22)
            axs[1, 0].set_title("Frequency of alpha", size=24)

            axs[1, 1].hist(
                theta,
                weights=weights,
                bins=[round(x, 4) for x in np.linspace(0.0001, 0.0009, 10).tolist()]
                + [round(x, 3) for x in np.arange(0.001, 0.011, 0.001).tolist()],
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[1, 1].set_xticks(
                [0.0001, 0.0005]
                + [round(x, 3) for x in np.arange(0.001, 0.011, 0.001).tolist()],
                labels=[0.0001, 0.0005]
                + [round(x, 3) for x in np.arange(0.001, 0.011, 0.001).tolist()],
                rotation=45.0,
            )
            axs[1, 1].tick_params(axis="both", which="major", labelsize=20.0)
            axs[1, 1].set_xlabel("theta (-)", fontsize=22)
            axs[1, 1].set_ylabel("Percentage of sites", fontsize=22)
            axs[1, 1].set_title("Frequency of theta", size=24)

            axs[1, 2].hist(
                theta,
                weights=weights,
                bins=[round(x, 3) for x in np.arange(0.01, 0.11, 0.01).tolist()],
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[1, 2].set_xticks(
                [round(x, 3) for x in np.arange(0.01, 0.11, 0.01).tolist()]
            )
            axs[1, 2].tick_params(axis="both", which="major", labelsize=20.0)
            axs[1, 2].set_xlabel("theta (-)", fontsize=22)
            axs[1, 2].set_ylabel("Percentage of sites", fontsize=22)
            axs[1, 2].set_title("Frequency of theta", size=24)

            axs[2, 0].hist(
                awc,
                weights=weights,
                bins=np.arange(0, 1100, 100).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[2, 0].set_xticks(np.arange(0, 1100, 100).tolist())
            axs[2, 0].tick_params(axis="both", which="major", labelsize=20.0)
            axs[2, 0].set_xlabel("AWC (mm)", fontsize=22)
            axs[2, 0].set_ylabel("Percentage of sites", fontsize=22)
            axs[2, 0].set_title("Frequency of AWC", size=24)

            axs[2, 1].hist(
                alpha_pt * 1.26,
                weights=weights,
                bins=np.linspace(0.0, 7.0, 15).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[2, 1].set_xticks(np.linspace(0.0, 7.0, 15).tolist())
            axs[2, 1].tick_params(axis="both", which="major", labelsize=20.0)
            axs[2, 1].set_xlabel("alpha in PT (-)", fontsize=22)
            axs[2, 1].set_ylabel("Percentage of sites", fontsize=22)
            axs[2, 1].set_title("Frequency of alpha in PT", size=24)

            axs[2, 2].hist(
                melt_rate_temp,
                weights=weights,
                bins=np.linspace(0.0, 0.5, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[2, 2].set_xticks(np.linspace(0.0, 0.5, 11).tolist())
            axs[2, 2].tick_params(axis="both", which="major", labelsize=20.0)
            axs[2, 2].set_xlabel(
                "snow melt rate for temp (mm $\u00b0C^{-1} hr^{-1}$)", fontsize=22
            )
            axs[2, 2].set_ylabel("Percentage of sites", fontsize=22)
            axs[2, 2].set_title("Frequency of $m_t$", size=24)

            axs[3, 0].hist(
                melt_rate_netrad,
                weights=weights,
                bins=np.linspace(0.0, 0.125, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[3, 0].set_xticks(np.linspace(0.0, 0.125, 11).tolist())
            axs[3, 0].tick_params(axis="both", which="major", labelsize=20.0)
            axs[3, 0].set_xlabel(
                "snow melt rate for net rad (mm $MJ^{-1} hr^{-1}$)", fontsize=22
            )
            axs[3, 0].set_ylabel("Percentage of sites", fontsize=22)
            axs[3, 0].set_title("Frequency of $m_r$", size=24)

            axs[3, 1].hist(
                sn_a,
                weights=weights,
                bins=np.linspace(0.0, 3.0, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[3, 1].set_xticks(np.linspace(0.0, 3.0, 11).tolist())
            axs[3, 1].tick_params(axis="both", which="major", labelsize=20.0)
            axs[3, 1].set_xlabel("sublimation resistance (-)", fontsize=22)
            axs[3, 1].set_ylabel("Percentage of sites", fontsize=22)
            axs[3, 1].set_title("Frequency of $sn_a$", size=24)

            fig.delaxes(axs[3, 2])

            fig.tight_layout()
            fig.savefig(os.path.join(fig_path, "optim_param_hist.png"))

            plt.close("all")
        ###########################################################################

        ############ LUE MODEL ############################
        elif model_settings["model_name"] == "LUE_model":
            fig, axs = plt.subplots(ncols=4, nrows=5, figsize=(50, 40))

            weights = (np.ones_like(lue_max) / len(lue_max)) * 100.0  # type: ignore

            axs[0, 0].hist(
                lue_max,  # type: ignore
                weights=weights,
                bins=np.linspace(0.0, 0.4, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[0, 0].set_xticks(np.linspace(0.0, 0.4, 11).tolist())
            axs[0, 0].tick_params(axis="both", which="major", labelsize=20.0)
            axs[0, 0].set_xlabel(
                ("$LUE_{max}$ \n (\u03bcmol C \u03bcmol photons$^{-1}$)"), fontsize=22
            )
            axs[0, 0].set_ylabel("Percentage of sites", fontsize=22)
            axs[0, 0].set_title("Frequency of Max LUE", size=24)

            axs[0, 1].hist(
                t_opt,  # type: ignore
                weights=weights,
                bins=np.linspace(0.0, 35.0, 8).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[0, 1].set_xticks(np.linspace(0.0, 35.0, 8).tolist())
            axs[0, 1].tick_params(axis="both", which="major", labelsize=20.0)
            axs[0, 1].set_xlabel("$T_{opt}$ ($\u00B0C$)", fontsize=22)
            axs[0, 1].set_ylabel("Percentage of sites", fontsize=22)
            axs[0, 1].set_title("Frequency of optimal temperature", size=24)

            axs[0, 2].hist(
                k_t,  # type: ignore
                weights=weights,
                bins=np.arange(0.0, 22.0, 2).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[0, 2].set_xticks(np.arange(0.0, 22.0, 2).tolist())
            axs[0, 2].tick_params(axis="both", which="major", labelsize=20.0)
            axs[0, 2].set_xlabel("$K_T$ ($\u00B0C^{-1}$)", fontsize=22)
            axs[0, 2].set_ylabel("Percentage of sites", fontsize=22)
            axs[0, 2].set_title(
                "Frequency of sensitivity to temperature changes", size=24
            )

            axs[0, 3].hist(
                alpha_ft_horn,  # type: ignore
                weights=weights,
                bins=np.linspace(0.0, 1.0, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[0, 3].set_xticks(np.linspace(0.0, 1.0, 11).tolist())
            axs[0, 3].tick_params(axis="both", which="major", labelsize=20.0)
            axs[0, 3].set_xlabel("$\u03B1_{fT}$ (-)", fontsize=22)
            axs[0, 3].set_ylabel("Percentage of sites", fontsize=22)
            axs[0, 3].set_title(
                "Frequency of lag parameter for temperature effect", size=24
            )

            axs[1, 0].hist(
                kappa_vpd,  # type: ignore
                weights=weights,
                bins=np.linspace(-0.01, -1e-5, 15).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[1, 0].set_xticks(
                [round(x, 4) for x in np.linspace(-0.01, -1e-5, 15).tolist()],
                labels=[round(x, 4) for x in np.linspace(-0.01, -1e-5, 15).tolist()],
                rotation=45.0,
            )
            axs[1, 0].tick_params(axis="both", which="major", labelsize=20.0)
            axs[1, 0].set_xlabel("$\u03BA_{VPD}$ ($Pa^{-1}$)", fontsize=22)
            axs[1, 0].set_ylabel("Percentage of sites", fontsize=22)
            axs[1, 0].set_title("Frequency of sensitivity to VPD changes", size=24)

            axs[1, 1].hist(
                ca_zero,  # type: ignore
                weights=weights,
                bins=np.linspace(340.0, 390.0, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[1, 1].set_xticks(np.linspace(340.0, 390.0, 11).tolist())
            axs[1, 1].tick_params(axis="both", which="major", labelsize=20.0)
            axs[1, 1].set_xlabel("$C_{a0}$ (ppm)", fontsize=22)
            axs[1, 1].set_ylabel("Percentage of sites", fontsize=22)
            axs[1, 1].set_title(
                "Frequency of minimum optimal atmospheric CO$_2$ concentration", size=24
            )

            axs[1, 2].hist(
                c_kappa,  # type: ignore
                weights=weights,
                bins=np.linspace(0.0, 10.0, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[1, 2].set_xticks(np.linspace(0.0, 10.0, 11).tolist())
            axs[1, 2].tick_params(axis="both", which="major", labelsize=20.0)
            axs[1, 2].set_xlabel("$C_{\u03BA}$ (-)", fontsize=22)
            axs[1, 2].set_ylabel("Percentage of sites", fontsize=22)
            axs[1, 2].set_title(
                "Frequency of sensitivity to atmospheric CO$_2$ changes", size=24
            )

            axs[1, 3].hist(
                c_m,  # type: ignore
                weights=weights,
                bins=np.linspace(100.0, 4000.0, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[1, 3].set_xticks(np.linspace(100.0, 4000.0, 11).tolist())
            axs[1, 3].tick_params(axis="both", which="major", labelsize=20.0)
            axs[1, 3].set_xlabel("$C_m$ (ppm)", fontsize=22)
            axs[1, 3].set_ylabel("Percentage of sites", fontsize=22)
            axs[1, 3].set_title(
                "Frequency of CO$_2$ fertilization intensity indicator", size=24
            )

            axs[2, 0].hist(
                gamma_fl_tal,  # type: ignore
                weights=weights,
                bins=np.linspace(0.0, 0.05, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[2, 0].set_xticks(np.linspace(0.0, 0.05, 11).tolist())
            axs[2, 0].tick_params(axis="both", which="major", labelsize=20.0)
            axs[2, 0].set_xlabel(
                "\u03B3 (\u03bcmol photons$^{-1}$ m$^2$s)", fontsize=22
            )
            axs[2, 0].set_ylabel("Percentage of sites", fontsize=22)
            axs[2, 0].set_title(
                "Frequency of Light saturation curve indicator", size=24
            )

            axs[2, 1].hist(
                mu_fci,  # type: ignore
                weights=weights,
                bins=np.linspace(0.001, 1.0, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[2, 1].set_xticks(
                np.linspace(0.001, 1.0, 11).tolist(),
                labels=np.linspace(0.001, 1.0, 11).tolist(),
                rotation=45.0,
            )
            axs[2, 1].tick_params(axis="both", which="major", labelsize=20.0)
            axs[2, 1].set_xlabel("\u03BC (-)", fontsize=22)
            axs[2, 1].set_ylabel("Percentage of sites", fontsize=22)
            axs[2, 1].set_title(
                "Frequency of Sensitivity to cloudiness index changes", size=24
            )

            axs[2, 2].hist(
                w_i,
                weights=weights,
                bins=[round(x, 1) for x in np.arange(0.0, 1.1, 0.1).tolist()],
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[2, 2].set_xticks(
                [round(x, 1) for x in np.arange(0.0, 1.1, 0.1).tolist()]
            )
            axs[2, 2].tick_params(axis="both", which="major", labelsize=20.0)
            axs[2, 2].set_xlabel("$W_I$ (mm/mm)", fontsize=22)
            axs[2, 2].set_ylabel("Percentage of sites", fontsize=22)
            axs[2, 2].set_title("Frequency of $W_I$", size=24)

            axs[2, 3].hist(
                k_w,
                weights=weights,
                bins=np.arange(-30, -2.5, 2.5).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[2, 3].set_xticks(np.arange(-30, -2.5, 2.5).tolist())
            axs[2, 3].tick_params(axis="both", which="major", labelsize=20.0)
            axs[2, 3].set_xlabel("$K_W$ (-)", fontsize=22)
            axs[2, 3].set_ylabel("Percentage of sites", fontsize=22)
            axs[2, 3].set_title("Frequency of $K_W$", size=24)

            axs[3, 0].hist(
                alpha,
                weights=weights,
                bins=[round(x, 1) for x in np.arange(0, 1.1, 0.1).tolist()],
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[3, 0].set_xticks([round(x, 1) for x in np.arange(0, 1.1, 0.1).tolist()])
            axs[3, 0].tick_params(axis="both", which="major", labelsize=20.0)
            axs[3, 0].set_xlabel("alpha (-)", fontsize=22)
            axs[3, 0].set_ylabel("Percentage of sites", fontsize=22)
            axs[3, 0].set_title("Frequency of alpha", size=24)

            axs[3, 1].hist(
                theta,
                weights=weights,
                bins=[round(x, 4) for x in np.linspace(0.0001, 0.0009, 10).tolist()]
                + [round(x, 3) for x in np.arange(0.001, 0.011, 0.001).tolist()],
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[3, 1].set_xticks(
                [0.0001, 0.0005]
                + [round(x, 3) for x in np.arange(0.001, 0.011, 0.001).tolist()],
                labels=[0.0001, 0.0005]
                + [round(x, 3) for x in np.arange(0.001, 0.011, 0.001).tolist()],
                rotation=45.0,
            )
            axs[3, 1].tick_params(axis="both", which="major", labelsize=20.0)
            axs[3, 1].set_xlabel("theta (-)", fontsize=22)
            axs[3, 1].set_ylabel("Percentage of sites", fontsize=22)
            axs[3, 1].set_title("Frequency of theta", size=24)

            axs[3, 2].hist(
                theta,
                weights=weights,
                bins=[round(x, 3) for x in np.arange(0.01, 0.11, 0.01).tolist()],
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[3, 2].set_xticks(
                [round(x, 3) for x in np.arange(0.01, 0.11, 0.01).tolist()]
            )
            axs[3, 2].tick_params(axis="both", which="major", labelsize=20.0)
            axs[3, 2].set_xlabel("theta (-)", fontsize=22)
            axs[3, 2].set_ylabel("Percentage of sites", fontsize=22)
            axs[3, 2].set_title("Frequency of theta", size=24)

            axs[3, 3].hist(
                awc,
                weights=weights,
                bins=np.arange(0, 1100, 100).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[3, 3].set_xticks(np.arange(0, 1100, 100).tolist())
            axs[3, 3].tick_params(axis="both", which="major", labelsize=20.0)
            axs[3, 3].set_xlabel("AWC (mm)", fontsize=22)
            axs[3, 3].set_ylabel("Percentage of sites", fontsize=22)
            axs[3, 3].set_title("Frequency of AWC", size=24)

            axs[4, 0].hist(
                alpha_pt * 1.26,
                weights=weights,
                bins=np.linspace(0.0, 7.0, 15).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[4, 0].set_xticks(np.linspace(0.0, 7.0, 15).tolist())
            axs[4, 0].tick_params(axis="both", which="major", labelsize=20.0)
            axs[4, 0].set_xlabel("alpha in PT (-)", fontsize=22)
            axs[4, 0].set_ylabel("Percentage of sites", fontsize=22)
            axs[4, 0].set_title("Frequency of alpha in PT", size=24)

            axs[4, 1].hist(
                melt_rate_temp,
                weights=weights,
                bins=np.linspace(0.0, 0.5, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[4, 1].set_xticks(np.linspace(0.0, 0.5, 11).tolist())
            axs[4, 1].tick_params(axis="both", which="major", labelsize=20.0)
            axs[4, 1].set_xlabel(
                "snow melt rate for temp (mm $\u00b0C^{-1} hr^{-1}$)", fontsize=22
            )
            axs[4, 1].set_ylabel("Percentage of sites", fontsize=22)
            axs[4, 1].set_title("Frequency of $m_t$", size=24)

            axs[4, 2].hist(
                melt_rate_netrad,
                weights=weights,
                bins=np.linspace(0.0, 0.125, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[4, 2].set_xticks(
                [round(x, 4) for x in np.linspace(0.0, 0.125, 11).tolist()],
                labels=[round(x, 4) for x in np.linspace(0.0, 0.125, 11).tolist()],
                rotation=45.0,
            )
            axs[4, 2].tick_params(axis="both", which="major", labelsize=20.0)
            axs[4, 2].set_xlabel(
                "snow melt rate for net rad (mm $MJ^{-1} hr^{-1}$)", fontsize=22
            )
            axs[4, 2].set_ylabel("Percentage of sites", fontsize=22)
            axs[4, 2].set_title("Frequency of $m_r$", size=24)

            axs[4, 3].hist(
                sn_a,
                weights=weights,
                bins=np.linspace(0.0, 3.0, 11).tolist(),
                facecolor="#FF8081",
                edgecolor="black",
                linewidth=1.2,
            )
            axs[4, 3].set_xticks(np.linspace(0.0, 3.0, 11).tolist())
            axs[4, 3].tick_params(axis="both", which="major", labelsize=20.0)
            axs[4, 3].set_xlabel("sublimation resistance (-)", fontsize=22)
            axs[4, 3].set_ylabel("Percentage of sites", fontsize=22)
            axs[4, 3].set_title("Frequency of $sn_a$", size=24)

            fig.tight_layout()
            fig.savefig(os.path.join(fig_path, "optim_param_hist.png"))

            plt.close("all")
        ###########################################################################

    ########## PLOT OPTIMIZED PARAMETERS (PER PFT OPTIMIZATION) ##################
    elif model_settings["opti_type"] == "per_pft":
        ############ P MODEL ############################
        if model_settings["model_name"] == "P_model":
            # plot the values of optimized parameters for each PFT
            fig, (
                (ax1, ax2),
                (ax3, ax4),
                (ax5, ax6),
                (ax7, ax8),
                (ax9, ax10),
            ) = plt.subplots(nrows=5, ncols=2, figsize=(30, 20))

            # add data to the plot
            ax1.bar(pft_arr, acclim_window, color="#714697", alpha=0.7)  # type: ignore
            ax2.bar(pft_arr, w_i, color="#714697", alpha=0.7)
            ax3.bar(pft_arr, k_w, color="#714697", alpha=0.7)
            ax4.bar(pft_arr, alpha, color="#714697", alpha=0.7)
            ax5.bar(pft_arr, awc, color="#714697", alpha=0.7)
            ax6.bar(pft_arr, theta, color="#714697", alpha=0.7)
            ax7.bar(pft_arr, alpha_pt, color="#714697", alpha=0.7)
            ax8.bar(pft_arr, melt_rate_temp, color="#714697", alpha=0.7)
            ax9.bar(pft_arr, melt_rate_netrad, color="#714697", alpha=0.7)
            ax10.bar(pft_arr, sn_a, color="#714697", alpha=0.7)

            # set the titles of subplots
            ax1.set_title("Acclimation window", fontsize=16)
            ax2.set_title("$W_I$ (mm/mm)", fontsize=16)
            ax3.set_title("$K_W$ (-)", fontsize=16)
            ax4.set_title("alpha", fontsize=16)
            ax5.set_title("AWC (mm)", fontsize=16)
            ax6.set_title("theta", fontsize=16)
            ax7.set_title("alphaPT", fontsize=16)
            ax8.set_title("snow melt rate for temp", fontsize=16)
            ax9.set_title("snow melt rate for net radiation", fontsize=16)
            ax10.set_title("sublimation resistance", fontsize=16)

            # set the y axis labels of subplots
            ax1.set_ylabel("Acclimation window (days)", fontsize=16)
            ax2.set_ylabel("$W_I$ (mm/mm)", fontsize=16)
            ax3.set_ylabel("$K_W$ (-)", fontsize=16)
            ax4.set_ylabel("alpha (-)", fontsize=16)
            ax5.set_ylabel("AWC (mm)", fontsize=16)
            ax6.set_ylabel("theta (-)", fontsize=16)
            ax7.set_ylabel("alpha in PT (-)", fontsize=16)
            ax8.set_ylabel("meltRate_temp \n (mm $\u00b0C^{-1} hr^{-1}$)", fontsize=16)
            ax9.set_ylabel("meltRate_netrad \n (mm $MJ^{-1} hr^{-1}$)", fontsize=16)
            ax10.set_ylabel("sn_a (-)", fontsize=16)

            # set the x axis labels of subplots
            for ax in fig.axes:
                ax.set_xlabel("PFT", fontsize=16)
                ax.set_xticks(pft_arr)
                ax.tick_params(axis="x", labelrotation=45, labelsize=16)

            # remove white space between subplots
            fig.tight_layout()

            # save the plot
            fig.savefig(os.path.join(fig_path, "optim_param_hist.png"))

            plt.close("all")
        ###########################################################################

        ############ LUE MODEL ############################
        elif model_settings["model_name"] == "LUE_model":  # for LUE model
            # plot the values of optimized parameters for each PFT
            fig, (
                (ax1, ax2, ax3, ax4),
                (ax5, ax6, ax7, ax8),
                (ax9, ax10, ax11, ax12),
                (ax13, ax14, ax15, ax16),
                (ax17, ax18, ax19, ax20),
            ) = plt.subplots(nrows=5, ncols=4, figsize=(30, 20))

            # add data to the plot
            ax1.bar(pft_arr, lue_max, color="#714697", alpha=0.7)  # type: ignore
            ax2.bar(pft_arr, t_opt, color="#714697", alpha=0.7)  # type: ignore
            ax3.bar(pft_arr, k_t, color="#714697", alpha=0.7)  # type: ignore
            ax4.bar(pft_arr, alpha_ft_horn, color="#714697", alpha=0.7)  # type: ignore
            ax5.bar(pft_arr, kappa_vpd, color="#714697", alpha=0.7)  # type: ignore
            ax6.bar(pft_arr, ca_zero, color="#714697", alpha=0.7)  # type: ignore
            ax7.bar(pft_arr, c_kappa, color="#714697", alpha=0.7)  # type: ignore
            ax8.bar(pft_arr, c_m, color="#714697", alpha=0.7)  # type: ignore
            ax9.bar(pft_arr, gamma_fl_tal, color="#714697", alpha=0.7)  # type: ignore
            ax10.bar(pft_arr, mu_fci, color="#714697", alpha=0.7)  # type: ignore
            ax11.bar(pft_arr, w_i, color="#714697", alpha=0.7)
            ax12.bar(pft_arr, k_w, color="#714697", alpha=0.7)
            ax13.bar(pft_arr, alpha, color="#714697", alpha=0.7)
            ax14.bar(pft_arr, awc, color="#714697", alpha=0.7)
            ax15.bar(pft_arr, theta, color="#714697", alpha=0.7)
            ax16.bar(pft_arr, alpha_pt, color="#714697", alpha=0.7)
            ax17.bar(pft_arr, melt_rate_temp, color="#714697", alpha=0.7)
            ax18.bar(pft_arr, melt_rate_netrad, color="#714697", alpha=0.7)
            ax19.bar(pft_arr, sn_a, color="#714697", alpha=0.7)

            # set the titles of subplots
            ax1.set_title("Maximum light use efficiency", fontsize=16)
            ax2.set_title("Optimal temperature", fontsize=16)
            ax3.set_title("Sensitivity to temperature changes", fontsize=16)
            ax4.set_title("Lag parameter for temperature effect", fontsize=16)
            ax5.set_title("Sensitivity to VPD changes", fontsize=16)
            ax6.set_title(
                "Minimum optimal atmospheric CO$_2$ concentration", fontsize=16
            )
            ax7.set_title(
                "Sensitivity to atmospheric CO$_2$ concentration changes", fontsize=16
            )
            ax8.set_title("CO$_2$ fertilization intensity indicator", fontsize=16)
            ax9.set_title("Light saturation curve indicator", fontsize=16)
            ax10.set_title("Sensitivity to cloudiness index changes", fontsize=16)
            ax11.set_title("Optimal soil moisture", fontsize=16)
            ax12.set_title("Sensitivity to soil moisture changes", fontsize=16)
            ax13.set_title("Lag parameter for soil moisture effect", fontsize=16)
            ax14.set_title("Available water capacity", fontsize=16)
            ax15.set_title("Decay of water bucket", fontsize=16)
            ax16.set_title("Multiplication factor for PET", fontsize=16)
            ax17.set_title("Snow melt rate for temp", fontsize=16)
            ax18.set_title("Snow melt rate for net radiation", fontsize=16)
            ax19.set_title("Sublimation resistance", fontsize=16)

            # set the y axis labels of subplots
            ax1.set_ylabel(
                "$LUE_{max}$ \n (\u03bcmol C \u03bcmol photons$^{-1}$)", fontsize=16
            )
            ax2.set_ylabel("$T_{opt}$ ($\u00B0C$)", fontsize=16)
            ax3.set_ylabel("$k_T$ ($\u00B0C^{-1}$)", fontsize=16)
            ax4.set_ylabel("$\u03B1_{fT}$ (-)", fontsize=16)
            ax5.set_ylabel("\u03BA ($Pa^{-1}$)", fontsize=16)
            ax6.set_ylabel("$C_{a0}$ (ppm)", fontsize=16)
            ax7.set_ylabel("$C_{\u03BA}$ (-)", fontsize=16)
            ax8.set_ylabel("$C_m$ (ppm)", fontsize=16)
            ax9.set_ylabel("\u03B3 (\u03bcmol photons$^{-1}$ m$^2$s)", fontsize=16)
            ax10.set_ylabel("\u03BC (-)", fontsize=16)
            ax11.set_ylabel("$W_I$ (mm/mm)", fontsize=16)
            ax12.set_ylabel("$K_W$ (-)", fontsize=16)
            ax13.set_ylabel("alpha (-)", fontsize=16)
            ax14.set_ylabel("AWC (mm)", fontsize=16)
            ax15.set_ylabel("theta (-)", fontsize=16)
            ax16.set_ylabel("alpha in PT (-)", fontsize=16)
            ax17.set_ylabel("meltRate_temp \n (mm $\u00b0C^{-1} hr^{-1}$)", fontsize=16)
            ax18.set_ylabel("meltRate_netrad \n (mm $MJ^{-1} hr^{-1}$)", fontsize=16)
            ax19.set_ylabel("$sn_a$ (-)", fontsize=16)

            fig.delaxes(ax20)  # delete extra subplot

            # set the x axis labels of subplots
            for ax in fig.axes:
                ax.set_xlabel("PFT", fontsize=16)
                ax.set_xticks(pft_arr)
                ax.tick_params(axis="x", labelrotation=45, labelsize=16)

            # remove white space between subplots
            fig.tight_layout()

            # save the plot
            fig.savefig(os.path.join(fig_path, "optim_param_hist.png"))

            plt.close("all")
        ###########################################################################

    ########## PLOT OPTIMIZED PARAMETERS (GLOBAL OPTIMIZATION) ##################
    elif model_settings["opti_type"] == "global_opti":
        ncols = 5  # set no of columns
        # determine no of rows based on total no of parameters
        nrows = math.ceil(len(param_name_arr) / 5)  # type: ignore

        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 12))

        # y axis label based on model
        if model_settings["model_name"] == "P_model":
            ylabels = [
                "Acclimation window (days)",
                "$W_I$ (mm/mm)",
                "$K_W$ (-)",
                "alpha (-)",
                "AWC (mm)",
                "theta (-)",
                "alpha in PT (-)",
                "meltRate_temp \n (mm $\u00b0C^{-1} hr^{-1}$)",
                "meltRate_netrad \n (mm $MJ^{-1} hr^{-1}$)",
                "sn_a (-)",
            ]
        elif model_settings["model_name"] == "LUE_model":
            ylabels = [
                "$LUE_{max}$ \n (\u03bcmol C \u03bcmol photons$^{-1}$)",
                "$T_{opt}$ ($\u00B0C$)",
                "$k_T$ ($\u00B0C^{-1}$)",
                "$\u03B1_{fT}$ (-)",
                "\u03BA ($Pa^{-1}$)",
                "$C_{a0}$ (ppm)",
                "$C_{\u03BA}$ (-)",
                "$C_m$ (ppm)",
                "\u03B3 (\u03bcmol photons$^{-1}$ m$^2$s)",
                "\u03BC (-)",
                "$W_I$ (mm/mm)",
                "$K_W$ (-)",
                "alpha (-)",
                "AWC (mm)",
                "theta (-)",
                "alpha in PT (-)",
                "meltRate_temp \n (mm $\u00b0C^{-1} hr^{-1}$)",
                "meltRate_netrad \n (mm $MJ^{-1} hr^{-1}$)",
                "$sn_a$ (-)",
            ]

        # add data to the plot
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx < len(param_name_arr):  # type: ignore
                    axs[i, j].bar(
                        param_name_arr[idx], # type: ignore
                        param_val_arr[idx],  # type: ignore
                        color="#714697",
                        alpha=0.7,
                    )
                    axs[i, j].set_xticks([])
                    axs[i, j].set_ylabel(ylabels[idx], fontsize=16)  # type: ignore

        if model_settings["model_name"] == "LUE_model":
            fig.delaxes(axs[3, 4]) # delete extra subplot

        # save the figure
        fig.tight_layout()
        plt.savefig(os.path.join(fig_path, "optim_param_hist.png"))
        plt.close("all")
