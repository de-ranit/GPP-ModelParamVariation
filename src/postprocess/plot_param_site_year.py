#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot optimized parameters for each site year for a 
particular site

author: rde
first created: 2023-12-22
"""

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_opti_param_site_year(result_dict, settings_dict):
    """
    Plot values of optimized parameters for each site year
    for a particular site

    parameters:
    result_dict (dict): dictionary with results from forward run
    settings_dict (dict): experiment settings

    returns:
    None (create and save the site year parameters plots)
    """
    # get the optimized parameters
    opti_params_dict = result_dict["Opti_par_val"]

    # collect the values of optimized parameters for each site year in arrays
    # common parameters
    site_year_arr = np.array(list(opti_params_dict.keys()))
    w_i_arr = np.zeros(len(site_year_arr))
    k_w_arr = np.zeros(len(site_year_arr))
    alpha_arr = np.zeros(len(site_year_arr))
    awc_arr = np.zeros(len(site_year_arr))
    theta_arr = np.zeros(len(site_year_arr))
    alpha_pt_arr = np.zeros(len(site_year_arr))
    meltrate_temp_arr = np.zeros(len(site_year_arr))
    meltrate_netrad_arr = np.zeros(len(site_year_arr))
    sn_a_arr = np.zeros(len(site_year_arr))
    if settings_dict["model_name"] == "P_model":  # P Model specific parameter
        acclim_window_arr = np.zeros(len(site_year_arr))
    elif settings_dict["model_name"] == "LUE_model":  # LUE Model specific parameters
        lue_max_arr = np.zeros(len(site_year_arr))
        t_opt_arr = np.zeros(len(site_year_arr))
        k_t_arr = np.zeros(len(site_year_arr))
        alpha_ft_horn_arr = np.zeros(len(site_year_arr))
        kappa_vpd_arr = np.zeros(len(site_year_arr))
        ca_zero_arr = np.zeros(len(site_year_arr))
        c_kappa_arr = np.zeros(len(site_year_arr))
        c_m_arr = np.zeros(len(site_year_arr))
        gamma_fl_tal_arr = np.zeros(len(site_year_arr))
        mu_fci_arr = np.zeros(len(site_year_arr))
    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {settings_dict['model_name']}"
            "is not implemented"
        )

    # initialize the flags to delete the subplots
    del_alpha_plot = False
    del_alpha_ft_horn_plot = False

    for ix, site_year in enumerate(site_year_arr):
        if settings_dict["model_name"] == "P_model":  # P Model specific parameter
            acclim_window_arr[ix] = opti_params_dict[site_year]["acclim_window"]  # type: ignore
        elif (
            settings_dict["model_name"] == "LUE_model"
        ):  # LUE Model specific parameters
            lue_max_arr[ix] = opti_params_dict[site_year]["LUE_max"]  # type: ignore
            t_opt_arr[ix] = opti_params_dict[site_year]["T_opt"]  # type: ignore
            k_t_arr[ix] = opti_params_dict[site_year]["K_T"]  # type: ignore
            try:
                alpha_ft_horn_arr[ix] = opti_params_dict[site_year]["alpha_fT_Horn"]  # type: ignore
            except (
                KeyError
            ):  # if alpha_ft_horn is not optimized, then it is not in the dictionary
                alpha_ft_horn_arr[ix] = np.nan  # type: ignore
                del_alpha_ft_horn_plot = True  # delete the alpha_ft_horn subplot
            kappa_vpd_arr[ix] = opti_params_dict[site_year]["Kappa_VPD"]  # type: ignore
            ca_zero_arr[ix] = opti_params_dict[site_year]["Ca_0"]  # type: ignore
            c_kappa_arr[ix] = opti_params_dict[site_year]["C_Kappa"]  # type: ignore
            c_m_arr[ix] = opti_params_dict[site_year]["c_m"]  # type: ignore
            gamma_fl_tal_arr[ix] = opti_params_dict[site_year]["gamma_fL_TAL"]  # type: ignore
            mu_fci_arr[ix] = opti_params_dict[site_year]["mu_fCI"]  # type: ignore

        # common parameters
        w_i_arr[ix] = opti_params_dict[site_year]["W_I"]
        k_w_arr[ix] = opti_params_dict[site_year]["K_W"]
        try:
            alpha_arr[ix] = opti_params_dict[site_year]["alpha"]
        except KeyError:  # if alpha is not optimized, then it is not in the dictionary
            alpha_arr[ix] = np.nan
            del_alpha_plot = True  # delete the alpha subplot
        awc_arr[ix] = opti_params_dict[site_year]["AWC"]
        theta_arr[ix] = opti_params_dict[site_year]["theta"]
        alpha_pt_arr[ix] = opti_params_dict[site_year]["alphaPT"]
        meltrate_temp_arr[ix] = opti_params_dict[site_year]["meltRate_temp"]
        meltrate_netrad_arr[ix] = opti_params_dict[site_year]["meltRate_netrad"]
        sn_a_arr[ix] = opti_params_dict[site_year]["sn_a"]

    # plot the values of optimized parameters for each site year
    if settings_dict["model_name"] == "P_model":  # P Model specific parameters
        fig, (
            (ax1, ax2),
            (ax3, ax4),
            (ax5, ax6),
            (ax7, ax8),
            (ax9, ax10),
        ) = plt.subplots(nrows=5, ncols=2, figsize=(30, 20))

        # add data to the plot
        ax1.bar(site_year_arr, acclim_window_arr, color="#714697", alpha=0.7)  # type: ignore
        ax2.bar(site_year_arr, w_i_arr, color="#714697", alpha=0.7)
        ax3.bar(site_year_arr, k_w_arr, color="#714697", alpha=0.7)
        ax4.bar(site_year_arr, alpha_arr, color="#714697", alpha=0.7)
        ax5.bar(site_year_arr, awc_arr, color="#714697", alpha=0.7)
        ax6.bar(site_year_arr, theta_arr, color="#714697", alpha=0.7)
        ax7.bar(site_year_arr, alpha_pt_arr, color="#714697", alpha=0.7)
        ax8.bar(site_year_arr, meltrate_temp_arr, color="#714697", alpha=0.7)
        ax9.bar(site_year_arr, meltrate_netrad_arr, color="#714697", alpha=0.7)
        ax10.bar(site_year_arr, sn_a_arr, color="#714697", alpha=0.7)

        # set the titles of subplots
        ax1.set_title("Acclimation window", fontsize=16)
        ax2.set_title("Optimal soil moisture", fontsize=16)
        ax3.set_title("Sensitivity to soil moisture changes", fontsize=16)
        ax4.set_title("Lag parameter for soil moisture effect", fontsize=16)
        ax5.set_title("Available water capacity", fontsize=16)
        ax6.set_title("Decay of water bucket", fontsize=16)
        ax7.set_title("Multiplication factor for PET", fontsize=16)
        ax8.set_title("Snow melt rate for temp", fontsize=16)
        ax9.set_title("Snow melt rate for net radiation", fontsize=16)
        ax10.set_title("Sublimation resistance", fontsize=16)

        # set the y axis labels of subplots
        ax1.set_ylabel("Acclimation window (days)", fontsize=16)
        ax2.set_ylabel("$W_I$ (mm $mm^{-1}$)", fontsize=16)
        ax3.set_ylabel("$K_W$ (-)", fontsize=16)
        ax4.set_ylabel("alpha (-)", fontsize=16)
        ax5.set_ylabel("AWC (mm)", fontsize=16)
        ax6.set_ylabel("theta (-)", fontsize=16)
        ax7.set_ylabel("alpha in PT (-)", fontsize=16)
        ax8.set_ylabel("meltRate_temp \n (mm $\u00b0C^{-1} hr^{-1}$)", fontsize=16)
        ax9.set_ylabel("meltRate_netrad \n (mm $MJ^{-1} hr^{-1}$)", fontsize=16)
        ax10.set_ylabel("$sn_a$ (-)", fontsize=16)

        # set the x axis labels of subplots
        for ax in fig.axes:
            ax.set_xlabel("Site year", fontsize=16)
            ax.set_xticks(site_year_arr)
            ax.tick_params(axis="x", labelrotation=45, labelsize=16)

        # set the title of the plot as SiteID
        fig.suptitle(f"{result_dict['SiteID']}", fontsize=20)
        # fig.text(0.5, 0.005, "Site year", ha="center", fontsize=20)

        # delete the alpha subplot if alpha is not optimized
        if del_alpha_plot:
            fig.delaxes(ax4)

        # remove white space between subplots
        fig.tight_layout()

    elif settings_dict["model_name"] == "LUE_model":  # LUE Model specific parameters
        fig, (
            (ax1, ax2),
            (ax3, ax4),
            (ax5, ax6),
            (ax7, ax8),
            (ax9, ax10),
            (ax11, ax12),
            (ax13, ax14),
            (ax15, ax16),
            (ax17, ax18),
            (ax19, ax20),
        ) = plt.subplots(nrows=10, ncols=2, figsize=(30, 40))

        # add data to the plot
        ax1.bar(site_year_arr, lue_max_arr, color="#714697", alpha=0.7)  # type: ignore
        ax2.bar(site_year_arr, t_opt_arr, color="#714697", alpha=0.7)  # type: ignore
        ax3.bar(site_year_arr, k_t_arr, color="#714697", alpha=0.7)  # type: ignore
        ax4.bar(site_year_arr, alpha_ft_horn_arr, color="#714697", alpha=0.7)  # type: ignore
        ax5.bar(site_year_arr, kappa_vpd_arr, color="#714697", alpha=0.7)  # type: ignore
        ax6.bar(site_year_arr, ca_zero_arr, color="#714697", alpha=0.7)  # type: ignore
        ax7.bar(site_year_arr, c_kappa_arr, color="#714697", alpha=0.7)  # type: ignore
        ax8.bar(site_year_arr, c_m_arr, color="#714697", alpha=0.7)  # type: ignore
        ax9.bar(site_year_arr, gamma_fl_tal_arr, color="#714697", alpha=0.7)  # type: ignore
        ax10.bar(site_year_arr, mu_fci_arr, color="#714697", alpha=0.7)  # type: ignore
        ax11.bar(site_year_arr, w_i_arr, color="#714697", alpha=0.7)
        ax12.bar(site_year_arr, k_w_arr, color="#714697", alpha=0.7)
        ax13.bar(site_year_arr, alpha_arr, color="#714697", alpha=0.7)
        ax14.bar(site_year_arr, awc_arr, color="#714697", alpha=0.7)
        ax15.bar(site_year_arr, theta_arr, color="#714697", alpha=0.7)
        ax16.bar(site_year_arr, alpha_pt_arr, color="#714697", alpha=0.7)
        ax17.bar(site_year_arr, meltrate_temp_arr, color="#714697", alpha=0.7)
        ax18.bar(site_year_arr, meltrate_netrad_arr, color="#714697", alpha=0.7)
        ax19.bar(site_year_arr, sn_a_arr, color="#714697", alpha=0.7)

        # set the titles of subplots
        ax1.set_title("Maximum light use efficiency", fontsize=16)
        ax2.set_title("Optimal temperature", fontsize=16)
        ax3.set_title("Sensitivity to temperature changes", fontsize=16)
        ax4.set_title("Lag parameter for temperature effect", fontsize=16)
        ax5.set_title("Sensitivity to VPD changes", fontsize=16)
        ax6.set_title("Minimum optimal atmospheric CO$_2$ concentration", fontsize=16)
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
        ax11.set_ylabel("$W_I$ (mm $mm^{-1}$)", fontsize=16)
        ax12.set_ylabel("$K_W$ (-)", fontsize=16)
        ax13.set_ylabel("alpha (-)", fontsize=16)
        ax14.set_ylabel("AWC (mm)", fontsize=16)
        ax15.set_ylabel("theta (-)", fontsize=16)
        ax16.set_ylabel("alpha in PT (-)", fontsize=16)
        ax17.set_ylabel("meltRate_temp \n (mm $\u00b0C^{-1} hr^{-1}$)", fontsize=16)
        ax18.set_ylabel("meltRate_netrad \n (mm $MJ^{-1} hr^{-1}$)", fontsize=16)
        ax19.set_ylabel("$sn_a$ (-)", fontsize=16)

        # set the x axis labels of subplots
        for ax in fig.axes:
            ax.set_xlabel("Site year", fontsize=16)
            ax.set_xticks(site_year_arr)
            ax.tick_params(axis="x", labelrotation=45, labelsize=16)

        fig.tight_layout()  # remove white space between subplots
        fig.subplots_adjust(top=0.97)  # add some white space at the top
        # add SiteID as title
        fig.text(
            0.5, 0.98, f"{result_dict['SiteID']}", va="center", ha="center", fontsize=20
        )

        # delete the alpha subplot if alpha was not optimized
        if del_alpha_ft_horn_plot:
            fig.delaxes(ax4)

        # delete the alpha_fT_Horn subplot if it was not optimized
        if del_alpha_plot:
            fig.delaxes(ax13)

        fig.delaxes(ax20)  # delete extra subplot

    # save the plot
    site_year_param_fig_path = Path(
        "model_results",
        settings_dict["model_name"],
        settings_dict["exp_name"],
        "site_level_plots",
        "site_year_params_plots",
    )
    os.makedirs(site_year_param_fig_path, exist_ok=True)
    site_year_param_fig_path_filename = os.path.join(
        site_year_param_fig_path,
        f"{result_dict['SiteID']}_optim_param_per_site_year.png",
    )
    plt.savefig(site_year_param_fig_path_filename)

    # close the plot
    plt.close("all")
