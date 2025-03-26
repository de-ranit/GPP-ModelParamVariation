#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
module to plot site level results

author: rde
first created: 2023-11-10
"""

import os
from pathlib import Path
import bottleneck as bn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


def plot_site_level_results(result_dict, ip_df_dict, settings_dict):
    """
    make timersies of obs vs. sim GPP (both sub-daily and daily scale)
    and ET (only subdaily) for each site

    parameters:
    result_dict (dict): dictionary with results from forward run of the model
    ip_df_dict (dict): dictionary with input forcing data
    settings_dict (dict): dictionary with experiment settings

    returns:
    create and save the site level timeseries plots
    """

    if settings_dict["model_name"] == "P_model":
        # replace the 1.0 (data used in cost func) with
        # max value of GPP and ET (to plot them as bars)
        gpp_drop_idx = result_dict[f"GPP_drop_idx_{result_dict['Temp_res']}"]
        max_gpp_val = max(
            bn.nanmax(result_dict[f"GPP_NT_{result_dict['Temp_res']}"]),
            bn.nanmax(result_dict[f"GPP_sim_{result_dict['Temp_res']}"]),
            bn.nanmax(result_dict[f"GPP_sim_no_moisture_{result_dict['Temp_res']}"]),
        )
        gpp_used_data_idx = np.where(gpp_drop_idx == 0.0, max_gpp_val, 0.0)

        max_gpp_d_val = max(
            bn.nanmax(result_dict["GPP_NT_daily"]),
            bn.nanmax(result_dict["GPP_sim_daily"]),
            bn.nanmax(result_dict["GPP_sim_no_moisture_daily"]),
        )
        gpp_d_used_data_idx = np.where(result_dict['good_gpp_d_idx'] == 1.0, max_gpp_d_val, 0.0)

        nse_sd_gpp_no_stress = round(
            result_dict["NSE_no_moisture_Stress"][f"NSE_{result_dict['Temp_res']}"], 3
        )
        r2_sd_gpp_no_stress = round(
            result_dict["R2_no_moisture_Stress"][f"R2_{result_dict['Temp_res']}"], 3
        )
        rmse_sd_gpp_no_stress = round(
            result_dict["RMSE_no_moisture_Stress"][f"RMSE_{result_dict['Temp_res']}"], 3
        )

        nse_d_gpp_no_stress = round(result_dict["NSE_no_moisture_Stress"]["NSE_d"], 3)
        r2_d_gpp_no_stress = round(result_dict["R2_no_moisture_Stress"]["R2_d"], 3)
        rmse_d_gpp_no_stress = round(
            result_dict["RMSE_no_moisture_Stress"]["RMSE_d"], 3
        )

    elif settings_dict["model_name"] == "LUE_model":
        # replace the 1.0 (data used in cost func)
        # with max value of GPP and ET (to plot them as bars)
        gpp_drop_idx = result_dict[f"GPP_drop_idx_{result_dict['Temp_res']}"]
        max_gpp_val = max(
            bn.nanmax(result_dict[f"GPP_NT_{result_dict['Temp_res']}"]),
            bn.nanmax(result_dict[f"GPP_sim_{result_dict['Temp_res']}"]),
        )
        gpp_used_data_idx = np.where(gpp_drop_idx == 0.0, max_gpp_val, 0.0)

        max_gpp_d_val = max(
            bn.nanmax(result_dict["GPP_NT_daily"]),
            bn.nanmax(result_dict["GPP_sim_daily"]),
        )
        gpp_d_used_data_idx = np.where(result_dict['good_gpp_d_idx'] == 1.0, max_gpp_d_val, 0.0)
    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {settings_dict['model_name']}"
            "is not implemented"
        )

    et_drop_idx = result_dict[f"ET_drop_idx_{result_dict['Temp_res']}"]
    max_et_val = max(
        bn.nanmax(result_dict[f"ET_{result_dict['Temp_res']}"]),
        bn.nanmax(result_dict[f"ET_sim_{result_dict['Temp_res']}"]),
    )
    et_used_data_idx = np.where(et_drop_idx == 0.0, max_et_val, 0.0)

    # get model perfromance metrices
    nse_sd = round(result_dict["NSE"][f"NSE_{result_dict['Temp_res']}"], 3)
    r2_sd = round(result_dict["R2"][f"R2_{result_dict['Temp_res']}"], 3)
    rmse_sd = round(result_dict["RMSE"][f"RMSE_{result_dict['Temp_res']}"], 3)

    nse_d = round(result_dict["NSE"]["NSE_d"], 3)
    r2_d = round(result_dict["R2"]["R2_d"], 3)
    rmse_d = round(result_dict["RMSE"]["RMSE_d"], 3)

    et_nse_sd = round(
        result_dict["ET_model_performance"][f"ET_NSE_{result_dict['Temp_res']}"], 3
    )
    et_r2_sd = round(
        result_dict["ET_model_performance"][f"ET_R2_{result_dict['Temp_res']}"], 3
    )
    et_rmse_sd = round(
        result_dict["ET_model_performance"][f"ET_RMSE_{result_dict['Temp_res']}"], 3
    )

    ############################################################################
    # # create a plot with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
    # set title with site info
    fig.suptitle(
        (
            f"SiteID: {result_dict['SiteID']},"
            f"PFT: {result_dict['PFT']},"
            f"KG: {result_dict['KG']},"
            f"avg. temp.: {result_dict['avg_temp']} \u00B0C, \n"
            f"avg. prec.: {result_dict['avg_precip']} {ip_df_dict['prec_unit']},"
            f"AI: {result_dict['arid_ind']}"
        ),
        x=0.5,
        y=0.98,
        fontsize=30,
    )
    fig.subplots_adjust(hspace=0.3)  # adjust the space between subplots

    if result_dict["Temp_res"] == "Daily":
        # create an empty subplot for subdaily GPP when using daily data
        ax1.text(
            0.5,
            0.5,
            "No sub-daily plot in case of model optim/run \nusing daily data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax1.transAxes,
            fontsize=22,
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
    else:
        # plot GPP subdaily
        ax1.bar(
            result_dict[f"Time_{result_dict['Temp_res']}"],
            gpp_used_data_idx,
            color="#FFD900",
            alpha=0.03,
        )  # data used in cost function and to evaluate model performance
        ax1.scatter(
            result_dict[f"Time_{result_dict['Temp_res']}"],
            result_dict[f"GPP_NT_{result_dict['Temp_res']}"],
            c="#F8766D",
            s=1,
            alpha=0.7,
        )  # observed GPP
        if settings_dict["model_name"] == "P_model":
            ax1.scatter(
                result_dict[f"Time_{result_dict['Temp_res']}"],
                result_dict[f"GPP_sim_no_moisture_{result_dict['Temp_res']}"],
                c="#017BFC",
                s=1,
                alpha=0.5,
            )  # simulated GPP without moisture stress
        ax1.scatter(
            result_dict[f"Time_{result_dict['Temp_res']}"],
            result_dict[f"GPP_sim_{result_dict['Temp_res']}"],
            c="#4BA34A",
            s=1,
            alpha=0.6,
        )  # simulated GPP
        ax1.set_ylabel("GPP (\u03bcmol C $m^{-2}$ $s^{-1}$)", fontsize=20)
        ax1.set_xlabel(f"Time[{result_dict['Temp_res']}]", fontsize=20)
        if settings_dict["model_name"] == "P_model":
            ax1.set_title(
                (
                    f"NSE: {nse_sd},"
                    f"R$^2$: {r2_sd},"
                    f"RMSE: {rmse_sd} \n"
                    f"NSE_no_moisture: {nse_sd_gpp_no_stress},"  # type: ignore
                    f"R$^2$_no_moisture: {r2_sd_gpp_no_stress},"  # type: ignore
                    f"RMSE_no_moisture: {rmse_sd_gpp_no_stress}"  # type: ignore
                ),
                size=22,
            )
        else:
            ax1.set_title(
                (f"NSE: {nse_sd}," f"R$^2$: {r2_sd}," f"RMSE: {rmse_sd}"),
                size=22,
            )
        ax1.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y")
        )  # set the x-axis label as year
        ax1.xaxis.set_major_locator(
            mdates.YearLocator(base=1)
        )  # set the x-axis ticks as every 2 years
        ax1.tick_params(
            axis="x", which="major", labelsize=18.0, labelrotation=45.0
        )  # set the x-axis tick label font size and rotation
        ax1.tick_params(
            axis="y", which="major", labelsize=18.0
        )  # set the y-axis tick label font size

    # plot daily GPP
    if result_dict["Temp_res"] == "Daily":
        # add the data quality flag in case of using daily input data
        ax2.bar(
            result_dict[f"Time_{result_dict['Temp_res']}"],
            gpp_used_data_idx,
            color="#FFD900",
            alpha=0.7,
        )  # data used in cost function and to evaluate model performance
    else:
        ax2.bar(
            result_dict["Time_daily"],
            gpp_d_used_data_idx,
            color="#FFD900",
            alpha=0.7,
        )
    ax2.scatter(
        result_dict["Time_daily"],
        result_dict["GPP_NT_daily"],
        c="#F8766D",
        s=3,
        # alpha=0.7,
    )  # observed daily GPP
    if settings_dict["model_name"] == "P_model":
        ax2.scatter(
            result_dict["Time_daily"],
            result_dict["GPP_sim_no_moisture_daily"],
            c="#017BFC",
            s=3,
            # alpha=0.7,
        )  # simulated daily GPP without moisture stress
    ax2.scatter(
        result_dict["Time_daily"],
        result_dict["GPP_sim_daily"],
        c="#4BA34A",
        s=3,
        # alpha=0.7,
    )  # simulated daily GPP
    ax2.set_ylabel("GPP (\u03bcmol C $m^{-2}$ $s^{-1}$)", fontsize=20)
    ax2.set_xlabel("Time[Daily]", fontsize=20)
    if settings_dict["model_name"] == "P_model":
        ax2.set_title(
            (
                f"NSE: {nse_d},"
                f"R$^2$: {r2_d},"
                f"RMSE: {rmse_d} \n"
                f"NSE_no_moisture: {nse_d_gpp_no_stress},"  # type: ignore
                f"R$^2$_no_moisture: {r2_d_gpp_no_stress},"  # type: ignore
                f"RMSE_no_moisture: {rmse_d_gpp_no_stress}"  # type: ignore
            ),
            size=22,
        )
    else:
        ax2.set_title(
            (f"NSE: {nse_d}," f"R$^2$: {r2_d}," f"RMSE: {rmse_d}"),
            size=22,
        )

    ax2.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )  # set the x-axis label as year
    ax2.xaxis.set_major_locator(
        mdates.YearLocator(base=1)
    )  # set the x-axis ticks as every 2 years
    ax2.tick_params(
        axis="x", which="major", labelsize=18.0, labelrotation=45.0
    )  # set the x-axis tick label font size and rotation
    ax2.tick_params(
        axis="y", which="major", labelsize=18.0
    )  # set the y-axis tick label font size

    # plot subdaily ET
    if result_dict["Temp_res"] == "Daily":
        bar_transparency = 0.7
        marker_size = 3
    else:
        bar_transparency = 0.03
        marker_size = 1
    ax3.bar(
        result_dict[f"Time_{result_dict['Temp_res']}"],
        et_used_data_idx,
        color="#FFD900",
        alpha=bar_transparency,
    )  # data used in cost function and to evaluate model performance
    ax3.scatter(
        result_dict[f"Time_{result_dict['Temp_res']}"],
        result_dict[f"ET_{result_dict['Temp_res']}"],
        c="#F8766D",
        s=marker_size,
        alpha=0.7,
    )
    ax3.scatter(
        result_dict[f"Time_{result_dict['Temp_res']}"],
        result_dict[f"ET_sim_{result_dict['Temp_res']}"],
        c="#4BA34A",
        s=marker_size,
        alpha=0.7,
    )
    ax3.set_ylabel("ET (mm $hr^{-1}$)", fontsize=20)
    ax3.set_xlabel(f"Time[{result_dict['Temp_res']}]", fontsize=20)
    ax3.set_title(f"NSE: {et_nse_sd}, R$^2$: {et_r2_sd}, RMSE: {et_rmse_sd}", size=22)
    ax3.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )  # set the x-axis label as year
    ax3.xaxis.set_major_locator(
        mdates.YearLocator(base=1)
    )  # set the x-axis ticks as every 2 years
    ax3.tick_params(
        axis="x", which="major", labelsize=18.0, labelrotation=45.0
    )  # set the x-axis tick label font size and rotation
    ax3.tick_params(
        axis="y", which="major", labelsize=18.0
    )  # set the y-axis tick label font size

    # create a legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="observed",
            markerfacecolor="#F8766D",
            markersize=20,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="simulated",
            markerfacecolor="#4BA34A",
            markersize=20,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="data used in cost function/ model evaluation",
            markerfacecolor="#FFD900",
            markersize=20,
        ),
    ]

    if settings_dict["model_name"] == "P_model":
        legend_elements.insert(
            2,
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="simulated without considering moisture stress",
                markerfacecolor="#017BFC",
                markersize=20,
            ),
        )

    fig.delaxes(ax4)  # delete the empty subplot

    plt.legend(
        handles=legend_elements,
        title="Legend",
        title_fontsize="26",
        bbox_to_anchor=(1.2, 0.5),
        fontsize="22",
        loc="upper left",
    )  # add legend

    # plt.tight_layout() # remove extra white spaces

    # save the plot
    site_fig_path = Path(
        "model_results",
        settings_dict["model_name"],
        settings_dict["exp_name"],
        "site_level_plots",
        "gpp_et_timeseries",
    )
    os.makedirs(site_fig_path, exist_ok=True)
    site_fig_path_filename = os.path.join(
        site_fig_path, f"{result_dict['SiteID']}_gpp_et_timeseries.png"
    )
    plt.savefig(site_fig_path_filename)
    plt.close()

    # ################################################################################
    if settings_dict["model_name"] == "LUE_model":
        # plot the timeseries of sensitivity functions
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
            nrows=3, ncols=3, figsize=(30, 20)
        )
        # set title with site info
        fig.suptitle(f"SiteID: {result_dict['SiteID']}", x=0.5, y=0.99, fontsize=25)

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        var_to_plt = ["fT", "fVPD", "fVPD_part", "fCO2_part", "fL", "fCI", "fW_Horn"]

        for ax, var in zip(axes, var_to_plt):
            ax.scatter(
                result_dict[f"Time_{result_dict['Temp_res']}"],
                result_dict[f"{var}_{result_dict['Temp_res']}"],
                c="#F8766D",
                s=1,
                alpha=0.7,
            )  # plot fX timeseries
            ax.set_ylabel(f"{var} (-)", fontsize=20)
            if var == "fW_Horn":  # change the label for fW_Horn
                var = "fW"
            ax.set_xlabel(f"Time[{result_dict['Temp_res']}]", fontsize=20)
            ax.set_title(f"{var}", size=22)
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y")
            )  # set the x-axis label as year
            ax.xaxis.set_major_locator(
                mdates.YearLocator(base=1)
            )  # set the x-axis ticks as every 2 years
            ax.tick_params(
                axis="x", which="major", labelsize=18.0, labelrotation=45.0
            )  # set the x-axis tick label font size and rotation
            ax.tick_params(
                axis="y", which="major", labelsize=18.0
            )  # set the y-axis tick label font size

        # delete empty subplots
        fig.delaxes(ax8)
        fig.delaxes(ax9)

        fig.tight_layout()

        # save the plot
        fx_ts_fig_path = Path(
            "model_results",
            settings_dict["model_name"],
            settings_dict["exp_name"],
            "site_level_plots",
            "fx_timeseries",
        )
        os.makedirs(fx_ts_fig_path, exist_ok=True)
        fx_ts_fig_path_filename = os.path.join(
            fx_ts_fig_path, f"{result_dict['SiteID']}_fx_timeseries.png"
        )
        plt.savefig(fx_ts_fig_path_filename)
        plt.close()

        ############################################################################
        # plot x vs fx (T vs fT, VPD vs fVPD, etc.)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            nrows=2, ncols=3, figsize=(30, 20)
        )
        # set title with site info
        fig.suptitle(f"SiteID: {result_dict['SiteID']}", x=0.5, y=0.99, fontsize=25)

        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        var_to_plt = ["fT", "fVPD_part", "fCO2_part", "fL", "fCI", "fW_Horn"]
        env_var = ["TA_GF", "VPD_GF", settings_dict["CO2_var"]]
        units = ["degC", "Pa", "ppm"]

        for idx, ax in enumerate(axes):
            if var_to_plt[idx] == "fL":
                ax.scatter(
                    ip_df_dict[settings_dict["fPAR_var"]] * ip_df_dict["PPFD_IN_GF"],
                    result_dict[f"{var_to_plt[idx]}_{result_dict['Temp_res']}"],
                    c="#F8766D",
                    s=1,
                    alpha=0.7,
                )  # plot fL timeseries
                ax.set_xlabel(
                    "L (PPFD X fPAR) [\u03bcmol photons $m^{-2}$ $s^{-1}$]", fontsize=20
                )
            elif var_to_plt[idx] == "fCI":
                ax.scatter(
                    result_dict["ci"],
                    result_dict[f"{var_to_plt[idx]}_{result_dict['Temp_res']}"],
                    c="#F8766D",
                    s=1,
                    alpha=0.7,
                )  # plot fL timeseries
                ax.set_xlabel("CI [-]", fontsize=20)
            elif var_to_plt[idx] == "fW_Horn":
                ax.scatter(
                    result_dict[f"WAI_nor_{result_dict['Temp_res']}"],
                    result_dict[f"{var_to_plt[idx]}_{result_dict['Temp_res']}"],
                    c="#F8766D",
                    s=1,
                    alpha=0.7,
                )  # plot fL timeseries
                var_to_plt[idx] = "fW"
                ax.set_xlabel("Normalized WAI (-)", fontsize=20)
            else:
                ax.scatter(
                    ip_df_dict[env_var[idx]],
                    result_dict[f"{var_to_plt[idx]}_{result_dict['Temp_res']}"],
                    c="#F8766D",
                    s=1,
                    alpha=0.7,
                )  # plot fX timeseries
                ax.set_xlabel(f"{env_var[idx]} [{units[idx]}]", fontsize=20)

            ax.set_ylim(
                0.0, max(1.2, ax.get_ylim()[1])
            )  # set the y-axis limit to 0.0 to max value
            ax.set_ylabel(f"{var_to_plt[idx]} (-)", fontsize=20)
            ax.tick_params(
                axis="x", which="major", labelsize=18.0, labelrotation=45.0
            )  # set the x-axis tick label font size and rotation
            ax.tick_params(
                axis="y", which="major", labelsize=18.0
            )  # set the y-axis tick label font size

        fig.tight_layout()

        # save the plot
        fx_ts_fig_path = Path(
            "model_results",
            settings_dict["model_name"],
            settings_dict["exp_name"],
            "site_level_plots",
            "x_vs_fx",
        )
        os.makedirs(fx_ts_fig_path, exist_ok=True)
        fx_ts_fig_path_filename = os.path.join(
            fx_ts_fig_path, f"{result_dict['SiteID']}_x_vs_fx.png"
        )
        plt.savefig(fx_ts_fig_path_filename)
        plt.close()
