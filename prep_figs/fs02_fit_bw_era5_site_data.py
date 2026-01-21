#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

author: rde
first created: Thu Dec 11 2025 17:18:03 CET
"""
from pathlib import Path
import os
import glob
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"  # use amsmath font
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


def energy_flux_conversion(var_arr):
    """
    Convert unit of energy fluxes from MJ/m2/hr to W/m2
    1 MJ = 1e6 J
    1 sec = (1/3600) hr
    1 J/m2.s = 1 W/m2
    """
    return var_arr * 1e6 * (1.0 / 3600.0)


def nnse(obs, sim):
    """
    Calculate the Normalized Nash-Sutcliffe Efficiency (NSE)

    Parameters
    ----------
    obs (np.ndarray): Observed values
    sim (np.ndarray): Simulated values

    Returns
    -------
    nnse (float): Normalized NSE value
    """

    nse = 1.0 - np.nansum((obs - sim) ** 2.0) / np.nansum(
        (obs - np.nanmean(obs)) ** 2.0
    )
    nnse = 1.0 / (2.0 - nse)

    return nnse


def calc_era5_vs_original_meteo_perform(
    nc_filename_list, nc_filename_list_era5, var_name_list
):

    collect_nnse = {}
    for var_name in var_name_list:
        collect_nnse[var_name] = []

    # collect_nnse["nnse_p_daily_sum"] = []

    for nc_file in nc_filename_list:

        ds = xr.open_dataset(nc_file)

        site_id = ds.Site_id

        ds_era5 = xr.open_dataset(
            os.path.join(nc_filename_list_era5, f"{site_id}.1989.2020.hourly.nc")
        )

        time_mask = np.isin(
            ds_era5["time"].values.reshape(-1), ds["time"].values.reshape(-1)
        )
        indices_in_ds = np.nonzero(time_mask)[0]

        for var_name in var_name_list:
            var_era5_arr = ds_era5[f"{var_name}_ERA5"].values.reshape(-1)[indices_in_ds]

            if var_name in ["SW_IN", "NETRAD"]:
                var_era5_arr = energy_flux_conversion(var_era5_arr)

            var_orig_arr = ds[f"{var_name}_GF"].values.reshape(-1)
            var_fill_flag = ds[f"{var_name}_FILL_FLAG"].values.reshape(-1)

            var_orig_arr_good_data = var_orig_arr[var_fill_flag == 0]
            var_era5_arr_good_data = var_era5_arr[var_fill_flag == 0]

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    nnse_site = nnse(var_orig_arr_good_data, var_era5_arr_good_data)
            except RuntimeWarning as e:
                print(f"Error calculating NNSE for {site_id} - {var_name}: {e}")
                nnse_site = np.nan
            collect_nnse[var_name].append(nnse_site)

    for k, v in collect_nnse.items():
        collect_nnse[k] = np.array(v)

    return collect_nnse


def plot_axs(ax, data, var_label):

    data = data[~np.isnan(data)]

    if var_label != r"$P$ -- daily sum":
        sns.histplot(
            x=data,
            stat="percent",
            kde=True,
            binrange=(0, 1.0),
            ax=ax,
            color="white",
            edgecolor="white",
        )

        ax.lines[0].set_color("#56B4E9")

    ax.axvline(x=np.median(data), linestyle=":", color="#56B4E9")

    ax.set_xticks(np.linspace(0.0, 1.0, 3))
    ax.set_xticklabels([round(x, 1) for x in np.linspace(0.0, 1.0, 3).tolist()])
    ax.tick_params(axis="x")
    ax.tick_params(axis="both", which="major", labelsize=26.0)

    ax.text(0.1, 1.8, f"{round(np.median(data),3)}", fontsize=20)

    ax.set_ylabel("")
    ax.set_title(f"{var_label} ({len(data)})", size=30)

    sns.despine(ax=ax, top=True, right=True)


def set_ticks_for_selected_subplots(axs, selected_indices):
    """
    hide subplot x axis ticks and only enable for given subplots
    """

    # Hide x-axis ticks for all subplots
    for row in axs:
        for ax in row:
            ax.tick_params(
                axis="x", which="both", top=False, labelbottom=False  # bottom=False
            )

    # Enable x-axis ticks for selected subplots
    for i, j in selected_indices:
        axs[i][j].tick_params(
            axis="x", which="both", bottom=True, top=False, labelbottom=True
        )


def plot_nnse_dist(nnse_dict):

    fig, axs = plt.subplots(
        ncols=3,
        nrows=2,
        figsize=(16, 9),
        sharex=True,
        sharey=True,
    )

    plot_axs(axs[0, 0], nnse_dict["SW_IN"], r"$\mathit{SW\_IN}$")
    plot_axs(axs[0, 1], nnse_dict["NETRAD"], r"$\mathit{NETRAD}$")
    plot_axs(axs[0, 2], nnse_dict["TA"], r"$T$")
    plot_axs(axs[1, 0], nnse_dict["VPD"], r"$\mathit{VPD}$")
    plot_axs(axs[1, 1], nnse_dict["P"], r"$P$")
    fig.delaxes(axs[1, 2])  # remove unused axis

    set_ticks_for_selected_subplots(axs, selected_indices=[(0, 2), (1, 0), (1, 1)])

    fig.supxlabel(
        "NNSE [-]",
        y=-0.02,
        fontsize=36,
    )
    fig.supylabel(r"Fraction of sites [\%]", x=0.06, fontsize=36)

    plt.savefig(
        "fs02_fit_bw_in_situ_and_era5.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        "fs02_fit_bw_in_situ_and_era5.pdf", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    data_path = Path(
        "/path/to/fluxnet/hourly/data/"
    )  # FLUXNET2015

    fluxnet_data_path_with_era5 = Path(
        (
            "/path/to/fluxnet/hourly/data/with_era5_meteo/"
        )
    )

    # get list of files
    filename_list = glob.glob(os.path.join(data_path, "*.hourly_for_PModel.nc"))
    filename_list = sorted(filename_list, key=str.lower)

    ##################METEO VS ERA5 match#############################################
    nnse_dict = calc_era5_vs_original_meteo_perform(
        filename_list,
        fluxnet_data_path_with_era5,
        ["TA", "SW_IN", "NETRAD", "VPD", "P"],
    )

    plot_nnse_dist(nnse_dict)
