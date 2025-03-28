#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot model performance results of the P-model where a group of parameters
were varied per year, while the rest of the parameters were kept constant
for a site. The performance metrics are calculated at hourly and annual scales.

author: rde
first created: Wed Feb 19 2025 18:11:11 CET
"""

import sys
import os
from pathlib import Path
import importlib
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from permetrics import RegressionMetric


# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

from src.common.get_data import read_nc_data  # pylint: disable=C0413
from src.common.get_data import df_to_dict  # pylint: disable=C0413
from src.postprocess.prep_results import (  # pylint: disable=C0413
    calc_variability_metrics,
)  # pylint: disable=C0413
from src.postprocess.prep_results import calc_bias_metrics  # pylint: disable=C0413

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}"  # use amsmath package
)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


def get_mod_res_perform_arr(res_path, perform_metric):
    """
    get the model values of a given performance metric

    Parameters:
    -----------
    res_path (str) : path to the directory containing the model results
    perform_metric (str) : performance metric to extract

    Returns:
    --------
    perform_metric_yy (np.ndarray) : array of performance metric values at annual scale
    perform_metric_hr (np.ndarray) : array of performance metric values at hourly scale
    """

    # find all the files with serialized model results
    mod_res_file_list = glob.glob(f"{res_path}/*.npy")
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

    perform_metric_hr = np.zeros(len(filtered_mod_res_file_list))
    perform_metric_yy = np.zeros(len(filtered_mod_res_file_list))

    for ix, res_file in enumerate(filtered_mod_res_file_list):
        # open the results file for P Model
        res_dict = np.load(res_file, allow_pickle=True).item()
        # site_id = res_dict["SiteID"]

        perform_metric_yy[ix] = res_dict[perform_metric][f"{perform_metric}_y"]
        perform_metric_hr[ix] = res_dict[perform_metric][
            f"{perform_metric}_{res_dict['Temp_res']}"
        ]

    return perform_metric_yy, perform_metric_hr


def get_perform_arr_dt_nt(data_path, perform_metric):
    """
    calculate model performance metric between GPP_NT as obs and GPP_DT as sim

    Parameters:
    -----------
    data_path (str) : path to the directory containing the model results
    perform_metric (str) : performance metric to calculate

    Returns:
    --------
    perform_metric_yy (np.ndarray) : array of performance metric values at annual scale
    perform_metric_hr (np.ndarray) : array of performance metric values at hourly scale
    """

    # find all the files with serialized model results
    mod_data_file_list = glob.glob(f"{data_path}/*.nc")
    mod_data_file_list.sort()  # sort the files by site ID

    # filter out bad sites
    filtered_data_file_list = [
        files
        for files in mod_data_file_list
        if not (
            "CG-Tch" in files
            or "MY-PSO" in files
            or "GH-Ank" in files
            or "US-LWW" in files
        )
    ]

    perform_metric_hr = np.zeros(len(filtered_data_file_list))
    perform_metric_yy = np.zeros(len(filtered_data_file_list))

    for ix, data_nc_file in enumerate(filtered_data_file_list):

        ip_df_dict = read_nc_data(data_nc_file, {"fPAR_var": "FPAR_FLUXNET_EO"})

        gpp_nan_indices = np.isnan(ip_df_dict["GPP_NT"]) | np.isnan(
            ip_df_dict["GPP_DT"]
        )

        bad_gpp_nt_indices = (ip_df_dict["NEE_QC"] == 0.0) | (
            ip_df_dict["NEE_QC"] == 0.5
        )
        bad_gpp_dt_indices = (ip_df_dict["NEE_QC_DT_50"] == 0.0) | (
            ip_df_dict["NEE_QC_DT_50"] == 0.5
        )

        ip_df_dict["GPP_NT"] = np.where(
            (ip_df_dict["SW_IN_GF"] <= 0.0) & (ip_df_dict["GPP_NT"] < 0.0),
            0.0,
            ip_df_dict["GPP_NT"],
        )

        negative_obs_gpp_indices = (
            ip_df_dict["GPP_NT"] < 0.0
        )  # remove noisy gpp_obs which are negative during daytime

        drop_gpp_data_indices = (
            gpp_nan_indices
            | bad_gpp_nt_indices
            | bad_gpp_dt_indices
            | negative_obs_gpp_indices
        )

        gpp_nt = ip_df_dict["GPP_NT"][~drop_gpp_data_indices]
        gpp_dt = ip_df_dict["GPP_DT"][~drop_gpp_data_indices]

        gpp_resample_df = pd.DataFrame(
            {
                "Time": ip_df_dict["Time"],
                "GPP_obs_NT": ip_df_dict["GPP_NT"],
                "GPP_obs_DT": ip_df_dict["GPP_DT"],
                "drop_gpp_idx": ~drop_gpp_data_indices,
            }
        )

        gpp_df_y = gpp_resample_df.resample("Y", on="Time").mean()
        gpp_df_y = gpp_df_y.reset_index()
        gpp_y_dict = df_to_dict(gpp_df_y)

        filter_idx = gpp_y_dict["drop_gpp_idx"] < 0.5
        good_data_mask = ~(filter_idx)
        gpp_nt_y_filtered = gpp_y_dict["GPP_obs_NT"][good_data_mask]
        gpp_dt_y_filtered = gpp_y_dict["GPP_obs_DT"][good_data_mask]

        if perform_metric == "NSE":
            evaluator = RegressionMetric(gpp_nt, gpp_dt, decimal=5)
            perform_metric_hr[ix] = evaluator.NSE()

            if (gpp_nt_y_filtered.size < 3) or (gpp_dt_y_filtered.size < 3):
                perform_metric_yy[ix] = np.nan
            else:
                evaluator_y = RegressionMetric(
                    gpp_nt_y_filtered, gpp_dt_y_filtered, decimal=5
                )
                perform_metric_yy[ix] = evaluator_y.NSE()

        elif perform_metric == "bias_coeff":
            perform_metric_hr[ix] = calc_bias_metrics(gpp_nt, gpp_dt)

            if (gpp_nt_y_filtered.size < 3) or (gpp_dt_y_filtered.size < 3):
                perform_metric_yy[ix] = np.nan
            else:
                perform_metric_yy[ix] = calc_bias_metrics(
                    gpp_nt_y_filtered, gpp_dt_y_filtered
                )

        elif perform_metric == "corr_coeff":
            evaluator = RegressionMetric(gpp_nt, gpp_dt, decimal=5)
            perform_metric_hr[ix] = (evaluator.PCC()) ** 2.0

            if (gpp_nt_y_filtered.size < 3) or (gpp_dt_y_filtered.size < 3):
                perform_metric_yy[ix] = np.nan
            else:
                evaluator_y = RegressionMetric(
                    gpp_nt_y_filtered, gpp_dt_y_filtered, decimal=5
                )
                perform_metric_yy[ix] = (evaluator_y.PCC()) ** 2.0

        elif perform_metric == "variability_coeff":
            perform_metric_hr[ix] = calc_variability_metrics(gpp_nt, gpp_dt)

            if (gpp_nt_y_filtered.size < 3) or (gpp_dt_y_filtered.size < 3):
                perform_metric_yy[ix] = np.nan
            else:
                perform_metric_yy[ix] = calc_variability_metrics(
                    gpp_nt_y_filtered, gpp_dt_y_filtered
                )

    return perform_metric_yy, perform_metric_hr


def calc_nnse_rm_nan(nse_arr):
    """
    calculate the normalized NSE and remove nan values

    Parameters:
    -----------
    nse_arr (np.ndarray) : array of NSE values

    Returns:
    --------
    nnse_arr (np.ndarray) : array of normalized NSE values with nan values removed
    """

    nnse_arr = 1.0 / (2.0 - nse_arr)
    nnse_arr = nnse_arr[~np.isnan(nnse_arr)]

    return nnse_arr


def plot_axs(
    ax,
    metric_g01,
    metric_g02,
    metric_g03,
    metric_g04,
    metric_syr,
    metric_dt_nt,
    title,
    metric_name,
    bw_adjust=None,
    cut=None,
):
    """
    plot the histograms and KDEs of the performance metric values for the different groups

    Parameters:
    -----------
    ax (matplotlib.axes.Axes) : axis to plot the histograms and KDEs
    metric_g01 (np.ndarray) : array of performance metric values when
    group 01 parameters varied per year
    metric_g02 (np.ndarray) : array of performance metric values when
    group 02 parameters varied per year
    metric_g03 (np.ndarray) : array of performance metric values
    when group 03 parameters varied per year
    metric_g04 (np.ndarray) : array of performance metric values when
    group 04 parameters varied per year
    title (str) : title of the plot
    metric_name (str) : name of the performance metric
    bw_adjust (float) : bandwidth adjustment for the KDE
    cut (float) : cut off value for the KDE

    Returns:
    --------
    median_dict (dict) : dictionary of median values of
    the performance metric for the different groups
    """

    if metric_name == "NSE":
        metric_g01 = calc_nnse_rm_nan(metric_g01)
        metric_g02 = calc_nnse_rm_nan(metric_g02)
        metric_g03 = calc_nnse_rm_nan(metric_g03)
        metric_g04 = calc_nnse_rm_nan(metric_g04)
        metric_syr = calc_nnse_rm_nan(metric_syr)
        metric_dt_nt = calc_nnse_rm_nan(metric_dt_nt)
    else:
        metric_g01 = metric_g01[~np.isnan(metric_g01)]
        metric_g02 = metric_g02[~np.isnan(metric_g02)]
        metric_g03 = metric_g03[~np.isnan(metric_g03)]
        metric_g04 = metric_g04[~np.isnan(metric_g04)]
        metric_syr = metric_syr[~np.isnan(metric_syr)]
        metric_dt_nt = metric_dt_nt[~np.isnan(metric_dt_nt)]

    # colors for the different groups
    # source: muted (https://packages.tesselle.org/khroma/articles/tol.html#muted)
    cols = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "black",
        "#AA4499",
    ]

    if metric_name == "NSE":
        # plot the histograms and KDE - but make histogram invisible and only show KDE
        sns.histplot(
            x=metric_g01,
            stat="percent",
            kde=True,
            kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
            color="white",
            edgecolor="white",
            binrange=(0, 1.0),
            binwidth=0.1,
            ax=ax,
        )
        ax.lines[0].set_color(cols[0])

        sns.histplot(
            x=metric_g02,
            stat="percent",
            kde=True,
            kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
            binrange=(0, 1.0),
            binwidth=0.1,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[1].set_color(cols[1])

        sns.histplot(
            x=metric_g03,
            stat="percent",
            kde=True,
            kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
            binrange=(0, 1.0),
            binwidth=0.1,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[2].set_color(cols[2])

        sns.histplot(
            x=metric_g04,
            stat="percent",
            kde=True,
            kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
            binrange=(0, 1.0),
            binwidth=0.1,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[3].set_color(cols[3])

        sns.histplot(
            x=metric_syr,
            stat="percent",
            kde=True,
            kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
            binrange=(0, 1.0),
            binwidth=0.1,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[4].set_color(cols[4])
        ax.lines[4].set_linewidth(4)

        sns.histplot(
            x=metric_dt_nt,
            stat="percent",
            kde=True,
            kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
            binrange=(0, 1.0),
            binwidth=0.1,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[5].set_color(cols[5])
        ax.lines[5].set_linestyle("-.")

    else:
        # plot the histograms and KDE - but make histogram invisible and only show KDE
        sns.histplot(
            x=metric_g01,
            stat="percent",
            kde=True,
            color="white",
            edgecolor="white",
            ax=ax,
        )
        ax.lines[0].set_color(cols[0])

        sns.histplot(
            x=metric_g02,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[1].set_color(cols[1])

        sns.histplot(
            x=metric_g03,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[2].set_color(cols[2])

        sns.histplot(
            x=metric_g04,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[3].set_color(cols[3])

        sns.histplot(
            x=metric_syr,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[4].set_color("black")
        ax.lines[4].set_linewidth(3)

        sns.histplot(
            x=metric_dt_nt,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[5].set_color(cols[5])
        ax.lines[5].set_linestyle("-.")

    # add vertical lines for the median values
    ax.axvline(
        x=np.median(metric_g01),
        linestyle=":",
        color=cols[0],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_g02),
        linestyle=":",
        color=cols[1],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_g03),
        linestyle=":",
        color=cols[2],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_g04),
        linestyle=":",
        color=cols[3],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_syr),
        linestyle=":",
        color=cols[4],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_dt_nt),
        linestyle=":",
        color=cols[5],
        linewidth=1.5,
        dashes=(4, 2),
    )

    # set the axis properties
    if metric_name == "NSE":
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_xticklabels([round(x, 1) for x in np.linspace(0.0, 1.0, 6).tolist()])

    elif (metric_name == "bias") and (
        title == r"(a) P$^{\text{W}}_{\text{hr}}$ model - hourly"
    ):
        ax.set_xticks(np.arange(-1, 6, 1))
        ax.set_xticklabels([int(x) for x in np.arange(-1, 6, 1).tolist()])

    ax.tick_params(axis="both", which="major", labelsize=26.0)
    ax.set_ylabel("")
    ax.set_title(f"{title} ({len(metric_g02)})", size=30)

    sns.despine(ax=ax, top=True, right=True)

    return {
        "g01": np.median(metric_g01),
        "g02": np.median(metric_g02),
        "g03": np.median(metric_g03),
        "g04": np.median(metric_g04),
        "syr": np.median(metric_syr),
        "dt_nt": np.median(metric_dt_nt),
    }


def plot_fig_main(result_paths):
    """
    plot the main figure

    Parameters:
    -----------
    result_paths (module) : module containing the paths to the model results

    Returns:
    --------
    None
    """

    nse_yy_g01, nse_hr_g01 = get_mod_res_perform_arr(
        result_paths.g01_vary_p_model_res_path, "NSE"
    )
    nse_yy_g02, nse_hr_g02 = get_mod_res_perform_arr(
        result_paths.g02_vary_p_model_res_path, "NSE"
    )
    nse_yy_g03, nse_hr_g03 = get_mod_res_perform_arr(
        result_paths.g03_vary_p_model_res_path, "NSE"
    )
    nse_yy_g04, nse_hr_g04 = get_mod_res_perform_arr(
        result_paths.g04_vary_p_model_res_path, "NSE"
    )
    nse_yy_syr, nse_hr_syr = get_mod_res_perform_arr(
        result_paths.per_site_yr_p_model_res_path, "NSE"
    )
    nse_yy_dt_nt, nse_hr_dt_nt = get_perform_arr_dt_nt(
        result_paths.hr_ip_data_path, "NSE"
    )

    bias_yy_g01, bias_hr_g01 = get_mod_res_perform_arr(
        result_paths.g01_vary_p_model_res_path, "bias_coeff"
    )
    bias_yy_g02, bias_hr_g02 = get_mod_res_perform_arr(
        result_paths.g02_vary_p_model_res_path, "bias_coeff"
    )
    bias_yy_g03, bias_hr_g03 = get_mod_res_perform_arr(
        result_paths.g03_vary_p_model_res_path, "bias_coeff"
    )
    bias_yy_g04, bias_hr_g04 = get_mod_res_perform_arr(
        result_paths.g04_vary_p_model_res_path, "bias_coeff"
    )
    bias_yy_syr, bias_hr_syr = get_mod_res_perform_arr(
        result_paths.per_site_yr_p_model_res_path, "bias_coeff"
    )
    bias_yy_dt_nt, bias_hr_dt_nt = get_perform_arr_dt_nt(
        result_paths.hr_ip_data_path, "bias_coeff"
    )

    corr_yy_g01, corr_hr_g01 = get_mod_res_perform_arr(
        result_paths.g01_vary_p_model_res_path, "corr_coeff"
    )
    corr_yy_g02, corr_hr_g02 = get_mod_res_perform_arr(
        result_paths.g02_vary_p_model_res_path, "corr_coeff"
    )
    corr_yy_g03, corr_hr_g03 = get_mod_res_perform_arr(
        result_paths.g03_vary_p_model_res_path, "corr_coeff"
    )
    corr_yy_g04, corr_hr_g04 = get_mod_res_perform_arr(
        result_paths.g04_vary_p_model_res_path, "corr_coeff"
    )
    corr_yy_syr, corr_hr_syr = get_mod_res_perform_arr(
        result_paths.per_site_yr_p_model_res_path, "corr_coeff"
    )
    corr_yy_dt_nt, corr_hr_dt_nt = get_perform_arr_dt_nt(
        result_paths.hr_ip_data_path, "corr_coeff"
    )

    variability_yy_g01, variability_hr_g01 = get_mod_res_perform_arr(
        result_paths.g01_vary_p_model_res_path, "variability_coeff"
    )
    variability_yy_g02, variability_hr_g02 = get_mod_res_perform_arr(
        result_paths.g02_vary_p_model_res_path, "variability_coeff"
    )
    variability_yy_g03, variability_hr_g03 = get_mod_res_perform_arr(
        result_paths.g03_vary_p_model_res_path, "variability_coeff"
    )
    variability_yy_g04, variability_hr_g04 = get_mod_res_perform_arr(
        result_paths.g04_vary_p_model_res_path, "variability_coeff"
    )
    variability_yy_syr, variability_hr_syr = get_mod_res_perform_arr(
        result_paths.per_site_yr_p_model_res_path, "variability_coeff"
    )
    variability_yy_dt_nt, variability_hr_dt_nt = get_perform_arr_dt_nt(
        result_paths.hr_ip_data_path, "variability_coeff"
    )

    # ###############################
    fig_width = 16
    fig_height = 9

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=2, nrows=1, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    hr_nnse_dict = plot_axs(
        axs[0],
        nse_hr_g01,
        nse_hr_g02,
        nse_hr_g03,
        nse_hr_g04,
        nse_hr_syr,
        nse_hr_dt_nt,
        r"(a) P$^{\text{W}}_{\text{hr}}$ model - hourly",
        "NSE",
        1.0,
        3,
    )

    yy_nnse_dict = plot_axs(
        axs[1],
        nse_yy_g01,
        nse_yy_g02,
        nse_yy_g03,
        nse_yy_g04,
        nse_yy_syr,
        nse_yy_dt_nt,
        r"(b) P$^{\text{W}}_{\text{hr}}$ model - annual",
        "NSE",
        1.0,
        3,
    )

    fig.supxlabel("NNSE [-]", y=-0.01, fontsize=36)
    fig.supylabel("Fraction of" + r" sites [\%]", x=0.05, fontsize=36)

    # Adding legend manually
    # source: muted (https://packages.tesselle.org/khroma/articles/tol.html#muted)
    colors = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "black",
        "#AA4499",
    ]

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=opti_type,
            markerfacecolor=colors[i],
            markersize=25,
        )
        for i, opti_type in enumerate(
            [
                r"Group 01 ($A_t$)",
                r"Group 02 ($fW$)",
                r"Group 03 ($WAI$)",
                r"Group 04 ($fW$ and $WAI$)",
            ]
        )
    ]

    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=r"Per site--year parameterization",
            markerfacecolor="black",
            markersize=25,
        )
    )

    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=r"Between $\mathit{GPP_{NT}}$ and $\mathit{GPP_{DT}}$",
            markerfacecolor=colors[-1],
            markersize=25,
        )
    )

    plt.legend(
        handles=legend_elements,
        fontsize=28,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(-0.1, -0.5),
    )

    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./figures/f02_nnse_p.png", dpi=300, bbox_inches="tight")
    plt.savefig("./figures/f02_nnse_p.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    print("median NNSE")
    df_nnse = pd.DataFrame([hr_nnse_dict, yy_nnse_dict])
    df_nnse = df_nnse.transpose()
    print(df_nnse.to_latex(index=False, float_format="%.3f"))
    print("###################")

    ########### BIAS ####################
    fig_width = 16
    fig_height = 9

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=2, nrows=1, figsize=(fig_width, fig_height), sharey=True
    )

    hr_bias_dict = plot_axs(
        axs[0],
        bias_hr_g01,
        bias_hr_g02,
        bias_hr_g03,
        bias_hr_g04,
        bias_hr_syr,
        bias_hr_dt_nt,
        r"(a) P$^{\text{W}}_{\text{hr}}$ model - hourly",
        "bias",
    )

    yy_bias_dict = plot_axs(
        axs[1],
        bias_yy_g01,
        bias_yy_g02,
        bias_yy_g03,
        bias_yy_g04,
        bias_yy_syr,
        bias_yy_dt_nt,
        r"(b) P$^{\text{W}}_{\text{hr}}$ model - annual",
        "bias",
    )

    fig.supxlabel("Bias [-]", y=-0.01, fontsize=36)
    fig.supylabel("Fraction of" + r" sites [\%]", x=0.05, fontsize=36)

    plt.legend(
        handles=legend_elements,
        fontsize=28,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(-0.1, -0.5),
    )

    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./supplement_figs/fs02_bias_p.png", dpi=300, bbox_inches="tight")
    plt.savefig("./supplement_figs/fs02_bias_p.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    df_bias = pd.DataFrame([hr_bias_dict, yy_bias_dict])
    df_bias = df_bias.transpose()

    print("median bias")
    print(df_bias.to_latex(index=False, float_format="%.3f"))
    print("###################")

    ########### CORR ####################
    fig_width = 16
    fig_height = 9

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=2, nrows=1, figsize=(fig_width, fig_height), sharey=True
    )

    hr_corr_dict = plot_axs(
        axs[0],
        corr_hr_g01,
        corr_hr_g02,
        corr_hr_g03,
        corr_hr_g04,
        corr_hr_syr,
        corr_hr_dt_nt,
        r"(a) P$^{\text{W}}_{\text{hr}}$ model - hourly",
        "corr_coeff",
    )

    yy_corr_dict = plot_axs(
        axs[1],
        corr_yy_g01,
        corr_yy_g02,
        corr_yy_g03,
        corr_yy_g04,
        corr_yy_syr,
        corr_yy_dt_nt,
        r"(b) P$^{\text{W}}_{\text{hr}}$ model - annual",
        "corr_coeff",
    )

    fig.supxlabel("Correlation coefficient [-]", y=-0.01, fontsize=36)
    fig.supylabel("Fraction of" + r" sites [\%]", x=0.05, fontsize=36)

    plt.legend(
        handles=legend_elements,
        fontsize=28,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(-0.1, -0.5),
    )

    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./supplement_figs/fs03_corr_p.png", dpi=300, bbox_inches="tight")
    plt.savefig("./supplement_figs/fs03_corr_p.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    df_corr = pd.DataFrame([hr_corr_dict, yy_corr_dict])
    df_corr = df_corr.transpose()

    print("median corr coeff")
    print(df_corr.to_latex(index=False, float_format="%.3f"))
    print("###################")

    ########### VAR ####################
    fig_width = 16
    fig_height = 9

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=2, nrows=1, figsize=(fig_width, fig_height), sharey=True
    )

    hr_varib_dict = plot_axs(
        axs[0],
        variability_hr_g01,
        variability_hr_g02,
        variability_hr_g03,
        variability_hr_g04,
        variability_hr_syr,
        variability_hr_dt_nt,
        r"(a) P$^{\text{W}}_{\text{hr}}$ model - hourly",
        "variability",
    )

    yy_varib_dict = plot_axs(
        axs[1],
        variability_yy_g01,
        variability_yy_g02,
        variability_yy_g03,
        variability_yy_g04,
        variability_yy_syr,
        variability_yy_dt_nt,
        r"(b) P$^{\text{W}}_{\text{hr}}$ model - annual",
        "variability",
    )

    fig.supxlabel("Relative variability [-]", y=-0.01, fontsize=36)
    fig.supylabel("Fraction of" + r" sites [\%]", x=0.05, fontsize=36)

    plt.legend(
        handles=legend_elements,
        fontsize=28,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(-0.1, -0.5),
    )

    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        "./supplement_figs/fs04_variability_p.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        "./supplement_figs/fs04_variability_p.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close("all")

    df_varib = pd.DataFrame([hr_varib_dict, yy_varib_dict])
    df_varib = df_varib.transpose()

    print("median variability")
    print(df_varib.to_latex(index=False, float_format="%.3f"))
    print("###################")


if __name__ == "__main__":
    # get the result paths collection module
    result_paths_coll = importlib.import_module("result_path_coll")

    # plot the figure
    plot_fig_main(result_paths_coll)
