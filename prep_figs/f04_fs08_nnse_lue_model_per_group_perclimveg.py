#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot model performance results of the Bao model where a group of parameters
were varied per year, while the rest of the parameters were kept constant
for a site. The performance metrics are calculated at hourly and annual scales for
each climate vegetation class.

author: rde
first created: Mon Feb 10 2025 15:46:44 CET
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


def determine_bioclim(pft, kg):
    """
    determine the bioclimatic zone based on the PFT and KG

    Parameters
    ----------
    pft (str) : plant functional type
    kg (str) : Koppen-Geiger climate zone

    Returns
    -------
    bioclim (str) : bioclimatic zone
    """
    if pft in ["EBF", "ENF", "DBF", "DNF", "MF", "WSA", "OSH", "CSH"]:
        bio = "forest"
    elif pft in ["GRA", "SAV", "CRO", "WET"]:
        bio = "grass"
    elif pft == "SNO":
        bio = "Polar"
    else:
        raise ValueError("PFT not recognized")

    if (kg[0] == "A") & (bio == "forest"):
        bioclim = "TropicalF"
    elif (kg[0] == "A") & (bio == "grass"):
        bioclim = "TropicalG"
    elif (kg[0] == "B") & (bio == "forest"):
        bioclim = "AridF"
    elif (kg[0] == "B") & (bio == "grass"):
        bioclim = "AridG"
    elif (kg[0] == "C") & (bio == "forest"):
        bioclim = "TemperateF"
    elif (kg[0] == "C") & (bio == "grass"):
        bioclim = "TemperateG"
    elif (kg[0] == "D") & (bio == "forest"):
        bioclim = "BorealF"
    elif (kg[0] == "D") & (bio == "grass"):
        bioclim = "BorealG"
    elif kg[0] == "E":
        bioclim = "Polar"
    else:
        raise ValueError(f"Bioclim could not be determined for {pft} and {kg}")

    return bioclim


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

    perform_metric_hr_dict = {
        "TropicalF": [],
        "TropicalG": [],
        "AridF": [],
        "AridG": [],
        "TemperateF": [],
        "TemperateG": [],
        "BorealF": [],
        "BorealG": [],
        "Polar": [],
    }

    perform_metric_yy_dict = {
        "TropicalF": [],
        "TropicalG": [],
        "AridF": [],
        "AridG": [],
        "TemperateF": [],
        "TemperateG": [],
        "BorealF": [],
        "BorealG": [],
        "Polar": [],
    }

    for res_file in filtered_mod_res_file_list:
        # open the results file for P Model
        res_dict = np.load(res_file, allow_pickle=True).item()
        # site_id = res_dict["SiteID"]

        perform_metric_hr = res_dict[perform_metric][
            f"{perform_metric}_{res_dict['Temp_res']}"
        ]
        perform_metric_yy = res_dict[perform_metric][f"{perform_metric}_y"]

        bioclim = determine_bioclim(res_dict["PFT"], res_dict["KG"])

        perform_metric_hr_dict[bioclim].append(perform_metric_hr)
        perform_metric_yy_dict[bioclim].append(perform_metric_yy)

    for k, v in perform_metric_hr_dict.items():
        perform_metric_yy_dict[k] = np.array(perform_metric_yy_dict[k])
        perform_metric_hr_dict[k] = np.array(v)

    return perform_metric_yy_dict, perform_metric_hr_dict


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

    perform_metric_hr_dict = {
        "TropicalF": [],
        "TropicalG": [],
        "AridF": [],
        "AridG": [],
        "TemperateF": [],
        "TemperateG": [],
        "BorealF": [],
        "BorealG": [],
        "Polar": [],
    }

    perform_metric_yy_dict = {
        "TropicalF": [],
        "TropicalG": [],
        "AridF": [],
        "AridG": [],
        "TemperateF": [],
        "TemperateG": [],
        "BorealF": [],
        "BorealG": [],
        "Polar": [],
    }

    for data_nc_file in filtered_data_file_list:

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
            perform_metric_hr = evaluator.NSE()

            if (gpp_nt_y_filtered.size < 3) or (gpp_dt_y_filtered.size < 3):
                perform_metric_yy = np.nan
            else:
                evaluator_y = RegressionMetric(
                    gpp_nt_y_filtered, gpp_dt_y_filtered, decimal=5
                )
                perform_metric_yy = evaluator_y.NSE()

        elif perform_metric == "bias_coeff":
            perform_metric_hr = calc_bias_metrics(gpp_nt, gpp_dt)

            if (gpp_nt_y_filtered.size < 3) or (gpp_dt_y_filtered.size < 3):
                perform_metric_yy = np.nan
            else:
                perform_metric_yy = calc_bias_metrics(
                    gpp_nt_y_filtered, gpp_dt_y_filtered
                )

        elif perform_metric == "corr_coeff":
            evaluator = RegressionMetric(gpp_nt, gpp_dt, decimal=5)
            perform_metric_hr = (evaluator.PCC()) ** 2.0

            if (gpp_nt_y_filtered.size < 3) or (gpp_dt_y_filtered.size < 3):
                perform_metric_yy = np.nan
            else:
                evaluator_y = RegressionMetric(
                    gpp_nt_y_filtered, gpp_dt_y_filtered, decimal=5
                )
                perform_metric_yy = (evaluator_y.PCC()) ** 2.0

        elif perform_metric == "variability_coeff":
            perform_metric_hr = calc_variability_metrics(gpp_nt, gpp_dt)

            if (gpp_nt_y_filtered.size < 3) or (gpp_dt_y_filtered.size < 3):
                perform_metric_yy = np.nan
            else:
                perform_metric_yy = calc_variability_metrics(
                    gpp_nt_y_filtered, gpp_dt_y_filtered
                )

        bioclim = determine_bioclim(ip_df_dict["PFT"], ip_df_dict["KG"])

        perform_metric_hr_dict[bioclim].append(perform_metric_hr)
        perform_metric_yy_dict[bioclim].append(perform_metric_yy)

    for k, v in perform_metric_hr_dict.items():
        perform_metric_yy_dict[k] = np.array(perform_metric_yy_dict[k])
        perform_metric_hr_dict[k] = np.array(v)

    return perform_metric_yy_dict, perform_metric_hr_dict


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
    metric_g05,
    metric_g06,
    metric_g07,
    metric_g08,
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
    metric_g05 (np.ndarray) : array of performance metric values when
    group 05 parameters varied per year
    metric_g06 (np.ndarray) : array of performance metric values when
    group 06 parameters varied per year
    metric_g07 (np.ndarray) : array of performance metric values when
    group 07 parameters varied per year
    metric_g08 (np.ndarray) : array of performance metric values when
    group 08 parameters varied per year
    title (str) : title of the plot
    metric_name (str) : name of the performance metric
    bw_adjust (float) : bandwidth adjustment for the KDE
    cut (float) : cut off value for the KDE

    Returns:
    --------
    median_dict (dict) : dictionary of median values
    of the performance metric for the different groups
    """

    if metric_name == "NSE":
        metric_g01 = calc_nnse_rm_nan(metric_g01)
        metric_g02 = calc_nnse_rm_nan(metric_g02)
        metric_g03 = calc_nnse_rm_nan(metric_g03)
        metric_g04 = calc_nnse_rm_nan(metric_g04)
        metric_g05 = calc_nnse_rm_nan(metric_g05)
        metric_g06 = calc_nnse_rm_nan(metric_g06)
        metric_g07 = calc_nnse_rm_nan(metric_g07)
        metric_g08 = calc_nnse_rm_nan(metric_g08)
        metric_syr = calc_nnse_rm_nan(metric_syr)
        metric_dt_nt = calc_nnse_rm_nan(metric_dt_nt)
        # metric_dt_nt = 1.0 / (2.0 - metric_dt_nt)
        # metric_dt_nt = metric_dt_nt[~np.isnan(metric_g01)]
    else:
        metric_g01 = metric_g01[~np.isnan(metric_g01)]
        metric_g02 = metric_g02[~np.isnan(metric_g02)]
        metric_g03 = metric_g03[~np.isnan(metric_g03)]
        metric_g04 = metric_g04[~np.isnan(metric_g04)]
        metric_g05 = metric_g05[~np.isnan(metric_g05)]
        metric_g06 = metric_g06[~np.isnan(metric_g06)]
        metric_g07 = metric_g07[~np.isnan(metric_g07)]
        metric_g08 = metric_g08[~np.isnan(metric_g08)]
        metric_syr = metric_syr[~np.isnan(metric_syr)]
        metric_dt_nt = metric_dt_nt[~np.isnan(metric_dt_nt)]

    cols = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
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
            x=metric_g05,
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

        sns.histplot(
            x=metric_g06,
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

        sns.histplot(
            x=metric_g07,
            stat="percent",
            kde=True,
            kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
            binrange=(0, 1.0),
            binwidth=0.1,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[6].set_color(cols[6])

        sns.histplot(
            x=metric_g08,
            stat="percent",
            kde=True,
            kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
            binrange=(0, 1.0),
            binwidth=0.1,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[7].set_color(cols[7])

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
        ax.lines[8].set_color(cols[8])
        ax.lines[8].set_linewidth(3)

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
        ax.lines[9].set_color(cols[9])
        ax.lines[9].set_linestyle("-.")

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
            x=metric_g05,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[4].set_color(cols[4])

        sns.histplot(
            x=metric_g06,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[5].set_color(cols[5])

        sns.histplot(
            x=metric_g07,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[6].set_color(cols[6])

        sns.histplot(
            x=metric_g08,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[7].set_color(cols[7])

        sns.histplot(
            x=metric_syr,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[8].set_color(cols[8])
        ax.lines[8].set_linewidth(3)

        sns.histplot(
            x=metric_dt_nt,
            stat="percent",
            kde=True,
            ax=ax,
            color="white",
            edgecolor="white",
        )
        ax.lines[9].set_color(cols[9])
        ax.lines[9].set_linestyle("-.")

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
        x=np.median(metric_g05),
        linestyle=":",
        color=cols[4],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_g06),
        linestyle=":",
        color=cols[5],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_g07),
        linestyle=":",
        color=cols[6],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_g08),
        linestyle=":",
        color=cols[7],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_syr),
        linestyle=":",
        color=cols[8],
        linewidth=1.5,
        dashes=(4, 2),
    )
    ax.axvline(
        x=np.median(metric_dt_nt),
        linestyle=":",
        color=cols[9],
        linewidth=1.5,
        dashes=(4, 2),
    )

    # set the axis properties
    if metric_name == "NSE":
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_xticklabels([round(x, 1) for x in np.linspace(0.0, 1.0, 6).tolist()])

    elif (metric_name == "bias") and (title == r"(a) Bao$_{\text{hr}}$ model - hourly"):
        ax.set_xticks(np.arange(-1, 0.5, 0.25))
        ax.set_xticklabels([round(x, 2) for x in np.arange(-1, 0.5, 0.25).tolist()])

    elif (metric_name == "bias") and (title == r"(b) Bao$_{\text{hr}}$ model - annual"):
        ax.set_xticks(np.arange(-15, 8, 3))
        ax.set_xticklabels([int(x) for x in np.arange(-15, 8, 3).tolist()])

    ax.tick_params(axis="both", which="major", labelsize=26.0)
    # ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("")
    ax.set_title(f"{title} ({len(metric_g02)})", size=30)

    sns.despine(ax=ax, top=True, right=True)

    return {
        "g01": np.median(metric_g01),
        "g02": np.median(metric_g02),
        "g03": np.median(metric_g03),
        "g04": np.median(metric_g04),
        "g05": np.median(metric_g05),
        "g06": np.median(metric_g06),
        "g07": np.median(metric_g07),
        "g08": np.median(metric_g08),
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
        result_paths.g01_vary_lue_model_res_path,
        "NSE",
    )
    nse_yy_g02, nse_hr_g02 = get_mod_res_perform_arr(
        result_paths.g02_vary_lue_model_res_path, "NSE"
    )
    nse_yy_g03, nse_hr_g03 = get_mod_res_perform_arr(
        result_paths.g03_vary_lue_model_res_path, "NSE"
    )
    nse_yy_g04, nse_hr_g04 = get_mod_res_perform_arr(
        result_paths.g04_vary_lue_model_res_path, "NSE"
    )
    nse_yy_g05, nse_hr_g05 = get_mod_res_perform_arr(
        result_paths.g05_vary_lue_model_res_path, "NSE"
    )
    nse_yy_g06, nse_hr_g06 = get_mod_res_perform_arr(
        result_paths.g06_vary_lue_model_res_path, "NSE"
    )
    nse_yy_g07, nse_hr_g07 = get_mod_res_perform_arr(
        result_paths.g07_vary_lue_model_res_path, "NSE"
    )
    nse_yy_g08, nse_hr_g08 = get_mod_res_perform_arr(
        result_paths.g08_vary_lue_model_res_path, "NSE"
    )
    nse_yy_syr, nse_hr_syr = get_mod_res_perform_arr(
        result_paths.per_site_yr_lue_model_res_path, "NSE"
    )
    nse_yy_dt_nt, nse_hr_dt_nt = get_perform_arr_dt_nt(
        result_paths.hr_ip_data_path, "NSE"
    )

    ax_index_dict_hr = {
        "TropicalF": [0, 0, "(a)"],
        "TropicalG": [0, 1, "(b)"],
        "AridF": [0, 2, "(c)"],
        "AridG": [1, 0, "(d)"],
        "TemperateF": [1, 1, "(e)"],
        "TemperateG": [1, 2, "(f)"],
        "BorealF": [2, 0, "(g)"],
        "BorealG": [2, 1, "(h)"],
        "Polar": [2, 2, "(i)"],
    }

    coll_median_dict_hr = {}

    ###############################
    fig_width = 16
    fig_height = 16

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=3, nrows=3, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    for bioclim, metric_arr in nse_hr_g01.items():
        med_dict = plot_axs(
            axs[ax_index_dict_hr[bioclim][0], ax_index_dict_hr[bioclim][1]],
            metric_arr,
            nse_hr_g02[bioclim],
            nse_hr_g03[bioclim],
            nse_hr_g04[bioclim],
            nse_hr_g05[bioclim],
            nse_hr_g06[bioclim],
            nse_hr_g07[bioclim],
            nse_hr_g08[bioclim],
            nse_hr_syr[bioclim],
            nse_hr_dt_nt[bioclim],
            f"{ax_index_dict_hr[bioclim][2]} {bioclim}",
            "NSE",
            bw_adjust=1.0,
            cut=3,
        )

        coll_median_dict_hr[bioclim] = med_dict

    fig.supxlabel("NNSE [-]", y=0.03, fontsize=36)
    fig.supylabel("Fraction of" + r" sites [\%]", x=0.05, fontsize=36)

    colors = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
        "black",
        "#AA4499",
    ]

    legend_elements = [
        Line2D(
            [0],
            [0],
            lmarker="s",
            color="w",
            label=opti_type,
            markerfacecolor=colors[i],
            markersize=25,
        )
        for i, opti_type in enumerate(
            [
                r"Group 01 ($\varepsilon_{max}$)",
                r"Group 02 ($fT$)",
                r"Group 03 ($fVPD$ and $fCO_2$)",
                r"Group 04 ($fL$)",
                r"Group 05 ($fCI$)",
                r"Group 06 ($fW$)",
                r"Group 07 ($WAI$)",
                r"Group 08 ($WAI$ + $fW$)",
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
        bbox_to_anchor=(-0.8, -1.3),
    )

    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./supplement_figs/fs08_nnse_lue_hr_climveg.png", dpi=300, bbox_inches="tight")
    plt.savefig("./supplement_figs/fs08_nnse_lue_hr_climveg.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    ###############################
    ax_index_dict_yr = {
        "TropicalG": [0, 0, "(a)"],
        "AridF": [0, 1, "(b)"],
        "AridG": [0, 2, "(c)"],
        "TemperateF": [1, 0, "(d)"],
        "TemperateG": [1, 1, "(e)"],
        "BorealF": [1, 2, "(f)"],
        "BorealG": [2, 0, "(g)"],
        "Polar": [2, 1, "(h)"],
    }

    coll_median_dict_yy = {}

    fig_width = 16
    fig_height = 16

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=3, nrows=3, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    for bioclim, metric_arr in nse_yy_g01.items():
        if bioclim == "TropicalF":
            pass
        else:
            med_dict = plot_axs(
                axs[ax_index_dict_yr[bioclim][0], ax_index_dict_yr[bioclim][1]],
                metric_arr,
                nse_yy_g02[bioclim],
                nse_yy_g03[bioclim],
                nse_yy_g04[bioclim],
                nse_yy_g05[bioclim],
                nse_yy_g06[bioclim],
                nse_yy_g07[bioclim],
                nse_yy_g08[bioclim],
                nse_yy_syr[bioclim],
                nse_yy_dt_nt[bioclim],
                f"{ax_index_dict_yr[bioclim][2]} {bioclim}",
                "NSE",
                bw_adjust=1.0,
                cut=3,
            )

            coll_median_dict_yy[bioclim] = med_dict

    fig.delaxes(axs[2, 2])

    fig.supxlabel("NNSE [-]", y=0.03, fontsize=36)
    fig.supylabel("Fraction of" + r" sites [\%]", x=0.05, fontsize=36)

    plt.legend(
        handles=legend_elements,
        fontsize=28,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, -1.3),
    )

    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./figures/f04_nnse_lue_yy_climveg.png", dpi=300, bbox_inches="tight")
    plt.savefig("./figures/f04_nnse_lue_yy_climveg.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    columns = [
        "type",
        "g01",
        "g02",
        "g03",
        "g04",
        "g05",
        "g06",
        "g07",
        "g08",
        "syr",
        "dt_nt",
    ]

    result_df_hr = pd.DataFrame(columns=columns)

    print("###################")
    print("Hourly")
    print("###################")
    for bioclim, median_dict in coll_median_dict_hr.items():
        row_df_hr = pd.DataFrame(
            [
                {
                    "type": bioclim,
                    "g01": median_dict["g01"],
                    "g02": median_dict["g02"],
                    "g03": median_dict["g03"],
                    "g04": median_dict["g04"],
                    "g05": median_dict["g05"],
                    "g06": median_dict["g06"],
                    "g07": median_dict["g07"],
                    "g08": median_dict["g08"],
                    "syr": median_dict["syr"],
                    "dt_nt": median_dict["dt_nt"],
                }
            ]
        )
        result_df_hr = pd.concat([result_df_hr, row_df_hr], ignore_index=True)

    print(result_df_hr.to_latex(index=False, float_format="%.2f"))

    print("###################")
    print("Annual")
    print("###################")

    result_df_yy = pd.DataFrame(columns=columns)
    for bioclim, median_dict in coll_median_dict_yy.items():
        row_df_yy = pd.DataFrame(
            [
                {
                    "type": bioclim,
                    "g01": median_dict["g01"],
                    "g02": median_dict["g02"],
                    "g03": median_dict["g03"],
                    "g04": median_dict["g04"],
                    "g05": median_dict["g05"],
                    "g06": median_dict["g06"],
                    "g07": median_dict["g07"],
                    "g08": median_dict["g08"],
                    "syr": median_dict["syr"],
                    "dt_nt": median_dict["dt_nt"],
                }
            ]
        )

        result_df_yy = pd.concat([result_df_yy, row_df_yy], ignore_index=True)
    print(result_df_yy.to_latex(index=False, float_format="%.2f"))


if __name__ == "__main__":
    # get the result paths collection module
    result_paths_coll = importlib.import_module("result_path_coll")

    # plot the figure
    plot_fig_main(result_paths_coll)
