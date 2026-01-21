#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pairwise significance testing between model performance of different parameter groups

first run to generate the required input files:
1. f02_fs03_fs04_fs05_nnse_p_model_per_group.py
2. f03_fs06_fs07_fs08_nnse_lue_model_per_group.py

author: rde
first created: Tue Dec 16 2025 11:55:40 CET
"""

from collections import OrderedDict
from itertools import combinations
import math
from scipy.stats import ks_2samp
from scipy.stats import kruskal
import numpy as np
import pandas as pd

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


def pairwise_significance_testing(nnse_group_dict, method):
    
    groups = sorted(nnse_group_dict.keys())
    pairs = list(combinations(groups, 2))

    results = OrderedDict()

    for group1, group2 in pairs:
        if method == "ks_2samp":
            stat, pval = ks_2samp(nnse_group_dict[group1], nnse_group_dict[group2])
        elif method == "kruskal":
            stat, pval = kruskal(nnse_group_dict[group1], nnse_group_dict[group2])
        
        results[(group1, group2)] = (stat, pval)

    return results


def format_pvalues(pval):

    if pval < 0.001:
        mark = "***"
    elif (pval < 0.01) and (pval >= 0.001):
        mark = "**"
    elif (pval < 0.05) and (pval >= 0.01):
        mark = "*"
    else:
        mark = "n.s."

    if pval == 0:
        p_val_str = f"0 \\textsuperscript{{{mark}}}"
    sign = "-" if pval < 0 else ""
    a = abs(float(pval))

    if a >= 1e-3:
        s = f"{pval:.{3}f}"
        p_val_str = f"{s} \\textsuperscript{{{mark}}}"
    else:
        # scientific notation
        exponent = int(math.floor(math.log10(a)))
        mantissa = a / (10.0**exponent)
        # format mantissa with significant digits
        mant_str = f"{mantissa:.{3}g}"
        # handle rounding that yields mantissa == 10.0 (e.g. 9.99 -> 10)
        if float(mant_str) >= 10.0:
            mantissa = float(mant_str) / 10.0
            exponent += 1
            mant_str = f"{mantissa:.{3}g}"

        p_val_str = (
            f"${sign}{mant_str} \\times 10^{{{exponent}}}$ \\textsuperscript{{{mark}}}"
        )

    return p_val_str


def calc_pairwise_significance(nse_group_dict, model_name, method):

    hr_dict = nse_group_dict["hr"]
    yy_dict = nse_group_dict["yy"]

    hr_dict.pop("dt_nt", None)
    yy_dict.pop("dt_nt", None)

    for group, nse_arr in yy_dict.items():
        yy_dict[group] = calc_nnse_rm_nan(nse_arr)
        hr_dict[group] = calc_nnse_rm_nan(hr_dict[group])

    test_results_hr = pairwise_significance_testing(hr_dict, method=method)
    test_results_yy = pairwise_significance_testing(yy_dict, method=method)
    
    group_pairs = []
    p_val_hr_list = []
    p_val_yy_list = []

    for pairs in test_results_hr.keys():

        group1, group2 = pairs

        if group1 != "syr":
            group1 = "Gr. " + group1[2:]
        if group2 != "syr":
            group2 = "Gr. " + group2[2:]

        if group1 == "syr":
            group1 = "site--year"
        if group2 == "syr":
            group2 = "site--year"

        group_pairs.append(f"{group1} and {group2}")

        _, pval_hr = test_results_hr[pairs]
        _, pval_yy = test_results_yy[pairs]

        p_val_hr_list.append(format_pvalues(pval_hr))
        p_val_yy_list.append(format_pvalues(pval_yy))

    p_val_df = pd.DataFrame(
        {   
            "empty": [""] * len(group_pairs),
            "Group Pair": group_pairs,
            "HR KS-2SAMP p-value": p_val_hr_list,
            "YY KS-2SAMP p-value": p_val_yy_list,
        }
    )

    print(f"{model_name} Pairwise Significance Testing Results:")
    print(p_val_df.to_latex(index=False))
    print("################################################")
    

if __name__ == "__main__":
    nse_group_dict_lue_model = np.load(
        "nse_model_per_group_lue_model.npy", allow_pickle=True
    ).item()

    nse_group_dict_p_model = np.load(
        "nse_model_per_group_p_model.npy", allow_pickle=True
    ).item()

    calc_pairwise_significance(nse_group_dict_lue_model, "lue_model", method="ks_2samp")
    calc_pairwise_significance(nse_group_dict_p_model, "p_model", method="ks_2samp")
