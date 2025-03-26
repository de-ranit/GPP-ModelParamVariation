#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this module uses a bucket model to calculate soil water availability index (WAI)
ref: https://doi.org/10.5194/hess-22-4061-2018; https://doi.org/10.5194/bg-16-2557-2019

author: rde, skoirala
first created: 2023-11-07
"""

import numpy as np
import numexpr as ne


def calc_etsub_pot(t_air, netrad, params):
    """
    function to calculate potential sublimation from snowfall
    from calc_etsub_pot.m (Bao et al., 2022)

    Parameters:
    t_air (np array): air temperature timeseries [degC]
    netrad (np array): net radiation timeseries [J/m2/s or W/m2]
    params (dict): dictionary with parameter values

    Returns:
    et_sub_pot (np array): potential sublimation of ice timeseries [mm/s]
    """
    # convert air temperature from Celsius to Kelvin
    tair_k = t_air + 273.15

    #### following comments are copied from calc_etsub_pot.m (Bao et al., 2022)
    # from Diego miralles
    # The majority of the parameters I use in GLEAM come from the equations
    # in Murphy and Koop (2005) here attached.
    # The slope of the vapour pressure over ice versus
    # temperature curve (Delta) is obtained from eq. (7). You may want to do this derivative
    # yourself because my calculus is not as good as it used to; what I get is:
    delta_eval_exp = (  # pylint: disable=unused-variable
        9.550426
        - 5723.265 / tair_k
        + 3.53068 * ne.evaluate("log(tair_k)")
        - 0.00728332 * tair_k
    )
    delta = (5723.265 / tair_k**2.0 + 3.53068 / (tair_k - 0.00728332)) * ne.evaluate(
        "exp(delta_eval_exp)"
    )
    delta = delta * 0.001  # convert from [Pa/K] to [kPa/K] by multiplying times 0.001.

    # The latent heat of sublimation of ice (Lambda) can be found in eq. (5):
    lhs_ice_eval_exp = -((tair_k / 123.75) ** 2.0)  # pylint: disable=unused-variable
    lhs_ice = (
        46782.5
        + 35.8925 * tair_k
        - 0.07414 * tair_k**2.0
        + 541.5 * ne.evaluate("exp(lhs_ice_eval_exp)")
    )
    lhs_ice = lhs_ice / (
        18.01528 * 1000.0
    )  # To convert from [J/mol] to [MJ/kg] assuming molecular mass of water of 18.01528 g/mol

    # Then the psychrometer 'constant' (Gamma) can be calculated in [kPa/K]
    # according to Brunt [1952] as:
    # Where P is the air pressure in [kPa], which is considered as a function of
    # the elevation (DEM) but can otherwise be set to 101.3, and pa is the
    # specific heat of air which I assume 0.001 MJ/kg/K.
    atmo_pres = (
        101.3 * ((293.0 - 0.0065 * params["ca"]) / 293.0) ** 5.26
    )  # atmospheric pressure in KPa at elevation ca in m
    cp_air = 1.013 / 1000.0  # specific heat of moist air in MJ/kg/K
    gamma = (atmo_pres * cp_air) / (0.622 * lhs_ice)  # psychrometer constant in kPa/K

    # calculate et_sub_pot
    netrad = netrad / 1e6  # convert netrad from J/m2.s to MJ/m2.s
    et_sub_pot = (
        params["sn_a"] * (netrad / lhs_ice) * (delta / (delta + gamma))
    )  # potential sublimation [mm/s]

    # replace nan and negative values with zero
    et_sub_pot = np.nan_to_num(et_sub_pot)
    et_sub_pot[et_sub_pot < 0] = 0.0

    return et_sub_pot  # mm/second


def calc_snofall(t_air, prec):
    """
    Calculate snowfall: if Tair <= 0, then snowfall = Prec, else snowfall = 0

    Parameters:
    t_air (np array): air temperature timeseries [degC]
    prec (np array): precipitation timeseries [mm]

    Returns:
    snofall (np array): snowfall timeseries [mm]
    """
    snofall = np.zeros_like(prec)
    snofall[t_air < 0] = prec[t_air < 0]
    return snofall


def calc_rain(snofall, prec):
    """
    Calculate liquid prec: if Prec - snofall > 0, then pl = Prec - snofall, else pl = 0

    Parameters:
    snofall (np array): snowfall timeseries [mm]
    prec (np array): precipitation timeseries [mm]

    Returns:
    prec_liquid (np array): liquid precipitation timeseries [mm]
    """
    prec_liquid = np.maximum(prec - snofall, 0.0)
    return prec_liquid


def calc_wai(
    input_data,
    wai_results,
    params,
    wai0=0.0,
    nloops=5,
    sno_prev=0.0,
    spinup=False,
    do_snow=True,
    normalize_wai=False,
    do_sublimation=True,
    nstepsday=24,
):
    """
    Calculate WAI and normalize it between 0 and 1 (W)

    Parameters:
    input_data (dict): dictionary with input data
    wai_results (dict): dictionary to store WAI output
    params (dict): dictionary with parameter values
    wai0 (float): initial value of WAI
    nloops (int): number of loops for spinup/ during actual WAI calculation
    sno_prev (float): initial value of snow bucket
    spinup (bool): if True, then spinup is performed
    do_snow (bool): if True, then snow calculation is considered in calculation of WAI
    normalize_wai (bool): if True, then WAI is normalized between 0 and 1
    do_sublimation (bool): if True, then loss of snow through
                           sublimation is considered in calculation of WAI
    nstepsday (int): number of sub-daily timesteps in a day

    Returns:
    wai_results (dict): dictionary with WAI output
    """
    # if no wai0 value is supplied, wai0 becomes AWC
    if wai0 is None:
        wai_prev = params["AWC"]
    else:
        wai_prev = wai0

    # get input data to array
    t_air_ts = input_data["TA_GF"]
    prec_ts = input_data["P_GF"]
    netrad_ts = input_data["NETRAD_GF"]
    # et_p_ts = input_data["PET"]
    et_p_ts = input_data["PET"] * params["alphaPT"]
    no_of_seconds = (24.0 / nstepsday) * 3600.0  # no. of seconds in each timestep

    # if snow calculation is considered in calculation of wai
    if do_snow is True:
        snofall_ts = calc_snofall(
            t_air_ts, prec_ts
        )  # get prec as amount of snowfall when t_air <= 0

        # scale the meltrate parameters based on the data frequency
        if nstepsday == 48:  # if half-hourly data
            scalar_meltrate = 0.5
        elif nstepsday == 24:  # if hourly data
            scalar_meltrate = 1.0
        elif nstepsday == 1:  # if daily data
            scalar_meltrate = 24.0
        else:
            raise ValueError(
                (
                    "nstepsday should be 1, 24 or 48 for daily,"
                    "hourly or half-hourly data, respectively"
                )
            )
        # calculate potential snowmelt
        # after converting netrad from J/m2.s to MJ/m2/timestep and meltrate from
        # either mm/°C/hr to mm/timestep or mm/MJ/hr to mm/MJ/timestep
        snomelt_pot_ts = (params["meltRate_temp"] * scalar_meltrate * t_air_ts) + (
            params["meltRate_netrad"]
            * scalar_meltrate
            * (netrad_ts / 1e6)
            * no_of_seconds
        )
        snomelt_pot_ts[t_air_ts <= 0] = 0.0  # no snowmelt when t_air <= 0
    else:  # else snow variables are set to zero
        snofall_ts = np.zeros_like(t_air_ts)
        snomelt_pot_ts = np.zeros_like(t_air_ts)

    # if loss of snow through sublimation is considered in calculation of wai
    if do_snow is True and do_sublimation is True:
        etsub_pot_ts = (
            calc_etsub_pot(t_air_ts, netrad_ts, params) * no_of_seconds
        )  # calculate potential sublimation
    else:  # else no sublimation
        etsub_pot_ts = np.zeros_like(t_air_ts)  # potential sublimation is zero

    # calculate amount of liquid prec
    prec_liquid_ts = calc_rain(snofall_ts, prec_ts)

    # if no spinup is required for wai calculation,
    # then just store the variables in the output dictionary
    if spinup is False:
        wai_results["etsub_pot"] = etsub_pot_ts
        wai_results["snofall"] = snofall_ts
        wai_results["pl"] = prec_liquid_ts

    # WAI Spinup
    for i in range(nloops):  # loop over number of spinup loops
        for t in range(t_air_ts.size):  # for each timestep
            snomelt = min(
                snomelt_pot_ts[t], sno_prev
            )  # snomelt is minimum of potential snowmelt and snow available from previous timestep
            etsub = min(
                etsub_pot_ts[t], sno_prev - snomelt
            )  # etsub is minimum of potential sublimation and
            # snow available from previous timestep minus snowmelt
            sno = (
                sno_prev + snofall_ts[t] - snomelt - etsub
            )  # available snow at current timestep
            pu = prec_liquid_ts[t] + snomelt

            # calculate wai
            min_prec = min(pu, (params["AWC"] - wai_prev))
            slet = params["theta"] * scalar_meltrate * (wai_prev + min_prec)
            et = min(et_p_ts[t], slet)
            wai = wai_prev + min_prec - et

            # assign the sno and wai value at current timestep as initials for the next timestep
            sno_prev = sno
            wai_prev = wai

            # collect the variables in the wai output dictionary
            if spinup is False and i == nloops - 1:
                wai_results["pu"][t] = pu
                wai_results["sno"][t] = sno
                wai_results["et"][t] = et
                wai_results["wai"][t] = wai
                if normalize_wai is True:
                    wai_results["wai_nor"][t] = wai / params["AWC"]
                else:
                    wai_results["wai_nor"][t] = wai
                wai_results["etsno"][t] = et + etsub
                wai_results["snomelt"][t] = snomelt
                wai_results["etsub"][t] = etsub

            elif (spinup is True) and (i == nloops - 1) and (t == t_air_ts.size - 1):
                # get the stable value of wai after spinup
                # to initialize the wai_prev for actual wai calculation
                wai_results["wai"][-1] = wai

    return wai_results
