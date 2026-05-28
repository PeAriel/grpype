"""Manual trigger classification tree and dataframe labeling (shared across notebooks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm


def manual_classification_tree(b, ecl_lat, Bs, Be, t, E):

    abs_b = abs(b)
    abs_beta = abs(ecl_lat)

    # ------- E_peak < 10 keV -------
    if E <= 10:
        if Bs > 1e-3:
            return "SF1"
        else:
            if abs_beta < 10:
                return "SF2"
            else:
                return "SGR1"

    # ------- 10 KeV < E_peak <= 20 keV -------
    elif E <= 20:
        if Bs < 1:
            if abs_b < 30:
                return "SGR2"
            else:
                return "(GRB/SGR)1"
        else:
            return "SF3"

    # ------- 20 KeV < E_peak <= 50 keV -------
    elif E <= 50:
        if abs_b < 20:
            if t < 0.85:
                return "SGR3"
            else:
                return "(GRB/SGR)2"
        else:
            if Bs > 1:
                return "SF4"
            else:
                return "GRB1"

    # ------- 50 keV < E_peak <= 1000 keV (NEW SPLIT) -------
    elif E <= 1000:
        if abs_b < 10:
            if t < 0.08:
                if Be > 1:
                    return "TGF1"
                else:
                    if E < 80:
                        return "SGR4"
                    else:
                        return "GRB2"
            else:
                return "GRB3"
        else:
            if t < 0.01:
                if Be > 0.01:
                    return "TGF2"
                else:
                    return "GRB4"
            else:
                return "GRB5"

    # ------- 1000 keV < E_peak <= 2000 keV (NEW SPLIT) -------
    elif E <= 2000:
        if t < 0.02:
            if Be > 0.006:
                return "TGF3"
            else:
                return "(GRB/TGF)1"
        else:
            return "GRB6"

    # ------- E_peak > 2000 keV -------
    else:
        if t < 0.06:
            if Be < 1e-3:
                return "(GRB/TGF)2"
            else:
                return "TGF4"
        else:
            if t < 4:
                return "GRB7"
            else:
                return "(GRB/SGR)3"


def get_classed_df(df, epeak_type):
    epeak_str = "epeak_" + epeak_type
    if epeak_str not in df.columns:
        df[epeak_str] = df.epeak

    df.loc[np.isnan(df.pe_earth_stat), "pe_earth_stat"] = df.earth_stat
    df.loc[np.isnan(df.pe_sun_stat), "pe_sun_stat"] = df.sun_stat

    X_trigs = pd.DataFrame(
        {
            "b": df.b_max,
            "ecl_lat": df.ecl_lat_max,
            "Bs": df.pe_sun_stat,
            "Be": df.pe_earth_stat,
            "t": df.timescale,
            "E": df[epeak_str],
        }
    )

    labels = np.empty(len(X_trigs), dtype=object)
    indices = np.empty(len(X_trigs), dtype=int)

    for i, (idx, row) in tqdm(enumerate(X_trigs.iterrows())):
        labels[i] = manual_classification_tree(
            row.b, row.ecl_lat, row.Bs, row.Be, row.t, row.E
        )
        indices[i] = idx

    labels_df = pd.DataFrame({"label": labels, "trigger_index": indices})
    return df.join(labels_df.set_index("trigger_index"), how="left")


# E-t bounds for each class leaf in manual_classification_tree.
# None means unconstrained on that axis (will fall back to plot limits).
CLASS_ET_BOUNDS = {
    # E <= 10
    "SF1": {"E_min": None, "E_max": 10, "t_min": None, "t_max": None},
    "SF2": {"E_min": None, "E_max": 10, "t_min": None, "t_max": None},
    "SGR1": {"E_min": None, "E_max": 10, "t_min": None, "t_max": None},
    # 10 < E <= 20
    "SGR2": {"E_min": 10, "E_max": 20, "t_min": None, "t_max": None},
    "(GRB/SGR)1": {"E_min": 10, "E_max": 20, "t_min": None, "t_max": None},
    "SF3": {"E_min": 10, "E_max": 20, "t_min": None, "t_max": None},
    # 20 < E <= 50
    "SGR3": {"E_min": 20, "E_max": 50, "t_min": None, "t_max": 0.85},
    "(GRB/SGR)2": {"E_min": 20, "E_max": 50, "t_min": 0.85, "t_max": None},
    "SF4": {"E_min": 20, "E_max": 50, "t_min": None, "t_max": None},
    "GRB1": {"E_min": 20, "E_max": 50, "t_min": None, "t_max": None},
    # 50 < E <= 1000
    "SGR4": {"E_min": 50, "E_max": 80, "t_min": None, "t_max": 0.08},
    "GRB2": {"E_min": 80, "E_max": 1000, "t_min": None, "t_max": 0.08},
    "GRB3": {"E_min": 50, "E_max": 1000, "t_min": 0.08, "t_max": None},
    "TGF1": {"E_min": 50, "E_max": 1000, "t_min": None, "t_max": 0.08},
    "TGF2": {"E_min": 50, "E_max": 1000, "t_min": None, "t_max": 0.01},
    "GRB4": {"E_min": 50, "E_max": 1000, "t_min": None, "t_max": 0.01},
    "GRB5": {"E_min": 50, "E_max": 1000, "t_min": 0.01, "t_max": None},
    # 1000 < E <= 2000
    "TGF3": {"E_min": 1000, "E_max": 2000, "t_min": None, "t_max": 0.02},
    "(GRB/TGF)1": {"E_min": 1000, "E_max": 2000, "t_min": None, "t_max": 0.02},
    "GRB6": {"E_min": 1000, "E_max": 2000, "t_min": 0.02, "t_max": None},
    # E > 2000
    "(GRB/TGF)2": {"E_min": 2000, "E_max": None, "t_min": None, "t_max": 0.06},
    "TGF4": {"E_min": 2000, "E_max": None, "t_min": None, "t_max": 0.06},
    "GRB7": {"E_min": 2000, "E_max": None, "t_min": 0.06, "t_max": 4},
    "(GRB/SGR)3": {"E_min": 2000, "E_max": None, "t_min": 4, "t_max": None},
}

__all__ = [
    "CLASS_ET_BOUNDS",
    "get_classed_df",
    "manual_classification_tree",
]
