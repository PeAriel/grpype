from __future__ import annotations

import os

from grpype.detection.global_params import DATAPATH


def already_done(date, burst_duration, res_str, slice_ind=None, full=False):
    if full:
        temp_str = res_str.split("/")
        temp_str[-1] = "finished.txt"
        res_str = "/".join(temp_str)

    if not os.path.exists(DATAPATH / res_str):
        return False

    str_cond = date.strftime("%Y-%m-%d %H:%M:%S") + f", bdur: {burst_duration}"
    if slice_ind is not None:
        str_cond += f", slice {slice_ind}"

    with open(DATAPATH / res_str, "r") as f:
        lines = f.readlines()
        for line in lines:
            if str_cond in line:
                return True
    return False
