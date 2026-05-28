from typing import Any


from numpy import dtype, ndarray


from pathlib import Path

import numpy as np

from grpype.detection.utils import logmatch, match
from grpype.detection.templates import TemplateBank, GlitchTemplates
from grpype.detection.pipeline import Detection
from grpype.detection.global_params import DATAPATH
from grpype.data_io.data_handlers import TTEData

def resolve_path(path_str, ptype='templates'):
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ptype and len(path.parts) > 1:
        path = Path(*path.parts[1:])
    return DATAPATH / ptype / path


def get_background(date, tbank, binning, bdur, rolling_window_sec):
    default_binning = 0.01
    burstdata = TTEData(date, default_binning, bdur)
    detection = Detection(default_binning, bdur, rolling_window_sec, 3*bdur)
    glitches = GlitchTemplates(default_binning, 3, hasamps=False)
    detection.just_bkg = True
    detection.match_filter(burstdata, tbank, glitches, slice_seconds=300)
    
    bkg = burstdata.fltr_bkgs[3]
    if binning < default_binning:
        bkg = bkg * (binning / default_binning)
    return bkg


def random_placement(reference_date, binning, bank_path, hasamps=False, metric='logmatch'):
    if metric == 'logmatch':
        match_func = logmatch
    elif metric == 'match':
        match_func = match
    else:
        raise ValueError(f'Invalid metric: {metric}')

    bank_path = resolve_path(bank_path, ptype='templates')
    
    # Would be better to have the detection limits calculate and loaded before doing the random placement!
    bdur = 0.04
    rolling_window_sec = 60
    tbank = TemplateBank(binning, hasamps=hasamps, alltemplates=True, kind=bank_path)
    tbank.load_templates()

    bkg = get_background(reference_date, tbank, binning, bdur, rolling_window_sec)

    initp = np.random.randint(0, tbank.templates.shape[0])
    bank = [tbank.templates[initp, :]]
    posdx = [initp]
    match_arr = np.zeros(len(bank))
    for i in range(tbank.templates.shape[0]):
        for b, bank_temp in enumerate(bank):
            cur_match = match_func(tbank.templates[i, :], bank_temp, bkg)
            if cur_match >= 0.95:
                match_arr[b] = cur_match
                break
            match_arr[b] = cur_match

        if np.max(match_arr) < 0.95:  # If the maximum match is less than 0.95, save the template
            temp = tbank.templates[i, :].copy()
            bank.append(temp)
            posdx.append(i)
            match_arr = np.zeros(len(bank))


    np.save(DATAPATH / f'templates/selected_indices_{metric}_{binning}.npy', posdx)

    return