import os
import numpy as np
from scipy.signal import windows

from grpype.detection.global_params import *


def generate_1d_glitch_templates(glitch_len_samples, std=1):
    datalen = len(detectors) * echans

    if not os.path.exists(DATAPATH / 'templates/glitches1d'):
        os.makedirs(DATAPATH / 'templates/glitches1d', exist_ok=True)

    glitchmat = np.zeros([len(detectors), datalen])
    for i in range(len(detectors)-2):
        glitchmat[i, i*echans:i*echans + glitch_len_samples] = windows.gaussian(glitch_len_samples, std=std)
    
    for det in [12, 13]:
        glitchmat[det, det*echans:det*echans + 3] = windows.gaussian(3, std=std)
        glitchmat[det, det*echans+14:det*echans + 24] = windows.gaussian(10, std=std)
        glitchmat[det, det*echans+29:det*echans + 34] = windows.gaussian(5, std=std)
    
    np.save(DATAPATH / 'templates/glitches1d' / f'glitchlensamples_{glitch_len_samples}.npy', glitchmat)

    return


def main():
    generate_1d_glitch_templates(3)
    return
