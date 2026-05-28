import argparse
from datetime import datetime

import numpy as np

from grpype.detection.global_params import DATAPATH
from grpype.detection.pipeline import Detection, TTEData
from grpype.detection.templates import GlitchTemplates, TemplateBank, TemplateGrid, resolve_path
from grpype.detection.utils import find_threshold


def _template_path(template_folder):
    return resolve_path(template_folder, ptype="templates")


def _template_kind(template_folder):
    template_path = _template_path(template_folder)
    if any(template_path.glob("positions*.npy")):
        return "bank"
    return "grid"


def _load_template_set(binning, template_folder):
    kind = _template_kind(template_folder)
    if kind == "bank":
        template_set = TemplateBank(
            binning, alltemplates=True, kind=template_folder, hasamps=False
        )
    else:
        template_set = TemplateGrid(binning, hasamps=False, kind=template_folder)
    template_set.load_templates()
    return template_set, kind


def templates_detection_limit(binning, idx0, idx1, template_folder):
    template_path = _template_path(template_folder)
    detlimpath = template_path / "detection_limit"
    if not detlimpath.exists():
        detlimpath.mkdir(parents=True)

    date = datetime(2016, 12, 18, 1)
    bdur = 0.04
    rolling_window_sec = 60
    burstdata = TTEData(date, binning, bdur, slice_time=[0.1, 0.3])
    detection = Detection(binning, bdur, rolling_window_sec, 3 * bdur)
    glitches = GlitchTemplates(binning, 3, hasamps=False)
    template_set, template_kind = _load_template_set(binning, template_folder)

    detection.just_bkg = True
    detection.match_filter(burstdata, template_set, glitches, slice_seconds=240)
    bkg = burstdata.fltr_bkgs[0]

    for idx in range(idx0, idx1):
        outfile = detlimpath / f"detlim_{binning}_{idx}.npy"
        if outfile.exists():
            continue

        if template_kind == "grid":
            template = template_set.templates[:, idx]
        else:
            template = template_set.templates[idx]
        amp = find_threshold(
            template,
            bkg,
            nsamples=10_000,
            amp_init=1,
            bins=500,
            niter=10,
            prec=1e-2,
        )
        np.save(outfile, amp)
        print(amp)

def glitches_detection_limit(binning):
    datapath = DATAPATH
    glitch_path = datapath / "templates/glitches1d"

    date = datetime(2016, 12, 18, 1)
    bdur = 0.04
    rolling_window_sec = 60
    burstdata = TTEData(date, binning, bdur, slice_time=[0.1, 0.3])
    detection = Detection(binning, bdur, rolling_window_sec, 3 * bdur)
    glitches = GlitchTemplates(binning, 3, hasamps=False)
    tbank = TemplateBank(binning)

    detection.just_bkg = True
    detection.match_filter(burstdata, tbank, glitches, slice_seconds=240)
    bkg = burstdata.fltr_bkgs[0]

    amps = np.zeros(len(glitches.glitch1d))
    for idx, glitch in enumerate(glitches.glitch1d):
        amps[idx] = find_threshold(
            glitch, bkg, nsamples=10_000, amp_init=1, bins=500, niter=10, prec=1e-2
        )
        print(amps[idx])

    np.save(glitch_path / f"glitches1d_amps_{binning}_{len(glitches)}.npy", amps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binning", type=float)
    parser.add_argument("idx0", type=int)
    parser.add_argument("idx1", type=int)
    parser.add_argument("--templates", type=str, default="")
    args = parser.parse_args(args=None)

    templates_detection_limit(args.binning, args.idx0, args.idx1, args.templates)

if __name__ == "__main__":
    main()

