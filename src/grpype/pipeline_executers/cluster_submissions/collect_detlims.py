import argparse

import numpy as np

from grpype.detection.templates import TemplateBank, TemplateGrid, resolve_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("binning", type=float)
    parser.add_argument("--templates", type=str, default="")
    return parser.parse_args()


def _template_kind(template_folder):
    template_path = resolve_path(template_folder, ptype="templates")
    if any(template_path.glob("positions*.npy")):
        return "bank"
    return "grid"


def _load_template_params(binning, template_folder):
    kind = _template_kind(template_folder)
    if kind == "bank":
        template_set = TemplateBank(
            binning, alltemplates=True, kind=template_folder, hasamps=False
        )
    else:
        template_set = TemplateGrid(binning, hasamps=False, kind=template_folder)
    return template_set, kind


def main():
    args = parse_args()

    template_set, template_kind = _load_template_params(args.binning, args.templates)
    template_path = resolve_path(args.templates, ptype="templates")
    detlimpath = template_path / "detection_limit"

    ntemplates = template_set.ntemplates
    detlims = np.zeros(ntemplates)
    for idx in range(ntemplates):
        detlims[idx] = np.load(detlimpath / f"detlim_{args.binning}_{idx}.npy")
        if idx % 1_000 == 0:
            print(f"finished {idx}/{ntemplates}")

    if template_kind == "grid":
        nangs = template_set.phis.shape[0]
        nquasi = template_set.alphas.shape[0]
        detlims = detlims.reshape(nquasi, nangs)

    np.save(template_path / f"amps_{args.binning}_{ntemplates}", detlims)


if __name__ == "__main__":
    main()


