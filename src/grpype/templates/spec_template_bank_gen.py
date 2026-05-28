import os
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from gdt.missions.fermi.time import Time

from grpype.detection.global_params import (
    DATAPATH,
    INTEGRATION_SPEC_FOLDER,
    echans,
    ndetectors,
)
from grpype.detection.utils import xyz2thetahpi, fibonacci_sphere
from grpype.templates.response_handlers import generate_gbm_responses_by_coordinates
from grpype.templates.spectral_models import BandFunction
from grpype.templates.template_utils import resolve_path


def generate_integration_spec_params(nangs, params_path):
    params_path = resolve_path(params_path, ptype='templates')
    os.makedirs(params_path, exist_ok=True)

    params = np.zeros([500, 3])
    i = 0
    for epeak in np.logspace(np.log10(10), np.log10(3000), 20):
        for alpha in np.linspace(-1.5, 3, 5):
            for beta in np.linspace(-5, -1.6, 5):
                params[i, :] = [alpha, beta, epeak]
                i += 1

    alphas = params[:, 0]
    betas = params[:, 1]
    epeaks = params[:, 2]

    nparams = len(alphas)
    np.save(params_path / f'alphas{nparams}.npy', alphas)
    np.save(params_path / f'betas{nparams}.npy', betas)
    np.save(params_path / f'Epeaks{nparams}.npy', epeaks)

    thetas, phis = xyz2thetahpi(*fibonacci_sphere(nangs), azel=True)
    theta_fermi = (-thetas + 90) % 180
    phi_fermi = (phis + 360) % 360

    np.save(params_path / f'phis_fermi{nangs}.npy', phi_fermi)
    np.save(params_path / f'thetas_fermi{nangs}.npy', theta_fermi)

    return nparams


def _load_params(params_path):
    params_path = resolve_path(params_path, ptype='templates')
    alphas = np.load(next(params_path.glob(f'alpha*.npy')))
    betas = np.load(next(params_path.glob(f'beta*.npy')))
    epeaks = np.load(next(params_path.glob(f'Epeaks*.npy')))
    phis = np.load(next(params_path.glob(f'phi*.npy')))
    thetas = np.load(next(params_path.glob(f'theta*.npy')))
    return phis, thetas, alphas, betas, epeaks


def generate_integration_spec_templates(
    nangs,
    ref_date,
    responses_path=None,
    bank_path=INTEGRATION_SPEC_FOLDER,
):
    bank_path = resolve_path(bank_path, ptype='templates')
    os.makedirs(bank_path, exist_ok=True)

    generate_integration_spec_params(nangs, bank_path)

    phis, thetas, alphas, betas, epeaks = _load_params(bank_path)
    phitheta = np.vstack([phis, thetas])
    nparams = len(alphas)
    nangs = len(phis)

    metobj = Time(ref_date, scale='utc')

    if responses_path is None:
        print('Starting to generate GBM responses...')
        resp_suffix = bank_path.name + '_' + ref_date.strftime("%y%m%d%H%M%S")
        rsps_path = generate_gbm_responses_by_coordinates(
            metobj,
            phitheta=phitheta,
            rsps_path_suffix=resp_suffix,
        )
        print('GBM responses generated successfully.')
    else:
        rsps_path = resolve_path(responses_path, ptype='rsp')

    model = BandFunction()
    nai_bin_centroids = np.load(rsps_path / 'nai_photon_bin_centroids.npy')
    nai_bin_widths = np.load(rsps_path / 'nai_photon_bin_widths.npy')
    bgo_bin_centroids = np.load(rsps_path / 'bgo_photon_bin_centroids.npy')
    bgo_bin_widths = np.load(rsps_path / 'bgo_photon_bin_widths.npy')

    print('Folding spectra through the responses...')
    spec_params = np.column_stack([alphas, betas, epeaks])
    templates = np.zeros((ndetectors * echans, nparams, nangs), dtype=np.float32)
    for angle_index in range(nangs):
        rsp = np.load(rsps_path / f'rsp_{angle_index}.npy')
        folded = model.fold_model(
            rsp,
            spec_params,
            nai_bin_centroids,
            nai_bin_widths,
            bgo_bin_centroids,
            bgo_bin_widths,
        )
        templates[:, :, angle_index] = folded.T.astype(np.float32)

    date_tag = ref_date.strftime("%y%m%d%H%M%S")
    np.save(
        bank_path / f'nangs_{nangs}_nrandom_{nparams}_referencedate{date_tag}',
        templates.reshape(nangs * nparams, ndetectors * echans),
    )

    return templates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate integration spectral templates")
    parser.add_argument("nangs", help="Number of angles for the integration grid", type=int)
    parser.add_argument("--responses-path", help="Path to the responses", type=str, default=None)
    args = parser.parse_args()

    ref_date = datetime(2021, 9, 27, 0)
    print(f'Generating integration spectral templates for {ref_date} using {args.nangs} angles')

    generate_integration_spec_templates(
        nangs=args.nangs,
        ref_date=ref_date,
        bank_path=INTEGRATION_SPEC_FOLDER,
        responses_path=args.responses_path,
    )
