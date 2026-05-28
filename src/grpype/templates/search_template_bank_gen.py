import os
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from gdt.missions.fermi.time import Time

from grpype.detection.utils import xyz2thetahpi, fibonacci_sphere
from grpype.detection.global_params import (
    DATAPATH,
    SEARCH_BANK_FOLDER,
    detectors,
    echans,
    ndetectors,
)
from grpype.templates.response_handlers import generate_gbm_responses_by_coordinates
from grpype.templates.spectral_models import BandFunction
from grpype.templates.template_utils import resolve_path

def gen_rand_band_params(nrandom, bank_path, method):
    """
    Generate random bank parameters for a given number of templates.

    Args:
        nrandom (int): Number of templates to generate
        bank_path (str): Path to the bank parameters
        method (str): Method to generate the bank parameters
            'qmc': Quasi-Monte Carlo sampling using Sobol sequence
            'random': Random sampling using uniform distribution
    Returns:
        phis (np.array): Phis in the satellite frame
        thetas (np.array): Thetas in the satellite frame
        alphas (np.array): 
        betas (np.array): Betas
        Epeaks (np.array): Epeaks
        positions (np.array): Positions
    """
    if method == 'qmc':
        sampler = qmc.Sobol(d=3, scramble=True)
        nsobol = int(np.log2(nrandom))
        sample = sampler.random_base2(m=nsobol)
        params = qmc.scale(sample, [-1.5, -5, 10], [3, -1.6, 3000])
        
        alphas = params[:, 0]
        betas =  params[:, 1]
        Epeaks = params[:, 2]
        nrandom = len(alphas)
        
    elif method == 'random':
        alphas = np.random.uniform(3, -1.5, nrandom)
        betas =  np.random.uniform(-1.6, -5, nrandom)
        Epeaks =  np.random.uniform(10, 3000, nrandom)

    positions = np.random.permutation(np.arange(nrandom))

    params_path = resolve_path(bank_path, ptype='templates')
    os.makedirs(params_path, exist_ok=True)

    np.save(params_path / f'alphas{nrandom}.npy', alphas)
    np.save(params_path / f'betas{nrandom}.npy', betas)
    np.save(params_path / f'Epeaks{nrandom}.npy', Epeaks)
    np.save(params_path / f'positions{nrandom}.npy', positions)

    thetas, phis = xyz2thetahpi(*fibonacci_sphere(nrandom), azel=True)
    thetas = (-thetas + 90) % 180
    phis = (phis + 360) % 360

    np.save(params_path / f'phisdeg{nrandom}.npy', phis)
    np.save(params_path / f'thetasdeg{nrandom}.npy', thetas)

    return phis, thetas, alphas, betas, Epeaks, positions


def gen_rand_band_search_bank(
    nrandom,
    ref_date,
    method,
    responses_path=None,
    bank_path=SEARCH_BANK_FOLDER,
):
    """
    Generate random band search bank parameters for a given number of templates.

    Args:
        nrandom (int): Number of templates to generate
        ref_date (datetime): Reference date
        bank_path (str): Path to the bank parameters
    Returns:
        phis (np.array): Phis in the satellite frame
        thetas (np.array): Thetas in the satellite frame
        alphas (np.array): 
        betas (np.array): Betas
        Epeaks (np.array): Epeaks
        positions (np.array): Positions
    """
    bank_path = resolve_path(bank_path, ptype='templates')
    os.makedirs(bank_path, exist_ok=True)

    metobj = Time(ref_date, scale='utc')
    
    phis, thetas, alphas, betas, Epeaks, positions = gen_rand_band_params(nrandom, bank_path, method)
    phitheta = np.vstack([phis, thetas])
    if responses_path is None:
        print('Starting to generate GBM responses...')
        resp_suffix = bank_path.name + '_' + ref_date.strftime("%y%m%d%H%M%S")
        rsps_path = generate_gbm_responses_by_coordinates(metobj, phitheta=phitheta, rsps_path_suffix=resp_suffix)
        print('GBM responses generated successfully.')
    else:
        rsps_path = resolve_path(responses_path, ptype='rsp')

    model = BandFunction()
    ntemplates = len(alphas)
    nai_bin_centroids = np.load(rsps_path / 'nai_photon_bin_centroids.npy')
    nai_bin_widths = np.load(rsps_path / 'nai_photon_bin_widths.npy')
    bgo_bin_centroids = np.load(rsps_path / 'bgo_photon_bin_centroids.npy')
    bgo_bin_widths = np.load(rsps_path / 'bgo_photon_bin_widths.npy')

    print('Folding spectra through the responses...')
    templates = np.zeros((ntemplates, echans * ndetectors), dtype=np.float32)
    for i in range(ntemplates):
        rsp_index = positions[i]
        rsp = np.load(rsps_path / f'rsp_{rsp_index}.npy')
        spec_params = np.array([alphas[i], betas[i], Epeaks[i]])
        templates[i] = model.fold_model(
            rsp,
            spec_params,
            nai_bin_centroids,
            nai_bin_widths,
            bgo_bin_centroids,
            bgo_bin_widths,
        )

    date_tag = ref_date.strftime("%y%m%d%H%M%S")
    np.save(bank_path / f'nrandom{ntemplates}_referencedate{date_tag}', templates)

    return phis, thetas, alphas, betas, Epeaks, positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a random band search bank")
    parser.add_argument("nrandom", help="Number of random templates to generate", type=int)
    parser.add_argument('--method',help="Method to generate the bank parameters", type=str, default='random')
    parser.add_argument("--responses-path", help="Path to the responses", type=str, default=None)
    args = parser.parse_args()

    ref_date = datetime(2021, 9, 27, 0)
    print(f'Generating {args.nrandom} random templates for {ref_date} using {args.method} method')

    gen_rand_band_search_bank(args.nrandom, ref_date, args.method)