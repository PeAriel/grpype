import os
import shutil
import subprocess
from time import sleep, time

import numpy as np

from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.response import GbmRsp as RSP

from grpype.data_io.data_handlers import DataLoaders
from grpype.detection.global_params import *
from grpype.detection.utils import xyz2thetahpi, fibonacci_sphere


def generate_gbm_responses_by_coordinates(metobj, phitheta=None, radec=None, rsps_path_suffix=None):
    """
    Generate GBM responses by coordinates. Responses are saved in the datadir/rsp/rsps_path_suffix (see below for the default path).

    Args:
        metobj: Met object
        phitheta (np.array [2,n]): array of phi and theta in the satellite frame. First row is phi, second row is theta.
        radec (np.array [2,n]): array of right ascension and declination. First row is right ascension, second row is declination.
        rsps_path_suffix (str): Suffix for the rsps path. If None, the rsps path is datadir/rsp/date.strftime("%y%m%d%H%M%S")
    Returns:
        None
    """

    date = metobj.utc.datetime

    loader = DataLoaders()
    while True:
        try:
            _, missing = loader.check_existing_cspec(date, return_missing=True)
            if missing:
                print(f"Downloading missing CSPEC files ({len(missing)}) for {date}...")
                loader.download_cspec(date, detectors=missing, verbose=False)
    
            poshist = loader.open_poshist_by_date(date, verbose=False)
            break
        except:
            sleep(10)
            continue

    if phitheta is not None:
        phis, thetas = phitheta
        ras, decs = poshist.to_equatorial(phis, thetas, metobj.fermi)
    elif radec is not None:
        ras, decs = radec
    else:
        raise ValueError("Either phi and theta or ra and dec must be provided")

    year, month, day, hour = loader._fix_time(date)

    if rsps_path_suffix is None:
        rsps_path = DATAPATH / f'rsp/{date.strftime("%y%m%d%H%M%S")}'
    else:
        rsps_path = DATAPATH / f'rsp/{rsps_path_suffix}'

    os.makedirs(rsps_path, exist_ok=True)
    shutil.copy(poshist.full_path, rsps_path / poshist.full_path.name)

    for detector in detectors:
        fname = loader.cspec_template.format(detector, date.strftime('%y%m%d'), "00")
        cspec_saved_folder = loader.cspec_folder / str(year) / str(month) / str(day) / fname
        cspec_move_name = rsps_path / fname
        shutil.copy(cspec_saved_folder, cspec_move_name)
        os.makedirs(rsps_path / detector, exist_ok=True)

    stime = metobj.fermi

    t0 = time()
    for i in range(len(ras)):
        t1 = time()

        gen_resp_command = [
            "SA_GBM_RSP_Gen.pl",
            f"-S{stime}",
            f"-R{ras[i]}",
            f"-D{decs[i]}",
            "-Ccspec",
            str(rsps_path) + '/.'
            ]
        
        result = subprocess.run(gen_resp_command, check=True, capture_output=True, text=True)

        rsp_all = []
        for j, detector in enumerate(detectors):
            rsp_name = f"glg_cspec_{detector}*.rsp"
            filelist = list(rsps_path.glob(rsp_name))
            while len(filelist) == 0:
                filelist = list(rsps_path.glob(rsp_name))
            
            shutil.move(filelist[0], rsps_path / detector / f'detector_{detector}_index_{i}.rsp')

            rsp = RSP.open(rsps_path / f'{detector}/detector_{detector}_index_{i}.rsp')
            drm = rsp.drm
            rsp_all.append(drm.matrix)

            if i == 0:
                if j == 0:
                    np.save(rsps_path / 'nai_photon_bin_centroids', drm.photon_bin_centroids)
                    np.save(rsps_path / 'nai_photon_bin_widths', drm.photon_bin_widths)
                if j == 13:
                    np.save(rsps_path / 'bgo_photon_bin_centroids', drm.photon_bin_centroids)
                    np.save(rsps_path / 'bgo_photon_bin_widths', drm.photon_bin_widths)
        
        rsp_stack = np.stack(rsp_all, axis=0)
        np.save(rsps_path / f'rsp_{i}', rsp_stack)

        log_name = "*.logfile"
        for log_path in list(rsps_path.glob(log_name)):
            os.remove(log_path)
        
        print(f"Response {i+1}/{len(ras)} done in {time()-t1:.2f} s, total time: {time()-t0:.2f} s")

    return rsps_path

def generate_gbm_responses_by_angular_resolution(ref_date, ang_res, rsps_path_suffix=None):
    """
    Generate GBM responses by angular resolution. Responses are saved in the datadir/rsp/rsps_path_suffix (see below for the default path).
    
    Args:
        ref_date (datetime): Reference date
        ang_res (float): Angular resolution
        rsps_path_suffix (str): Suffix for the rsps path. If None, the rsps path is datadir/rsp/date.strftime("%y%m%d%H%M%S")
    Returns:
        None
    """

    metobj = Time(ref_date, scale='utc')
    loader = DataLoaders()
    poshist = loader.open_poshist_by_date(ref_date, verbose=False)

    year, month, day, hour = loader._fix_time(ref_date)

    if rsps_path_suffix is None:
        rsps_path = DATAPATH / f'rsp/{ref_date.strftime("%y%m%d%H%M%S")}'
    else:
        rsps_path = DATAPATH / f'rsp/{rsps_path_suffix}'
    os.makedirs(rsps_path, exist_ok=True)
    shutil.copy(poshist.full_path, rsps_path / poshist.full_path.name)

    nsky = int(4*np.pi * (180 / np.pi)**2 / ang_res)
    thetas, phis = xyz2thetahpi(*fibonacci_sphere(nsky), azel=True)
    thetas = (-thetas + 90) % 180
    phis = (phis + 360) % 360
    ras, decs = poshist.to_equatorial(phis, thetas, metobj.fermi)
    np.save(rsps_path / 'thetas', thetas)
    np.save(rsps_path / 'phis', phis)
    
    rsps_path = DATAPATH / f'rsp/{ref_date.strftime("%y%m%d%H%M%S")}'
    os.makedirs(rsps_path, exist_ok=True)
    shutil.copy(poshist.full_path, rsps_path / poshist.full_path.name)

    for detector in detectors:
        fname = loader.cspec_template.format(detector, ref_date.strftime('%y%m%d'), "00")
        cspec_saved_folder = loader.cspec_folder / str(year) / str(month) / str(day) / fname
        cspec_move_name = rsps_path / fname
        shutil.copy(cspec_saved_folder, cspec_move_name)
        os.makedirs(rsps_path / detector, exist_ok=True)

    stime = metobj.fermi

    t0 = time()
    for i in range(len(ras)):
        t1 = time()

        gen_resp_command = [
            "SA_GBM_RSP_Gen.pl",
            f"-S{stime}",
            f"-R{ras[i]}",
            f"-D{decs[i]}",
            "-Ccspec",
            str(rsps_path) + '/.'
            ]
        
        result = subprocess.run(gen_resp_command, check=True, capture_output=True, text=True)

        rsp_all = []
        for j, detector in enumerate(detectors):
            rsp_name = f"glg_cspec_{detector}*.rsp"
            filelist = list(rsps_path.glob(rsp_name))
            while len(filelist) == 0:
                filelist = list(rsps_path.glob(rsp_name))
            
            shutil.move(filelist[0], rsps_path / detector / f'detector_{detector}_index_{i}.rsp')
            
            rsp = RSP.open(rsps_path / f'{detector}/detector_{detector}_index_{i}.rsp')
            drm = rsp.drm
            rsp_all.append(drm.matrix)

            if i == 0:
                if j == 0:
                    np.save(rsps_path / 'nai_photon_bin_centroids', drm.photon_bin_centroids)
                    np.save(rsps_path / 'nai_photon_bin_widths', drm.photon_bin_widths)
                if j == 13:
                    np.save(rsps_path / 'bgo_photon_bin_centroids', drm.photon_bin_centroids)
                    np.save(rsps_path / 'bgo_photon_bin_widths', drm.photon_bin_widths)

        rsp_stack = np.stack(rsp_all, axis=0)
        np.save(rsps_path / f'rsp_{i}', rsp_stack)

        log_name = "*.logfile"
        for log_path in list(rsps_path.glob(log_name)):
            os.remove(log_path)
    
        print(f"Response {i+1}/{len(ras)} done in {time()-t1:.2f} s, total time: {time()-t0:.2f} s")

    return rsps_path


def fold_model(
    rsp,
    spec_model,
    spec_params,
    nai_bin_centroids,
    nai_bin_widths,
    bgo_bin_centroids,
    bgo_bin_widths,
    normalize=True,
):
    """
    Fold a model into a response to generate a template.

    Args:
        rsp (np.array): Response
        spec_model (SpectralModel): Spectral model
        spec_params (np.array): Spectral parameters
        nai_bin_centroids (np.array): NAI bin centroids
        nai_bin_widths (np.array): NAI bin widths
        bgo_bin_centroids (np.array): BGO bin centroids
        bgo_bin_widths (np.array): BGO bin widths
        normalize (bool): Normalize the template
    Returns:
        np.array: Template
    """

    spec_nai = np.atleast_2d(spec_model(spec_params, nai_bin_centroids))
    spec_bgo = np.atleast_2d(spec_model(spec_params, bgo_bin_centroids))

    weighted_nai = spec_nai * nai_bin_widths
    weighted_bgo = spec_bgo * bgo_bin_widths

    # i - detector, j - input photon bin, k - output photon bin, n - number of templates (spec_params.shape[0])
    folded_nai = np.einsum("ijk,nj->nik", rsp[:12], weighted_nai)
    folded_bgo = np.einsum("ijk,nj->nik", rsp[12:], weighted_bgo)

    folded = np.concatenate([folded_nai, folded_bgo], axis=1)
    templates = folded.reshape(folded.shape[0], -1)

    if normalize:
        templates = templates / (np.sum(templates, axis=1, keepdims=True) + EPS)

    if templates.shape[0] == 1:
        return templates[0]
    return templates


if __name__ == "__main__":
    pass
