import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os
from copy import deepcopy
from datetime import datetime, timedelta
from functools import partial
from math import ceil, floor
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pickle
import shutil
from time import sleep

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d
from tqdm import tqdm

from gdt.core.binning.binned import rebin_by_time
from gdt.core.binning.unbinned import bin_by_time
from gdt.missions.fermi.gbm.phaii import Cspec, Ctime
from gdt.missions.fermi.gbm.tte import GbmTte as TTE
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.time import Time

from grpype._compat import PosHistCompat
from grpype.detection.global_params import *
from grpype.detection.utils import rolling_mean_padded


class DataLoaders:
    def __init__(self):
        self.detectors = detectors

        datapath = DATAPATH

        self.ctime_folder = datapath / "ctime"
        self.cspec_folder = datapath / "cspec"
        self.tte_folder = datapath / "tte"
        self.poshist_folder = datapath / "poshist"
        self.tmp_folder = datapath / ".tmp"

        if os.path.exists(self.tmp_folder) is False:
            os.mkdir(self.tmp_folder)

        self.daily_template = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{}/{}/{}/current/"
        self.ctime_template = "glg_ctime_{}_{}_v{}.pha"
        self.cspec_template = "glg_cspec_{}_{}_v{}.pha"
        self.tte_template = "glg_tte_{}_{}_{}z_v{}.fit.gz"
        self.poshist_template = "glg_poshist_all_{}_v{}.fit"

    @staticmethod
    def _fix_time(date):
        """
        fix date format tp match the one in the url. Append 0 if needed
        :param date: date to fix
        :return: fixed date
        """
        month = "0" + str(date.month) if date.month < 10 else str(date.month)
        day = "0" + str(date.day) if date.day < 10 else str(date.day)
        hour = "0" + str(date.hour) if date.hour < 10 else str(date.hour)
        return str(date.year), month, day, hour

    def download_ctime(self, date, detectors=None, skip_existing=True, version="00", verbose=True):
        """
        download a ctime file and save it
        :param date (datetime object): date of the file
        :param detector (str of list of str): detector to download. If none, download all
        :param skip_existing (bool): if True, skip download if file already exists
        """

        year, month, day, hour = self._fix_time(date)

        if detectors is not None and type(detectors) is not list:
            detectors = [detectors]
        else:
            detectors = self.detectors

        for detector in detectors:
            ctime_template = self.ctime_template.format(detector, date.strftime("%y%m%d"), version)
            url = self.daily_template.format(year, month, day, hour) + ctime_template
            outfolder = self.ctime_folder / str(year) / str(month) / str(day)
            outpath = outfolder / ctime_template
            os.makedirs(outfolder, exist_ok=True)
            tmp_outpath = self.tmp_folder / ctime_template
            if os.path.exists(outpath) and skip_existing:
                if verbose:
                    print("File {} already exists".format(outpath))
                continue
            else:
                self._download(url, tmp_outpath)
                shutil.move(tmp_outpath, outpath)

    def download_cspec(self, date, detectors=None, npar=14, skip_existing=True, version="00", verbose=True):
        """
        download a cspec file and save it
        :param date (datetime object): date of the file
        :param detector (str of list of str): detector to download. If none, download all
        :param skip_existing (bool): if True, skip download if file already exists
        """

        year, month, day, hour = self._fix_time(date)

        if detectors is not None and type(detectors) is not list:
            detectors = [detectors]
        else:
            detectors = self.detectors

        partial_downlaod = partial(
            self._parallel_cspec_download, date=date, version=version, skip_existing=skip_existing, verbose=verbose
        )
        npar = min(len(detectors), 14, cpu_count(), npar)
        with Pool(npar) as p:
            p.map(partial_downlaod, detectors)

    def download_tte(self, date, detectors=None, npar=1, skip_existing=True, version="00", verbose=False):
        """
        download a tte file and save it
        :param date (datetime object): date of the file. Must include hour! Otherwise, 0 is assumed.
        :param detector (str of list of str): detector to download. If none, download all
        :param skip_existing (bool): if True, skip download if file already exists

        TODO: Implement support for dates before 2013
        """

        assert date.year >= 2013, "TTE format is different before 2013. Not implemented yet"

        if detectors is not None and type(detectors) is not list:
            detectors = [detectors]
        else:
            detectors = self.detectors

        if npar is None:
            npar = 1

        partial_downlaod = partial(
            self._parallel_tte_download, date=date, version=version, skip_existing=skip_existing, verbose=verbose
        )
        npar = min(len(detectors), 14, cpu_count(), npar)
        if npar <= 1:
            for detector in detectors:
                partial_downlaod(detector)
            return

        with Pool(npar) as p:
            p.map(partial_downlaod, detectors)

    def check_existing_tte(self, date, detectors=None, version="00", return_missing=False):
        """
        Check how many TTE files exist locally for a date.
        :param date (datetime object): date of the file. Must include hour! Otherwise, 0 is assumed.
        :param detector (str or list of str): detectors to check. If none, check all
        :param version (str): tte version to check
        :param return_missing (bool): if True, also return missing detectors
        :return: number of existing files (and optional missing detector list)
        """
        year, month, day, hour = self._fix_time(date)

        if detectors is not None and type(detectors) is not list:
            detectors = [detectors]
        else:
            detectors = self.detectors

        existing = 0
        missing = []
        for detector in detectors:
            tte_template = self.tte_template.format(detector, date.strftime("%y%m%d"), hour, version)
            filename = self.tte_folder / str(year) / str(month) / str(day) / tte_template
            if os.path.exists(filename):
                existing += 1
            else:
                missing.append(detector)

        if return_missing:
            return existing, missing
        return existing

    def check_existing_cspec(self, date, detectors=None, version="00", return_missing=False):
        """
        Check how many CSPEC files exist locally for a date.
        :param date (datetime object): date of the file.
        :param detector (str or list of str): detectors to check. If none, check all
        :param version (str): cspec version to check
        :param return_missing (bool): if True, also return missing detectors
        :return: number of existing files (and optional missing detector list)
        """
        year, month, day, hour = self._fix_time(date)

        if detectors is not None and type(detectors) is not list:
            detectors = [detectors]
        else:
            detectors = self.detectors

        existing = 0
        missing = []
        for detector in detectors:
            cspec_template = self.cspec_template.format(detector, date.strftime("%y%m%d"), version)
            filename = self.cspec_folder / str(year) / str(month) / str(day) / cspec_template
            if os.path.exists(filename):
                existing += 1
            else:
                missing.append(detector)

        if return_missing:
            return existing, missing
        return existing

    def _parallel_tte_download(self, detector, date, version, skip_existing, verbose):
        year, month, day, hour = self._fix_time(date)
        tte_template = self.tte_template.format(detector, date.strftime("%y%m%d"), hour, version)
        url = self.daily_template.format(year, month, day, hour) + tte_template
        outfolder = self.tte_folder / str(year) / str(month) / str(day)
        outpath = outfolder / tte_template
        os.makedirs(outfolder, exist_ok=True)

        tmp_outpath = self.tmp_folder / tte_template
        if os.path.exists(outpath) and skip_existing:
            if verbose:
                print("File {} already exists".format(outpath))
            return
        else:
            self._download(url, tmp_outpath)
            shutil.move(tmp_outpath, outpath)

    def _parallel_cspec_download(self, detector, date, version, skip_existing, verbose):
        year, month, day, hour = self._fix_time(date)
        cspec_template = self.cspec_template.format(detector, date.strftime("%y%m%d"), version)
        url = self.daily_template.format(year, month, day, hour) + cspec_template
        outfolder = self.cspec_folder / str(year) / str(month) / str(day)
        outpath = outfolder / cspec_template
        os.makedirs(outfolder, exist_ok=True)

        tmp_outpath = self.tmp_folder / cspec_template
        if os.path.exists(outpath) and skip_existing:
            if verbose:
                print("File {} already exists".format(outpath))
            return
        else:
            self._download(url, tmp_outpath)
            shutil.move(tmp_outpath, outpath)

    def _download(self, url, filename):
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split("/")[-1]) as progress_bar:
            for i in range(10):
                response = requests.get(url, stream=True)
                if response.status_code == 404:
                    url = url.replace(f"v0{i}", f"v0{i+1}")
                elif response.status_code != 200:
                    while True:
                        sleep(np.random.uniform(2, 20))
                        response = requests.get(url, stream=True)
                        if response.status_code == 200:
                            break

            block_size = 1024
            with open(filename, "wb") as file:
                for data in response.iter_content(chunk_size=block_size):
                    file.write(data)
                    progress_bar.update(len(data))

    def open_ctime_by_date(self, date, detector, time_bins=None, skip_existing=True, version="00", verbose=False):
        """
        open a ctime file from a date and a detector
        :param date (datetime object): date of the file
        :time_bins (float): sample rate to rebin the ctime file
        :param detector (str): detector to open
        :param skip_existing (bool): if True, skip download if file already exists
        :return: ctime object
        """
        year, month, day, hour = self._fix_time(date)
        ctime_template = self.ctime_template.format(detector, date.strftime("%y%m%d"), version)
        filename = self.ctime_folder / str(year) / str(month) / str(day) / ctime_template
        if os.path.exists(filename) is False:
            self.download_ctime(date, detector, skip_existing=skip_existing, version=version, verbose=verbose)

        ctime = Ctime.open(filename)

        if time_bins is not None:
            rebinned = ctime.rebin_time(rebin_by_time, time_bins)
            return rebinned

        return ctime

    def open_cspec_by_date(
        self, date, detector, time_bins=None, npar=1, skip_existing=True, version="00", verbose=False
    ):
        """
        open a cspec file from a date and a detector
        :param date (datetime object): date of the file
        :param detector (str): detector to open
        :time_bins (float): sample rate to rebin the cspec file
        :npar (int): number of parallel downloads to use
        :param skip_existing (bool): if True, skip download if file already exists
        :return: cspec object
        """
        year, month, day, hour = self._fix_time(date)
        cspec_template = self.cspec_template.format(detector, date.strftime("%y%m%d"), version)
        filename = self.cspec_folder / str(year) / str(month) / str(day) / cspec_template
        if os.path.exists(filename) is False:
            self.download_cspec(date, detector, npar, skip_existing=skip_existing, version=version, verbose=verbose)

        cspec = Cspec.open(filename)

        if time_bins is not None:
            rebinned = cspec.rebin_time(rebin_by_time, time_bins)
            return rebinned

        return cspec

    def open_tte_by_date(
        self, date, detector, time_bins=2.048, npar=1, slice_time=None, skip_existing=True, version="00", verbose=False
    ):
        """
        open a tte file from a date and a detector
        :param date (datetime object): date of the file
        :param detector (str): detector to open
        :time_bins (float): sample rate to bin the tte file. If none, return the unbinned tte file
        :npar (int): number of parallel downloads to use
        :slice_time (tuple): fractional time range to slice the tte file
        :param skip_existing (bool): if True, skip download if file already exists
        :return: tte object
        """
        year, month, day, hour = self._fix_time(date)
        tte_template = self.tte_template.format(detector, date.strftime("%y%m%d"), hour, version)

        hour = "0" + str(date.hour) if date.hour < 10 else str(date.hour)

        filename = self.tte_folder / str(year) / str(month) / str(day) / tte_template
        if os.path.exists(filename) is False:
            self.download_tte(date, detector, npar, skip_existing=skip_existing, version=version, verbose=verbose)

        tte = TTE.open(filename)

        if slice_time is not None:
            fullrng = tte.time_range[1] - tte.time_range[0]
            t0 = tte.time_range[0] + fullrng * slice_time[0]
            tf = tte.time_range[0] + fullrng * slice_time[1]
            st = [t0, tf]
            tte = tte.slice_time(st)

        if time_bins is not None:
            binned = tte.to_phaii(bin_by_time, time_bins)
            return binned
        else:
            return tte

    def open_poshist_by_date(self, date, skip_existing=True, version="00", verbose=False):
        year, month, day, hour = self._fix_time(date)

        poshist_template = self.poshist_template.format(date.strftime("%y%m%d"), version)
        url = self.daily_template.format(year, month, day) + poshist_template

        tmp_outpath = self.tmp_folder / poshist_template
        filename = self.poshist_folder / poshist_template

        if os.path.exists(filename) is False:
            self._download(url, tmp_outpath)
            shutil.move(tmp_outpath, filename)

        return PosHistCompat(GbmPosHist.open(filename), filepath=filename)

    def get_extended_poshist(self, date, delta=1):
        """
        Return a list of poshist objects spanning date-delta to date+delta days.

        GbmPosHist does not support merging; callers should select the
        appropriate poshist by checking time coverage.
        """
        return [
            self.open_poshist_by_date(date + timedelta(days=-delta)),
            self.open_poshist_by_date(date),
            self.open_poshist_by_date(date + timedelta(days=delta)),
        ]


class TTEData:
    """
    A class to load and manipulate tte data for a particular date. Allows to load data from different detectors and remove outliers.
    :param date (datetime object): date of the data. Must include hour
    :param detector (str): detector to load. If None, load all detectors
    :param binning (float): time in seconds to bin the data
    :param timeslides (float): time in minutes to slide the data. If None, don't slide the data. 13 minutes will give 1 minute slide for each detector.
    """

    def __init__(
        self,
        date,
        binning,
        burst_duration_sec,
        timeslides=None,
        simulate=False,
        cut_len_seconds=10,
        npar=1,
        slice_time=None,
        save_obj=False,
        debug_slides=None,
        old_load=False
    ):
        self.ndetectors = ndetectors
        self.debug_slides = debug_slides
        self.timeslides = timeslides
        self.timeslides_minutes = None
        self.timeslides_samples = None
        self.full_det_data = None
        self.full_time_data = None
        self.full_chan_data = None
        self.simulate = simulate

        self.has_bkg = False
        self.bkg_binning = None
        self.bkgs = None
        self.fltr_bkgs = None
        self.fltr_bkgs_inds = None
        self.sharptime_reduce = 0

        self.poshist = None
        self.data = None
        self.time = None
        self.gti = None
        self.gti_inds = None
        self.nai_edges = None
        self.bgo_edges = None

        if type(date) == str:
            self.date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
        else:
            self.date = datetime(date.year, date.month, date.day, date.hour)

        self.detectors = detectors
        self.binning = binning
        self.slice_time = slice_time
        self.set_burst_duration(burst_duration_sec)

        self.cut_len_seconds = cut_len_seconds
        self.cut_len_samples = int(self.cut_len_seconds / self.binning)
        self.npar = npar

        t = "all" if slice_time is None else slice_time[0]
        gtisfile = f"gtis/gtis{self.date.strftime('%y%m%d%H')}_{t}.npy"
        self.gtisfile = (
            DATAPATH / f"results/{self.date.year}" / gtisfile
            if self.timeslides is None
            else DATAPATH / f"results_timeslides/{self.date.year}" / gtisfile
        )
        self.gtisfile = DATAPATH / f"results_simul/{self.date.year}" / gtisfile if self.simulate else self.gtisfile
        self.gtisfile_all = Path("_".join(self.gtisfile.as_posix().split("_")[:-1] + ["all.npy"]))

        self.loader = DataLoaders()
        year, month, day, hour = self.loader._fix_time(self.date)
        self.data_path = (
            DATAPATH
            / "tte"
            / str(year)
            / str(month)
            / str(day)
            / f"TTEData_all_bins_{self.binning}_{self.date.strftime('%y%m%d%H')}.pkl"
        )

        self.burstcat = pd.read_csv(DATAPATH / "catalogs/burstcat.csv")[["trigger_time", "t90"]]
        self.burstcat["trigger_time"] = pd.to_datetime(self.burstcat["trigger_time"])

        self.trigcat = pd.read_csv(DATAPATH / "catalogs/trigcat.csv")[["trigger_time", "trigger_type", "trigger_timescale"]]
        self.trigcat["trigger_time"] = pd.to_datetime(self.trigcat["trigger_time"])

        if not self.data_path.exists():
            if not old_load:
                self.data = self._get_data()
            else:
                self.data = self._get_data_old()
            if self.timeslides is not None or self.debug_slides is not None:
                self.apply_timeslides(self.timeslides, self.debug_slides)
            if save_obj:
                self._save_data()
        else:
            self._load_data()

    def _get_data(self):
        """
        Get the data for the given date. Some detectors sometimes have more points than others, so take the minimum length.
        Also, get the good time intervals indices for the given cut length in seconds.
        """
        if self.npar and self.npar != 1:
            _, missing = self.loader.check_existing_tte(self.date, return_missing=True)
            if missing:
                self.loader.download_tte(self.date, detectors=missing, npar=self.npar)

        data = []
        gtis = []
        data_containers = []

        min_len = np.inf
        for detector in self.detectors:
            tte = self.loader.open_tte_by_date(
                self.date, detector, time_bins=self.binning, npar=self.npar, slice_time=self.slice_time
            )

            for i in range(tte.gti.num_intervals):
                gti = tte.gti[i]
                gtis.append([ceil(gti.tstart), floor(gti.tstop)])

            if detector == "n0":
                self.nai_edges = np.append(tte.data.emin, tte.data.emax[-1]).astype(np.float32)
            if detector == "b0":
                self.bgo_edges = np.append(tte.data.emin, tte.data.emax[-1]).astype(np.float32)

            data_containers.append({"counts": np.int16(tte.data.counts), "t_start": tte.data.time_range[0]})

        global_start_time = max(d["t_start"] for d in data_containers)

        aligned_arrays = []
        valid_lengths = []

        for item in data_containers:
            time_diff = global_start_time - item["t_start"]
            offset_idx = int(np.round(time_diff / self.binning))

            aligned_slice = item["counts"][offset_idx:]
            aligned_arrays.append(aligned_slice)
            valid_lengths.append(len(aligned_slice))

        min_len = min(valid_lengths)
        data = [arr[:min_len] for arr in aligned_arrays]

        data = np.concatenate(data, axis=1, dtype=np.int16)

        global_end_time = global_start_time + (min_len * self.binning)
        self.time = np.linspace(global_start_time, global_end_time, min_len)

        self.poshist = self.loader.open_poshist_by_date(self.date)

        if not (self.gtisfile.exists() or self.gtisfile_all.exists()):
            self.handle_gtis(data, gtis)
        elif self.gtisfile_all.exists():
            self.gtis = np.load(self.gtisfile_all)
            self.full_gti_inds = [
                [np.searchsorted(self.time, gti[0]), np.searchsorted(self.time, gti[1]) - 1] for gti in self.gtis
            ]
            rm = []
            for i, gti in enumerate(self.full_gti_inds):
                if gti[1] - gti[0] < self.cut_len_samples:
                    rm.append(i)
            self.full_gti_inds = np.delete(self.full_gti_inds, rm, axis=0)
        elif self.gtisfile.exists():
            self.gtis = np.load(self.gtisfile)
            self.full_gti_inds = [
                [np.searchsorted(self.time, gti[0]), np.searchsorted(self.time, gti[1])] for gti in self.gtis
            ]
            rm = []
            for i, gti in enumerate(self.full_gti_inds):
                if gti[1] - gti[0] < self.cut_len_samples:
                    rm.append(i)
            self.full_gti_inds = np.delete(self.full_gti_inds, rm, axis=0)

        data = data.astype(np.int16)

        return data

    def _get_data_old(self):
        """
        Get the data for the given date. Some detectors sometimes have more points than others, so take the minimum length.
        Also, get the good time intervals indices for the given cut length in seconds.
        """
        # else:
        #    self.loader.download_tte(self.date, npar=self.npar)  # will skip is data already exists
        
        data = []
        gtis = []
        
        min_len = np.inf
        for detector in self.detectors:
            tte = self.loader.open_tte_by_date(self.date, detector, time_bins=self.binning, npar=self.npar, slice_time=self.slice_time)

            for i in range(tte.gti.num_intervals):
                gti = tte.gti[i]
                gtis.append([ceil(gti.tstart), floor(gti.tstop)])
            
            if detector == 'n0':
                self.nai_edges = np.append(tte.data.emin, tte.data.emax[-1]).astype(np.float32)
            if detector == 'b0':
                self.bgo_edges = np.append(tte.data.emin, tte.data.emax[-1]).astype(np.float32)
            if tte.data.counts.shape[0] < min_len:
                min_len = tte.data.counts.shape[0]
                time_range = tte.data.time_range
            data.append(np.int16(tte.data.counts))

        data = [d[:min_len] for d in data]
        data = np.concatenate(data, axis=1, dtype=np.int16)

        self.time = np.linspace(time_range[0], time_range[1], min_len)
        self.poshist = self.loader.open_poshist_by_date(self.date)

        if not (self.gtisfile.exists() or self.gtisfile_all.exists()):
            self.handle_gtis(data, gtis)
        elif self.gtisfile_all.exists():
            self.gtis = np.load(self.gtisfile_all)
            self.full_gti_inds = [[np.searchsorted(self.time, gti[0]), np.searchsorted(self.time, gti[1])-1] for gti in self.gtis]
            rm = []
            for i, gti in enumerate(self.full_gti_inds):
                if gti[1] - gti[0] < self.cut_len_samples:
                    rm.append(i)
            self.full_gti_inds = np.delete(self.full_gti_inds, rm, axis=0)
        elif self.gtisfile.exists():
            self.gtis = np.load(self.gtisfile)
            self.full_gti_inds = [[np.searchsorted(self.time, gti[0]), np.searchsorted(self.time, gti[1])] for gti in self.gtis]
            rm = []
            for i, gti in enumerate(self.full_gti_inds):
                if gti[1] - gti[0] < self.cut_len_samples:
                    rm.append(i)
            self.full_gti_inds = np.delete(self.full_gti_inds, rm, axis=0)

        data = data.astype(np.int16)
        
        return data
   
    def handle_gtis(self, data, gtis):
        gtis = self.merge_gtis(gtis)
        gtis = self._outliers2gti(gtis, data, avg_window_min=2)

        if self.timeslides is not None or self.simulate or self.debug_slides is not None:
            gtis = self._bursts2gti(gtis)
            gtis = self._trigcat2gti(gtis)
        else:
            gtis = self._trigcat2gti(gtis, "pipeline")

        self.gti = self.merge_gtis(gtis)
        self.gti_inds = self.get_gti_inds(self.cut_len_seconds)

        self.full_gti_inds = self.cut_ppu_glitch(data, self.gti_inds)
        if self.binning > 0.001:
            self.full_gti_inds = self.cut_deadtime(data, self.full_gti_inds)

        self.full_gti_inds = self.merge_gtis(self.full_gti_inds, gtype="samples")

        gtis = np.array([[self.time[gtiind[0]], self.time[gtiind[1]]] for gtiind in self.full_gti_inds])
        if not self.gtisfile.parent.exists():
            self.gtisfile.parent.mkdir(parents=True)

        np.save(self.gtisfile, gtis)

    def _save_data(self):
        self.datashape = self.data.shape
        self.t0 = self.time[0]
        nonzero_bins_t, nonzero_bins_e = np.nonzero(self.data)
        self.nonzero_bins = (np.int32(nonzero_bins_t), np.int32(nonzero_bins_e))

        self.nonzero_values = self.data[self.nonzero_bins].astype(np.int16)

        tmpdata = self.data.copy()
        del self.data, self.burstcat, self.trigcat

        with open(self.data_path, "wb") as f:
            pickle.dump(self, f)

        self.data = tmpdata

    def _load_data(self):
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        simulate_ = self.simulate
        self.__dict__.update(data.__dict__)
        self.simulate = simulate_
        self.time = self.time.astype(np.float64)

        self.data = np.zeros(self.datashape)
        self.data[self.nonzero_bins] = self.nonzero_values

    def get_gti_inds(self, cut_length_seconds):
        """
        Get the good time intervals indices for the given cut length in seconds.
        """
        cut_length_samples = int(cut_length_seconds / self.binning)
        gti_inds = []
        for gti in self.gti:
            gtistart = np.searchsorted(self.time, gti[0])
            gtistop = np.searchsorted(self.time, gti[1])

            start = gtistart + cut_length_samples // 2
            stop = gtistop - cut_length_samples // 2

            if stop - start >= 120 // self.binning:
                pass
            elif self.slice_time is not None:
                pass
            else:
                continue

            if stop > start:
                if start < len(self.time) and stop < len(self.time):
                    gti_inds.append((start, stop))

        return gti_inds

    def cut_ppu_glitch(self, data, gtis):
        mean_wind = int(1 * 60 // self.binning)
        min_gti = int(2 * 60 // self.binning)
        glitch_cut = int(10 * 60 // self.binning)

        meandat = data.reshape(-1, self.ndetectors, echans).mean(axis=2)
        new_gti = []
        for gti in gtis:
            gdat = rolling_mean_padded(meandat[gti[0] : gti[1]], window=mean_wind)
            badlocs = np.where((gdat > 4 * self.binning / 0.01))
            zeros = badlocs[0] + gti[0]
            nbaddet = len(np.unique(badlocs[1]))

            if len(zeros) > 0 and nbaddet > 1:
                gti0end = np.min(zeros) - glitch_cut
                gti1start = np.max(zeros) + glitch_cut

                if (gti0end - gti[0]) > min_gti:
                    new_gti.append([gti[0], gti0end])
                if (gti[1] - gti1start) > min_gti:
                    new_gti.append([gti1start, gti[1]])

            else:
                new_gti.append(gti)

        if len(new_gti) == 0:
            return []

        return new_gti

    def cut_deadtime(self, data, gtis):
        mean_wind = 8
        min_gti = int(2 * 60 // self.binning)

        meandat = data.reshape(-1, self.ndetectors, echans).mean(axis=2)
        new_gti = []
        for gti in gtis:
            gdat = rolling_mean_padded(meandat[gti[0] : gti[1]], window=mean_wind)
            zeros = np.where(gdat == 0)[0] + gti[0]
            if len(zeros) == 0:
                new_gti.append(gti)
                continue

            zeros = np.unique(zeros)
            zeros = np.concatenate([[gti[0]], zeros, [gti[1]]])

            zeros_diff = np.diff(zeros)
            inds = np.where(zeros_diff > 1)[0]
            inds = np.concatenate([inds, [-1]])

            if gti[1] - zeros[inds[0]] <= min_gti:
                continue

            start = zeros[0]
            stop = zeros[1]
            for i in range(len(inds) - 2):
                if stop - start >= min_gti:
                    new_gti.append([start, stop])
                start = zeros[inds[i + 1]] + self.cut_len_samples // 2
                stop = zeros[inds[i + 2]] - self.cut_len_samples // 2 - mean_wind

            if stop - start >= min_gti:
                new_gti.append([start, stop])

        return new_gti

    def _outliers2gti(self, gtis, data, avg_window_min=2):
        avg_window_samp = int(avg_window_min * 60 / self.binning)
        meandat = data.mean(axis=1)
        local_mean = rolling_mean_padded(meandat, window=avg_window_samp)
        local_std = np.sqrt(rolling_mean_padded(meandat ** 2, window=avg_window_samp) - local_mean**2)
        outliers = np.where(meandat - local_mean < -5 * local_std)[0]
        outliers = np.unique(outliers)

        outdiff = np.diff(np.insert(outliers, 0, 0))
        outliers = outliers[np.where(outdiff > self.cut_len_samples)[0]]
        outliers_time = [self.time[outlier] for outlier in outliers]

        new_gtis = []
        for gti in gtis:
            for outlier in outliers_time:
                outstart = max(outlier, gti[0])
                outend = min(outlier, gti[1])
                if outstart > gti[0] and outend < gti[1]:
                    new_gtis.append([gti[0], outstart - 1])
                    new_gtis.append([outend + 1, gti[1]])
                    gti[0] = outend + 1
                else:
                    new_gtis.append(gti)

        if len(new_gtis) == 0:
            new_gtis = gtis

        return new_gtis

    def _bursts2gti(self, gtis):
        sdate = self.date - pd.Timedelta(hours=1)
        edate = self.date + pd.Timedelta(hours=1)

        burstdf = self.burstcat[
            (self.burstcat.trigger_time >= sdate) & (self.burstcat.trigger_time <= edate)
        ][["trigger_time", "t90"]]

        new_gtis = []
        for gti in gtis:
            for i in range(len(burstdf)):
                burst_time = Time(burstdf.iloc[i].trigger_time, scale='utc').fermi
                bststart = max(burst_time - burstdf.iloc[i].t90, gti[0])
                bstend = min(burst_time + burstdf.iloc[i].t90, gti[1])
                if bststart >= gti[0] and bststart <= gti[1]:
                    if bststart - gti[0] >= 120:
                        new_gtis.append([gti[0], bststart])
                if bstend >= gti[0] and bstend <= gti[1]:
                    if gti[1] - bstend >= 120:
                        new_gtis.append([bstend, gti[1]])

        if len(new_gtis) == 0:
            new_gtis = gtis

        return new_gtis

    def _trigcat2gti(self, gtis, kind="timelides"):
        sdate = self.date - pd.Timedelta(hours=1)
        edate = self.date + pd.Timedelta(hours=1)

        burstdf = self.trigcat[(self.trigcat.trigger_time >= sdate) & (self.trigcat.trigger_time <= edate)]
        if kind == "pipeline":
            burstdf = burstdf[burstdf.trigger_type == "SFLARE"]

        new_gtis = []
        for gti in gtis:
            for i in range(len(burstdf)):
                cuttime = 10 * 60 if burstdf.iloc[i].trigger_type == "SFLARE" else 0.5
                burst_time = Time(burstdf.iloc[i].trigger_time, scale='utc').fermi
                if burst_time < gti[0] or burst_time > gti[1]:
                    new_gtis.append(gti)
                    continue
                bststart = burst_time - cuttime
                bstend = burst_time + cuttime
                if bststart >= gti[0] and bststart <= gti[1]:
                    if bststart - gti[0] >= 120:
                        new_gtis.append([gti[0], bststart])
                if bstend >= gti[0] and bstend <= gti[1]:
                    if gti[1] - bstend >= 120:
                        new_gtis.append([bstend, gti[1]])

        if len(new_gtis) == 0:
            new_gtis = gtis

        return new_gtis

    def apply_symmetric_timeslides(self, minutes=1.0, slides=None):
        """
        Slide the data in time for the determination of the null triggers. We randomly slide
        each detector between linspace(0, minutes, ndetectors) (shuffeled).
        The dataloss is the paramter "minutes".
        The method is as follows:
        - For each gti, slide the data from gti[0] to gti[0]+seglen, where seglen is the length of the
          segment we keep (in order for all the slides to have the same length)
        - The gti is changed accordingly and used in the matched filter.

        Note: if there is more than one gti, the dataloss is per gti.

        params:
        -------
        minutes (float): the number of minutes to slide the data (minutes/13 is the slide between adjacent detectors)
        slides (str): used for debugging, the slides to use for each detector in minutes, seperated by '-'. If None, use random slides.
        """
        if slides is None:
            self.timeslides_minutes = np.random.permutation(np.linspace(-minutes / 2, minutes / 2, self.ndetectors))
            self.timeslides_samples = (self.timeslides_minutes * 60 // self.binning).astype(int)
        else:
            self.timeslides_minutes = np.array([float(j) for j in slides.split("-")])
            self.timeslides_samples = (self.timeslides_minutes * 60 // self.binning).astype(int)
            minutes = np.max(self.timeslides_minutes)

        maxshift = int(minutes * 60 // self.binning)
        self.data = self.data.reshape([self.data.shape[0], self.ndetectors, echans])
        newgtis = []
        for gti in self.full_gti_inds:
            seglen = gti[1] - gti[0] - maxshift
            if seglen > 0:
                slided = np.zeros([seglen, self.data.shape[1], self.data.shape[2]])
                start = gti[0] + maxshift // 2
                end = gti[0] + maxshift // 2 + seglen
                for i in range(self.ndetectors):
                    slided[:, i, :] = self.data[start + self.timeslides_samples[i] : end + self.timeslides_samples[i], i, :]

                self.data[gti[0] : gti[0] + seglen, :, :] = slided
                newgtis.append([gti[0], gti[0] + seglen])

        self.data = self.data.reshape([-1, self.ndetectors * echans])
        self.full_gti_inds = newgtis

    def apply_timeslides(self, minutes=1.0, slides=None):
        """
        Slide the data in time for the determination of the null triggers. We randomly slide
        each detector between linspace(0, minutes, ndetectors) (shuffeled).
        The dataloss is the paramter "minutes".
        The method is as follows:
        - For each gti, slide the data from gti[0] to gti[0]+seglen, where seglen is the length of the
          segment we keep (in order for all the slides to have the same length)
        - The gti is changed accordingly and used in the matched filter.

        Note: if there is more than one gti, the dataloss is per gti.

        params:
        -------
        minutes (float): the number of minutes to slide the data (minutes/13 is the slide between adjacent detectors)
        slides (str): used for debugging, the slides to use for each detector in minutes, seperated by '-'. If None, use random slides.
        """
        if slides is None:
            self.timeslides_minutes = np.random.permutation(np.linspace(0, minutes, self.ndetectors))
            self.timeslides_samples = (self.timeslides_minutes * 60 // self.binning).astype(int)

        else:
            self.timeslides_minutes = np.array([float(j) for j in slides.split("-")])
            self.timeslides_samples = (self.timeslides_minutes * 60 // self.binning).astype(int)
            minutes = np.max(self.timeslides_minutes)

        maxshift = int(minutes * 60 // self.binning)
        self.data = self.data.reshape([self.data.shape[0], self.ndetectors, echans])
        newdata = np.zeros_like(self.data)
        newgtis = []
        for gti in self.full_gti_inds:
            seglen = gti[1] - gti[0] - maxshift
            if seglen > 0:
                slided = np.zeros([seglen, self.data.shape[1], self.data.shape[2]])
                for i in range(self.ndetectors):
                    slided[:, i, :] = self.data[
                        gti[0] + self.timeslides_samples[i] : gti[0] + self.timeslides_samples[i] + seglen, i, :
                    ]

                newdata[gti[0] : gti[0] + seglen, :, :] = slided
                newgtis.append([gti[0], gti[0] + seglen])

        self.data = newdata.reshape([-1, self.ndetectors * echans])
        self.full_gti_inds = newgtis

    def zoomin(self, time_samples, length_samples, cache=True):
        """
        Zoom in the data around a given time and length (in units of samples).
        It changes the data, time and gti.
        """
        if cache:
            self.recover_time()
            self.full_time_data = self.data.copy()
            self.full_time = self.time.copy()
            self.full_gti_inds_cache = self.full_gti_inds.copy()

        start = np.maximum(time_samples - length_samples // 2, 0)
        stop = np.minimum(time_samples + length_samples // 2, len(self.time))

        self.data = self.data[start:stop, :]
        self.time = self.time[start:stop]
        self.full_gti_inds = [(0, len(self.time))]

        return

    def recover_time(self):
        """
        Recover the time that was zoomed in. It changes the data, time and gti.
        """
        if self.full_time_data is not None:
            self.data = self.full_time_data.copy()
            self.time = self.full_time.copy()
            self.full_gti_inds = self.full_gti_inds_cache.copy()

            self.full_time_data = None

    def remove_dets(self, dets):
        """
        Remove detectors from the data. It changes the data.
        params:
        -------
        dets (list): the detectors to remove
        """
        self.recover_dets()
        self.ndetectors -= len(dets)
        self.full_det_data = self.data.copy()
        self.data = np.concatenate(
            [self.full_det_data[:, echans * i : echans * (i + 1)] for i, val in enumerate(range(ndetectors)) if i not in dets],
            axis=1,
        )

    def keep_chans(self, chans):
        """
        Remove channels from the data. It changes the data.
        params:
        -------
        chans (list): the channels to keep
        """
        self.recover_chans()
        self.full_chan_data = deepcopy(self.data)
        self.full_chan_data = self.full_chan_data.reshape([self.data.shape[0], self.ndetectors, echans])
        self.data = self.full_chan_data[:, :, chans].reshape(-1, len(chans) * self.ndetectors)

    def keep1det(self, det):
        """
        Keep only one detector. It changes the data.
        params:
        -------
        det (int): the detector to keep
        """
        self.recover_dets()
        self.full_det_data = self.data.copy()
        self.data = self.data[:, echans * det : echans * (det + 1)]

    def recover_dets(self):
        """
        Recover the detectors that were removed. It changes the data.
        """
        if self.full_det_data is not None:
            self.data = self.full_det_data.copy()

            self.full_det_data = None

            self.ndetectors = 14

    def recover_chans(self):
        """
        Recover the detectors that were removed. It changes the data.
        """
        if self.full_chan_data is not None:
            self.data = deepcopy(self.full_chan_data).reshape(-1, self.ndetectors * echans)

            self.full_chan_data = None

    def merge_gtis(self, gtis, gtype="seconds"):
        if len(gtis) <= 1:
            return gtis

        min_dist = self.cut_len_seconds if gtype == "seconds" else self.cut_len_samples

        gtis = np.array(gtis)
        gtis = gtis[np.argsort(gtis[:, 0])]

        diffs = np.diff(gtis, axis=0)[:, 0]

        inds = np.where(diffs > min_dist)[0]
        inds = np.concatenate([[-1], inds, [len(gtis) - 1]])

        new_gtis = []
        for i in range(len(inds) - 1):
            start = gtis[inds[i + 1]][0]
            end = np.min(gtis[inds[i] + 1 : inds[i + 1] + 1, 1])
            new_gtis.append([start, end])

        new_gtis_diff = np.diff(new_gtis, axis=0)
        pop_inds = []
        for i in range(len(new_gtis_diff)):
            if new_gtis_diff[i][0] < 0:
                new_gtis[i + 1][0] = new_gtis[i][0]
            if new_gtis_diff[i][1] <= 0:
                pop_inds.append(i)

        new_gtis = np.delete(new_gtis, pop_inds, axis=0)

        return new_gtis

    def set_burst_duration(self, burst_duration_sec):
        self.burst_duration_sec = burst_duration_sec
        self.burst_duration_samp = int(burst_duration_sec / self.binning)

    def interp_bkg(self, bkg, old_binning, old_time):
        t0_ind = np.searchsorted(old_time, self.time[0])
        t1_ind = np.searchsorted(old_time, self.time[-1])

        t_new = np.linspace(0, t1_ind - t0_ind, len(self.time))
        t_new = np.clip(t_new, 0, t1_ind - t0_ind - 1)

        f = interp1d(np.arange(t1_ind - t0_ind), bkg[t0_ind:t1_ind], axis=0)

        self.bkgs = f(t_new)
        self.bkgs = self.bkgs * self.binning / old_binning

        self.has_bkg = True

    @property
    def total_time_used(self):
        """
        Calculate the total time used in the data.
        It is the sum of the good time intervals minus the time lost due to sharptimes.
        """
        gti_used = np.sum([gti[1] - gti[0] for gti in self.full_gti_inds]) * self.binning
        tot_used = gti_used - self.sharptime_reduce
        return tot_used


TTEHandler = TTEData

__all__ = ["DataLoaders", "TTEData", "TTEHandler"]
