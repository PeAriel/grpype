from datetime import datetime

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from grpype.detection.global_params import *
from grpype.data_io.data_handlers import DataLoaders, TTEData
from grpype.detection.utils import (
   calc_best_amp,
   calc_glitch_statistic,
   calc_mf,
   calibrate_from_norm,
   exp2_tail,
   find_peaks_2d,
   find_threshold,
   interp_posterior,
   psd_drift_svd,
   quad_rolling_mean_padded,
   rolling_mean_padded,
   square_convolve,
   fit_bkg,
)
from grpype.detection.templates import TemplateBank, TemplateGrid

from gdt.missions.fermi.time import Time
from grpype._compat import get_sun_loc


def get_detection_limit(templates, binning, kind, no_burst_date='2016-12-18T01', npar=12):
   """
   Loading the detection limit for the given binning. 
   """
   path = DATAPATH / f'detection_limit/{kind}_bins_{binning}_ntemps_{templates.shape[0]}.npy'
   if path.exists():
      amps = np.load(path)
      return amps
   
   no_burst_date = datetime.fromisoformat(no_burst_date)
   tte = TTEData(no_burst_date, binning, save_obj=False)  # TODO: change to save_obj=True
   bkg = tte.data.mean(axis=0)

   amps = find_threshold(templates*binning, bkg=bkg, nsamples=10_000, amp_init=10, prec=1e-2)

   amps = np.array(amps)
   # np.save(path, amps)
   return amps


class Detection:
   """
      A class to perform the detection.
      params:
      -------
      template_bank (TemplateBank): the template object bank to use
      glitches (GlitchTemplates): the glitch template object to use
      binning (float): the time binning in seconds
      burst_duration (float): the duration of the burst in seconds
      rolling_window_sec (float): Window for the rolling mean background calculation in seconds.
      rolling_gap_sec (float): Gap between the rolling mean windows in seconds.
   """
   def __init__(self, binning, burst_duration, rolling_window_sec=10, rolling_gap_sec=10):
      self.binning = binning
      self.set_burst_duration(burst_duration)
      self.bkg_window = int(rolling_window_sec//binning)
      self.set_rolling_gap(rolling_gap_sec)

      self.just_bkg = False

      self._reset_params()

   def match_filter(self, burstdata, tbank, glitches, slice_seconds, mf_threshold=8, min_dist_sec=60, glitch_threshold=8, glitch_extend=0, overlap=1, drift_corr=True, pe_drift_corr=True, calibrate=True):
      """
      Calculates the matched filter for the data and glitches and cleans the glitches.
      Glitches are replaced with random noise in the matched filter array.

      params:
      -------
      burstdata (TTEData): the data to calculate the matched filter
      tbank (TemplateBank): the template bank object
      glitches (GlitchTemplates): the glitch template object
      slice_seconds (int): the number of seconds to slice the data for the mean bkg. If 0, use the full data.
      mf_threshold (float): the threshold to use in the matched filter to detect grbs in units of sigma
      min_dist_sec (float): the minimum distance between peaks in seconds
      glitch_threshold (float): the threshold to use to clean the glitches in units of sigma
      glitch_extend (int): the number of samples to extend the glitch trigger in the matched filter in units of burst duration samples
      singledet (bool): if True, calculate the single detector matched filters

      returns:
      --------
      mf (np.ndarray): the glitch cleaned matched filter, shape [time, ntemplates]
      maxtimes (np.ndarray): the times of the peaks in the matched filter
      maxtemps (np.ndarray): the templates of the peaks in the matched filter
      triggers_met (list): a list of Met objects with the times of the triggers
      """
      self.ndetectors = burstdata.ndetectors
      min_dist_samp = min_dist_sec//self.binning

      if not self.just_bkg:
         tbankfull = TemplateBank(self.binning, alltemplates=True, kind=SEARCH_BANK_FOLDER)
         skytgrid = TemplateGrid(self.binning, kind=INTEGRATION_SKY_FOLDER)
         spectgrid = TemplateGrid(self.binning, kind=INTEGRATION_SPEC_FOLDER)
    
      nsing = 150

      if calibrate:
         self.tailparams = np.load(DATAPATH / f'dist_coeffs/coefficients{self.bdur_sec:.3f}.npy')[0]
         mf_threshold = calibrate_from_norm(exp2_tail, self.tailparams, mf_threshold)
         glitch_threshold = calibrate_from_norm(exp2_tail, self.tailparams, glitch_threshold)

      # mf = stats.norm.rvs(size=[burstdata.data.shape[0], tbank.ntemplates])
      mf = np.zeros([burstdata.data.shape[0], tbank.ntemplates], dtype=np.float32)
      glitch1d_mf = np.zeros([burstdata.data.shape[0], glitches.glitch1d.shape[0]], dtype=np.float32)

      # fullbkg = np.zeros(burstdata.data.shape, dtype=np.float32)
      if not burstdata.has_bkg:
         burstdata.bkgs = np.zeros(burstdata.data.shape, dtype=np.float32)

      burstdata.fltr_bkgs = []
      burstdata.fltr_bkgs_inds = []
      burstdata.trig_bkgs = []
      
      maxtimes = np.array([], dtype=int)
      maxtemps = np.array([], dtype=int)
      if not self.just_bkg:
         allzvar0 = np.array([])  # the mean of the matched filter
         allzvar = np.array([])  # the variance of the matched filter
         
         drift1s = np.empty((0, tbankfull.ntemplates), dtype=np.float32)
         drift2s = np.empty((0, tbankfull.ntemplates), dtype=np.float32)
         drift2s_sky = np.empty((0, skytgrid.ntemplates), dtype=np.float32)
         drift1s_sky = np.empty((0, skytgrid.ntemplates), dtype=np.float32)
         drift2s_spec = np.empty((0, spectgrid.ntemplates), dtype=np.float32)
         drift1s_spec = np.empty((0, spectgrid.ntemplates), dtype=np.float32)

         trigs_numers = np.empty((self.ndetectors, 0), dtype=np.float32)
         trigs_vars = np.empty([self.ndetectors, 0], dtype=np.float32)
         trigs_test_zvars0 = np.empty([len(detectors), 0], dtype=np.float32)
         trigs_test_zvars = np.empty([len(detectors), 0], dtype=np.float32)
         trigs_single_zvars0 = np.empty([len(detectors), 0], dtype=np.float32)
         trigs_single_zvars = np.empty([len(detectors), 0], dtype=np.float32)

      mean_test_th = 1.5

      sharpflag = [False]
      self.glitch_trigs = 0
      # for inds in burstdata.gti_inds:
      for inds in burstdata.full_gti_inds:
         i = inds[0]
         f = inds[1]

         if (f - i) < min_dist_samp:
            continue

         segments = int(((f - i) * self.binning) // slice_seconds) if slice_seconds != 0 else 1
         if segments > 0:
            inds = np.linspace(i, f, segments+1).astype(int)
         else:
            inds = np.array([i, f])

         for j in range(len(inds)-1):
            # ii = inds[j] if j == 0 else inds[j] - self.bdur_samp - (self.bkg_window + self.bkg_window_gap)  # We want the end of the previous segment to be the start of the next but it changes the background which messes the optimization
            ii = inds[j]
            ff = inds[j+1]
            burstdata.fltr_bkgs_inds.append([ii, ff])
            iconv = ii + self.bdur_samp//2 - (self.bdur_samp+1)%2
            fconv = ff - self.bdur_samp//2
            # trimend = (self.bkg_window + self.bkg_window_gap) * (j != len(inds)-2)
            
            print(Time(burstdata.time[iconv], format='fermi', scale='utc').utc.datetime, Time(burstdata.time[fconv], format='fermi', scale='utc').utc.datetime)

            if (ff - ii) < self.bkg_window:
               continue
            
            if not burstdata.has_bkg:
               burstdata.bkgs[ii:ff] = quad_rolling_mean_padded(burstdata.data[ii:ff], self.bkg_window, gap=self.bkg_window_gap, clipneg=True)
               burstdata.bkgs[ii:ff] = fit_bkg(burstdata.bkgs[ii:ff])

            if burstdata.simulate:
               simbkg = burstdata.bkgs[ii:ff].mean(axis=0)*np.ones_like(burstdata.bkgs[ii:ff])
               burstdata.data[ii:ff] = stats.poisson(simbkg).rvs()
               burstdata.bkgs[ii:ff] = quad_rolling_mean_padded(burstdata.data[ii:ff], self.bkg_window, gap=self.bkg_window_gap, clipneg=True)
               burstdata.bkgs[ii:ff] = fit_bkg(burstdata.bkgs[ii:ff])

            fltr_bkg = burstdata.bkgs[ii:ff].mean(axis=0)

            if (burstdata.bkgs[ii:ff].mean(axis=1).max()/fltr_bkg.mean() > mean_test_th or fltr_bkg.mean()/burstdata.bkgs[ii:ff].mean(axis=1).min() > mean_test_th) or sharpflag[-1]:
               burstdata.bkgs[ii:ff] = np.zeros_like(burstdata.bkgs[ii:ff])
               burstdata.fltr_bkgs.append(np.zeros_like(fltr_bkg))
               mf[iconv:fconv] = np.zeros([fconv-iconv, tbank.ntemplates])  #stats.norm.rvs(size=[fconv-iconv, tbank.ntemplates])
               self.sharptimes += f'{burstdata.time[ii]} {burstdata.time[ff]}\n'
               burstdata.sharptime_reduce += (burstdata.time[ff] - burstdata.time[ii])
               if len(sharpflag) == 4:
                  sharpflag = [False]
               continue

            burstdata.fltr_bkgs.append(fltr_bkg)
            if self.just_bkg:  # If we just want the background for other purposes no need to calculate the matched filter
               continue

            temp_mf, mf_numers, mf_vars, zvar0, zvar = self.roll_mf(burstdata.data[ii:ff], tbank.templates, burstdata.bkgs[ii:ff], fltr_bkg, drift_corr=drift_corr, split=True)

            jump = max(int(self.bdur_samp * (1 - overlap)), 1)
            mf[iconv:fconv:jump] = temp_mf[::jump]
            del temp_mf

            glitch1d_mf[iconv:fconv], _, _, _, _ = self.roll_mf(burstdata.data[ii:ff], glitches.glitch1d, burstdata.bkgs[ii:ff], fltr_bkg, drift_corr=False, split=False)
            del fltr_bkg
            
            glitch1d_maxtimes = self.get_glitch_times(mf, glitch1d_mf, iconv, fconv, glitch_threshold, glitch_extend)
            self.glitch_trigs += len(glitch1d_maxtimes)
            mf[np.minimum(glitch1d_maxtimes + iconv, fconv-1), :] = np.zeros([len(glitch1d_maxtimes), tbank.ntemplates])  #stats.norm.rvs(size=[len(glitch1d_maxtimes), tbank.ntemplates])
            del glitch1d_maxtimes

            maxtime, maxtemp = find_peaks_2d(mf[iconv:fconv], min_peak_dist=min_dist_samp, min_peak_height=mf_threshold)
            zeromsk = (mf[iconv:fconv][maxtime, maxtemp] < 1/np.sqrt(EPS)/10)
            maxtime = maxtime[zeromsk]
            maxtemp = maxtemp[zeromsk]

            if len(maxtime) > 0:
               maxtimes = np.concatenate([maxtimes, maxtime + iconv])
               maxtemps = np.concatenate([maxtemps, maxtemp])
               # allzvar = np.concatenate([allzvar, zvar[maxtime]], axis=0)
               allzvar0 = np.concatenate([allzvar0, zvar0[maxtime, maxtemp]])
               allzvar = np.concatenate([allzvar, zvar[maxtime, maxtemp]])

               for i in range(len(maxtime)):
                  tleft = maxtime[i] + iconv - self.bdur_samp//2 + (self.bdur_samp+1)%2
                  tright = maxtime[i] + iconv + self.bdur_samp//2 + 1
                  burstdata.trig_bkgs.append(burstdata.bkgs[tleft:tright].sum(axis=0))
                  
                  if pe_drift_corr:
                     tbankfull.load_templates()
                     drift1, drift2 = psd_drift_svd(tbankfull, [ii, ff], burstdata, self.bkg_window, self.bkg_window_gap, maxtime[i], nsing)
                     drift1s = np.concatenate([drift1s, drift1[None, :]], axis=0)
                     drift2s = np.concatenate([drift2s, drift2[None, :]], axis=0)
                     del tbankfull.templates

                     skytgrid.load_templates()
                     skytgrid.templates = skytgrid.templates.reshape(DATALEN, -1).T
                     drift1, drift2 = psd_drift_svd(skytgrid, [ii, ff], burstdata, self.bkg_window, self.bkg_window_gap, maxtime[i], nsing)
                     drift1s_sky = np.concatenate([drift1s_sky, drift1[None, :]], axis=0)
                     drift2s_sky = np.concatenate([drift2s_sky, drift2[None, :]], axis=0)
                     del skytgrid.templates

                     spectgrid.load_templates()
                     spectgrid.templates = spectgrid.templates.reshape(DATALEN, -1).T
                     drift1, drift2 = psd_drift_svd(spectgrid, [ii, ff], burstdata, self.bkg_window, self.bkg_window_gap, maxtime[i], nsing)
                     drift1s_spec = np.concatenate([drift1s_spec, drift1[None, :]], axis=0)
                     drift2s_spec = np.concatenate([drift2s_spec, drift2[None, :]], axis=0)
                     del spectgrid.templates
                  
                  else:
                     drift1s = np.concatenate([drift1s, np.zeros((1, tbankfull.ntemplates), dtype=np.float32)], axis=0)
                     drift2s = np.concatenate([drift2s, np.ones((1, tbankfull.ntemplates), dtype=np.float32)], axis=0)
                     drift1s_sky = np.concatenate([drift1s_sky, np.zeros((1, skytgrid.ntemplates), dtype=np.float32)], axis=0)
                     drift2s_sky = np.concatenate([drift2s_sky, np.zeros((1, skytgrid.ntemplates), dtype=np.float32)], axis=0)
                     drift1s_spec = np.concatenate([drift1s_spec, np.zeros((1, spectgrid.ntemplates), dtype=np.float32)], axis=0)
                     drift2s_spec = np.concatenate([drift2s_spec, np.zeros((1, spectgrid.ntemplates), dtype=np.float32)], axis=0)

               trigs_numers = np.concatenate([trigs_numers, mf_numers[:, maxtime, maxtemp]], axis=1)
               trigs_vars = np.concatenate([trigs_vars, mf_vars[:, maxtime, maxtemp]], axis=1)

               # Drift correction expecpt a detector (test_zvars) and drift correction for a single detector (single_zvars)
               test_zvars0 = np.zeros_like(mf_numers)
               test_zvars = np.zeros_like(mf_numers)
               single_zvars0 = np.zeros_like(mf_numers)
               single_zvars = np.zeros_like(mf_numers)
               for det in range(ndetectors):
                  dets = np.delete(np.arange(ndetectors), det)
                  test_zvars0[det] = rolling_mean_padded((mf_numers[dets].sum(axis=0)/np.sqrt(mf_vars[dets].sum(axis=0) + EPS)), self.bkg_window, gap=self.bkg_window_gap)
                  mf2 = mf_numers[dets].sum(axis=0)**2 / (mf_vars[dets].sum(axis=0) + EPS)
                  test_zvars[det] = rolling_mean_padded(mf2, self.bkg_window, gap=self.bkg_window_gap)
                  del mf2

                  single_zvars0[det] = rolling_mean_padded((mf_numers[det]/np.sqrt(mf_vars[det] + EPS)), self.bkg_window, gap=self.bkg_window_gap)
                  single_zvars[det] = rolling_mean_padded((mf_numers[det]**2/(mf_vars[det] + EPS)), self.bkg_window, gap=self.bkg_window_gap)
               
               del mf_numers, mf_vars

               trigs_test_zvars0 = np.concatenate([trigs_test_zvars0, test_zvars0[:, maxtime, maxtemp]], axis=1)  # All except the one that is being tested
               trigs_test_zvars = np.concatenate([trigs_test_zvars, test_zvars[:, maxtime, maxtemp]], axis=1)  # All except the one that is being tested
               trigs_single_zvars0 = np.concatenate([trigs_single_zvars0, single_zvars0[:, maxtime, maxtemp]], axis=1)  # Only the one that is being tested
               trigs_single_zvars = np.concatenate([trigs_single_zvars, single_zvars[:, maxtime, maxtemp]], axis=1)  # Only the one that is being tested
               del test_zvars0, test_zvars, single_zvars0, single_zvars

      triggers_met = []
      for i in range(len(maxtimes)):
         triggers_met.append(Time(burstdata.time[maxtimes[i]], format='fermi', scale='utc'))
      triggers_met = np.array(triggers_met)

      if self.just_bkg:
         return
      
      occ_stat = self.calc_occultation_stat(burstdata, tbank, maxtimes, maxtemps)
      shower_stat = self.calc_shower_stat(burstdata, tbank, maxtimes)
      
      tbankfull.load_templates()
      fullbank_snr = self.optimize_snr(burstdata, tbankfull, drift1s, drift2s, maxtimes)
      del tbankfull.templates
      
      skytgrid.load_templates()
      sun_statistic, earth_statistic, gcen_statistic, timing_glitch, maxtemps_skyopt = self.calc_vetoes(burstdata, tbank, skytgrid, triggers_met, maxtimes, maxtemps, drift1s_sky, drift2s_sky)
      del skytgrid.templates

      spectgrid.load_templates()
      maxtemps_specopt = self.estimate_spectrum(burstdata, spectgrid, maxtimes, maxtemps, drift1s_spec, drift2s_spec)
      del spectgrid.templates

      self._extract_params(skytgrid, spectgrid, burstdata, fullbank_snr, mf, maxtimes, maxtemps, maxtemps_specopt, maxtemps_skyopt, triggers_met, sun_statistic, earth_statistic, gcen_statistic, timing_glitch, occ_stat, shower_stat, trigs_numers, trigs_vars, allzvar, trigs_test_zvars0, trigs_test_zvars, trigs_single_zvars0, trigs_single_zvars)

      return mf, maxtimes, maxtemps, triggers_met

   def fast_d_bkg_slice(self, burstdata, trigger_time_met, slice_seconds, min_dist_sec=60, return_mf=False):
      """
      Calculates the matched filter for the data and glitches and cleans the glitches.
      Glitches are replaced with random noise in the matched filter array.

      params:
      -------
      burstdata (TTEData): the data to calculate the matched filter
      tbank (TemplateBank): the template bank object
      glitches (GlitchTemplates): the glitch template object
      slice_seconds (int): the number of seconds to slice the data for the mean bkg. If 0, use the full data.
      mf_threshold (float): the threshold to use in the matched filter to detect grbs in units of sigma
      min_dist_sec (float): the minimum distance between peaks in seconds
      glitch_threshold (float): the threshold to use to clean the glitches in units of sigma
      glitch_extend (int): the number of samples to extend the glitch trigger in the matched filter in units of burst duration samples
      singledet (bool): if True, calculate the single detector matched filters

      returns:
      --------
      mf (np.ndarray): the glitch cleaned matched filter, shape [time, ntemplates]
      maxtimes (np.ndarray): the times of the peaks in the matched filter
      maxtemps (np.ndarray): the templates of the peaks in the matched filter
      triggers_met (list): a list of Met objects with the times of the triggers
      """
      min_dist_samp = min_dist_sec//self.binning
      
      bkgs = np.zeros(burstdata.data.shape, dtype=np.float32)
      
      # for inds in burstdata.gti_inds:
      for inds in burstdata.full_gti_inds:
         i = inds[0]
         f = inds[1]

         if (f - i) < min_dist_samp:
            continue

         segments = int(((f - i) * self.binning) // slice_seconds) if slice_seconds != 0 else 1
         if segments > 0:
            inds = np.linspace(i, f, segments+1).astype(int)
         else:
            inds = np.array([i, f])

         for j in range(len(inds)-1):
            ii = inds[j]
            ff = inds[j+1]
            iconv = ii + self.bdur_samp//2 - (self.bdur_samp+1)%2
            fconv = ff - self.bdur_samp//2

            trigger_time_samp = np.searchsorted(burstdata.time, trigger_time_met)
            if not (iconv <= trigger_time_samp <= fconv):
               continue
            
            print(Time(burstdata.time[iconv], format='fermi', scale='utc').utc.datetime, Time(burstdata.time[fconv], format='fermi', scale='utc').utc.datetime)

            if (ff - ii) < self.bkg_window:
               continue
            
            bkgs[ii:ff] = quad_rolling_mean_padded(burstdata.data[ii:ff], self.bkg_window, gap=self.bkg_window_gap, clipneg=True)
            bkgs[ii:ff] = fit_bkg(bkgs[ii:ff])

            if return_mf:
               fltr_bkg = bkgs[ii:ff].mean(axis=0)
               tbank = TemplateBank(self.binning)
               tbank.load_templates()
               temp_mf, mf_numers, mf_vars, zvar0, zvar = self.roll_mf(burstdata.data[ii:ff], tbank.templates, bkgs[ii:ff], fltr_bkg, drift_corr=True, split=False)
               return temp_mf, mf_numers, mf_vars, zvar0, zvar

            # tleft = trigger_time_samp - self.bdur_samp//2 + (self.bdur_samp+1)%2
            # tright = trigger_time_samp + self.bdur_samp//2 + 1

            # trig_data = burstdata.data[tleft:tright]#.sum(axis=0).astype(np.int64)
            # trig_bkg = bkgs[tleft:tright]#.sum(axis=0)

            trigind = trigger_time_samp - ii
            return burstdata.data[ii:ff], bkgs[ii:ff], trigind

      return None, None, None

   def calc_shower_stat(self, burstdata, tbank, maxtimes):
      shower_stats = np.zeros(len(maxtimes))

      for i in range(len(maxtimes)):
         tleft = maxtimes[i] - self.bdur_samp//2 + (self.bdur_samp+1)%2
         tright = maxtimes[i] + self.bdur_samp//2 + 1

         d = burstdata.data[tleft:tright].sum(axis=0)
         bkg = burstdata.bkgs[tleft:tright].sum(axis=0)

         mf = calc_mf(d, bkg, tbank.templates)
         bf_ind = np.argmax(mf)
         amp = calc_best_amp(d, bkg, tbank.templates[bf_ind])
         
         ind_mfs = np.zeros(ndetectors-2)
         for det in range(ndetectors-2):
            dsing = d[det*echans:(det+1)*echans]
            bkgsing = bkg[det*echans:(det+1)*echans]
            temp_sing = tbank.templates[bf_ind, det*echans:(det+1)*echans]
            ind_mfs[det] = calc_mf(dsing, bkgsing, temp_sing)[0]  # All the values are the same since we broadcast

         bf_det = np.argmax(ind_mfs)
         shower_template = tbank.templates[bf_ind, bf_det*echans:(bf_det+1)*echans]
         shower_template = np.tile(shower_template, ndetectors)
         shower_template[echans*12:] = 0  # BGO detectors are not used in the shower template

         # shower_mf = calc_mf(d, bkg, shower_template)[0]  # Not really needed. Just for debugging/understanding
         shower_amp = calc_best_amp(d, bkg, shower_template)

         fltr = np.log((1 + shower_amp*shower_template/(bkg + EPS))/(1 + amp*tbank.templates[bf_ind]/(bkg + EPS)))
         stat_numer = np.dot(fltr, d - amp*tbank.templates[bf_ind] - bkg)
         stat_var = np.dot(fltr**2, bkg + amp*tbank.templates[bf_ind])

         shower_stats[i] = stat_numer / np.sqrt(stat_var + EPS)
         
      return shower_stats

   def calc_occultation_stat(self, burstdata, tbank, maxtimes, maxtemps):
      occ_stats = np.zeros(len(maxtimes))

      for i in range(len(maxtimes)):
         tleft = maxtimes[i] - self.bdur_samp//2 + (self.bdur_samp+1)%2
         tright = maxtimes[i] + self.bdur_samp//2 + 1

         offsets = np.linspace(2*self.bkg_window_gap, self.bkg_window/2, 1000)
         mfs = np.zeros(len(offsets))

         for j in range(len(offsets)):
            delta = int(offsets[j])

            if (tleft - delta < 0) or (tright + delta > burstdata.bkgs.shape[0]):
               if self.bdur_sec > 2:
                  mfs[j] = 999
               continue

            d = burstdata.data[tleft:tright].sum(axis=0)
            d_left = burstdata.data[tleft-delta:tright-delta].sum(axis=0)
            
            bkg = burstdata.bkgs[tleft:tright].sum(axis=0)
            bkg_right = burstdata.bkgs[tleft+delta:tright+delta].sum(axis=0)

            template = tbank.templates[maxtemps[i]]

            amp = calc_best_amp(d, bkg, template)
            mfs[j] = calc_mf(d_left, bkg_right, amp*template)[0]

         occ_stats[i] = np.median(mfs)
      
      return occ_stats

   def optimize_snr(self, burstdata, tbankfull, drift1s, drift2s, maxtimes):
      fullbank_snr = np.zeros(len(maxtimes), dtype=np.float32)

      for i in range(len(maxtimes)):
         tleft = maxtimes[i] - self.bdur_samp//2 + (self.bdur_samp+1)%2
         tright = maxtimes[i] + self.bdur_samp//2 + 1
         d = burstdata.data[tleft:tright].sum(axis=0).astype(np.int64)
         bkgw = burstdata.bkgs[tleft:tright].sum(axis=0)

         # b1 = np.clip(drift1s[i], -0.25, 0.25)
         # b2 = np.clip(drift2s[i], 0.85**2, 1.25**2)
         b1 = drift1s[i]
         b2 = drift2s[i]
         mfopt = calc_mf(d, bkgw, tbankfull.templates)
         mfopt = (mfopt - b1) / np.sqrt(b2)
         
         mfoptind = np.argmax(mfopt)
         fullbank_snr[i] = mfopt[mfoptind]

      return fullbank_snr

   def estimate_spectrum(self, burstdata, spectgrid, maxtimes, maxtemps, drift1s_spec, drift2s_spec):
      maxtemps_specopt = np.zeros_like(maxtemps)
      for i in range(len(maxtimes)):
         tleft = maxtimes[i] - self.bdur_samp//2 + (self.bdur_samp+1)%2
         tright = maxtimes[i] + self.bdur_samp//2 + 1
         d = burstdata.data[tleft:tright].sum(axis=0).astype(np.int64)
         bkgw = burstdata.trig_bkgs[i]

         b1 = drift1s_spec[i].reshape(spectgrid.templates.shape[1], spectgrid.templates.shape[2])
         b2 = drift2s_spec[i].reshape(spectgrid.templates.shape[1], spectgrid.templates.shape[2])
         integrand = spectgrid.calc_posterior(d, bkgw, slc=50, psd_drift=[b1, b2])
         spec_posterior = integrand.sum(axis=1)

         maxtemps_specopt[i] = np.argmax(spec_posterior)

      return maxtemps_specopt

   def calc_vetoes(self, burstdata, tbank, tgrid, triggers_met, maxtimes, maxtemps, drift1s_sky, drift2s_sky):
      maxtemps_skyopt = np.zeros_like(maxtemps)
      sun_statistic = np.zeros(len(maxtimes), dtype=np.float32)
      earth_statistic = np.zeros(len(maxtimes), dtype=np.float32)
      gcen_statistic = np.zeros(len(maxtimes), dtype=np.float32)
      timing_glitch = np.zeros(len(maxtimes))

      gcen = np.array([[266.4], [-28.9]])
      for i in range(len(maxtimes)):
         tleft = maxtimes[i] - self.bdur_samp//2 + (self.bdur_samp+1)%2
         tright = maxtimes[i] + self.bdur_samp//2 + 1
         d = burstdata.data[tleft:tright].sum(axis=0).astype(np.int64)
         bkgw = burstdata.trig_bkgs[i]
         timing_glitch[i] = calc_glitch_statistic(d, bkgw, tbank.templates, bkgw, return_best=False)

         slc = 50

         b1 = drift1s_sky[i].reshape(tgrid.templates.shape[1], tgrid.templates.shape[2])
         b2 = drift2s_sky[i].reshape(tgrid.templates.shape[1], tgrid.templates.shape[2])
         integrand = tgrid.calc_posterior(d, bkgw, slc=slc, psd_drift=[b1, b2])

         sky_posterior = integrand.sum(axis=0)
         maxtemps_skyopt[i] = np.argmax(sky_posterior)
         
         sky_posterior, ra_pix, dec_pix = interp_posterior(tgrid, burstdata, triggers_met[i].fermi, sky_posterior, nside=128)

         inflate = 2
         sun_mask = tgrid.is_occulted(burstdata, triggers_met[i].fermi, 'sun', ras=ra_pix, decs=dec_pix, inflate=inflate)
         earth_mask = tgrid.is_occulted(burstdata, triggers_met[i].fermi, 'earth', ras=ra_pix, decs=dec_pix)
         gcen_mask = tgrid.is_occulted(burstdata, triggers_met[i].fermi, 'gcen', rad=0.5, cen=gcen, ras=ra_pix, decs=dec_pix, inflate=inflate)
         no_es_mask = ~(sun_mask | earth_mask | gcen_mask)

         res_e = np.sum(sky_posterior[earth_mask]/earth_mask.sum())
         res_s = np.sum(sky_posterior[sun_mask]/sun_mask.sum()/inflate)
         res_gcen = np.sum(sky_posterior[gcen_mask]/gcen_mask.sum()/inflate)
         res_nes = np.sum(sky_posterior[no_es_mask]/no_es_mask.sum())

         sun_statistic[i] = res_s/res_nes
         earth_statistic[i] = res_e/res_nes
         gcen_statistic[i] = res_gcen/res_nes

         del sun_mask, earth_mask, no_es_mask, d, bkgw, integrand, sky_posterior

      return sun_statistic, earth_statistic, gcen_statistic, timing_glitch, maxtemps_skyopt

   def get_glitch_times(self, mf, glitch1d_mf, iconv, fconv, glitch_threshold, glitch_extend):
      glitch1d_maxtimes, glitch1d_maxtemps = np.where(np.logical_and(np.abs(glitch1d_mf[iconv:fconv]) >= glitch_threshold, np.abs(glitch1d_mf[iconv:fconv]) < 1/np.sqrt(EPS)))
      
      glitchmask = glitch1d_mf[glitch1d_maxtimes + iconv, glitch1d_maxtemps] > mf[glitch1d_maxtimes + iconv, :].max(axis=1)
      glitch1d_maxtimes = glitch1d_maxtimes[glitchmask]
      glitch1d_maxtemps = glitch1d_maxtemps[glitchmask]

      extend = int(glitch_extend*self.bdur_samp//2)
      indices = np.arange(-extend, extend + 1)
      glitch1d_maxtimes = np.unique((glitch1d_maxtimes[:, np.newaxis] + indices).flatten()).astype(int)
      
      return glitch1d_maxtimes

   def roll_mf(self, d, templates, bkg, fltr_bkg, drift_corr=True, split=True):
      fltr = np.log1p(templates/(fltr_bkg + EPS))
      mf_numers = square_convolve(d - bkg, self.bdur_samp, fltr, split)
      mf_vars = square_convolve(bkg, self.bdur_samp, fltr**2, split)
      del fltr

      if split:
         mf_numer = mf_numers.sum(axis=0)
         mf_var = mf_vars.sum(axis=0)
         temp_mf = mf_numer / (np.sqrt(mf_var) + EPS)
         del mf_numer, mf_var
      else:
         temp_mf = mf_numers / (np.sqrt(mf_vars) + EPS)

      if drift_corr:
         zvar0 = rolling_mean_padded(temp_mf, self.bkg_window, gap=self.bkg_window_gap)
         zvar = rolling_mean_padded(temp_mf**2, self.bkg_window, gap=self.bkg_window_gap)
         zvar -= zvar0**2
         temp_mf = (temp_mf - zvar0) / np.sqrt(zvar + EPS)
         return temp_mf, mf_numers, mf_vars, zvar0, zvar
      
      zvar0 = np.zeros_like(temp_mf)
      zvar = np.ones_like(temp_mf)
      return temp_mf, mf_numers, mf_vars, zvar0, zvar

   def plot_mf(self, burstdata, mf, maxtimes, maxtemps, triggers_met, timeslides=None, simulate=False, save=False, show=False, return_figax=False):
      if len(maxtimes) == 0:
         return
      res_str = f'results/{burstdata.date.year}/figures' if timeslides is None else f'results_timeslides/{burstdata.date.year}/figures'
      res_str = f'results_simul/{burstdata.date.year}/figures' if simulate else res_str
      results_path = DATAPATH / res_str / f"{triggers_met[0].utc.datetime.strftime('%y%m%d%H')}_mf"
   
      # msk = np.zeros([burstdata.data.shape[0]], dtype=bool)
      # for gti in burstdata.full_gti_inds:
      #    msk[gti[0]:gti[1]] = True
      msk = (mf != 0)

      fig, axes = plt.subplots(1, 2, figsize=(14, 4))
      axes[0].plot(burstdata.time, mf[:, maxtemps], alpha=0.5, label=f'statistic for all trigger templates ({len(maxtemps)})')
      axes[0].vlines(burstdata.time[maxtimes], 0, np.max(mf)+5, color='orange', linestyle='--', linewidth=1)
      for i in range(len(maxtimes)):
         axes[0].text(burstdata.time[maxtimes][i], 7, s=f'{mf[maxtimes, maxtemps][i]:.2f}', alpha=0.5, fontsize=10)

      axes[0].legend([f"Trigger time: {triggers_met[i].utc.datetime.strftime('%H:%M:%S:%f')}" for i in range(len(maxtimes))], fontsize=12, loc='upper left')

      axes[0].set_xticklabels([datetime.utcfromtimestamp(t).strftime('%H:%M:%S') for t in axes[0].get_xticks()], rotation=25)
      axes[0].set_title(f'match filter time series for triggers', fontsize=12)

      # for i in range(len(maxtimes)):
      #    msk[maxtimes[i]-200:maxtimes[i]+200] = False
      # for i in range(len(maxtimes)):
      #    msk[maxtimes[i]] = True
      for i in range(len(maxtimes)):
         axes[1].hist(mf[:, maxtemps[i]][msk[:, i]], bins=100, density=True, histtype='step', linewidth=2)
      
      x = np.linspace(-5, 5, 1000)
      axes[1].plot(x, stats.norm.pdf(x), color='k', linestyle='--', linewidth=1)

      axes[1].set_yscale('log')
      axes[1].set_title(f'match filter statistic histogram for triggers', fontsize=12)

      fig.suptitle(f'date: {triggers_met[0].utc.datetime.strftime("%Y-%m-%d")}, binning: {self.binning} sec/samp, square pulse duration: {self.bdur_sec} sec, bkg window: 2$\\times${self.bkg_window*self.binning:.3f} sec, gap: {self.bkg_window_gap*self.binning:.3f} sec', fontsize=12)
      if save:
         if not results_path.parent.exists():
            results_path.parent.mkdir(parents=True)

         fig.savefig(results_path.as_posix() + '.png', dpi=100)
      if show:
         plt.show()
         return
      if return_figax:
         return fig, axes

      plt.close('all')
      return

   def plot_bkg(self, burstdata, mf, maxtimes, maxtemps, triggers_met, timeslides=None, simulate=False, save=False, show=False, return_figax=False):
      if len(maxtimes) == 0:
         return
      
      res_str = f'results/{burstdata.date.year}/figures' if timeslides is None else f'results_timeslides/{burstdata.date.year}' + '/figures'
      res_str = f'results_simul/{burstdata.date.year}/figures' if simulate else res_str
      results_path = DATAPATH / res_str / f"{triggers_met[0].utc.datetime.strftime('%y%m%d%H')}_bkg"

      fig, ax = plt.subplots(figsize=(10, 4))
      ax.plot(burstdata.time, burstdata.data.mean(axis=1), alpha=0.5)
      ax.plot(burstdata.time, burstdata.bkgs.mean(axis=1), alpha=0.5)
      for i, fltrbkg in enumerate(burstdata.fltr_bkgs):
         ax.hlines(fltrbkg.mean(), burstdata.time[burstdata.fltr_bkgs_inds[i][0]], burstdata.time[burstdata.fltr_bkgs_inds[i][1]])
      for i, t in enumerate(maxtimes):
         ax.axvline(burstdata.time[t], color='k', alpha=0.25, linestyle='--')
         ax.text(burstdata.time[t], ax.get_yticks()[1], s=f'{mf[maxtimes[i], maxtemps[i]]:.1f}', alpha=0.5, fontsize=12)

      ax.set_xticklabels([datetime.utcfromtimestamp(t).strftime('%H:%M:%S') for t in ax.get_xticks()], rotation=25, fontsize=12)
      ax.set_title('mean data over axis=1 (dets*echans) vs bkg')
      ax.legend(['data', 'bkg', 'fltr bkg'])

      if save:
         fig.savefig(results_path.as_posix() + '.png', dpi=100)
      if show:
         plt.show()
         return
      if return_figax:
         return fig, ax
      
      plt.close('all')
      return

   def plot_burst(self, trignum, tbank, rng, burstdata, mf, maxtimes, maxtemps, triggers_met, timeslides=None, simulate=False, save=False, show=False, return_figax=False):
      res_str = f'results/{burstdata.date.year}/figures' if timeslides is None else f'results_timeslides/{burstdata.date.year}' + '/figures'
      res_str = f'results_simul/{burstdata.date.year}/figures' if simulate else res_str
      results_path = DATAPATH / res_str / f"{mf[maxtimes[trignum], maxtemps[trignum]]:.2f}_{triggers_met[0].utc.datetime.strftime('%y%m%d%H')}_detectors"

      for i, interval in enumerate(burstdata.fltr_bkgs_inds):
         if interval[0] <= maxtimes[trignum] <= interval[1]:
            trig_ind = i
            break
      
      trigs_numers = [[float(n) for n in numer.split(',')] for numer in self.single_numers[0]]
      trigs_vars = [[float(v) for v in var.split(',')] for var in self.single_vars[0]]

      trigs_numers = np.array(trigs_numers).T
      trigs_vars = np.array(trigs_vars).T

      singles = trigs_numers/np.sqrt(trigs_vars.sum(axis=0))
      
      fig, ax = plt.subplots(14, 2, figsize=(16, 20))
      for i, det in enumerate(detectors):
         ax[i, 0].imshow(burstdata.data[maxtimes[trignum]-rng:maxtimes[trignum]+rng, echans*i:echans*(i+1)], aspect='auto')
         ax[i, 0].set_ylabel(f'{det} = {singles[i, trignum]:.2f}')

         ax[i, 1].plot(burstdata.data[maxtimes[trignum]-rng:maxtimes[trignum]+rng, echans*i:echans*(i+1)].mean(axis=0), label='mean data')
         ax[i, 1].plot(tbank.templates[maxtemps[trignum], :].reshape([len(detectors), echans]).T[:, i], label='template')
         ax[i, 1].plot(burstdata.fltr_bkgs[trig_ind].reshape([len(detectors), echans]).T[:, i], label='bkg')

      ax[0, 1].legend()
      
      if save:
         fig.savefig(results_path.as_posix() + '.png', dpi=100)
      if show:
         plt.show()
         return
      if return_figax:
         return fig, ax
      
      plt.close('all')
      return

   def plot_detectors(self, trignum, tbank, rng, burstdata, mf, maxtimes, maxtemps, triggers_met, timeslides=None, simulate=False, save=False, show=False, return_figax=False, rebin=0):
      res_str = f'results/{burstdata.date.year}/figures' if timeslides is None else f'results_timeslides/{burstdata.date.year}' + '/figures'
      res_str = f'results_simul/{burstdata.date.year}/figures' if simulate else res_str
      results_path = DATAPATH / res_str / f"{mf[maxtimes[trignum], maxtemps[trignum]]:.2f}_{triggers_met[0].utc.datetime.strftime('%y%m%d%H')}_detectors"

      for i, interval in enumerate(burstdata.fltr_bkgs_inds):
         if interval[0] <= maxtimes[trignum] <= interval[1]:
            trig_ind = i
            break
      
      trigs_numers = [[float(n) for n in numer.split(',')] for numer in self.single_numers[0]]
      trigs_vars = [[float(v) for v in var.split(',')] for var in self.single_vars[0]]

      trigs_numers = np.array(trigs_numers).T
      trigs_vars = np.array(trigs_vars).T

      singles = trigs_numers/np.sqrt(trigs_vars.sum(axis=0))
      real_singles = trigs_numers**2/(trigs_vars+1e-6)
      
      x = np.arange(-rng, rng)
      if rebin:
         dat = burstdata.data.reshape(burstdata.data.shape[0]//rebin, rebin, -1).sum(axis=1)
         dat = dat[maxtimes[trignum]//rebin-rng:maxtimes[trignum]//rebin+rng]
         bins = self.binning*rebin
      
      else:
         dat = burstdata.data[maxtimes[trignum]-rng:maxtimes[trignum]+rng]
         bins = self.binning

      fig, ax = plt.subplots(5, 3, figsize=(16, 20), sharey=True)
      idx = 0
      for i in range(5):
         for j in range(3):
            ax[i, j].imshow(dat[:, echans*idx:echans*(idx+1)], aspect='auto')
            ax[i, j].set_title(f'{detectors[idx]} $snr^2$ = {real_singles[idx, trignum]:.2f}', fontsize=14)
            
            if j == 0:
               ax[i, j].set_ylabel(f'time [{bins} second bins]', fontsize=14)
               ax[i, j].set_yticks(np.arange(0, 2 * rng, 25))  # Set y-ticks at intervals of 10 within the data range
               ax[i, j].set_yticklabels(np.arange(-rng, rng, 25))  # Set y-tick labels to match the tick positions
            
            ax[i, j].set_xticks(np.arange(8, 128, 32))
            if i < 4:
               ax[i, j].set_xticklabels([round(burstdata.nai_edges[e], 1) for e in ax[i, j].get_xticks()], fontsize=14)
            else:
               ax[i, j].set_xticklabels([round(burstdata.bgo_edges[e], 1) for e in ax[i, j].get_xticks()], fontsize=14)
               ax[i, j].set_xlabel('Energy [keV]', fontsize=14)

            idx += 1
            if idx == len(detectors):
               break

      fig.delaxes(ax[-1, -1])
      fig.tight_layout()

      if save:
         fig.savefig(results_path.as_posix() + '.png', dpi=100)
      if show:
         plt.show()
         return
      if return_figax:
         return fig, ax
      
      plt.close('all')
      return

   def save_triggers(self, date, timeslides=None, simulate=False, clean=True, filename='triggers', slc_ind=None):
      res_str = f'results/{date.year}/{filename}.csv' if timeslides is None else f'results_timeslides/{date.year}/{filename}.csv'
      res_str = f'results_simul/{date.year}/{filename}.csv' if simulate else res_str
      trigpath = DATAPATH / res_str

      sharp_str = f'results/{date.year}/sharptimes.txt' if timeslides is None else f'results_timeslides/{date.year}/sharptimes.txt'
      sharp_str = f'results_simul/{date.year}/sharptimes.txt' if simulate else sharp_str
      sharp_path = DATAPATH / sharp_str

      if not sharp_path.exists():
         sharp_path.parent.mkdir(exist_ok=True)

      with open(sharp_path, 'a') as f:
         f.write(self.sharptimes)

      if not trigpath.exists():
         detsnumer = ','.join([d + '_numer' for d in detectors])
         detsvar = ','.join([d + '_var' for d in detectors])
         dets_test_zvars0 = ','.join([d + '_test_zvar0' for d in detectors])
         dets_test_zvars = ','.join([d + '_test_zvar' for d in detectors])
         dets_single_zvars0 = ','.join([d + '_single_zvar0' for d in detectors])
         dets_single_zvars = ','.join([d + '_single_zvar' for d in detectors])
         
         with open(trigpath, 'a') as f:
            f.write('')
            # f.write('trigtime,snr,fullbank_snr,timescale,binning,,slc_ind,glitch_trigs,template_num,phi_fermi,theta_fermi,phi_sun_fermi,theta_sun_fermi,phi_earth_fermi,theta_earth_fermi,ra,dec,alpha,beta,epeak,sun_stat,earth_stat,gcen_stat,timing_stat,occultation_stat,shower_stat,mcilwin,saa_passage,timeslides,' + detsnumer + ',' + detsvar + ',allzvar,'  + dets_test_zvars0 + ','  + dets_test_zvars + ',' + dets_single_zvars0 + ',' + dets_single_zvars +'\n')

      if timeslides is not None:
         timeslides = "-".join(map(str, timeslides))
      
      for i in range(len(self.trigtimes)):
         trigtime = self.trigtimes[i]
         snr = self.snrs[i]
         fullbank_snr = self.fullbank_snrs[i]
         template_num = self.template_num[i]
         phi_fermi = self.phis_fermi[i]
         theta_fermi = self.thetas_fermi[i]
         alpha = self.alphas[i]
         beta = self.betas[i]
         epeak = self.epeaks[i]
         sun_stat = self.sun_statistic[i]
         earth_stat = self.earth_statistic[i]
         gcen_stat = self.gcen_statistic[i]
         timing_glitch = self.timing_glitch[i]
         occ_stat = self.occ_stat[i]
         shower_stat = self.shower_stat[i]
         ra = self.ras[i]
         dec = self.decs[i]
         sun_phi = self.sun_phis[i]
         sun_theta = self.sun_thetas[i]
         earth_phi = self.earth_phis[i]
         earth_theta = self.earth_thetas[i]
         mcilwin = self.mcilwains[i]
         saa_passage = self.saa_passages[i]
         single_numers = self.single_numers[0][i]
         single_vars = self.single_vars[0][i]
         all_zvars = self.allzvars[i]
         test_zvars0 = self.trigs_test_zvars0[0][i]
         test_zvars = self.trigs_test_zvars[0][i]
         single_zvars0 = self.trigs_single_zvars0[0][i]
         single_zvars = self.trigs_single_zvars[0][i]
         
         with open(trigpath, 'a') as f:
            f.write(f'{trigtime},{snr:.2f},{fullbank_snr:.2f},{self.bdur_sec:.3f},{self.binning},{slc_ind},{self.glitch_trigs:.3f},{template_num},{phi_fermi:.2f},{theta_fermi:.2f},{sun_phi:.2f},{sun_theta:.2f},{earth_phi:.2f},{earth_theta:.2f},{ra:.2f},{dec:.2f},{alpha:.2f},{beta:.2f},{epeak:.2f},{sun_stat:.3f},{earth_stat:.3f},{gcen_stat:.3f},{timing_glitch:.3f},{occ_stat:.5f},{shower_stat:.3f},{mcilwin:.3f},{saa_passage},{timeslides},{single_numers},{single_vars},{all_zvars:.3f},{test_zvars0},{test_zvars},{single_zvars0},{single_zvars}\n')

      if clean:
         self._reset_params()

   def _extract_params(self, skytgrid, spectgrid, burstdata, fullbank_snr, mf, maxtimes, maxtemps, maxtemps_specopt, maxtemps_skyopt, triggers_met, sun_statistic, earth_statistic, gcen_statistic, timing_glitch, occ_stat, shower_stat, trigs_numers, trigs_vars, allzvar, trigs_test_zvars0, trigs_test_zvars, trigs_single_zvars0, trigs_single_zvars):
      for i in range(len(maxtimes)):
         self.trigtimes.append(triggers_met[i].utc.datetime.strftime('%Y-%m-%d %H:%M:%S.%f'))
         self.snrs.append(mf[maxtimes[i], maxtemps[i]])
         self.fullbank_snrs.append(fullbank_snr[i])
         self.template_num.append(maxtemps[i])
         self.phis_fermi.append(skytgrid.phis[maxtemps_skyopt[i]])
         self.thetas_fermi.append(skytgrid.thetas[maxtemps_skyopt[i]])
         self.alphas.append(spectgrid.alphas[maxtemps_specopt[i]])
         self.betas.append(spectgrid.betas[maxtemps_specopt[i]])
         self.epeaks.append(spectgrid.epeaks[maxtemps_specopt[i]])
         self.allzvars.append(allzvar[i])

         self.sun_statistic.append(sun_statistic[i])
         self.earth_statistic.append(earth_statistic[i])
         self.gcen_statistic.append(gcen_statistic[i])
         self.timing_glitch.append(timing_glitch[i])
         self.occ_stat.append(occ_stat[i])
         self.shower_stat.append(shower_stat[i])
         
         if not (triggers_met[i].fermi >= burstdata.poshist.time_range[0] and triggers_met[i].fermi <= burstdata.poshist.time_range[1]):
            loader = DataLoaders()
            burstdata.poshist = loader.open_poshist_by_date(triggers_met[i].utc.datetime)

         ras, decs = burstdata.poshist.to_equatorial(skytgrid.phis[[maxtemps_skyopt[i]]], skytgrid.thetas[[maxtemps_skyopt[i]]], triggers_met[i].fermi)
         sun_phi, sun_theta = burstdata.poshist.to_fermi_frame(*get_sun_loc(triggers_met[i].fermi), triggers_met[i].fermi)
         earth_phi, earth_theta = burstdata.poshist.to_fermi_frame(*burstdata.poshist.get_geocenter_radec(triggers_met[i].fermi), triggers_met[i].fermi)
         saa_passage = burstdata.poshist.get_saa_passage(burstdata.time[maxtimes[i]])
         mcilwain = burstdata.poshist.get_mcilwain_l(burstdata.time[maxtimes[i]])
         
         self.ras.append(np.float32(ras))
         self.decs.append(np.float32(decs))
         self.sun_phis.append(np.float32(sun_phi))
         self.sun_thetas.append(np.float32(sun_theta))
         self.earth_phis.append(np.float32(earth_phi))
         self.earth_thetas.append(np.float32(earth_theta))
         self.mcilwains.append(np.float32(mcilwain))
         self.saa_passages.append(bool(saa_passage))
      
      self.single_numers.append([','.join(map(str, np.around(trigs_numers[:, i], 10))) for i in range(trigs_numers.shape[1])])
      self.single_vars.append([','.join(map(str, np.around(trigs_vars[:, i], 10))) for i in range(trigs_vars.shape[1])])
      
      self.trigs_test_zvars0.append([','.join(map(str, np.around(trigs_test_zvars0[:, i], 10))) for i in range(trigs_test_zvars.shape[1])])
      self.trigs_test_zvars.append([','.join(map(str, np.around(trigs_test_zvars[:, i], 10))) for i in range(trigs_test_zvars.shape[1])])
      self.trigs_single_zvars0.append([','.join(map(str, np.around(trigs_single_zvars0[:, i], 10))) for i in range(trigs_single_zvars0.shape[1])])
      self.trigs_single_zvars.append([','.join(map(str, np.around(trigs_single_zvars[:, i], 10))) for i in range(trigs_single_zvars.shape[1])])

   def _reset_params(self):
      self.trigtimes = []
      self.snrs = []
      self.fullbank_snrs = []
      self.template_num = []
      self.phis_fermi = []
      self.thetas_fermi = []
      self.alphas = []
      self.betas = []
      self.epeaks = []
      self.sun_statistic = []
      self.earth_statistic = []
      self.gcen_statistic = []
      self.timing_glitch = []
      self.occ_stat = []
      self.shower_stat = []
      self.ras = []
      self.decs = []
      self.sun_phis = []
      self.sun_thetas = []
      self.earth_phis = []
      self.earth_thetas = []
      self.saa_passages = []
      self.mcilwains = []
      self.single_numers = []
      self.single_vars = []
      self.allzvars = []
      self.trigs_test_zvars0 = []
      self.trigs_test_zvars = []
      self.trigs_single_zvars0 = []
      self.trigs_single_zvars = []
      self.sharptimes = ''

   def set_burst_duration(self, burst_duration):
      self.bdur_sec = burst_duration
      self.bdur_samp = int(burst_duration / self.binning)

   def set_rolling_gap(self, rolling_gap_sec):
      self.bkg_window_gap = np.int16(rolling_gap_sec//self.binning)


