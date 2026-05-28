import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from gdt.missions.fermi.time import Time
from grpype._compat import get_sun_loc, haversine, find_greedy_credible_levels

from grpype.detection.global_params import *
from grpype.data_io.data_handlers import DataLoaders
from grpype.detection.utils import interp_posterior


def resolve_path(path_str, ptype='templates'):
   path = Path(path_str)
   if path.is_absolute():
      return path
   if path.parts and path.parts[0] == ptype and len(path.parts) > 1:
      path = Path(*path.parts[1:])
   return DATAPATH / ptype / path


class TemplateBank:
   """
      A class to load the template bank and parameters. The templates are normalized to and are
      multiplied by the detection limit and binning to get the correct amplitude.

      params:
      -------
      binning (float): the time binning in seconds
      quantile (float): the quantile to use to remove bad templates

      attributes:
      -----------
      templates (np.ndarray): the templates, shape [ntemplates, nenergies*ndetectors]
      amps (np.ndarray): the detection limit for each template
      alphas (np.ndarray): the alpha parameters of the templates
      betas (np.ndarray): the beta parameters of the templates
      epeaks (np.ndarray): the epeak parameters of the templates
      phis (np.ndarray): the phi direction of the templates in degrees
      thetas (np.ndarray): the theta direction of the templates in degrees
   """
   def __init__(self, binning, quantile=0, alltemplates=False, kind=SEARCH_BANK_FOLDER, hasamps=True):
      self.ndetectors = ndetectors
      self.binning = binning
      self.fulltemplates = None
      self.fullchantemplates = None
      self.hasamps = hasamps
      self.alltemplates = alltemplates
      self.quantile = quantile

      self.template_path = resolve_path(kind, ptype='templates')

      self._load_params()
      if not self.alltemplates:
         self._set_param_bank()

      self.ntemplates = self.phis.shape[0]

   def _load_params(self):
      if not self.alltemplates:
         self.load_bank_inds()

      self.alphas = np.load(next(self.template_path.glob(f'alpha*.npy'))).astype(np.float32)
      self.betas = np.load(next(self.template_path.glob(f'beta*.npy'))).astype(np.float32)
      self.epeaks = np.load(next(self.template_path.glob(f'Epeak*.npy'))).astype(np.float32)
      
      self.anginds = np.load(next(self.template_path.glob(f'positions*.npy')))
      self.phis = np.load(next(self.template_path.glob(f'phi*.npy'))).astype(np.float32)
      self.thetas = np.load(next(self.template_path.glob(f'theta*.npy'))).astype(np.float32)
      self.phis = self.phis[self.anginds]
      self.thetas = self.thetas[self.anginds]
      
      if self.hasamps:
         self.amps = np.load(self.template_path / f'amps_{self.binning}_{self.alphas.shape[0]}.npy').astype(np.float32)
      else:
         self.amps = np.ones(self.alphas.shape[0], dtype=np.float32)

   def load_templates(self):
      self.templates = np.load(next(self.template_path.glob(f'*reference*.npy')))

      self.templates = self.normalize_templates(self.templates)
      
      self.templates *= (self.amps[:, np.newaxis] * self.binning)

      if not self.alltemplates:
         self._set_template_bank()

      self._clean_bad_templates(self.quantile)

   def _set_template_bank(self):
      self.templates = self.templates[self.bank_inds]

   def _set_param_bank(self):
      self.alphas = self.alphas[self.bank_inds]
      self.betas = self.betas[self.bank_inds]
      self.epeaks = self.epeaks[self.bank_inds]
      self.phis = self.phis[self.bank_inds]
      self.thetas = self.thetas[self.bank_inds]

   def _clean_bad_templates(self, quantile):
      quant_val = np.quantile(self.templates.sum(axis=1), quantile)
      quant_inds = np.where(self.templates.sum(axis=1) >= quant_val)[0]

      self.templates = self.templates[quant_inds]

      self.alphas = self.alphas[quant_inds]
      self.betas = self.betas[quant_inds]
      self.epeaks = self.epeaks[quant_inds]
      self.phis = self.phis[quant_inds]
      self.thetas = self.thetas[quant_inds]

   def load_bank_inds(self):
      self.bank_inds = np.load(self.template_path / f'selected_indices_logmatch_{self.binning}.npy')
      # self.bank_inds = np.load(self.template_path / f'selected_indices_{self.binning}.npy')

   def normalize_templates(self, templates):
      sums = np.sum(templates, axis=1)[:, np.newaxis] + EPS
      templates /= sums
      return templates

   def is_occulted(self, burstdata, time, body, inflate=1):
      """
      Used to calculate the templates that are from the directions of the sun or
      of the earth. Note that the time is imoprtant because fermi moves.

      params:
      -------
      burstdata (TTEData): the burst data object (conataining the poshist)
      time (float): the time of the trigger in met
      body (str): either 'earth' or 'sun'

      returns:
      --------
      occulted (array): boolean array of the same shape as the tbank templates indicating
                        whether the template is occulted by the body.
      """
      if time >= burstdata.poshist.time_range[0] and time <= burstdata.poshist.time_range[1]:
         poshist = burstdata.poshist
      elif time > burstdata.poshist.time_range[1]:
         loader = DataLoaders()
         date = Time(time, format='fermi', scale='utc').utc.datetime + pd.Timedelta(minutes=5)
         poshist = loader.open_poshist_by_date(date)
      elif time < burstdata.poshist.time_range[0]:
         loader = DataLoaders()
         date = Time(time, format='fermi', scale='utc').utc.datetime - pd.Timedelta(minutes=5)
         poshist = loader.open_poshist_by_date(date)

      ras, decs = poshist.to_equatorial(self.phis, self.thetas, time)

      if body == 'earth':
         center = np.array(poshist.get_geocenter_radec(time)).reshape(2, -1)
         angular_radius = poshist.get_earth_radius(time) * inflate
      elif body == 'sun':
         center = np.array(get_sun_loc(time)).reshape(2, -1)
         angular_radius = 0.5 * inflate

      angle = haversine(center[0, :], center[1, :], ras, decs)
      occulted = (angle <= angular_radius)

      return occulted

   def remove_dets(self, dets):
      """
      Remove detectors from the data. It changes the data.
      params:
      -------
      dets (list): the detectors to remove
      """
      self.recover_dets()
      self.ndetectors -= len(dets)
      self.fulltemplates = self.templates.copy()
      self.templates = np.concatenate([self.fulltemplates[:, echans*i:echans*(i+1)] for i, val in enumerate(range(ndetectors)) if i not in dets], axis=1)

   def keep_chans(self, chans):
      """
      Remove channels from the data. It changes the data.
      params:
      -------
      chans (list): the channels to keep
      """
      self.recover_chans()
      self.fullchantemplates = deepcopy(self.templates)
      self.fullchantemplates = self.fullchantemplates.reshape([self.templates.shape[0], self.ndetectors, echans])
      self.templates = self.fullchantemplates[:, :, chans].reshape(-1, len(chans)*self.ndetectors)

   def keep1det(self, det):
      """
      Keep only one detector. It changes the data.
      params:
      -------
      det (int): the detector to keep
      """
      self.recover_dets()
      self.fulltemplates = self.templates.copy()
      self.templates = self.templates[:, echans*det:echans*(det+1)]

   def recover_dets(self):
      """
      Recover the detectors that were removed. It changes the data.
      """
      if self.fulltemplates is not None:
         self.templates = self.fulltemplates.copy()

         self.ndetectors = ndetectors

   def recover_chans(self):
      """
      Recover the detectors that were removed. It changes the data.
      """
      if self.fullchantemplates is not None:
         self.templates = deepcopy(self.fullchantemplates).reshape(-1, self.ndetectors*echans)

         self.fullchantemplates = None

   def to_cartesian(self):
      sf = np.arctan(1.0) / 45.0
      plat = self.thetas * sf
      plon = self.phis * sf

      x = np.sin(plat)*np.cos(plon)
      y = np.sin(plat)*np.sin(plon)
      z = np.cos(plat)

      return x, y, z


class TemplateGrid:
   """
   A class to represent the template grid for integration (i.e. the full Neyman-Pearson test).
   The templates are normalized to and are multiplied by the detection limit and binning to get the correct amplitude.
   Angles are ready to use in radians and degrees. The degrees are following the convention of the fermi data, and
   the radians are for the integration. They are sorted consistently.

   params:
   -------
   binning (float): the time binning in seconds

   attributes:
   -----------
   templates (np.ndarray): the templates, shape [ntemplates, nenergies*self.ndetectors]
   amps (np.ndarray): the detection limit for each template
   alphas (np.ndarray): the alpha parameters of the templates
   betas (np.ndarray): the beta parameters of the templates
   epeaks (np.ndarray): the epeak parameters of the templates
   phisrad (np.ndarray): the phi direction of the templates in radians
   thetasrad (np.ndarray): the theta direction of the templates in radians
   phisdeg (np.ndarray): the phi direction of the templates in degrees
   thetasdeg (np.ndarray): the theta direction of the templates in degrees
   """
   def __init__(self, binning, hasamps=True, kind=None):
      self.ndetectors = ndetectors
      self.binning = binning
      self.fulltemplates = None
      self.hasamps = hasamps

      if kind is None:
         raise ValueError("TemplateGrid requires an explicit bank folder path.")

      self.template_path = resolve_path(kind, ptype='templates')
      self._load_params()

      self.ntemplates = self.phis.shape[0] * self.alphas.shape[0]

   def _load_params(self):
      self.alphas = np.load(next(self.template_path.glob(f'alpha*.npy'))).astype(np.float32)
      self.betas = np.load(next(self.template_path.glob(f'beta*.npy'))).astype(np.float32)
      self.epeaks = np.load(next(self.template_path.glob(f'Epeak*.npy'))).astype(np.float32)
      self.phis = np.load(next(self.template_path.glob(f'phi*.npy'))).astype(np.float32)  # degrees, fermi convention
      self.thetas = np.load(next(self.template_path.glob(f'theta*.npy'))).astype(np.float32)  # degrees fermi convention
      
      if self.hasamps:
         self.amps = np.load(self.template_path / f'amps_{self.binning}_{self.alphas.shape[0]*self.phis.shape[0]}.npy').astype(np.float32)
      else:
         self.amps = np.ones([self.alphas.shape[0], self.phis.shape[0]], dtype=np.float32)

   def load_templates(self):
      self.templates = np.load(next(self.template_path.glob(f'*reference*.npy')))      

      self.templates = self.normalize_templates(self.templates)
      
      self.templates *= (self.amps[None, :] * self.binning)

   def normalize_templates(self, templates):
      sums = np.sum(templates, axis=0)[None, :] + EPS
      templates /= sums
      return templates

   def is_occulted(self, burstdata, time, body, inflate=1, rad=None, cen=None, opp=False, ras=None, decs=None):
      """
      Used to calculate the templates that are from the directions of the sun or
      of the earth. Note that the time is imoprtant because fermi moves.

      params:
      -------
      burstdata (TTEData): the burst data object (conataining the poshist)
      time (float): the time of the trigger in met
      body (str): either 'earth' or 'sun'
      inflate (float): the factor to inflate the angular radius of the body
      rad (float): the angular radius of the body in degrees.
      cen (np.ndarray): the center of the body in ra and dec.
      opp (bool): whether to return the opposite of the occulted templates

      returns:
      --------
      occulted (array): boolean array of the same shape as the tbank templates indicating
                        whether the template is occulted by the body.
      """
      if time >= burstdata.poshist.time_range[0] and time <= burstdata.poshist.time_range[1]:
         poshist = burstdata.poshist
      elif time > burstdata.poshist.time_range[1]:
         loader = DataLoaders()
         date = Time(time, format='fermi', scale='utc').utc.datetime + pd.Timedelta(minutes=5)
         poshist = loader.open_poshist_by_date(date)
      elif time < burstdata.poshist.time_range[0]:
         loader = DataLoaders()
         date = Time(time, format='fermi', scale='utc').utc.datetime - pd.Timedelta(minutes=5)
         poshist = loader.open_poshist_by_date(date)

      if ras is None and decs is None:
         ras, decs = poshist.to_equatorial(self.phis, self.thetas, time)

      if body == 'earth':
         center = np.array(poshist.get_geocenter_radec(time)).reshape(2, -1)
         angular_radius = poshist.get_earth_radius(time) * inflate
      elif body == 'sun':
         center = np.array(get_sun_loc(time)).reshape(2, -1)
         angular_radius = 0.5 * inflate

      if rad is not None and cen is not None:
         angular_radius = rad
         center = cen

      if opp:
         ras *= -1

      angle = haversine(center[0, :], center[1, :], ras, decs).astype(np.float32)
      occulted = (angle <= angular_radius)

      return occulted

   def remove_dets(self, dets):
      """
      Remove detectors from the data. It changes the data.
      params:
      -------
      dets (list): the detectors to remove
      """
      self.recover_dets()
      self.ndetectors -= len(dets)
      self.fulltemplates = deepcopy(self.templates)
      self.templates = np.concatenate([self.fulltemplates[echans*i:echans*(i+1)] for i, val in enumerate(range(ndetectors)) if i not in dets], axis=0)

   def keep1det(self, det):
      """
      Keep only one detector. It changes the data.
      params:
      -------
      det (int): the detector to keep
      """
      self.recover_dets()
      self.fulltemplates = self.templates.copy()
      self.templates = self.templates[echans*det:echans*(det+1)]

   def recover_dets(self):
      """
      Recover the detectors that were removed. It changes the data.
      """
      if self.fulltemplates is not None:
         self.templates = deepcopy(self.fulltemplates)

         self.ndetectors = ndetectors

   def calc_amps(self, d, bkg, psd_drift=None, slc=None, inds=None):
      """
      Calculate the amplitudes of the templates at the time of the detection.
      params:
      -------
      d (np.ndarray): the data
      bkg (np.ndarray): the background

      returns:
      --------
      amps (np.ndarray): the amplitudes of the templates
      """
      d = d[:, None, None]
      bkg += EPS

      if psd_drift is None:
         pscorr1 = np.zeros_like(self.templates[0])
         pscorr2 = np.ones_like(self.templates[0])
      else:
         # pscorr1 = psd_drift[0]
         # pscorr2 = psd_drift[1]
         pscorr1 = np.clip(psd_drift[0], -0.25, 0.25)
         pscorr2 = np.clip(psd_drift[1], 0.85**2, 1.25**2)
      
      if inds is None:
         slc = self.templates.shape[2] if slc is None else slc
         amps = np.zeros([self.templates.shape[1], self.templates.shape[2]])
         for js in range(0, self.templates.shape[2], slc):
            fltr = np.log1p(self.templates[:, :, js:js+slc]/bkg)
            b1 = pscorr1[:, js:js+slc]
            b2 = np.sqrt(pscorr2[:, js:js+slc])
            numer = b2*np.sum((d - bkg) * fltr, axis=0) + b1*np.sqrt(np.sum(bkg * fltr**2, axis=0))
            denom = np.sum(self.templates[:, :, js:js+slc] * fltr, axis=0) + EPS
            amps[:, js:js+slc] = numer / denom
      else:
         fltr = np.log1p(self.templates[:, :, inds[0]:inds[1]]/bkg)
         b1 = pscorr1[:, inds[0]:inds[1]]
         b2 = np.sqrt(pscorr2[:, inds[0]:inds[1]])
         numer = b2*np.sum((d - bkg) * fltr, axis=0) + b1 * np.sqrt(np.sum(bkg * fltr**2, axis=0))
         denom = np.sum(self.templates[:, :, inds[0]:inds[1]] * fltr, axis=0) + EPS
         amps = numer / denom
      
      return amps

   def to_cartesian(self):
      sf = np.arctan(1.0) / 45.0
      plat = self.thetas * sf
      plon = self.phis * sf

      x = np.sin(plat)*np.cos(plon)
      y = np.sin(plat)*np.sin(plon)
      z = np.cos(plat)

      return x, y, z
   
   def calc_posterior(self, d, bkg, psd_drift=None, slc=100):
      """
      calculate the posterior of the templates given the data and the background.
      params:
      -------
      d (np.ndarray): the data
      bkg (np.ndarray): the background
      slc (int): the number of templates to calculate at once

      returns:
      --------
      posterior (np.ndarray): the posterior
      """
      posterior = np.zeros([self.templates.shape[1], self.templates.shape[2]], dtype=np.float64)
      for js in range(0, self.templates.shape[2], slc):
         amps = self.calc_amps(d, bkg[:, None, None], inds=[js, js+slc], psd_drift=psd_drift)
         temp = amps*self.templates[:, :, js:js+slc]/(bkg[:, None, None] + 1e-4)
         temp = np.clip(temp, -1, np.inf)
         temp = np.log(1 + temp + 1e-4)
         posterior[:, js:js+slc] = np.tensordot(d, temp, axes=([0], [0]))
         posterior[:, js:js+slc] -= np.sum(amps*self.templates[:, :, js:js+slc], axis=0)

      np.subtract(posterior, np.max(posterior), out=posterior)
      np.exp(posterior, out=posterior)

      return posterior

   def plot_skymap(self, burstdata, maxtimes, trigind, save=False, show=False):
      from gdt.core.healpix import HealPixLocalization
      from gdt.missions.fermi.gbm.localization import GbmHealPix
      from gdt.core.plot.sky import EquatorialPlot

      maxtime = maxtimes[trigind]

      tleft = maxtime - burstdata.burst_duration_samp//2 + (burstdata.burst_duration_samp+1)%2
      tright = maxtime + burstdata.burst_duration_samp//2 + 1
      
      d = burstdata.data[tleft:tright].sum(axis=0).astype(np.int64)
      bkgw = burstdata.trig_bkgs[trigind]
      integrand = self.calc_posterior(d, bkgw, slc=100)

      skypost = integrand.sum(axis=0)
      skypost /= np.sum(skypost)
      
      probmap, ra_pix, dec_pix = interp_posterior(self, burstdata, burstdata.time[maxtime], skypost)

      best_ra = ra_pix[np.argmax(probmap)]
      best_dec = dec_pix[np.argmax(probmap)]

      try:
         frame = burstdata.poshist._frame_at(burstdata.time[maxtime])
         hpmap = GbmHealPix.from_data(
            probmap,
            trigtime=burstdata.time[maxtime],
            quaternion=frame.quaternion,
            scpos=frame.obsgeoloc,
         )
      except Exception:
         hpmap = HealPixLocalization.from_data(probmap)

      fermiplot = EquatorialPlot()
      add_kwargs = dict(gradient=True, clevels=[0.9], sun=True, earth=True)
      if not hasattr(hpmap, "frame"):
         add_kwargs.update(detectors=[], sun=False, earth=False)
      fermiplot.add_localization(hpmap, **add_kwargs)
      fermiplot.ax.set_facecolor('white')
      fermiplot.ax.set_title(f'{hpmap.area(0.9):.2f} square degrees at 90% confidence level. Best fit: RA={best_ra:.2f}, DEC={best_dec:.2f}')

      if save:
         fermiplot.fig.tight_layout()
         dname = Time(burstdata.time[maxtime], format='fermi', scale='utc').utc.datetime.strftime('%Y-%m-%d_%H:%M:%S.%f')
         figpath = DATAPATH / f'results/{burstdata.date.year}/figures/skymap_{dname}.png'
         fermiplot.fig.savefig(figpath)

         plt.close(fermiplot.fig)

      if show:
         plt.show()

      return hpmap
   
   @property
   def phisrad(self):
      return np.deg2rad(self.phis)

   @property
   def thetasrad(self):
      return np.deg2rad(self.thetas)


class GlitchTemplates:
   """
      A class to load the glitches. The glitches are normalized to and are
      multiplied by the detection limit and binning to get the correct amplitude.

      params:
      -------
      binning (float): the time binning in seconds
      glitch1d_len (int or list): the length of the 1d glitches to load. Can be a list of lengths to load multiple lengths.

      attributes:
      -----------
      glitch1d (np.ndarray): the 1d glitches, shape [ntemplates, nenergies*ndetectors]
   """
   def __init__(self, binning, glitch1d_len=3, hasamps=True):
      self.binning = binning
      self.fullglitch1d = None
      self.fullchanglitch1d = None
      self.ndetectors = ndetectors
      self.hasamps = hasamps
      self.glitch1d_len = glitch1d_len

      self.load_templates()

   def load_templates(self):
      self._load_1d(self.glitch1d_len)
      self._set_1d()
   
   def _load_1d(self, glitch1d_len):
      self.glitch1d_len = glitch1d_len if type(glitch1d_len) is not int else [glitch1d_len]
      self.glitch1d_path = DATAPATH.as_posix() + '/templates/glitches1d/glitchlensamples_{}.npy'
      self.glitch1d = []
      for length in self.glitch1d_len:
         glitch = np.load(self.glitch1d_path.format(length))
         self.glitch1d.append(glitch)
      self.glitch1d = np.concatenate(self.glitch1d, axis=0)

      if self.hasamps:
         self.amps1d = np.load(DATAPATH.as_posix() + f'/templates/glitches1d/glitches1d_amps_{self.binning}_{len(self.glitch1d_len)*self.ndetectors}.npy')
      else:
         self.amps1d = np.ones(len(self.glitch1d_len)*self.ndetectors, dtype=np.float32)

   def _set_1d(self):
      self.glitch1d = self.normalize_template(self.glitch1d)

      self.glitch1d *= (self.amps1d[:, np.newaxis] * self.binning)

   def normalize_template(self, templates):
      templates = templates / (np.sum(templates, axis=1)[:, np.newaxis] + EPS)
      return templates

   def remove_dets(self, dets):
      """
      Remove detectors from the data. It changes the data.
      params:
      -------
      dets (list): the detectors to remove
      """
      self.recover_dets()
      self.ndetectors -= len(dets)
      self.fullglitch1d = self.glitch1d.copy()
      self.glitch1d = np.concatenate([self.fullglitch1d[:, echans*i:echans*(i+1)] for i, val in enumerate(range(ndetectors)) if i not in dets], axis=1)
   
   def keep_chans(self, chans):
      """
      Remove channels from the data. It changes the data.
      params:
      -------
      chans (list): the channels to keep
      """
      self.recover_chans()
      self.fullchanglitch1d = deepcopy(self.glitch1d)
      self.fullchanglitch1d = self.fullchanglitch1d.reshape([self.glitch1d.shape[0], self.ndetectors, echans])
      self.glitch1d = self.fullchanglitch1d[:, :, chans].reshape(-1, len(chans)*self.ndetectors)

   def keep1det(self, det):
      """
      Keep only one detector. It changes the data.
      params:
      -------
      det (int): the detector to keep
      """
      self.recover_dets()
      self.fullglitch1d = self.glitch1d.copy()
      self.glitch1d = self.glitch1d[:, echans*det:echans*(det+1)]

   def recover_dets(self):
      """
      Recover the detectors that were removed. It changes the data.
      """
      if self.fullglitch1d is not None:
         self.glitch1d = self.fullglitch1d.copy()

         self.ndetectors = ndetectors

   def recover_chans(self):
      """
      Recover the detectors that were removed. It changes the data.
      """
      if self.fullchanglitch1d is not None:
         self.glitch1d = deepcopy(self.fullchanglitch1d).reshape(-1, self.ndetectors*echans)

         self.fullchanglitch1d = None
