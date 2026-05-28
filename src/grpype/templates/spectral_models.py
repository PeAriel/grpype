import numpy as np

from grpype.detection.global_params import EPS


def band_function(spec_params, energy_bin_centroids):
    """
    Use Band function for folding in order to generate a spectrum (via blarog drm of gbm drm).
    Using the definition of GMB:
    $F(E) = A \begin{cases} \left(\frac{E}{E_{\text{piv}}}\right)^{\alpha} e^{-(2+\alpha)\frac{E}{E_{\text{peak}}}}, \text{ if } E < \frac{(\alpha-\beta)E_{\text{peak}}}{2+\alpha} \\ \left(\frac{(\alpha-\beta)E_{\text{peak}}}{(2+\alpha)E_{\text{piv}}}\right)^{\alpha-\beta} \exp(\beta-\alpha)\left(\frac{E}{E_{\text{piv}}}\right)^{\beta}, \text{ otherwise} \end{cases}$
    """
    if len(spec_params) == 3:
        alpha, beta, epeak = spec_params
        epiv = 100
        A = 1
    else:
        A, alpha, beta, epeak, epiv = spec_params
        
    if alpha < beta:
        return np.zeros_like(energy_bin_centroids)
    if epeak <= 0 or epiv <= 0 or not np.isfinite(epeak):
        return np.zeros_like(energy_bin_centroids)
    if abs(alpha + 2.0) < 1e-3:
        return np.zeros_like(energy_bin_centroids)

    spectrum = np.zeros(energy_bin_centroids.shape)
    
    threshold = epeak*(alpha - beta)/(alpha + 2)

    low_e = A*(energy_bin_centroids/epiv)**alpha*np.exp(-(alpha + 2)*energy_bin_centroids/epeak)
    high_e = A*(energy_bin_centroids/epiv)**beta*np.exp(beta - alpha)*((alpha - beta)*epeak/(epiv*(alpha + 2)))**(alpha - beta)

    spectrum[energy_bin_centroids < threshold] = low_e[energy_bin_centroids < threshold]
    spectrum[energy_bin_centroids >= threshold] = high_e[energy_bin_centroids >= threshold]
    spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)

    return spectrum

def cutoff_powerlaw(spec_params, energy_bin_centroids):
    if len(spec_params) == 2:
        alpha, epeak = spec_params
        A = 1
        epiv = 100
    else:
        A, alpha, epeak, epiv = spec_params
    
    x = energy_bin_centroids

    return A * (x/epiv)**alpha * np.exp(-(alpha+2)*x/epeak)


class SpectralModel:
    def __call__(self, spec_params, energy_bin_centroids):
        raise NotImplementedError

    def log_prior(self, spec_params):
        raise NotImplementedError

    def fold_model(
        self,
        rsp,
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
            rsp (np.array): Response stack [ndet, nchan, nphoton]
            spec_params (array-like or scalar): Spectral parameters
            nai_bin_centroids (np.array): Nai photon bin centroids
            nai_bin_widths (np.array): Nai photon bin widths
            bgo_bin_centroids (np.array): BGO photon bin centroids
            bgo_bin_widths (np.array): BGO photon bin widths
            normalize (bool): Normalize the template
        Returns:
            np.array: Template
        """
        spec_nai = np.atleast_2d(self(spec_params, nai_bin_centroids))
        spec_bgo = np.atleast_2d(self(spec_params, bgo_bin_centroids))

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


class BandFunction(SpectralModel):
    def __init__(self, epiv=100, A=1, alpha_prior=[-2, 3], beta_prior=[-7, 1], epeak_prior=[8, 5000]):
        self.epiv = epiv
        self.A = A
        
        self.min_alpha = alpha_prior[0]
        self.max_alpha = alpha_prior[1]
        
        self.min_beta = beta_prior[0]
        self.max_beta = beta_prior[1]
        
        self.min_epeak = epeak_prior[0]
        self.max_epeak = epeak_prior[1]

        self.param_bounds = [alpha_prior, beta_prior, epeak_prior]
        self.param_labels = ['alpha', 'beta', 'epeak']

    def __call__(self, spec_params, energy_bin_centroids):
        spec_params = np.atleast_2d(spec_params)
        alpha = spec_params[:, 0]
        beta = spec_params[:, 1]
        epeak = spec_params[:, 2]
        epiv, A = self.epiv, self.A

        valid = (
            (alpha >= beta)
            & (epeak > 0)
            & np.isfinite(epeak)
            & (epiv > 0)
            & (np.abs(alpha + 2.0) >= 1e-3)
        )

        x = energy_bin_centroids[None, :]
        spectrum = np.zeros((spec_params.shape[0], x.shape[1]), dtype=float)

        if np.any(valid):
            alpha_v = alpha[valid][:, None]
            beta_v = beta[valid][:, None]
            epeak_v = epeak[valid][:, None]

            threshold = epeak_v * (alpha_v - beta_v) / (alpha_v + 2)
            low_e = A * (x / epiv) ** alpha_v * np.exp(-(alpha_v + 2) * x / epeak_v)
            high_e = (
                A
                * (x / epiv) ** beta_v
                * np.exp(beta_v - alpha_v)
                * ((alpha_v - beta_v) * epeak_v / (epiv * (alpha_v + 2))) ** (alpha_v - beta_v)
            )

            spectrum_valid = np.where(x < threshold, low_e, high_e)
            spectrum[valid] = spectrum_valid

        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)

        if spec_params.shape[0] == 1:
            return spectrum[0]
        return spectrum

    def log_prior(self, spec_params):
        alpha, beta, epeak = spec_params
        if (self.min_alpha <= alpha <= self.max_alpha and
             self.min_beta <= beta <= self.max_beta and
               self.min_epeak <= epeak <= self.max_epeak and
                 alpha >= beta):
            return 0.0
        return -np.inf


class CutoffPowerLaw(SpectralModel):
    def __init__(self, epiv=100, A=1, alpha_prior=[-2, 3], epeak_prior=[8, 5000]):
        self.epiv = epiv
        self.A = A

        self.min_alpha = alpha_prior[0]
        self.max_alpha = alpha_prior[1]
        
        self.min_epeak = epeak_prior[0]
        self.max_epeak = epeak_prior[1]

        self.param_bounds = [alpha_prior, epeak_prior]
        self.param_labels = ['alpha', 'epeak']

    def __call__(self, spec_params, bin_centroids):
        spec_params = np.atleast_2d(spec_params)
        alpha = spec_params[:, 0][:, None]
        epeak = spec_params[:, 1][:, None]
        x = bin_centroids[None, :]
        spectrum = self.A * (x / self.epiv) ** alpha * np.exp(-(alpha + 2) * x / epeak)
        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)
        if spec_params.shape[0] == 1:
            return spectrum[0]
        return spectrum

    def log_prior(self, spec_params):
        alpha, epeak = spec_params
        if (self.min_alpha <= alpha <= self.max_alpha and
            self.min_epeak <= epeak <= self.max_epeak):
            return 0.0
        return -np.inf


class PowerLaw(SpectralModel):
    def __init__(self, epiv=100, A=1, alpha_prior=[-10, 10]):
        self.epiv = epiv
        self.A = A

        self.min_alpha = alpha_prior[0]
        self.max_alpha = alpha_prior[1]

        self.param_bounds = [alpha_prior]
        self.param_labels = ['alpha']

    def __call__(self, spec_params, bin_centroids):
        if np.isscalar(spec_params):
            spec_params = np.array([[spec_params]], dtype=float)
        else:
            spec_params = np.atleast_2d(spec_params)
        alpha = spec_params[:, 0][:, None]
        x = bin_centroids[None, :]
        spectrum = self.A * (x / self.epiv) ** alpha
        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)
        if spec_params.shape[0] == 1:
            return spectrum[0]
        return spectrum

    def log_prior(self, spec_params):
        if np.isscalar(spec_params):
            alpha = spec_params
        else:
            alpha = spec_params[0]
        if (self.min_alpha <= alpha <= self.max_alpha):
            return 0.0
        return -np.inf


class Blackbody(SpectralModel):
    def __init__(self, A=1, kT_prior=[1, 1000]):
        self.A = A

        self.min_kT = kT_prior[0]
        self.max_kT = kT_prior[1]

        self.param_bounds = [kT_prior]
        self.param_labels = ['kT']

    def __call__(self, spec_params, bin_centroids):
        if np.isscalar(spec_params):
            spec_params = np.array([[spec_params]], dtype=float)
        else:
            spec_params = np.atleast_2d(spec_params)
        kT = spec_params[:, 0][:, None]
        x = bin_centroids[None, :]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            spectrum = self.A * (x**2) / (np.exp(x / kT) - 1)
        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)
        if spec_params.shape[0] == 1:
            return spectrum[0]
        return spectrum

    def log_prior(self, spec_params):
        if np.isscalar(spec_params):
            kT = spec_params
        else:
            kT = spec_params[0]
        if (self.min_kT <= kT <= self.max_kT):
            return 0.0
        return -np.inf


if __name__ == "__main__":
    pass
