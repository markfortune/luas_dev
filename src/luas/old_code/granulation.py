"""
Standalone amplitude scaling and hyperparameter loading for solar-like
oscillations and granulation.

Implements the bandpass amplitude scaling from Morris et al. (2020), Eq. 11,
and loads the gadfly solar hyperparameter data without importing gadfly.

Dependencies: numpy, astropy.
gadfly must be *installed* (so its data files are locatable) but is never
imported.  Alternatively, pass explicit file paths to get_wavelength_scaling.
"""

import json
import os

import numpy as np
import astropy.units as u
from astropy.modeling.models import BlackBody
from astropy.table import QTable
import tinygp


# ── Planck-function amplitude scaling (Morris et al. 2020, Eq. 11) ───────────

def _alpha(center_um, temperature_K=5777, fwhm_frac=0.05, n=10_000):
    """
    Amplitude scaling factor alpha for a tophat bandpass centred at
    `center_um` microns relative to a bolometric (SOHO VIRGO) observation.

    alpha > 1  →  more variability than bolometric (typical for optical/NIR).
    alpha < 1  →  less variability (rare, long-wavelength filters).

    Parameters
    ----------
    center_um     : float  – bandpass centre in microns
    temperature_K : float  – stellar effective temperature in K
    fwhm_frac     : float  – fractional bandwidth of the tophat (FWHM/centre)
    n             : int    – number of wavelength points for numerical integration
    """
    T  = temperature_K * u.K
    wl = np.logspace(-1.5, 1.5, n) * u.um   # 0.032 – 31.6 µm
    x  = wl.to(u.um).value

    # Planck function and numerical dI/dT (±10 K finite difference)
    I     = BlackBody(T)(wl)
    dI_dT = (BlackBody(T + 10*u.K)(wl) - BlackBody(T - 10*u.K)(wl)) / (20 * u.K)

    # Tophat transmittance for the target filter
    half   = center_um * fwhm_frac / 2
    T_filt = np.where((x >= center_um - half) & (x <= center_um + half), 1.0, 0.0)
    T_bolo = np.ones_like(x)    # SOHO VIRGO ≈ bolometric

    # Equation 11 — the units cancel in the two ratios
    r0 = (np.trapezoid((dI_dT * x * T_filt).value, x) /
          np.trapezoid((dI_dT * x * T_bolo).value, x))
    r1 = (np.trapezoid((I    * x * T_bolo).value, x) /
          np.trapezoid((I    * x * T_filt).value, x))

    return float(r0 * r1)


# ── Hyperparameter loading ────────────────────────────────────────────────────

def _gadfly_data_dir():
    """Return gadfly's bundled data directory without importing gadfly."""
    spec = importlib.util.find_spec('gadfly')
    if spec is None:
        raise ImportError(
            "gadfly is not installed.  Either install it or pass explicit "
            "hp_path / broomhall_path / param_vector_path to get_wavelength_scaling."
        )
    return os.path.join(os.path.dirname(spec.origin), 'data')


def _load_hyperparameters(
    hp_path=None,
    broomhall_path=None,
    param_vector_path=None,
    include_pmodes=False,
    pmode_degrees=None,
    max_pmodes=None,
):
    """
    Load solar GP hyperparameters from the gadfly data files.

    Granulation terms
    -----------------
    Five SHO (simple harmonic oscillator) kernels with Q = 0.6 (overdamped),
    modelling supergranulation through to mesogranulation at progressively
    higher characteristic frequencies w0 (in rad day⁻¹).  Each kernel
    contributes a broad, power-law-like hump in the power spectrum.

    P-mode oscillation terms (include_pmodes=True)
    -----------------------------------------------
    Solar p-modes are standing acoustic waves with frequencies near 3000 µHz
    (≈ 5 minute oscillations).  Each individual mode is modelled as a
    high-Q SHO kernel (narrow peak).

    The modes are labelled by their spherical harmonic degree ℓ:

      ℓ = 0  Radial modes — the whole star breathes in and out uniformly.
      ℓ = 1  Dipole — one hemisphere expands while the other contracts.
      ℓ = 2  Quadrupole.
      ℓ = 3  Octupole.

    When a star is observed as a point source, higher-ℓ modes partially
    cancel across the disk, so ℓ = 1 has the largest photometric amplitude
    and ℓ = 3 the smallest.

    Each mode's SHO parameters come from parameter_vector.txt, which stores
    gadfly's fitted parameter vector.  The file layout is: 6 preamble floats
    (Voigt envelope / background parameters), then 81 triplets of
    (S0_i, w0_i [rad s⁻¹], Q_i) in the same row order as
    broomhall2009_table2_labeled.ecsv.

    S0_i already incorporates the Voigt envelope weighting centred on ν_max,
    so it is the correct amplitude to pass directly to a celerite2 SHOTerm
    kernel — unlike the degree-level S0 stored in hyperparameters.json, which
    is only the unweighted amplitude scale c_ℓ and must be multiplied by the
    envelope before use.

    Unphysical modes (ν ≤ 0 or Q ≤ 0) from optimiser artefacts are always
    silently removed before any further filtering.

    Parameters
    ----------
    hp_path           : str or None   – path to hyperparameters.json
    broomhall_path    : str or None   – path to broomhall2009_table2_labeled.ecsv
    param_vector_path : str or None   – path to parameter_vector.txt
    include_pmodes    : bool          – whether to include p-mode oscillation terms
    pmode_degrees     : list or None  – restrict to these ℓ values, e.g. [0, 1]
                                        (None keeps all degrees)
    max_pmodes        : int or None   – keep only this many modes, ranked by S0
                                        descending (None keeps all passing modes)

    Returns
    -------
    dict with keys:
      'granulation'  : list of dicts, each {S0, w0, Q}
                       w0 in rad day⁻¹
      'oscillations' : list of dicts (only if include_pmodes=True), each:
                         nu_uHz         – mode frequency in µHz
                         w0_rad_per_day – angular frequency in rad day⁻¹
                         degree         – spherical harmonic degree ℓ
                         S0             – SHO amplitude (Voigt-envelope-weighted)
                         Q              – mode quality factor
                       Sorted by nu_uHz ascending.
    """
    data_dir = "."

    if hp_path is None:
        hp_path = os.path.join(data_dir, 'hyperparameters.json')
    if include_pmodes:
        if broomhall_path is None:
            broomhall_path = os.path.join(
                data_dir, 'broomhall2009_table2_labeled.ecsv'
            )
        if param_vector_path is None:
            param_vector_path = os.path.join(data_dir, 'parameter_vector.txt')

    with open(hp_path) as fh:
        raw = json.load(fh)

    granulation = []
    for entry in raw:
        hp  = entry['hyperparameters']
        src = entry['metadata']['source']
        if src == 'granulation':
            granulation.append({'S0': hp['S0']*1e-12, 'w0': hp['w0'], 'Q': hp['Q']})

    result = {'granulation': granulation}

    if include_pmodes:
        # Mode frequencies and degree labels from Broomhall et al. (2009)
        tbl = QTable.read(broomhall_path, format='ascii.ecsv')

        # Pre-computed per-mode SHO parameters from gadfly's parameter vector.
        # Layout: 6 preamble values, then 81 triplets of (S0_i, w0_i, Q_i)
        # where w0_i is in rad s⁻¹.
        pv       = np.loadtxt(param_vector_path)
        triplets = pv[6:].reshape(-1, 3)   # shape (81, 3)

        _s_per_day = 86400.0   # rad s⁻¹ → rad day⁻¹

        oscillations = []
        for i, row in enumerate(tbl):
            S0_i, w0_s, Q_i = triplets[i]
            nu_i = float(row['nu'].value)
            ell  = int(row['degree'])

            # Drop optimiser artefacts (unphysical negative frequency or Q)
            if w0_s <= 0 or Q_i <= 0:
                continue

            # Degree filter
            if pmode_degrees is not None and ell not in pmode_degrees:
                continue

            oscillations.append({
                'nu_uHz'        : nu_i,
                'w0': w0_s * _s_per_day,
                'degree'        : ell,
                'S0'            : S0_i*1e-12,
                'Q'             : Q_i,
            })

        # Keep only the top-N modes by S0 amplitude
        if max_pmodes is not None and len(oscillations) > max_pmodes:
            oscillations.sort(key=lambda m: m['S0'], reverse=True)
            oscillations = oscillations[:max_pmodes]

        # Return in frequency order (natural for GP kernel construction)
        oscillations.sort(key=lambda m: m['nu_uHz'])

        result['oscillations'] = oscillations

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def get_wavelength_scaling(
    wavelengths_um,
    temperature_K     = 5777,
    include_pmodes    = False,
    fwhm_frac         = 0.05,
    hp_path           = None,
    broomhall_path    = None,
    param_vector_path = None,
    pmode_degrees     = None,
    max_pmodes        = None,
):
    """
    Compute bandpass amplitude scaling factors and load solar GP
    hyperparameters — no gadfly imports required.

    Parameters
    ----------
    wavelengths_um    : array_like
        Central wavelengths of the observing bandpasses in microns.
    temperature_K     : float
        Stellar effective temperature in K (default: 5777, solar).
    include_pmodes    : bool
        If True, include individual p-mode oscillation terms in hp
        (parameters from gadfly's fitted parameter_vector.txt, frequencies
        from Broomhall et al. 2009).  Default False returns only the five
        granulation SHO kernels.
    fwhm_frac         : float
        Fractional bandwidth of the tophat used to compute alpha
        (FWHM / centre wavelength).  Default 0.05 (5 %).
    hp_path           : str or None
        Path to hyperparameters.json.  If None, located from the gadfly
        installation automatically.
    broomhall_path    : str or None
        Path to broomhall2009_table2_labeled.ecsv.  Only used when
        include_pmodes=True.  If None, located from gadfly automatically.
    param_vector_path : str or None
        Path to parameter_vector.txt.  Only used when include_pmodes=True.
        If None, located from gadfly automatically.
    pmode_degrees     : list of int or None
        Restrict p-modes to the given spherical harmonic degrees, e.g.
        [0, 1] keeps only radial and dipole modes (the two most photometrically
        visible).  None (default) keeps all degrees.
    max_pmodes        : int or None
        After degree filtering, keep only this many p-modes ranked by S0
        amplitude (largest first).  Useful to limit the number of SHO kernels
        passed to a GP package.  None (default) keeps all passing modes.

        Practical guidance:
          max_pmodes=20  — captures ~90 % of total p-mode power, fast GPs
          max_pmodes=10  — dominant modes only, suitable for quick fits
          pmode_degrees=[0,1] alone  — drops ℓ=2,3, reduces from ~80 to ~43 terms

    Returns
    -------
    alphas : ndarray, shape (n_wavelengths,)
        Amplitude scaling factor at each wavelength relative to SOHO VIRGO
        (bolometric).  Multiply any bolometric GP amplitude by alpha to get
        the wavelength-specific value.
    hp : dict
        Solar GP hyperparameters:
          hp['granulation']  – list of 5 dicts, each {S0, w0, Q}
                               with w0 in rad day⁻¹
          hp['oscillations'] – list of dicts per p-mode (only if
                               include_pmodes=True), sorted by frequency:
                                 {nu_uHz, w0_rad_per_day, degree, S0, Q}
                               S0 and Q are ready to use directly as
                               celerite2 SHOTerm parameters.
    """
    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    alphas = np.array([_alpha(wv, temperature_K, fwhm_frac)
                       for wv in wavelengths_um])
    hp = _load_hyperparameters(
        hp_path, broomhall_path, param_vector_path,
        include_pmodes, pmode_degrees, max_pmodes,
    )
    return alphas, hp




def solar_kernel(x_l, include_pmodes = False, **kwargs):

    alpha_vec, hp = get_wavelength_scaling(np.linspace(0.6, 1., 10), include_pmodes = include_pmodes, **kwargs)

    if include_pmodes:
        hp_list = hp["granulation"] + hp["oscillations"]
    else:
        hp_list = hp["granulation"]
    
    
    kernel = sum(alpha_vec[0]**2 * tinygp.kernels.quasisep.SHO(sigma = np.sqrt(hp["S0"]), omega = hp["w0"], quality = hp["Q"]) for hp in hp_list)
    
    return (luas.kernels.Outer(alpha_vec), luas.kernels.quasisep.Custom(kernel))
    
N_l = 16
N_t = 1000

x_l = np.logspace(np.log10(0.4e-6), np.log10(5.0e-6), N_l)
x_t = np.linspace(-0.15, 0.15, N_t)

kernel = solar_kernel(x_l, include_pmodes = False, max_pmodes = 20)
Kt = kernel(x_t, x_t) + 10e-6**2 * np.eye(N_t)

r = np.random.multivariate_normal(np.zeros(N_t), Kt)

plt.plot(24*x_t, 1e6*r)
# plt.colorbar()