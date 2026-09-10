import numpy as np
from scipy.integrate import quad

# Cosntants
c = 299792.458  # Speed of light in km/s

# Friedmann equation E(z) = H(z)/H0
def E(z, Omega_m, Omega_k, Omega_Lambda):
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_Lambda)

# Comoving distance calculation
def comoving_distance(z, H0, Omega_m, Omega_k, Omega_Lambda):
    integral, _ = quad(lambda z_prime: 1 / E(z_prime, Omega_m, Omega_k, Omega_Lambda), 0, z)
    return (c / H0) * integral  # in Mpc

# Luminosity distance calculation
def luminosity_distance(z, H0, Omega_m, Omega_k, Omega_Lambda):
    d_c = comoving_distance(z, H0, Omega_m, Omega_k, Omega_Lambda)
    # Transverse comoving distance d_m calculation based on curvature
    if Omega_k > 0:    # Open universe
        d_m = (c / H0) / np.sqrt(Omega_k) * np.sinh(np.sqrt(Omega_k) * d_c * H0 / c)
    elif Omega_k < 0:  # Closed universe
        d_m = (c / H0) / np.sqrt(-Omega_k) * np.sin(np.sqrt(-Omega_k) * d_c * H0 / c)
    else:              # Flat universe
        d_m = d_c
    return (1 + z) * d_m

# Angular diameter distance calculation
def angular_diameter_distance(z, H0, Omega_m, Omega_k, Omega_Lambda):
    d_l = luminosity_distance(z, H0, Omega_m, Omega_k, Omega_Lambda)
    return d_l / (1 + z)**2

# Lookback time calculation
def lookback_time(z, H0, Omega_m, Omega_k, Omega_Lambda):
    integral, _ = quad(lambda z_prime: 1 / ((1 + z_prime) * E(z_prime, Omega_m, Omega_k, Omega_Lambda)), 0, z)
    h = H0 / 100 # Convert H0 to units of 100 km/s/Mpc
    return (9.78 / h) * integral  # Convert to Gyr = 10^9 years

# Age at redshift z calculation in three decimals
def age_at_redshift(z, H0, Omega_m, Omega_k, Omega_Lambda):
    integral, _ = quad(lambda z_prime: 1 / ((1 + z_prime) * E(z_prime, Omega_m, Omega_k, Omega_Lambda)), z, np.inf)
    Tyr = 977.8 # Coefficient for converting 1/H into Gyr
    return (Tyr / H0) * integral  # Into Gyr


# =============================================================================
# K correction  (Hogg, Baldry, Blanton & Eisenstein 2002, astro-ph/0210394)
# =============================================================================
# We use equation (13) of that paper: wavelength units, source described by its
# rest-frame luminosity density L_lambda:
#
#                        1    Int[ dLo Lo L_lam(Lo/(1+z)) R(Lo) ]   Int[ dLe Le gQ(Le) Q(Le) ]
#   K_QR = -2.5 log10 [ --- * --------------------------------- * -------------------------- ]
#                       1+z   Int[ dLo Lo gR(Lo) R(Lo) ]           Int[ dLe Le L_lam(Le) Q(Le) ]
#
#   with   m_R = M_Q + DM + K_QR                                   (Hogg eq. 2)
#
#   Lo = observed-frame wavelength, Le = emitted (rest-frame) wavelength, Le = Lo/(1+z)
#   R  = observed-frame bandpass (the filter on the telescope)
#   Q  = rest-frame bandpass in which the absolute magnitude M_Q is defined
#   g  = flux density of the zero-magnitude "standard" source. For AB magnitudes
#        g_nu = 3631 Jy is constant, so in wavelength units g_lam ~ lambda^-2.
#
# Source spectrum: instead of a black body we use a power law
#        L_nu ~ nu^alpha    <=>    L_lam ~ lambda^-(alpha+2)      (nu L_nu = lambda L_lam)
#   alpha ~ -0.5 is the usual optical/UV quasar continuum slope (Vanden Berk+ 2001,
#   Richards+ 2006). alpha = -1 means flat in nu*L_nu, alpha = 0 flat in L_nu.
#
# Useful special case: for a pure power law with Q = R (same filter for the
# observed and rest-frame magnitudes, same zero point) everything cancels and
#        K = -2.5 (1 + alpha) log10(1 + z)
# which does NOT depend on the filter at all ("no filter" and "luminance filter"
# give the same K).  The filter only matters when Q != R, when the two bands
# have different zero points, or when the spectrum is not a pure power law.
# All wavelengths below are in nanometres.

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def power_law_L_lambda(alpha):
    """L_lambda(lambda) for a power-law source L_nu ~ nu**alpha (normalisation is arbitrary,
    it cancels in the K correction)."""
    beta = -(alpha + 2.0)
    return lambda lam: np.asarray(lam, dtype=float) ** beta


def ab_standard_L_lambda(lam):
    """AB zero-magnitude source in wavelength units: g_lam = c lam^-2 g_nu, g_nu = 3631 Jy.
    Only the lambda^-2 shape matters (constants cancel when both bands are AB)."""
    return np.asarray(lam, dtype=float) ** -2.0


class Filter:
    """Bandpass R(lambda): transmission as a function of wavelength (nm) plus the
    wavelength range outside of which it is zero. With transmission=None it is a
    top hat (1 inside [lam_min, lam_max], 0 outside)."""

    def __init__(self, name, lam_min, lam_max, transmission=None):
        if lam_max <= lam_min:
            raise ValueError("lam_max must be larger than lam_min")
        self.name = name
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self._transmission = transmission

    def __call__(self, lam):
        lam = np.asarray(lam, dtype=float)
        inside = (lam >= self.lam_min) & (lam <= self.lam_max)
        if self._transmission is None:
            return inside.astype(float)
        return np.where(inside, self._transmission(lam), 0.0)

    def __repr__(self):
        return f"Filter({self.name!r}, {self.lam_min:g}-{self.lam_max:g} nm)"


def tophat_filter(lam_min, lam_max, name=None):
    """Convenience constructor for a top-hat (box) filter between lam_min and lam_max (nm)."""
    name = name or f"Top hat {lam_min:g}-{lam_max:g} nm"
    return Filter(name, lam_min, lam_max)


# Ready-made bandpasses (top-hat approximations; replace with a tabulated curve
# via Filter(..., transmission=...) when you have the real one).
FILTERS = {
    "Luminance (400-700 nm)": tophat_filter(400.0, 700.0, "Luminance"),
    "Johnson B (~390-490 nm)": tophat_filter(390.0, 490.0, "Johnson B"),
    "Johnson V (~505-595 nm)": tophat_filter(505.0, 595.0, "Johnson V"),
    "Johnson R (~560-720 nm)": tophat_filter(560.0, 720.0, "Johnson R"),
    "SDSS i (~690-840 nm)":   tophat_filter(690.0, 840.0, "SDSS i"),
}


def _band_integral(func, band, n_points=4001):
    """Int[ d(lam) lam * func(lam) * band(lam) ] over the support of `band`."""
    lam = np.linspace(band.lam_min, band.lam_max, n_points)
    return _trapezoid(lam * func(lam) * band(lam), lam)


def k_correction(z, L_lambda, R, Q=None, gR=ab_standard_L_lambda, gQ=None, n_points=4001):
    """Numerical K correction K_QR from Hogg et al. (2002) eq. (13).

    z         : redshift
    L_lambda  : callable, rest-frame luminosity density L_lambda(lam) (any normalisation),
                e.g. power_law_L_lambda(alpha)
    R         : Filter, observed-frame bandpass
    Q         : Filter, rest-frame bandpass of the absolute magnitude (default: same as R)
    gR, gQ    : callables, zero-magnitude standard source g_lambda for each band
                (default: AB for both)

    Returns K in magnitudes, to be used as  m_R = M_Q + DM + K.
    """
    Q = R if Q is None else Q
    gQ = gR if gQ is None else gQ
    one_plus_z = 1.0 + z

    src_obs = _band_integral(lambda lam: L_lambda(lam / one_plus_z), R, n_points)  # source seen through R
    src_rest = _band_integral(L_lambda, Q, n_points)                                 # source at rest through Q
    std_Q = _band_integral(gQ, Q, n_points)                                          # standard source through Q
    std_R = _band_integral(gR, R, n_points)                                          # standard source through R

    ratio = (src_obs * std_Q) / (one_plus_z * std_R * src_rest)
    return -2.5 * np.log10(ratio)


def k_correction_power_law(z, alpha):
    """Analytic eq. (13) for a power law L_nu ~ nu**alpha with Q = R and equal zero points.
    Independent of the filter shape, so this is also the "no filter" case."""
    return -2.5 * (1.0 + alpha) * np.log10(1.0 + z)


def rest_frame_range(band, z):
    """Rest-frame wavelength range (nm) sampled by observed-frame `band` at redshift z."""
    return band.lam_min / (1.0 + z), band.lam_max / (1.0 + z)


LYMAN_ALPHA_NM = 121.567  # blueward of this the IGM absorbs almost all quasar light at high z

if __name__ == "__main__":
    H0 = 70 # Hubble constant in km/s/Mpc
    Omega_m = 0.3 # Matter density parameter
    Omega_Lambda = 0.7 # Dark energy density parameter
    Omega_k = 1.0 - Omega_m - Omega_Lambda # Curvature density parameter
    z = 15 # Redshift

# If we want a flat universe with Omega_k = 0, we can set Omega_Lambda = 1 - Omega_m

    print(f"Comoving Distance: {comoving_distance(z, H0, Omega_m, Omega_k, Omega_Lambda):.2f} Mpc")
    print(f"Luminosity Distance: {luminosity_distance(z, H0, Omega_m, Omega_k, Omega_Lambda):.2f} Mpc")
    print(f"Angular Diameter Distance: {angular_diameter_distance(z, H0, Omega_m, Omega_k, Omega_Lambda):.2f} Mpc")
    print(f"Lookback Time: {lookback_time(z, H0, Omega_m, Omega_k, Omega_Lambda):.2f} Gyr")
    print(f"Age at Redshift {z}: {age_at_redshift(z, H0, Omega_m, Omega_k, Omega_Lambda):.3f} Gyr")

    # ---- K correction demo -------------------------------------------------
    alpha = -0.5                         # quasar continuum slope, L_nu ~ nu^alpha
    L = power_law_L_lambda(alpha)
    lum = FILTERS["Luminance (400-700 nm)"]
    print(f"K (analytic, any filter, Q = R):        {k_correction_power_law(z, alpha):.4f} mag")
    print(f"K (numeric eq.13, Luminance, Q = R):    {k_correction(z, L, lum):.4f} mag")
    print(f"K (numeric eq.13, R = Luminance, Q = V): {k_correction(z, L, lum, FILTERS['Johnson V (~505-595 nm)']):.4f} mag")
    lo, hi = rest_frame_range(lum, z)
    print(f"Luminance filter at z = {z} samples rest-frame {lo:.1f}-{hi:.1f} nm (Lyman alpha = {LYMAN_ALPHA_NM} nm)")
