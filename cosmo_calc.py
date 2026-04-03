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