import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import streamlit as st

# -15 to -5

# Constants
h = 6.62607015e-34
k = 1.380649e-23
c = 299792458

# Streamlit configuration
st.set_page_config(page_title="Blackbody Radiation Energy Fraction Calculator", layout="wide")
st.title("Blackbody Radiation Energy Fraction Calculator")
st.markdown("Calculate the fraction of energy emitted by a blackbody within a specific frequency range.")

# Planck function
def planck(v, T):
    exponent = (h * v) / (k * T)
    if exponent > 700: # Python crashes
        return 0.0
    return (2 * h * v**3) / c**2 / (np.exp(exponent) - 1)

# Fraction function
def blackbody_energy_fraction(v1, v2, T):
    # Numerator 
    energy_band, _ = quad(lambda v: planck(v, T), v1, v2)

    # Denominator 
    total_energy = (2 * (k * T)**4 * np.pi**4) / (15 * c**2 * h**3)

    return energy_band / total_energy

# Streamlit UI
st.sidebar.header("Input Parameters")
T = st.sidebar.number_input("Temperature (K)", min_value=1.0, value=3000.0, step=500.0)

st.sidebar.markdown("Frequency Range (Hz)")
st.sidebar.markdown("*TIP: you can input scientific notation like 1e13 for 10^13 directly in the boxes below*")
v1 = st.sidebar.number_input("Lower Frequency (v1)", min_value=4e13, value=4e14, step=1e9, format="%.1e")
v2 = st.sidebar.number_input("Upper Frequency (v2)", min_value=8e14, value=8e14, step=1e9, format="%.1e")

# add number input for y axis limits
y_min = st.sidebar.number_input("Y-axis Minimum", min_value=-20.0, max_value=15.0, value=-15.0, step=1.0)
y_max = st.sidebar.number_input("Y-axis Maximum", min_value=-20.0, max_value=15.0, value=-5.0, step=1.0)

# error
if v1 >= v2:
    st.error("Error: Lower frequency (v1) must be less than upper frequency (v2). Please adjust the values.")
    st.stop()
else:

# Compute fraction
    fraction = blackbody_energy_fraction(v1, v2, T)

# Streamlit output
    st.subheader("Results")
    st.metric(label=f"Energy Fraction in Band emitted between {v1:.1e} and {v2:.1e} Hz", value=f"{fraction:.4%}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    frequencies = np.logspace(12, 18, 1000)
    intensities = np.array([planck(v, T) for v in frequencies])

    log_intensities = np.log10(intensities + 1e-50)

    ax.plot(frequencies, log_intensities, linewidth=2,
        color='black', label='Blackbody Spectrum')

    mask = (frequencies >= v1) & (frequencies <= v2)
    ax.fill_between(frequencies[mask], log_intensities[mask], y2=-30,
                alpha=0.3, label=f"Fraction = {fraction:.2%}", color='orange')

    ax.axvline(v1, linestyle='--', color='red', label=f'v1 = {v1:.1e} Hz')
    ax.axvline(v2, linestyle='--', color='red', label=f'v2 = {v2:.1e} Hz')

    ax.set_xscale('log')
    ax.set_ylim(-15, -5)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('log10(Spectral Radiance)')
    ax.set_title(f'Blackbody Spectrum (T = {T} K)')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
    
        plt.show()
        st.pyplot(fig, width="content")
