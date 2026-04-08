import streamlit as st
import numpy as np
import cosmo_calc as cc  

# Title
st.title("Observability Dashboard")
st.markdown("Change the parameters below to see if the Quasar is observable in real-time.")

if "Omega_m" not in st.session_state:
    st.session_state.Omega_m = 0.300
if "Omega_Lambda" not in st.session_state:
    st.session_state.Omega_Lambda = 0.700

# Sidebar Sliders
st.sidebar.header("Cosmological Parameters")
z = st.sidebar.number_input("Redshift (z)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
Omega_m = st.sidebar.slider("Matter Density (Ωm)", min_value=0.0, max_value=1.5, step=0.001, format="%.3f", key="Omega_m")
Omega_Lambda = st.sidebar.slider("Dark Energy (ΩΛ)", min_value=0.0, max_value=1.5, step=0.001, format="%.3f", key="Omega_Lambda")

H0 = st.sidebar.slider("Hubble Constant (H0)", min_value=50.0, max_value=100.0, value=70.0, step=0.1)

def set_flat_universe():
    st.session_state.Omega_m = 0.3
    st.session_state.Omega_Lambda = 0.7
st.sidebar.button("Flat Universe (Ωk=0)", on_click=set_flat_universe) # Button to change to flat

st.sidebar.header("Telescope & Object")
M_quasar = st.sidebar.number_input("Absolute Magnitude (M)", value=-26.7)
limiting_mag = st.sidebar.number_input("Telescope Limiting Mag", value=26.0)

# Math
Omega_k = 1.0 - Omega_m - Omega_Lambda

# Calculate Luminosity Distance using your CC code
DL = cc.luminosity_distance(z, H0, Omega_m, Omega_k, Omega_Lambda)  # in Mpc

# Calculate age at z 
age_z = cc.age_at_redshift(z, H0, Omega_m, Omega_k, Omega_Lambda)

# Calculate Magnitudes
mu = 5 * np.log10(DL) + 25 # Distance modulus (DL in Mpc, so we add 25)
m = mu + M_quasar          # Apparent magnitude
detectable = "Yes" if m <= limiting_mag else "No"

# --- Display Results
st.subheader("Results")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Curvature (Ωk)", value=f"{Omega_k:.3f}")
    st.metric(label="Luminosity Distance (DL)", value=f"{DL:.2f} Mpc")
    st.metric(label="Age at Redshift (t)", value=f"{age_z:.3f} Gyr")
    
with col2:
    st.metric(label="Distance Modulus (μ)", value=f"{mu:.4f}")
    st.metric(label="Apparent Magnitude (m)", value=f"{m:.4f}")

st.markdown("---")
if m <= limiting_mag:
    st.success(f"**Observable? {detectable}** (m = {m:.2f} is brighter than the limit of {limiting_mag})")
else:
    st.error(f"**Observable? {detectable}** (m = {m:.2f} is too faint for the limit of {limiting_mag})")

