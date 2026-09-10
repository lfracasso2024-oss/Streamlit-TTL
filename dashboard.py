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

# K correction inputs (Hogg et al. 2002, eq. 13, power-law spectrum L_nu ~ nu^alpha)
st.sidebar.header("K Correction (Hogg+02 eq. 13)")
alpha = st.sidebar.number_input("Spectral index α (L_ν ∝ ν^α)", value=-0.5, step=0.1,
                                help="Power-law slope of the quasar continuum. α ≈ -0.5 is typical for the optical/UV.")
NO_FILTER = "None (filter-independent)"
R_choice = st.sidebar.selectbox("Observed filter R", [NO_FILTER] + list(cc.FILTERS))
same_band = st.sidebar.checkbox("Rest-frame band Q = R", value=True, disabled=(R_choice == NO_FILTER),
                                help="Untick if the absolute magnitude M is defined in a different rest-frame band than the one you observe in.")
Q_choice = R_choice
if R_choice != NO_FILTER and not same_band:
    Q_choice = st.sidebar.selectbox("Rest-frame filter Q (band of M)", list(cc.FILTERS))

# Math
Omega_k = 1.0 - Omega_m - Omega_Lambda

# Calculate Luminosity Distance using your CC code
DL = cc.luminosity_distance(z, H0, Omega_m, Omega_k, Omega_Lambda)  # in Mpc

# Calculate age at z 
age_z = cc.age_at_redshift(z, H0, Omega_m, Omega_k, Omega_Lambda)

# K correction
if R_choice == NO_FILTER:
    # Power law with Q = R: eq. 13 collapses to K = -2.5 (1 + alpha) log10(1 + z), no filter needed
    K = cc.k_correction_power_law(z, alpha)
    rest_range = None
else:
    R_band = cc.FILTERS[R_choice]
    Q_band = cc.FILTERS[Q_choice]
    K = cc.k_correction(z, cc.power_law_L_lambda(alpha), R_band, Q_band)
    rest_range = cc.rest_frame_range(R_band, z)

# Calculate Magnitudes
mu = 5 * np.log10(DL) + 25 # Distance modulus (DL in Mpc, so we add 25)
m = mu + M_quasar + K      # Apparent magnitude: m_R = M_Q + DM + K_QR (Hogg eq. 2)
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
    st.metric(label="K Correction (K)", value=f"{K:.4f}")
    st.metric(label="Apparent Magnitude (m)", value=f"{m:.4f}")

if rest_range is not None:
    lo, hi = rest_range
    st.caption(f"{R_choice} at z = {z:g} samples rest-frame wavelengths {lo:.1f}-{hi:.1f} nm "
               f"(rest-frame band Q: {Q_choice}).")
    if hi < cc.LYMAN_ALPHA_NM:
        st.warning(f"The whole observed band sits blueward of Lyman α ({cc.LYMAN_ALPHA_NM} nm) in the rest frame. "
                   "At high z the IGM absorbs nearly all of that light, so a pure power-law K correction is not reliable here.")
    elif lo < cc.LYMAN_ALPHA_NM:
        st.info(f"Part of the observed band is blueward of Lyman α ({cc.LYMAN_ALPHA_NM} nm) in the rest frame; "
                "the power-law K correction ignores IGM absorption there.")

st.markdown("---")
if m <= limiting_mag:
    st.success(f"**Observable? {detectable}** (m = {m:.2f} is brighter than the limit of {limiting_mag})")
else:
    st.error(f"**Observable? {detectable}** (m = {m:.2f} is too faint for the limit of {limiting_mag})")

