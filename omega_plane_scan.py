"""
omega_plane_scan.py  --  which cosmologies give the same apparent magnitude?

Idea from the 2026-10-14 meeting (whiteboard):

  1. Pick a redshift z (2, 3 or 4) and an absolute magnitude M.
  2. At the fiducial cosmology (Omega_m, Omega_Lambda) = (0.3, 0.7) this gives one
     apparent magnitude m0 (e.g. 19.57 at z = 3 with the K correction included).
  3. Treat +-TOL (e.g. 0.03 mag) as the measurement error on m.
  4. Choose an angle in the (Omega_m, Omega_Lambda) plane and step outward from the
     fiducial point until |m - m0| > TOL, or until the cosmology stops being physical
     (no Big Bang / past the antipode of a closed universe).
  5. Repeat for every angle -> the boundary of the region of cosmologies that a single
     magnitude measurement cannot tell apart from the fiducial one.

Run:   python omega_plane_scan.py                (interactive plots)
       python omega_plane_scan.py --z 3 --tol 0.05 --save figures
       python omega_plane_scan.py --help
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cosmo_calc as cc

# default settings 
Z_LIST = [2.0, 3.0, 4.0]
M_ABS = -26.7          # absolute magnitude of the quasar
H0 = 70.0              # km/s/Mpc
ALPHA = -0.5           # power-law slope L_nu ~ nu^alpha (K correction, Hogg convention)
TOL = 0.03             # +- tolerance in magnitudes
OM0, OL0 = 0.3, 0.7    # fiducial cosmology (start of every ray)
STEP = 0.005           # step length in the (Omega_m, Omega_Lambda) plane
MAX_R = 2.0            # give up after walking this far from the fiducial point
N_ANGLES = 360         # one ray every degree
GRID_N = 121           # resolution of the background map of |m - m0|


# physics helpers 
def is_physical(Om, OL, z, H0=H0, n=400):
    """True if the cosmology can be used at redshift z.

    Requires Omega_m >= 0 and E(z')^2 > 0 for all 0 <= z' <= z (otherwise the
    universe bounced and never reached that redshift), and for a closed universe
    that the light has not yet passed the antipode (where D_L would hit zero).
    """
    if Om < 0:
        return False
    Ok = 1.0 - Om - OL
    zz = np.linspace(0.0, z, n)
    E2 = Om * (1 + zz) ** 3 + Ok * (1 + zz) ** 2 + OL
    if np.any(E2 <= 0):
        return False
    if Ok < 0:
        d_c = cc.comoving_distance(z, H0, Om, Ok, OL)
        if np.sqrt(-Ok) * d_c * H0 / cc.c >= np.pi:
            return False
    return True


def apparent_magnitude(z, Om, OL, H0=H0, M=None, alpha=ALPHA):
    """m = M + DM + K  (Hogg+02 eq. 2) with the power-law K correction."""
    if M is None:
        M = M_ABS
    Ok = 1.0 - Om - OL
    DL = cc.luminosity_distance(z, H0, Om, Ok, OL)   # Mpc
    mu = 5.0 * np.log10(DL) + 25.0
    K = cc.k_correction_power_law(z, alpha)          # independent of cosmology
    return M + mu + K


# the ray walk
def step_out(z, theta_deg, m0, tol=TOL, step=STEP, max_r=MAX_R, origin=(OM0, OL0)):
    """Walk from the fiducial point along angle theta (degrees, measured from the
    +Omega_m axis towards +Omega_Lambda) until |m - m0| > tol.

    Returns a dict with the whole path (r, Om, OL, m) and why the walk stopped:
      'tolerance'  -> last point is the first one outside m0 +- tol
      'unphysical' -> next step would leave the physical region; last point is the
                      last physical one
      'max_r'      -> walked max_r without leaving the band (degenerate direction)
    """
    th = np.radians(theta_deg)
    ux, uy = np.cos(th), np.sin(th)
    rs, Oms, OLs, ms = [], [], [], []
    r, reason = 0.0, "max_r"
    while r <= max_r:
        Om, OL = origin[0] + r * ux, origin[1] + r * uy
        if not is_physical(Om, OL, z):
            reason = "unphysical"
            break
        m = apparent_magnitude(z, Om, OL)
        if abs(m - m0) > tol:
            # crossed the band: bisect between the last inside point and this one
            # so the final point sits on |m - m0| = tol instead of overshooting
            lo, hi = r - step, r
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                m_mid = apparent_magnitude(z, origin[0] + mid * ux, origin[1] + mid * uy)
                if abs(m_mid - m0) > tol:
                    hi = mid
                else:
                    lo = mid
            r = hi
            Om, OL = origin[0] + r * ux, origin[1] + r * uy
            m = apparent_magnitude(z, Om, OL)
            reason = "tolerance"
        rs.append(r)
        Oms.append(Om)
        OLs.append(OL)
        ms.append(m)
        if reason == "tolerance":
            break
        r += step
    return dict(theta=theta_deg, r=np.array(rs), Om=np.array(Oms), OL=np.array(OLs),
                m=np.array(ms), reason=reason)


def scan(z, tol=TOL, step=STEP, n_angles=N_ANGLES, max_r=MAX_R):
    m0 = apparent_magnitude(z, OM0, OL0)
    angles = np.arange(0.0, 360.0, 360.0 / n_angles)
    rays = [step_out(z, th, m0, tol, step, max_r) for th in angles]
    return m0, rays


def magnitude_map(z, m0, om_range, ol_range, n=GRID_N):
    """m - m0 on a grid (NaN where unphysical). Background check of the ray walk."""
    oms = np.linspace(*om_range, n)
    ols = np.linspace(*ol_range, n)
    dm = np.full((n, n), np.nan)
    for j, OL in enumerate(ols):
        for i, Om in enumerate(oms):
            if is_physical(Om, OL, z):
                dm[j, i] = apparent_magnitude(z, Om, OL) - m0
    return oms, ols, dm


# plot

def plot_omega_plane(results, tol, show_map=True):
    """results: list of (z, m0, rays)."""
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    fig, ax = plt.subplots(figsize=(8, 7))

    # axis range from the boundary points
    allx = np.concatenate([[ray["Om"][-1] for ray in rays] for _, _, rays in results])
    ally = np.concatenate([[ray["OL"][-1] for ray in rays] for _, _, rays in results])
    pad = 0.1
    om_range = (max(0.0, allx.min() - pad), allx.max() + pad)
    ol_range = (ally.min() - pad, ally.max() + pad)

    if show_map:
        # shade the full +-tol band of each z from a grid (independent check of the rays)
        for (z, m0, _), col in zip(results, colors):
            oms, ols, dm = magnitude_map(z, m0, om_range, ol_range)
            ax.contourf(oms, ols, np.abs(dm) <= tol, levels=[0.5, 1.5],
                        colors=[col], alpha=0.18)
            ax.contour(oms, ols, dm, levels=[-tol, tol], colors=[col], linewidths=0.6, alpha=0.6)

    for (z, m0, rays), col in zip(results, colors):
        xb = np.array([ray["Om"][-1] for ray in rays])
        yb = np.array([ray["OL"][-1] for ray in rays])
        reason = np.array([ray["reason"] for ray in rays])
        hit = reason == "tolerance"
        ax.plot(xb[hit], yb[hit], "o", color=col, ms=3.5,
                label=f"z = {z:g}  (m0 = {m0:.2f})")
        ax.plot(xb[~hit], yb[~hit], "x", color=col, ms=6)

    ax.plot(OM0, OL0, "r*", ms=14, label=f"fiducial ({OM0}, {OL0})")
    xx = np.linspace(om_range[0], om_range[1], 10)
    ax.plot(xx, 1 - xx, "k--", lw=1, label="flat: Ωm + ΩΛ = 1")
    ax.set_xlim(om_range)
    ax.set_ylim(ol_range)
    ax.set_xlabel("Ωm")
    ax.set_ylabel("ΩΛ")
    ax.set_title(f"Cosmologies with |m − m0| ≤ {tol} mag   (M = {M_ABS}, H0 = {H0}, α = {ALPHA})\n"
                 "points: where each ray leaves the band (o tolerance, x physical limit); "
                 "shading: band from a grid", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_ray_profiles(z, m0, rays, tol, angles=(0, 45, 90, 135, 180, 225, 270, 315)):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(-tol, tol, color="gold", alpha=0.25, label=f"±{tol} mag band")
    for ray in rays:
        if min(abs(ray["theta"] - a) for a in angles) < 1e-6:
            ax.plot(ray["r"], ray["m"] - m0, "-", lw=1.5,
                    label=f"θ = {ray['theta']:.0f}°  ({ray['reason']})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("distance r from fiducial point in the (Ωm, ΩΛ) plane")
    ax.set_ylabel("m − m0  [mag]")
    ax.set_title(f"Magnitude change along rays at z = {z:g}  (m0 = {m0:.3f})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# main
def main():
    global M_ABS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z", type=float, nargs="+", default=Z_LIST, help="redshift(s) to scan")
    p.add_argument("--M", type=float, default=M_ABS, help="absolute magnitude")
    p.add_argument("--tol", type=float, default=TOL, help="tolerance in mag")
    p.add_argument("--step", type=float, default=STEP, help="step length in the Omega plane")
    p.add_argument("--angles", type=int, default=N_ANGLES, help="number of rays")
    p.add_argument("--max-r", type=float, default=MAX_R, help="max walk distance")
    p.add_argument("--profile-z", type=float, default=None,
                   help="z for the ray-profile plot (default: middle z)")
    p.add_argument("--no-map", action="store_true", help="skip the background |m-m0| map (faster)")
    p.add_argument("--save", metavar="DIR", default=None, help="save PNGs into DIR")
    p.add_argument("--no-show", action="store_true", help="do not open plot windows")
    a = p.parse_args()
    M_ABS = a.M

    results = []
    for z in a.z:
        m0, rays = scan(z, a.tol, a.step, a.angles, a.max_r)
        results.append((z, m0, rays))
        rb = np.array([ray["r"][-1] for ray in rays])
        reasons = [ray["reason"] for ray in rays]
        print(f"\nz = {z:g}:  m0 = {m0:.4f} mag  (M = {M_ABS}, H0 = {H0}, "
              f"K = {cc.k_correction_power_law(z, ALPHA):+.3f})")
        print(f"  boundary distance: min {rb.min():.3f} at θ = {rays[rb.argmin()]['theta']:.0f}°, "
              f"max {rb.max():.3f} at θ = {rays[rb.argmax()]['theta']:.0f}°")
        print(f"  stopped by: tolerance {reasons.count('tolerance')}, "
              f"physical limit {reasons.count('unphysical')}, max_r {reasons.count('max_r')}")
        print("   θ(deg)     r      Ωm      ΩΛ     m-m0   reason")
        for ray in rays:
            if ray["theta"] % 45 == 0:
                print(f"   {ray['theta']:5.0f}  {ray['r'][-1]:6.3f}  {ray['Om'][-1]:6.3f}  "
                      f"{ray['OL'][-1]:6.3f}  {ray['m'][-1] - m0:+6.3f}   {ray['reason']}")

    fig1 = plot_omega_plane(results, a.tol, show_map=not a.no_map)
    zp = a.profile_z if a.profile_z is not None else a.z[len(a.z) // 2]
    z, m0, rays = min(results, key=lambda t: abs(t[0] - zp))
    fig2 = plot_ray_profiles(z, m0, rays, a.tol)

    if a.save:
        os.makedirs(a.save, exist_ok=True)
        fig1.savefig(os.path.join(a.save, "omega_plane_region.png"), dpi=150)
        fig2.savefig(os.path.join(a.save, f"ray_profiles_z{z:g}.png"), dpi=150)
        print(f"\nsaved figures to {a.save}")
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
