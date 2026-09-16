# ══════════════════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN & KINEMATIK  –  Frenet-Serret einer 3D-Raumkurve (VCG)
#
#  Hinweis Migration: Im ursprünglichen Notebook-Export (helper_annontation.py)
#  fehlte der Import von `gaussian_filter1d` (und `numpy`/`pandas`) – diese
#  kamen implizit aus dem Notebook-Namespace anderer Zellen. Für ein
#  eigenständig lauffähiges Modul wurden alle benötigten Imports unten
#  ergänzt.
# ══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d


def smooth(sig: np.ndarray, sigma: float) -> np.ndarray:
    """Gaußglättung; sigma ≤ 0 gibt das Signal unverändert zurück."""
    if sigma <= 0:
        return sig.astype(float)
    return gaussian_filter1d(sig.astype(float), sigma=sigma)


def _rank_norm(sig: np.ndarray) -> np.ndarray:
    """Rang-Normalisierung auf [0, 1]."""
    r = np.argsort(np.argsort(sig.astype(float)))
    return r / max(len(r) - 1, 1)


def _local_stat(sig: np.ndarray, fn, lw: int) -> np.ndarray:
    """Rollendes Fenster ±lw Samples (am Rand kleiner statt gepolstert),
    Statistik `fn` wird angewendet.

    Vektorisierte Fast-Pfade für `np.max`/`np.min`/`np.std` (die einzigen im
    Code tatsächlich genutzten Varianten, siehe `annotation/features.py`):

    - `np.max`/`np.min`: `scipy.ndimage` mit `mode="nearest"` ist hier exakt
      äquivalent zum ursprünglichen Clipping-Fenster, nicht nur eine
      Näherung — die am Rand von "nearest" gespiegelten Zusatzwerte sind
      immer Duplikate von Werten, die im geclippten Fenster ohnehin schon
      enthalten sind, und Duplikate ändern Max/Min einer Menge nicht.
      Getestet (siehe `tests/test_kinematics.py`): 0 Abweichung, auch am Rand.
    - `np.std`: Padding wäre hier NICHT exakt (Mittelwert/Varianz sind im
      Gegensatz zu Max/Min nicht duplikat-invariant) — stattdessen eine
      exakte O(n)-Berechnung über Präfixsummen mit der tatsächlichen, am
      Rand kleineren Fenstergröße (kein Padding nötig, keine Näherung).
    - Jede andere Funktion: Fallback auf die langsame, aber garantiert
      korrekte Python-Schleife von vorher.
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    size = 2 * lw + 1

    if fn is np.max:
        return maximum_filter1d(sig, size=size, mode="nearest")
    if fn is np.min:
        return minimum_filter1d(sig, size=size, mode="nearest")
    if fn is np.std:
        idx = np.arange(n)
        lo_idx = np.clip(idx - lw, 0, None)
        hi_idx = np.clip(idx + lw + 1, None, n)
        counts = (hi_idx - lo_idx).astype(float)
        csum  = np.concatenate(([0.0], np.cumsum(sig)))
        csum2 = np.concatenate(([0.0], np.cumsum(sig * sig)))
        s  = csum[hi_idx]  - csum[lo_idx]
        s2 = csum2[hi_idx] - csum2[lo_idx]
        mean = s / counts
        var  = np.clip(s2 / counts - mean * mean, 0.0, None)  # numerische Robustheit
        return np.sqrt(var)

    out = np.empty_like(sig)
    for i in range(n):
        lo, hi = max(0, i - lw), min(n, i + lw + 1)
        out[i] = fn(sig[lo:hi])
    return out


def ensure_spherical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt sphärische Koordinaten zum DataFrame hinzu, falls noch nicht vorhanden.

    Konvention (Physik):
        r     – Zeigerlänge (Betrag des VCG-Vektors)
        theta – Azimutwinkel in der XY-Ebene  [-π, π]   (wie arctan2)
        phi   – Polarwinkel von der Z-Achse   [ 0,  π]

    Hinweis: theta und phi in Radiant.
    """
    if 'r' in df.columns:
        return df
    df   = df.copy()
    x, y, z  = df['X'].values, df['Y'].values, df['Z'].values
    r        = np.sqrt(x**2 + y**2 + z**2)
    df['r']     = r
    df['theta'] = np.arctan2(y, x)                  # Azimut
    df['phi']   = np.arccos(z / (r + 1e-10))        # Polar
    return df


# ══════════════════════════════════════════════════════════════════════════
#  KINEMATIK  –  Frenet-Serret einer 3D-Raumkurve
# ══════════════════════════════════════════════════════════════════════════

def compute_vcg_kinematics(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Berechnet Geschwindigkeit, Beschleunigung, Jerk, Krümmung, Torsion
    und sphärische Koordinaten (r, theta, phi) aus den XYZ-Komponenten des VCG.

    Differentialgeometrische Grundlage (Frenet-Serret):
        κ  =  |v × a|  /  |v|³          Krümmung
        τ  =  (v × a) · j  /  |v × a|²  Torsion

    Parameters
    ----------
    df : pd.DataFrame
        Muss 'Time', 'X', 'Y', 'Z' enthalten.

    Returns
    -------
    df : pd.DataFrame
        In-place ergänzt um kinematische und sphärische Spalten.
    dt : float
        Mittlerer Zeitschritt [s].
    """
    dt = np.mean(np.diff(df['Time']))

    # ── 1. Ableitung – Geschwindigkeit
    V_X   = np.gradient(df['X'], dt)
    V_Y   = np.gradient(df['Y'], dt)
    V_Z   = np.gradient(df['Z'], dt)
    V_abs = np.sqrt(V_X**2 + V_Y**2 + V_Z**2)

    # ── 2. Ableitung – Beschleunigung
    A_X   = np.gradient(V_X, dt)
    A_Y   = np.gradient(V_Y, dt)
    A_Z   = np.gradient(V_Z, dt)
    A_abs = np.sqrt(A_X**2 + A_Y**2 + A_Z**2)

    # ── 3. Ableitung – Jerk
    J_X = np.gradient(A_X, dt)
    J_Y = np.gradient(A_Y, dt)
    J_Z = np.gradient(A_Z, dt)

    # ── Krümmung κ und Torsion τ
    v             = np.stack([V_X, V_Y, V_Z], axis=-1)
    a             = np.stack([A_X, A_Y, A_Z], axis=-1)
    j             = np.stack([J_X, J_Y, J_Z], axis=-1)
    cross_va      = np.cross(v, a)
    cross_va_norm = np.linalg.norm(cross_va, axis=-1)
    v_norm        = np.linalg.norm(v, axis=-1)

    curvature = cross_va_norm / (v_norm**3 + 1e-15)
    torsion   = (
        np.einsum('ij,ij->i', cross_va, j)
        / (cross_va_norm**2 + 1e-15)
    )

    # ── Sphärische Koordinaten des VCG-Zeigers
    x, y, z = df['X'].values, df['Y'].values, df['Z'].values
    r       = np.sqrt(x**2 + y**2 + z**2)
    theta   = np.arctan2(y, x)          # Azimutwinkel in XY-Ebene  [-π, π]
    phi     = np.arccos(z / (r + 1e-10))  # Polarwinkel von Z-Achse  [ 0,  π]

    # ── Ins DataFrame schreiben
    df['V_X']       = V_X
    df['V_Y']       = V_Y
    df['V_Z']       = V_Z
    df['V_abs']     = V_abs
    df['A_X']       = A_X
    df['A_Y']       = A_Y
    df['A_Z']       = A_Z
    df['A_abs']     = A_abs
    df['Curvature'] = curvature
    df['Torsion']   = torsion
    df['r']         = r
    df['theta']     = theta
    df['phi']       = phi

    return df, dt
