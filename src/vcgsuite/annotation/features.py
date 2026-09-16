# ══════════════════════════════════════════════════════════════════════════
#  FEATURE-EXTRAKTION
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd

from ..kinematics.frenet_serret import ensure_spherical, smooth, _local_stat, _rank_norm


def extract_features(df: pd.DataFrame, twin: np.ndarray, r_peak_t: float,
                     local_window_ms: float = 20.0) -> pd.DataFrame:
    """
    Extrahiert kinematische Features für ein einzelnes Beat-Fenster.

    Parameters
    ----------
    df           : pd.DataFrame  –  Vollständiges Signal-DataFrame.
    twin         : array-like    –  Zeitachse des Fensters [s].
    r_peak_t     : float         –  Zeitpunkt des R_peak [s] (Referenz für rel_t_ms).
    local_window_ms : float      –  Halbfenster für lokale Statistiken [ms].

    Returns
    -------
    feat : pd.DataFrame  –  Feature-Matrix (Samples × FEATURE_COLS + 't_abs').
    """
    df   = ensure_spherical(df)
    dt   = float(np.median(np.diff(df['Time'].values)))
    lw   = max(1, int(local_window_ms / 1000.0 / dt))
    mask = (df['Time'] >= twin[0]) & (df['Time'] <= twin[-1])
    sub  = df[mask].copy().reset_index(drop=True)
    tsub = sub['Time'].values

    # Rohdaten
    curv      = sub['Curvature'].values.astype(float)
    aabs      = sub['A_abs'].values.astype(float)
    vabs      = sub['V_abs'].values.astype(float)
    tors      = np.abs(sub['Torsion'].values.astype(float))
    r_raw     = sub['r'].values.astype(float)
    theta_raw = np.unwrap(sub['theta'].values.astype(float))
    phi_raw   = np.unwrap(sub['phi'].values.astype(float))

    # Sphärische Ableitungen
    dr_dt     = np.gradient(r_raw,     dt)
    dtheta_dt = np.gradient(theta_raw, dt)
    dphi_dt   = np.gradient(phi_raw,   dt)

    # Geglättete Signale & ihre Ableitungen
    curv_sm = smooth(curv, 3);  aabs_sm = smooth(aabs, 3)
    vabs_sm = smooth(vabs, 3);  r_sm    = smooth(r_raw, 3)
    dcurv   = np.gradient(curv_sm, dt)
    daabs   = np.gradient(aabs_sm, dt)
    dvabs   = np.gradient(vabs_sm, dt)

    # Normierungen
    r_norm  = r_raw / (r_raw.max() + 1e-15)
    r_rel   = r_raw - r_raw.mean()
    r_slope = np.gradient(r_sm, dt)

    return pd.DataFrame({
        'rel_t_ms':        (tsub - r_peak_t) * 1000,
        # Rohdaten
        'Curvature_raw':   curv,
        'A_abs_raw':       aabs,
        'V_abs_raw':       vabs,
        'Torsion_abs_raw': tors,
        'r_raw':           r_raw,
        'dr_dt_raw':       dr_dt,
        'dtheta_dt_raw':   np.abs(dtheta_dt),
        'dphi_dt_raw':     np.abs(dphi_dt),
        # Geglättet
        'Curvature_sm3':   curv_sm,
        'A_abs_sm3':       aabs_sm,
        'V_abs_sm3':       vabs_sm,
        'r_sm3':           r_sm,
        # Ableitungen (geglättet)
        'dCurvature_dt':   dcurv,
        'dA_abs_dt':       daabs,
        'dV_abs_dt':       dvabs,
        # Lokale Maxima
        'Curvature_lmax':  _local_stat(curv_sm, np.max, lw),
        'A_abs_lmax':      _local_stat(aabs_sm, np.max, lw),
        'V_abs_lmax':      _local_stat(vabs_sm, np.max, lw),
        'r_lmax':          _local_stat(r_sm,    np.max, lw),
        # Lokale Standardabweichung
        'Curvature_lstd':  _local_stat(curv_sm, np.std, lw),
        'A_abs_lstd':      _local_stat(aabs_sm, np.std, lw),
        # Rang-Normalisierungen
        'Curvature_rank':  _rank_norm(curv_sm),
        'A_abs_rank':      _rank_norm(aabs_sm),
        'V_abs_rank':      _rank_norm(vabs_sm),
        'r_rank':          _rank_norm(r_sm),
        # Krümmungsradius
        'CurvRadius_raw':  1.0 / (curv     + 1e-15),
        'CurvRadius_sm3':  1.0 / (curv_sm  + 1e-15),
        # r-Metriken
        'r_rel':           r_rel,
        'r_normalized':    r_norm,
        'r_slope':         r_slope,
        # Absoluter Zeitstempel
        't_abs':           tsub,
    })


def _extract_features_for_window(df: pd.DataFrame, anchor_t: float,
                                 lo_ms: float, hi_ms: float) -> pd.DataFrame | None:
    """Bestimmt das Zeitfenster [anchor_t+lo_ms, anchor_t+hi_ms] und ruft
    `extract_features()` genau einmal dafür auf. Gemeinsam genutzter
    Rechenschritt für `detect_in_window()` und `detect_consensus()` — Letztere
    fragt oft mehrere Strategien mit demselben Fenster ab (siehe dort)."""
    t    = df['Time'].values
    tlo  = anchor_t + lo_ms / 1000.0
    thi  = anchor_t + hi_ms / 1000.0
    twin = t[(t >= tlo) & (t <= thi)]
    if len(twin) < 3:
        return None
    return extract_features(df, twin, anchor_t)


def _pick_from_features(feat: pd.DataFrame | None, fcol: str, op: str) -> float:
    """MAX/MIN einer bereits berechneten Feature-Spalte → Zeitpunkt oder NaN.
    Reiner Lookup, keine erneute `extract_features()`-Berechnung."""
    if feat is None or fcol not in feat.columns:
        return np.nan
    fsig = feat[fcol].values.astype(float)
    if np.std(fsig) < 1e-10:
        return np.nan
    idx = int(np.argmax(fsig)) if op == 'MAX' else int(np.argmin(fsig))
    return float(feat['t_abs'].values[idx])


def detect_in_window(df: pd.DataFrame, anchor_t: float, lo_ms: float, hi_ms: float,
                     fcol: str, op: str) -> float:
    """MAX/MIN eines Features im Fenster → Zeitpunkt oder NaN."""
    feat = _extract_features_for_window(df, anchor_t, lo_ms, hi_ms)
    return _pick_from_features(feat, fcol, op)


def detect_consensus(df: pd.DataFrame, anchor_t: float, lo_ms: float, hi_ms: float,
                     strategies: list) -> tuple[float, float]:
    """Mehrere (feature, op)-Votes → Median-Konsens.

    strategies: Liste von (fcol, op) Tupeln
    Gibt (zeitpunkt, spread_ms) zurück.

    Performance-Hinweis: alle Strategien teilen sich dasselbe Fenster
    (lo_ms/hi_ms/anchor_t) — `extract_features()` wird deshalb nur EINMAL
    für alle Strategien gemeinsam berechnet (vorher: einmal pro Strategie,
    identische Neuberechnung — reine Redundanz). Das Ergebnis ist dadurch
    unverändert, siehe `tests/test_annotation.py`."""
    feat = _extract_features_for_window(df, anchor_t, lo_ms, hi_ms)
    votes = [_pick_from_features(feat, fcol, op) for fcol, op in strategies]
    votes = [v for v in votes if not np.isnan(v)]
    if not votes:
        return np.nan, np.nan
    spread_ms = (max(votes) - min(votes)) * 1000 if len(votes) > 1 else 0.0
    return float(np.median(votes)), spread_ms
