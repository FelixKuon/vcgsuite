# ══════════════════════════════════════════════════════════════════════════
#  HIERARCHISCHE BEAT-ANNOTATION
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import detect_consensus
from ..kinematics.constants import HIERARCHICAL_WINDOWS, Q_OFF_OFFSET_S


def annotate_beat_hierarchical(df: pd.DataFrame, r_peak_t: float,
                               r_turn_t: float | None = None) -> dict:
    """
    Annotiert einen einzelnen Beat hierarchisch: ausgehend von R_peak+ (und
    optional R_turn) werden nacheinander S/Q-Grenzen, P-Komplex und T-Welle
    per Konsens-Voting (siehe `detect_consensus`) lokalisiert. Spätere
    Marker referenzieren dabei frühere Marker als Anker (z. B. sucht T_on
    relativ zu S_off, nicht relativ zu R_peak).

    Die Suchfenstergrenzen (lo_ms/hi_ms je Marker) stammen aus
    `vcgsuite.kinematics.constants.HIERARCHICAL_WINDOWS` statt hartcodierter
    Zahlen im Funktionskörper (siehe dortiger Kommentar zur Konsolidierung
    der alten WINDOWS_LOCAL-Konstante).

    Parameters
    ----------
    r_turn_t : float oder None
        Wenn übergeben, wird R_turn direkt gesetzt (deterministisch).
        Sonst Fallback auf detect_consensus.

    Returns
    -------
    res : dict  –  Marker-Name → Zeitpunkt [s] (oder NaN).
    """
    W = HIERARCHICAL_WINDOWS
    res = {'R_peak+': r_peak_t}

    # ── Stufe 2: relativ zu R_peak ────────────────────────────────────────
    if r_turn_t is not None and not np.isnan(r_turn_t):
        res['R_turn'] = r_turn_t                        # ← deterministisch
    else:
        w = W['R_turn']
        res['R_turn'], _ = detect_consensus(df, r_peak_t, w['lo_ms'], w['hi_ms'], [
            ('dtheta_dt_raw', 'MAX'),
            ('dphi_dt_raw',   'MIN'),
        ])

    w = W['S_on']
    res['S_on'], _ = detect_consensus(df, r_peak_t, w['lo_ms'], w['hi_ms'], [
        ('Curvature_raw',  'MIN'),
        ('CurvRadius_raw', 'MAX'),
        ('r_slope',        'MIN'),
    ])
    w = W['S_off']
    res['S_off'], _ = detect_consensus(df, r_peak_t, w['lo_ms'], w['hi_ms'], [
        ('A_abs_lstd', 'MAX'),
        ('r_slope',    'MAX'),
    ])
    w = W['Q_on']
    res['Q_on'], _ = detect_consensus(df, r_peak_t, w['lo_ms'], w['hi_ms'], [
        ('Curvature_raw', 'MAX'),
        ('V_abs_raw',     'MIN'),
    ])
    res['Q_off'] = r_peak_t + Q_OFF_OFFSET_S

    # ── Stufe 3a: P-Komplex ───────────────────────────────────────────────
    w = W['P_peak']
    res['P_peak'], _ = detect_consensus(df, r_peak_t, w['lo_ms'], w['hi_ms'], [
        ('r_raw',        'MAX'),
        ('r_normalized', 'MAX'),
    ])
    if not np.isnan(res['P_peak']):
        w = W['P_on']
        res['P_on'], _  = detect_consensus(df, res['P_peak'], w['lo_ms'], w['hi_ms'], [
            ('dA_abs_dt',     'MAX'),
            ('dCurvature_dt', 'MIN'),
        ])
        w = W['P_off']
        res['P_off'], _ = detect_consensus(df, res['P_peak'], w['lo_ms'], w['hi_ms'], [
            ('Curvature_raw', 'MAX'),
            ('Curvature_sm3', 'MAX'),
        ])
    else:
        res['P_on'] = res['P_off'] = np.nan

    # ── Stufe 3b: T-Welle ─────────────────────────────────────────────────
    t_soff = res.get('S_off', np.nan)
    w = W['T_on']
    res['T_on'] = detect_consensus(df, t_soff, w['lo_ms'], w['hi_ms'], [
        ('CurvRadius_raw', 'MIN'),
        ('dCurvature_dt',  'MIN'),
    ])[0] if not np.isnan(t_soff) else np.nan

    t_on = res.get('T_on', np.nan)
    w = W['T_turn1']
    res['T_turn1'] = detect_consensus(df, t_on, w['lo_ms'], w['hi_ms'], [
        ('r_raw',         'MAX'),
        ('r_normalized',  'MAX'),
        ('Curvature_raw', 'MAX'),
    ])[0] if not np.isnan(t_on) else np.nan

    t_turn1 = res.get('T_turn1', np.nan)
    w = W['T_turn2']
    res['T_turn2'] = detect_consensus(df, t_turn1, w['lo_ms'], w['hi_ms'], [
        ('r_raw',        'MAX'),
        ('r_normalized', 'MAX'),
        ('r_rel',        'MAX'),
    ])[0] if not np.isnan(t_turn1) else np.nan

    t_turn2 = res.get('T_turn2', np.nan)
    w = W['T_off']
    res['T_off'] = detect_consensus(df, t_turn2, w['lo_ms'], w['hi_ms'], [
        ('dA_abs_dt', 'MIN'),
        ('dV_abs_dt', 'MIN'),
    ])[0] if not np.isnan(t_turn2) else np.nan

    return res
