# ══════════════════════════════════════════════════════════════════════════
#  P-LOOP FEATURES
#
#  Migriert aus P-loop_features.py ("BLOCK A — P-Loop Features").
#  Input:  df_analysis (Spalten: Time, X, Y, Z), df_annotations, df_r
#  Output: df_p
#
#  Die 8 gemeinsamen Loop-Helper (safe, t2idx, loop_svd, ...) leben jetzt in
#  `._shared`, statt hier als eigene Kopie definiert zu sein. `fs`/`dt`
#  werden nicht mehr hartcodiert (siehe `_shared.resolve_fs`).
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import (
    safe, t2idx, loop_area_3d, loop_svd, loop_dipol_norm,
    loop_asym, loop_roundness_fwhm, angle_between, resolve_fs,
)


def compute_p_wave_features(df_analysis: pd.DataFrame,
                            df_annotations: pd.DataFrame,
                            df_r: pd.DataFrame,
                            fs: float | None = None) -> pd.DataFrame:
    """
    Berechnet P-Loop-Features (Zeit- und Geometrie-Merkmale der P-Welle)
    pro Beat.

    Parameters
    ----------
    df_analysis    : pd.DataFrame  –  Spalten Time, X, Y, Z.
    df_annotations : pd.DataFrame  –  Ausgabe von `annotate_all_beats()`.
    df_r           : pd.DataFrame  –  Ausgabe von `compute_beat_rotation()`,
                                      muss 'beat_id', 'Rpeak_r', 'Rpeak_ca',
                                      'Rturn_r', 'Rturn_ca', 'RR_ms' enthalten.
    fs             : float | None  –  Abtastrate. None → aus
                                      `df_analysis.attrs["fs"]` bzw.
                                      `vcgsuite.config.FS` abgeleitet
                                      (siehe `_shared.resolve_fs`).

    Returns
    -------
    df_p : pd.DataFrame  –  eine Zeile pro Beat mit P-Loop-Features.
    """
    fs_val = resolve_fs(df_analysis, fs)
    dt = 1.0 / fs_val

    vcg = df_analysis[['X', 'Y', 'Z']].to_numpy()
    t_vcg = df_analysis['Time'].to_numpy()

    # ── Merge ─────────────────────────────────────────────────────────────
    extra_r_cols = ['beat_id', 'Rpeak_r', 'Rpeak_ca', 'Rturn_r', 'Rturn_ca', 'RR_ms']
    df_merged = df_annotations.merge(df_r[extra_r_cols], on='beat_id', how='left')
    df_merged = df_merged.sort_values('beat_id').reset_index(drop=True)

    # ── QRS-Hauptachse vorberechnen (für θ_P_QRS) ────────────────────────
    # Nutzt t_Q_on → t_S_off aus df_annotations als QRS-Fenster.
    # TODO: theta_P_QRS (hier) und theta_QRS_P (in qrs_complex.py) werden
    # unabhängig über zwei separate SVD-Berechnungen auf demselben
    # (Q_on, S_off)-Fenster bestimmt statt eine gemeinsame Quelle zu nutzen
    # und können dadurch leicht divergieren, siehe Analysebericht /
    # Migrations-Kommentar in qrs_complex.py.
    qrs_axes = {}
    for _, row in df_merged.iterrows():
        bid = int(row['beat_id'])
        t_qon = safe(row, 't_Q_on')
        t_soff = safe(row, 't_S_off')
        if np.isfinite(t_qon) and np.isfinite(t_soff):
            i0 = t2idx(t_qon, t_vcg)
            i1 = t2idx(t_soff, t_vcg)
            seg = vcg[i0:i1 + 1]
            if len(seg) >= 4:
                _, Vt = loop_svd(seg)
                qrs_axes[bid] = Vt[0]   # Hauptachsenvektor

    # ── Feature-Berechnung pro Beat ───────────────────────────────────────
    records = []

    for _, row in df_merged.iterrows():
        bid = int(row['beat_id'])
        t_pon = safe(row, 't_P_on')
        t_ppk = safe(row, 't_P_peak')
        t_poff = safe(row, 't_P_off')
        t_qon = safe(row, 't_Q_on')
        t_rpk = safe(row, 't_R_peak+')
        rr = safe(row, 'RR_ms')
        resp = row.get('resp_phase_x', np.nan)

        # ── Skalare Zeit-Features ─────────────────────────────────────────
        p_dur_ms = (t_poff - t_pon) * 1000 if np.isfinite(t_poff) and np.isfinite(t_pon) else np.nan
        pq_ms = (t_qon - t_pon) * 1000 if np.isfinite(t_qon) and np.isfinite(t_pon) else np.nan
        p_rise_ms = (t_ppk - t_pon) * 1000 if np.isfinite(t_ppk) and np.isfinite(t_pon) else np.nan

        if np.isfinite(t_ppk) and np.isfinite(t_pon) and np.isfinite(t_poff):
            denom = t_poff - t_pon
            p_symmetry = (t_ppk - t_pon) / denom if denom > 0 else np.nan
        else:
            p_symmetry = np.nan

        # ── Loop-Features ─────────────────────────────────────────────────
        p_area = p_rho = p_phi = p_asym = p_dipol = p_round = np.nan
        p_axis = np.array([np.nan, np.nan, np.nan])
        theta_p_qrs = np.nan

        if np.isfinite(t_pon) and np.isfinite(t_poff):
            i0 = t2idx(t_pon, t_vcg)
            i1 = t2idx(t_poff, t_vcg)
            seg = vcg[i0:i1 + 1]

            if len(seg) >= 4:
                p_area = loop_area_3d(seg)
                p_dipol = loop_dipol_norm(seg, dt)
                p_asym = loop_asym(seg)
                p_round = loop_roundness_fwhm(seg)

                S, Vt = loop_svd(seg)
                lam = S ** 2
                if lam.sum() > 0:
                    p_rho = lam[1] / lam[0]        # Rundheit: λ2/λ1
                    p_phi = lam[2] / lam.sum()      # Planarität: λ3/Σλ
                p_axis = Vt[0]                       # Hauptachsenvektor

                # Winkel P-Achse ↔ QRS-Achse
                if bid in qrs_axes:
                    theta_p_qrs = angle_between(p_axis, qrs_axes[bid])

        records.append({
            'beat_id': bid,
            'RR_ms': rr,
            # Zeit-Features
            'P_dur_ms': p_dur_ms,
            'PQ_ms': pq_ms,
            'P_rise_ms': p_rise_ms,
            'P_symmetry': p_symmetry,
            # Loop-Features
            'P_area': p_area,
            'P_rho': p_rho,
            'P_phi': p_phi,
            'P_asym': p_asym,
            'P_dipol_norm': p_dipol,
            'P_round': p_round,
            # Geometrie
            'P_axis_x': p_axis[0],
            'P_axis_y': p_axis[1],
            'P_axis_z': p_axis[2],
            'theta_P_QRS': theta_p_qrs,
            # Zeitstempel
            't_P_on': t_pon,
            't_P_peak': t_ppk,
            't_P_off': t_poff,
            't_Q_on': t_qon,
            't_R_peak': t_rpk,
            'resp_phase': resp,
        })

    df_p = pd.DataFrame(records)

    feat_cols = ['P_dur_ms', 'PQ_ms', 'P_rise_ms', 'P_symmetry',
                 'P_area', 'P_rho', 'P_phi', 'P_asym', 'P_dipol_norm', 'P_round', 'theta_P_QRS']
    print(f"Beats: {len(df_p)}  |  VCG-Samples: {len(vcg)}  |  fs={fs_val} Hz")
    print(df_p[feat_cols].describe().round(4))
    print(f"\nFehlende Werte:\n{df_p[feat_cols].isna().sum()}")

    return df_p
