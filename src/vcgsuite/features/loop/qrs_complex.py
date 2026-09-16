# ══════════════════════════════════════════════════════════════════════════
#  QRS-LOOP FEATURES  (VCG-basiert)
#
#  Migriert aus QRS-loop_features.py ("BLOCK A_QRS — QRS-Loop Features").
#  Input:  df_annotations, df_analysis, df_vagus, df_p, df_r
#  Output: df_qrs
#
#  df_vagus (T-/QRS-Achsen, QT/QTc, ...) muss vorher via
#  `features.loop.t_wave.build_vagus_features()` erzeugt worden sein — im
#  Original wurde df_vagus im "T-Loop"-Block inline erzeugt und stand daher
#  bereits zur Verfügung, wenn "Block A_QRS" lief; das wird hier durch die
#  explizite Übergabe als Parameter sichtbar gemacht.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import (
    safe, t2idx, loop_area_3d, loop_svd, loop_dipol_norm,
    loop_asym, loop_roundness_fwhm, angle_between, resolve_fs,
)


def compute_qrs_features(df_analysis: pd.DataFrame,
                         df_annotations: pd.DataFrame,
                         df_vagus: pd.DataFrame,
                         df_p: pd.DataFrame,
                         df_r: pd.DataFrame,
                         fs: float | None = None) -> pd.DataFrame:
    """
    Berechnet QRS-Loop-Features (Zeit- und Geometrie-Merkmale des QRS-
    Komplexes) pro Beat.

    Parameters
    ----------
    df_analysis    : pd.DataFrame  –  Spalten Time, X, Y, Z.
    df_annotations : pd.DataFrame  –  Ausgabe von `annotate_all_beats()`.
    df_vagus       : pd.DataFrame  –  Ausgabe von
                                      `features.loop.t_wave.build_vagus_features()`.
    df_p           : pd.DataFrame  –  Ausgabe von `compute_p_wave_features()`
                                      (für θ_QRS_P).
    df_r           : pd.DataFrame  –  Ausgabe von `compute_beat_rotation()`.
    fs             : float | None  –  Abtastrate, siehe `_shared.resolve_fs`.

    Returns
    -------
    df_qrs : pd.DataFrame
    """
    fs_val = resolve_fs(df_analysis, fs)
    dt = 1.0 / fs_val

    vcg = df_analysis[['X', 'Y', 'Z']].to_numpy()
    t_vcg = df_analysis['Time'].to_numpy()

    # ── Merge ─────────────────────────────────────────────────────────────
    extra_r_cols = ['beat_id', 'Rpeak_r', 'Rpeak_ca', 'Rturn_r', 'Rturn_ca', 'RR_ms']
    df_merged = df_annotations.merge(df_r[extra_r_cols], on='beat_id', how='left')
    df_merged = df_merged.sort_values('beat_id').reset_index(drop=True)

    # ── Vagus-Lookup (QRS-Achse + theta_QT bereits berechnet) ────────────
    vagus_cols = ['QRS_axis_x', 'QRS_axis_y', 'QRS_axis_z',
                  'T_axis_x', 'T_axis_y', 'T_axis_z',
                  'theta_QT_deg', 'T_mean_speed', 'QT_ms', 'QTc_Bazett']
    vagus_avail = [c for c in vagus_cols if c in df_vagus.columns]
    vagus_lut = df_vagus.set_index('beat_id')[vagus_avail]

    # ── P-Achsen-Lookup (aus df_p) ─────────────────────────────────────────
    p_axis_lut = df_p.set_index('beat_id')[['P_axis_x', 'P_axis_y', 'P_axis_z']] \
        if all(c in df_p.columns for c in ['P_axis_x', 'P_axis_y', 'P_axis_z']) \
        else None

    # ── Feature-Berechnung ────────────────────────────────────────────────
    records = []

    for _, row in df_merged.iterrows():
        bid = int(row['beat_id'])
        t_qon = safe(row, 't_Q_on')
        t_rpk = safe(row, 't_R_peak+')
        t_soff = safe(row, 't_S_off')
        t_qoff = safe(row, 't_Q_off')   # Ende QRS in manchen Annotationen
        rr = safe(row, 'RR_ms')
        resp = row.get('resp_phase_x', np.nan)

        # S_off bevorzugen, fallback Q_off
        t_qrs_end = t_soff if np.isfinite(t_soff) else t_qoff

        rec = {
            'beat_id': bid,
            'RR_ms': rr,
            'resp_phase': resp,
            't_R_peak': t_rpk,
        }

        # Vagus-Features übernehmen
        if bid in vagus_lut.index:
            for c in vagus_avail:
                rec[c] = vagus_lut.loc[bid, c]
        else:
            for c in vagus_avail:
                rec[c] = np.nan

        # ── Skalare Zeit-Features ─────────────────────────────────────────
        rec['QRS_dur_ms'] = (t_qrs_end - t_qon) * 1000 \
            if np.isfinite(t_qrs_end) and np.isfinite(t_qon) else np.nan
        rec['QRS_rise_ms'] = (t_rpk - t_qon) * 1000 \
            if np.isfinite(t_rpk) and np.isfinite(t_qon) else np.nan
        rec['QRS_fall_ms'] = (t_qrs_end - t_rpk) * 1000 \
            if np.isfinite(t_qrs_end) and np.isfinite(t_rpk) else np.nan

        # QRS-Symmetrie: rise/(rise+fall) → 0.5=sym, >0.5=rechtssteil
        if np.isfinite(rec['QRS_rise_ms']) and np.isfinite(rec['QRS_fall_ms']):
            total = rec['QRS_rise_ms'] + rec['QRS_fall_ms']
            rec['QRS_sym'] = rec['QRS_rise_ms'] / total if total > 0 else np.nan
        else:
            rec['QRS_sym'] = np.nan

        # ── Loop-Features ─────────────────────────────────────────────────
        QRS_area = QRS_rho = QRS_phi = QRS_dipol = QRS_asym = QRS_round = np.nan
        QRS_axis = np.array([np.nan, np.nan, np.nan])
        # Winkel-Features
        # TODO: theta_QRS_P (hier) und theta_P_QRS (in p_wave.py) werden
        # unabhängig über zwei separate SVD-Berechnungen auf demselben
        # (Q_on, QRS-Ende)-Fenster bestimmt statt eine gemeinsame Quelle zu
        # nutzen und können dadurch leicht divergieren, siehe Analysebericht.
        theta_QRS_P = theta_QRS_T_svd = np.nan
        # Maximale Momentangeschwindigkeit (R-Peak Steilheit)
        QRS_max_speed = QRS_mean_speed = np.nan
        # G_QRS: Aktivierungsgradient früh/spät (analog G_APD)
        G_QRS = np.nan

        if np.isfinite(t_qon) and np.isfinite(t_qrs_end):
            i0 = t2idx(t_qon, t_vcg)
            i1 = t2idx(t_qrs_end, t_vcg)
            seg = vcg[i0:i1 + 1]

            if len(seg) >= 4:
                QRS_area = loop_area_3d(seg)
                QRS_dipol = loop_dipol_norm(seg, dt)
                QRS_asym = loop_asym(seg)
                QRS_round = loop_roundness_fwhm(seg)

                S, Vt = loop_svd(seg)
                lam = S ** 2
                if lam.sum() > 0:
                    QRS_rho = lam[1] / lam[0]     # Rundheit: λ2/λ1
                    QRS_phi = lam[2] / lam.sum()   # Planarität: λ3/Σλ
                QRS_axis = Vt[0]                   # Hauptachsenvektor (SVD)

                # Momentangeschwindigkeit
                speed = np.linalg.norm(np.diff(seg, axis=0), axis=1)
                QRS_max_speed = float(speed.max())
                QRS_mean_speed = float(speed.mean())

                # G_QRS: Fläche früh (Depolarisation) vs. spät (Endpolarisation)
                # Trennpunkt: R-Peak
                if np.isfinite(t_rpk):
                    i_rpk = t2idx(t_rpk, t_vcg)
                    split = i_rpk - i0
                    split = max(1, min(split, len(seg) - 1))
                    a_early = loop_area_3d(seg[:split])
                    a_late = loop_area_3d(seg[split:])
                    G_QRS = a_late / (a_early + 1e-12) \
                        if (np.isfinite(a_early) and np.isfinite(a_late)) else np.nan

                # θ QRS-SVD ↔ P-Achse
                if p_axis_lut is not None and bid in p_axis_lut.index:
                    p_ax = p_axis_lut.loc[bid].values.astype(float)
                    if np.all(np.isfinite(p_ax)):
                        theta_QRS_P = angle_between(QRS_axis, p_ax)

                # θ QRS-SVD ↔ T-SVD (aus df_vagus Achsen)
                t_ax = np.array([rec.get('T_axis_x', np.nan),
                                 rec.get('T_axis_y', np.nan),
                                 rec.get('T_axis_z', np.nan)])
                if np.all(np.isfinite(t_ax)) and np.all(np.isfinite(QRS_axis)):
                    theta_QRS_T_svd = angle_between(QRS_axis, t_ax)

        rec.update({
            'QRS_area': QRS_area,
            'QRS_rho': QRS_rho,
            'QRS_phi': QRS_phi,
            'QRS_dipol_norm': QRS_dipol,
            'QRS_asym': QRS_asym,
            'QRS_round': QRS_round,
            'QRS_max_speed': QRS_max_speed,
            'QRS_mean_speed': QRS_mean_speed,
            'G_QRS': G_QRS,
            'QRS_axis_svd_x': QRS_axis[0],
            'QRS_axis_svd_y': QRS_axis[1],
            'QRS_axis_svd_z': QRS_axis[2],
            'theta_QRS_P': theta_QRS_P,
            'theta_QRS_T_svd': theta_QRS_T_svd,
        })

        records.append(rec)

    df_qrs = pd.DataFrame(records)

    # Surrogate einpflegen
    for col in ['Rpeak_ca', 'Rpeak_r']:
        if col not in df_qrs.columns:
            lut = dict(zip(df_r['beat_id'].astype(int), df_r[col]))
            df_qrs[col] = df_qrs['beat_id'].map(lut)
    for col in ['theta_QT_deg', 'T_mean_speed']:
        if col not in df_qrs.columns and col in df_vagus.columns:
            lut = dict(zip(df_vagus['beat_id'].astype(int), df_vagus[col]))
            df_qrs[col] = df_qrs['beat_id'].map(lut)

    new_cols = ['QRS_dur_ms', 'QRS_rise_ms', 'QRS_fall_ms', 'QRS_sym',
                'QRS_area', 'QRS_rho', 'QRS_phi', 'QRS_dipol_norm',
                'QRS_asym', 'QRS_round', 'QRS_max_speed', 'QRS_mean_speed',
                'G_QRS', 'theta_QRS_P', 'theta_QRS_T_svd']

    print(f"df_qrs: {len(df_qrs)} Beats, {df_qrs.shape[1]} Spalten")
    print(df_qrs[new_cols].describe().round(4))
    print(f"\nFehlende Werte:\n{df_qrs[new_cols].isna().sum()}")

    return df_qrs
