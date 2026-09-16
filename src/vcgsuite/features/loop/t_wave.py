# ══════════════════════════════════════════════════════════════════════════
#  T-LOOP FEATURES
#
#  Migriert aus T-loop_features.py ("BLOCK A_T — T-Loop Features").
#
#  Das Original war als "Standalone"-Block geschrieben, der df_vagus IMMER
#  inline neu erzeugte (auch wenn qrs_complex.py bereits ein df_vagus
#  benötigt hätte). Das wird hier in zwei Funktionen aufgeteilt, die den
#  Datenfluss explizit machen:
#
#    build_vagus_features()      – entspricht dem inline-Block
#                                   ("_compute_t_wave_params" + Aufruf-
#                                   Schleife), erzeugt df_vagus
#                                   (T-/QRS-Achsen, QT/QTc, G_APD, ...).
#    compute_t_wave_features()   – entspricht dem "Rest von Block A_T"
#                                   (T-Loop-Geometrie via SVD, θ_P_T, ...),
#                                   nutzt df_vagus als Lookup.
#
#  Empfohlene Aufrufreihenfolge (siehe pipeline.py / features/loop/__init__):
#    df_p     = compute_p_wave_features(...)
#    df_vagus = build_vagus_features(...)
#    df_qrs   = compute_qrs_features(..., df_vagus=df_vagus, df_p=df_p, ...)
#    df_t     = compute_t_wave_features(..., df_p=df_p, df_vagus=df_vagus)
#    df_full  = merge_loop_features(df_p, df_qrs, df_t, df_r)
#
#  Die 8 gemeinsamen Loop-Helper leben in `._shared`. `fs`/`dt` werden nicht
#  mehr hartcodiert (siehe `_shared.resolve_fs`).
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import (
    safe, t2idx, loop_area_3d, loop_svd, loop_dipol_norm,
    loop_asym, loop_roundness_fwhm, angle_between, resolve_fs,
)


def _slice(df_s: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    mask = (df_s['Time'] >= t0) & (df_s['Time'] <= t1)
    return df_s[mask].reset_index(drop=True)


def _compute_t_wave_params(row, df_sig: pd.DataFrame) -> dict:
    """Berechnet T-/QRS-Achsen, QT(c), G_APD, T-Asymmetrie etc. für einen Beat."""
    def g(name):
        return float(row.get(f't_{name}', np.nan))

    t_ton = g('T_on')
    t_toff = g('T_off')
    t_t1 = g('T_turn1')
    t_qon = g('Q_on')
    t_soff = g('S_off')
    t_rpk = g('R_peak+')

    rec = {'beat_id': int(row['beat_id']), 't_R_peak+': t_rpk}
    if any(pd.isna(v) for v in [t_ton, t_toff]):
        return rec

    t_df = _slice(df_sig, t_ton, t_toff)
    qrs_df = _slice(df_sig, t_qon, t_soff) \
        if not any(pd.isna(v) for v in [t_qon, t_soff]) else None

    rec['T_width_ms'] = round((t_toff - t_ton) * 1000, 2)
    rec['QT_ms'] = round((t_toff - t_qon) * 1000, 2) if not pd.isna(t_qon) else np.nan

    # T-Achse
    e_T_hat = None
    if len(t_df) >= 3:
        tv = t_df['Time'].to_numpy()
        e_T = np.array([np.trapz(t_df['X'], tv),
                        np.trapz(t_df['Y'], tv),
                        np.trapz(t_df['Z'], tv)])
        norm_T = np.linalg.norm(e_T)
        e_T_hat = e_T / (norm_T + 1e-12)
        rec.update({'T_axis_x': float(e_T_hat[0]),
                    'T_axis_y': float(e_T_hat[1]),
                    'T_axis_z': float(e_T_hat[2]),
                    'T_axis_magnitude': float(norm_T)})
    else:
        rec.update({'T_axis_x': np.nan, 'T_axis_y': np.nan,
                    'T_axis_z': np.nan, 'T_axis_magnitude': np.nan})

    # QRS-Achse + QRS-T-Winkel
    e_QRS_hat = None
    if qrs_df is not None and len(qrs_df) >= 3:
        tq = qrs_df['Time'].to_numpy()
        e_QRS = np.array([np.trapz(qrs_df['X'], tq),
                          np.trapz(qrs_df['Y'], tq),
                          np.trapz(qrs_df['Z'], tq)])
        norm_Q = np.linalg.norm(e_QRS)
        e_QRS_hat = e_QRS / (norm_Q + 1e-12)
        rec.update({'QRS_axis_x': float(e_QRS_hat[0]),
                    'QRS_axis_y': float(e_QRS_hat[1]),
                    'QRS_axis_z': float(e_QRS_hat[2])})
    else:
        rec.update({'QRS_axis_x': np.nan, 'QRS_axis_y': np.nan, 'QRS_axis_z': np.nan})

    if e_T_hat is not None and e_QRS_hat is not None:
        dot = float(np.clip(np.dot(e_T_hat, e_QRS_hat), -1.0, 1.0))
        rec['theta_QT_deg'] = float(np.degrees(np.arccos(dot)))
    else:
        rec['theta_QT_deg'] = np.nan

    # Geschwindigkeit + G_APD + T_asym
    if len(t_df) >= 4:
        xyz = t_df[['X', 'Y', 'Z']].to_numpy()
        t_arr = t_df['Time'].to_numpy()
        dt_arr = np.diff(t_arr)
        speed = np.linalg.norm(
            np.diff(xyz, axis=0) / (dt_arr[:, None] + 1e-12), axis=1)
        mid = t_t1 if not pd.isna(t_t1) else (t_ton + t_toff) / 2.0
        t_mid = t_arr[:-1]
        early = speed[t_mid <= mid]
        dt_e = dt_arr[t_mid <= mid]
        late = speed[t_mid > mid]
        dt_l = dt_arr[t_mid > mid]
        if len(early) and len(late):
            G_e = float(np.sum(early * dt_e))
            G_l = float(np.sum(late * dt_l))
            rec.update({'G_APD': G_l / (G_e + 1e-12),
                        'G_APD_early': G_e, 'G_APD_late': G_l})
        else:
            rec.update({'G_APD': np.nan, 'G_APD_early': np.nan, 'G_APD_late': np.nan})
        rec['T_mean_speed'] = float(np.mean(speed))
        rec['T_max_speed'] = float(np.max(speed))
        if not pd.isna(t_t1):
            rising = speed[t_mid <= t_t1]
            falling = speed[t_mid > t_t1]
            rec['T_asym'] = float(np.mean(falling)) / (float(np.mean(rising)) + 1e-12) \
                if len(rising) and len(falling) else np.nan
        else:
            rec['T_asym'] = np.nan
    else:
        rec.update({'G_APD': np.nan, 'G_APD_early': np.nan, 'G_APD_late': np.nan,
                    'T_mean_speed': np.nan, 'T_max_speed': np.nan, 'T_asym': np.nan})

    # Amplitude
    if 'r' in t_df.columns and len(t_df):
        rec['T_r_max'] = float(t_df['r'].max())
        rec['T_r_mean'] = float(t_df['r'].mean())
    else:
        rec['T_r_max'] = rec['T_r_mean'] = np.nan

    # Timing relativ zu R_peak+
    if not pd.isna(t_rpk):
        rec['T_on_rel_ms'] = round((t_ton - t_rpk) * 1000, 2)
        rec['T_off_rel_ms'] = round((t_toff - t_rpk) * 1000, 2)
    else:
        rec['T_on_rel_ms'] = rec['T_off_rel_ms'] = np.nan

    rec['T_off_minus_Soff_ms'] = \
        round((t_toff - t_soff) * 1000, 2) if not pd.isna(t_soff) else np.nan

    return rec


def build_vagus_features(df_analysis: pd.DataFrame,
                         df_annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Erzeugt df_vagus: T-/QRS-Achsen, QT(c)-Intervalle, G_APD, T-Asymmetrie
    etc. pro Beat. Wird sowohl von `compute_qrs_features()` (Vagus-Lookup)
    als auch von `compute_t_wave_features()` benötigt.

    Migriert aus dem inline-Block in T-loop_features.py
    ("df_vagus inline erzeugen (immer, ohne externen Block)").

    Parameters
    ----------
    df_analysis    : pd.DataFrame  –  Spalten Time, X, Y, Z (+ optional 'r').
    df_annotations : pd.DataFrame  –  Ausgabe von `annotate_all_beats()`.

    Returns
    -------
    df_vagus : pd.DataFrame
    """
    rows = []
    for _, row in df_annotations.iterrows():
        try:
            rows.append(_compute_t_wave_params(row, df_analysis))
        except Exception as e:
            rows.append({'beat_id': int(row.get('beat_id', -1)), 'error': str(e)})

    df_vagus = pd.DataFrame(rows)
    rr = np.diff(df_vagus['t_R_peak+'].to_numpy(dtype=float), prepend=np.nan) * 1000
    df_vagus['RR_ms'] = rr
    df_vagus['delta_APD_rel'] = df_vagus['T_off_minus_Soff_ms'] / (df_vagus['RR_ms'] + 1e-6)
    df_vagus['QTc_Bazett'] = df_vagus['QT_ms'] / np.sqrt(df_vagus['RR_ms'] / 1000.0 + 1e-6)
    print(f"[T-Wave] df_vagus erzeugt: {df_vagus.shape}")
    return df_vagus


def compute_t_wave_features(df_analysis: pd.DataFrame,
                            df_annotations: pd.DataFrame,
                            df_r: pd.DataFrame,
                            df_p: pd.DataFrame,
                            df_vagus: pd.DataFrame | None = None,
                            fs: float | None = None) -> pd.DataFrame:
    """
    Berechnet T-Loop-Features (Geometrie der T-Welle) pro Beat.

    Parameters
    ----------
    df_analysis    : pd.DataFrame  –  Spalten Time, X, Y, Z.
    df_annotations : pd.DataFrame  –  Ausgabe von `annotate_all_beats()`.
    df_r           : pd.DataFrame  –  Ausgabe von `compute_beat_rotation()`.
    df_p           : pd.DataFrame  –  Ausgabe von `compute_p_wave_features()`
                                      (für θ_P_T).
    df_vagus       : pd.DataFrame | None  –  Ausgabe von
                                      `build_vagus_features()`. None → wird
                                      intern automatisch berechnet.
    fs             : float | None  –  Abtastrate, siehe `_shared.resolve_fs`.

    Returns
    -------
    df_t : pd.DataFrame
    """
    fs_val = resolve_fs(df_analysis, fs)
    dt = 1.0 / fs_val

    vcg = df_analysis[['X', 'Y', 'Z']].to_numpy()
    t_vcg = df_analysis['Time'].to_numpy()

    if df_vagus is None:
        df_vagus = build_vagus_features(df_analysis, df_annotations)

    # ── Merge: Zeitmarker aus df_annotations, Surrogate aus df_r ──────────
    extra_r_cols = ['beat_id', 'Rpeak_r', 'Rpeak_ca', 'Rturn_r', 'Rturn_ca', 'RR_ms']
    df_merged = df_annotations.merge(df_r[extra_r_cols], on='beat_id', how='left')
    df_merged = df_merged.sort_values('beat_id').reset_index(drop=True)

    # ── Vorhandene df_vagus-Features als Lookup ────────────────────────────
    vagus_cols = ['T_width_ms', 'QT_ms', 'QTc_Bazett', 'theta_QT_deg',
                  'G_APD', 'G_APD_early', 'G_APD_late',
                  'T_mean_speed', 'T_max_speed', 'T_asym',
                  'T_r_max', 'T_r_mean',
                  'T_on_rel_ms', 'T_off_rel_ms',
                  'QRS_axis_x', 'QRS_axis_y', 'QRS_axis_z',
                  'T_axis_x', 'T_axis_y', 'T_axis_z', 'T_axis_magnitude']
    vagus_avail = [c for c in vagus_cols if c in df_vagus.columns]
    vagus_lut = df_vagus.set_index('beat_id')[vagus_avail]

    # ── Feature-Berechnung pro Beat ───────────────────────────────────────
    records = []

    for _, row in df_merged.iterrows():
        bid = int(row['beat_id'])
        t_ton = safe(row, 't_T_on')
        t_toff = safe(row, 't_T_off')
        t_t1 = safe(row, 't_T_turn1')   # Wendepunkt früh (T-Peak-Proxy)
        t_qon = safe(row, 't_Q_on')
        t_soff = safe(row, 't_S_off')
        t_rpk = safe(row, 't_R_peak+')
        rr = safe(row, 'RR_ms')
        resp = row.get('resp_phase_x', np.nan)

        rec = {
            'beat_id': bid,
            'RR_ms': rr,
            'resp_phase': resp,
            't_R_peak': t_rpk,
        }

        # Vorhandene df_vagus-Features übernehmen
        if bid in vagus_lut.index:
            for c in vagus_avail:
                rec[c] = vagus_lut.loc[bid, c]
        else:
            for c in vagus_avail:
                rec[c] = np.nan

        # ── Loop-Extraktion ─────────────────────────────────────────────
        T_area = T_rho = T_phi = T_dipol = T_asym_loop = T_round = np.nan
        T_axis = np.array([np.nan, np.nan, np.nan])
        theta_P_T = np.nan

        if np.isfinite(t_ton) and np.isfinite(t_toff):
            i0 = t2idx(t_ton, t_vcg)
            i1 = t2idx(t_toff, t_vcg)
            seg = vcg[i0:i1 + 1]

            if len(seg) >= 4:
                T_area = loop_area_3d(seg)
                T_dipol = loop_dipol_norm(seg, dt)
                T_asym_loop = loop_asym(seg)
                T_round = loop_roundness_fwhm(seg)

                S, Vt = loop_svd(seg)
                lam = S ** 2
                if lam.sum() > 0:
                    T_rho = lam[1] / lam[0]      # Rundheit: λ2/λ1
                    T_phi = lam[2] / lam.sum()    # Planarität: λ3/Σλ
                T_axis = Vt[0]                     # Hauptachsenvektor

                # θ P-Achse ↔ T-Achse (P-Dipol aus df_p falls vorhanden)
                if bid in df_p['beat_id'].values:
                    p_row = df_p[df_p['beat_id'] == bid].iloc[0]
                    p_axis = np.array([p_row.get('P_axis_x', np.nan),
                                       p_row.get('P_axis_y', np.nan),
                                       p_row.get('P_axis_z', np.nan)])
                    if np.all(np.isfinite(p_axis)):
                        theta_P_T = angle_between(p_axis, T_axis)

        # ── Normalisierungen ──────────────────────────────────────────────
        T_rise_ms = (t_t1 - t_ton) * 1000 if np.isfinite(t_t1) and np.isfinite(t_ton) else np.nan
        T_fall_ms = (t_toff - t_t1) * 1000 if np.isfinite(t_t1) and np.isfinite(t_toff) else np.nan

        rec.update({
            'T_area': T_area,
            'T_rho': T_rho,
            'T_phi': T_phi,
            'T_dipol_norm': T_dipol,
            'T_asym_loop': T_asym_loop,
            'T_round': T_round,
            'T_axis_svd_x': T_axis[0],
            'T_axis_svd_y': T_axis[1],
            'T_axis_svd_z': T_axis[2],
            'theta_P_T': theta_P_T,
            'T_rise_ms': T_rise_ms,
            'T_fall_ms': T_fall_ms,
        })

        records.append(rec)

    df_t = pd.DataFrame(records)

    # ── Surrogate einpflegen ──────────────────────────────────────────────
    for col in ['Rpeak_ca', 'Rpeak_r']:
        if col not in df_t.columns:
            lut = dict(zip(df_r['beat_id'].astype(int), df_r[col]))
            df_t[col] = df_t['beat_id'].map(lut)

    new_loop_cols = ['T_area', 'T_rho', 'T_phi', 'T_dipol_norm',
                      'T_asym_loop', 'T_round', 'theta_P_T',
                      'T_rise_ms', 'T_fall_ms']
    scalar_cols = ['T_width_ms', 'QT_ms', 'QTc_Bazett', 'theta_QT_deg',
                   'G_APD', 'T_mean_speed', 'T_asym']

    print(f"df_t: {len(df_t)} Beats, {df_t.shape[1]} Spalten")
    print("\n── Neue Loop-Features ────────────────────────────────────────")
    print(df_t[new_loop_cols].describe().round(4))
    print("\n── Vorhandene Skalare ────────────────────────────────────────")
    print(df_t[scalar_cols].describe().round(3))
    print(f"\nFehlende Werte:\n{df_t[new_loop_cols + scalar_cols].isna().sum()}")

    return df_t
