# ══════════════════════════════════════════════════════════════════════════
#  MERGE  –  P + QRS + T Loop-Features vereinen
#
#  Migriert aus merge_features.py ("BLOCK MERGE — df_full").
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd

P_COLS = ['beat_id',
    'theta_P_QRS', 'P_asym', 'P_phi', 'P_rho',
    'P_dipol_norm', 'P_area', 'P_dur_ms', 'PQ_ms',
]

QRS_COLS = ['beat_id',
    'QRS_area', 'QRS_dipol_norm', 'QRS_rho', 'QRS_phi',
    'QRS_asym', 'G_QRS', 'QRS_max_speed', 'QRS_mean_speed',
    'theta_QRS_T_svd', 'theta_QRS_P', 'QRS_dur_ms', 'QRS_sym',
]

T_COLS = ['beat_id',
    'T_area', 'T_dipol_norm', 'T_rho', 'T_phi',
    'T_asym_loop', 'T_round', 'theta_P_T', 'theta_QT_deg',
    'G_APD', 'T_mean_speed', 'T_asym', 'T_width_ms', 'QTc_Bazett',
    'T_rise_ms', 'T_fall_ms',
]

META_COLS = ['beat_id', 'RR_ms', 'resp_phase', 't_R_peak',
             'Rpeak_ca', 'Rpeak_r']

META_ONLY = {'beat_id', 'RR_ms', 'resp_phase', 't_R_peak', 'Rpeak_ca', 'Rpeak_r'}


def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Filtert `cols` auf die tatsächlich in `df` vorhandenen Spalten."""
    return [c for c in cols if c in df.columns]


def merge_loop_features(df_p: pd.DataFrame, df_qrs: pd.DataFrame,
                        df_t: pd.DataFrame, df_r: pd.DataFrame,
                        meta: list[str] | None = None) -> pd.DataFrame:
    """
    Vereint P-/QRS-/T-Loop-Features zu einem df_full (ein Beat pro Zeile).

    Parameters
    ----------
    df_p, df_qrs, df_t : pd.DataFrame
        Ausgabe von `compute_p_wave_features()`, `compute_qrs_features()`,
        `compute_t_wave_features()`.
    df_r : pd.DataFrame
        Ausgabe von `compute_beat_rotation()`. Dient als Fallback-Quelle für
        META_COLS, falls eine Meta-Spalte (z. B. durch abweichende
        Aufrufreihenfolge) unerwartet weder in df_qrs noch df_full vorhanden
        sein sollte — im Original wurden diese Spalten implizit als bereits
        in df_qrs gemergt vorausgesetzt.
    meta : list[str] | None
        Zusätzliche Spaltennamen, die (falls vorhanden) aus df_r ergänzt
        werden sollen, über die Standard-META_COLS hinaus.

    Returns
    -------
    df_full : pd.DataFrame
    """
    df_full = (df_p[safe_cols(df_p, P_COLS)]
               .merge(df_qrs[safe_cols(df_qrs, QRS_COLS)], on='beat_id', how='outer')
               .merge(df_t[safe_cols(df_t, T_COLS)], on='beat_id', how='outer'))

    # Meta-Spalten: bevorzugt aus df_qrs (hat i. d. R. alle Surrogate bereits
    # gemergt), Fallback auf df_r falls dort fehlend.
    meta_src = df_qrs[safe_cols(df_qrs, META_COLS)]
    for col in META_COLS[1:]:   # beat_id schon drin
        if col in meta_src.columns and col not in df_full.columns:
            df_full = df_full.merge(meta_src[['beat_id', col]], on='beat_id', how='left')
        elif col not in df_full.columns and col in df_r.columns:
            df_full = df_full.merge(df_r[['beat_id', col]], on='beat_id', how='left')

    if meta:
        for col in meta:
            if col not in df_full.columns and col in df_r.columns:
                df_full = df_full.merge(df_r[['beat_id', col]], on='beat_id', how='left')

    df_full = df_full.sort_values('beat_id').reset_index(drop=True)

    # Alle numerischen Feature-Spalten (ohne Meta)
    feat_cols = [c for c in df_full.columns
                 if c not in META_ONLY and df_full[c].dtype in [np.float64, np.float32, float]]

    print(f"df_full: {len(df_full)} Beats × {len(feat_cols)} Features")
    print(f"Fehlende Werte (%):\n"
          f"{(df_full[feat_cols].isna().mean() * 100).round(1).sort_values(ascending=False).head(10)}")

    return df_full
