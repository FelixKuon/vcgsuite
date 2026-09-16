# ══════════════════════════════════════════════════════════════════════════
#  HRV_PREP — df_rr aus df_r konstruieren
#
#  Migriert aus HRV_prep.py ("BLOCK HRV_PREP").
#
#  Quelle:  df_r  ['beat_id', 't_R_peak+', 'Rpeak_r', 'Rpeak_ca',
#                  'Rturn_r', 'Rturn_ca', 'RR_ms']
#           (Ausgabe von `vcgsuite.beats.compute_beat_rotation()`)
#
#  Ausgabe: df_rr  ['Time', 'RR_interval', 'Rpeak_ca', 'Rpeak_r',
#                   'RR_derived']
#
#  Drei EDR-Kanäle (für Dropdown/Auswahl in hrv.analysis.extract_edr_signal):
#    Rpeak_ca   — Herzrotations-Surrogat (mechanisch)   → 0.05–0.5 Hz Ord.4
#    Rpeak_r    — Amplitude-Surrogat     (Vorlast)      → 0.05–0.5 Hz Ord.4
#    RR_derived — RR-Intervall selbst   (neuro-veget.)  → 0.15–0.5 Hz Ord.2
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import pandas as pd


def build_rr_dataframe(df_r: pd.DataFrame) -> pd.DataFrame:
    """
    Baut df_rr (RR-Tachogramm + EDR-Kandidatenkanäle) aus df_r auf.

    Parameters
    ----------
    df_r : pd.DataFrame  –  Ausgabe von `vcgsuite.beats.compute_beat_rotation()`.

    Returns
    -------
    df_rr : pd.DataFrame  –  Spalten Time, RR_interval, Rpeak_ca, Rpeak_r, RR_derived.
    """
    src = df_r.sort_values('beat_id').copy()

    # ── RR_ms: erste Zeile ist NaN (kein Vorgänger) ───────────────────────
    src['RR_ms'] = (src['RR_ms']
                    .interpolate(method='linear')
                    .ffill()
                    .bfill())

    # ── Kernkonstruktion ──────────────────────────────────────────────────
    df_rr = pd.DataFrame({
        'Time': src['t_R_peak+'].astype(float),         # Sekunden
        'RR_interval': src['RR_ms'].astype(float) / 1000.0,    # ms → s
        'Rpeak_ca': src['Rpeak_ca'].astype(float),
        'Rpeak_r': src['Rpeak_r'].astype(float),
    }).reset_index(drop=True)

    # RR_derived = Rohes RR-Signal (wird in extract_edr_signal gefiltert)
    df_rr['RR_derived'] = df_rr['RR_interval'].values

    # ── Validierung ───────────────────────────────────────────────────────
    dur = df_rr['Time'].max() - df_rr['Time'].min()
    print("df_rr bereit")
    print(f"   Beats:       {len(df_rr)}")
    print(f"   Dauer:       {dur:.1f} s  ({dur / 60:.1f} min)")
    print(f"   ∅ RR:        {df_rr['RR_interval'].mean() * 1000:.0f} ms  "
          f"(∅ HR = {60 / df_rr['RR_interval'].mean():.0f} bpm)")
    print(f"   ∅ Rpeak_ca:  {df_rr['Rpeak_ca'].mean():.4f}  "
          f"(NaN: {df_rr['Rpeak_ca'].isna().sum()})")
    print(f"   ∅ Rpeak_r:   {df_rr['Rpeak_r'].mean():.4f}  "
          f"(NaN: {df_rr['Rpeak_r'].isna().sum()})")
    print(f"   t-Bereich:   [{df_rr['Time'].min():.2f}, {df_rr['Time'].max():.2f}] s")

    return df_rr
