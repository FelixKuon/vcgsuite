# ============================================================
#  DATEN LADEN  –  12-KANAL ECG
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FS, LEAD_ORDER
from ..utils import parse_subject_id


def load_ecg12(filepath: str, sep: str = ";",
               start_sec: float = 0, duration: float | None = None,
               lead_order: list[str] = LEAD_ORDER,
               fs: int = FS) -> dict:
    """
    Lädt eine 12-Kanal-EKG-Datei und schneidet das gewünschte Zeitfenster aus.

    Parameters
    ----------
    start_sec : float
        Startzeit in Sekunden.
    duration : float | None
        Analysedauer in Sekunden ab start_sec.
        None  →  lädt alles ab start_sec bis zum Ende der Aufnahme.

    Returns
    -------
    dict mit keys:
        proband_id, subject_id, fs, t,
        leads   (dict[str, np.ndarray]),
        df      (pd.DataFrame, gefenstert)
    """
    proband_id, subject_id = parse_subject_id(filepath)
    df_raw = pd.read_csv(filepath, sep=sep)

    start_idx = int(start_sec * fs)
    end_idx   = start_idx + int(duration * fs) if duration is not None else len(df_raw)
    df_win    = df_raw.iloc[start_idx:end_idx].reset_index(drop=True)
    t         = np.arange(len(df_win)) / fs

    leads = {lead: df_win[lead].astype(float).to_numpy() for lead in lead_order}

    print(f"12-CH |  Proband: {proband_id}  |  Fenster: {start_sec:.1f} s"
          f"  →  {start_sec + len(df_win)/fs:.1f} s"
          f"  ({len(df_win)} Samples = {len(df_win)/fs:.1f} s)")
    print(f"       Spalten gesamt: {list(df_raw.columns)}")
    print(f"       Geladene Leads: {lead_order}")

    return {
        "proband_id": proband_id,
        "subject_id": subject_id,
        "fs":    fs,
        "t":     t,
        "leads": leads,
        "df":    df_win,
    }
