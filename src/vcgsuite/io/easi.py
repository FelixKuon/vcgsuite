# ============================================================
#  DATEN LADEN  –  EASI
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FS
from ..utils import parse_subject_id


def load_easi(filepath: str, sep: str = ";",
              col_i: str = "R", col_e: str = "M", col_s: str = "L",
              start_sec: float = 0, duration: float | None = None,
              fs: int = FS) -> dict:
    """
    Lädt eine EASI-EKG-Datei und schneidet das gewünschte Zeitfenster aus.

    Parameters
    ----------
    start_sec : float
        Startzeit in Sekunden. Samples vor diesem Zeitpunkt werden
        verworfen, z. B. um Artefakte am Aufnahme-Anfang zu entfernen.
    duration : float | None
        Analysedauer in Sekunden ab start_sec.
        None  →  lädt alles ab start_sec bis zum Ende der Aufnahme.

    Returns
    -------
    dict mit keys:
        proband_id, subject_id, fs, t,
        I, E, S   (je np.ndarray),
        df        (pd.DataFrame, gefenstert)
    """
    proband_id, subject_id = parse_subject_id(filepath)
    df_raw = pd.read_csv(filepath, sep=sep)

    start_idx = int(start_sec * fs)
    end_idx   = start_idx + int(duration * fs) if duration is not None else len(df_raw)
    df_win    = df_raw.iloc[start_idx:end_idx].reset_index(drop=True)
    t         = np.arange(len(df_win)) / fs

    print(f"EASI  |  Proband: {proband_id}  |  Fenster: {start_sec:.1f} s"
          f"  →  {start_sec + len(df_win)/fs:.1f} s"
          f"  ({len(df_win)} Samples = {len(df_win)/fs:.1f} s)")
    print(f"       Spalten: {list(df_raw.columns)}")

    return {
        "proband_id": proband_id,
        "subject_id": subject_id,
        "fs":  fs,
        "t":   t,
        "I":   df_win[col_i].astype(float).to_numpy(),
        "E":   df_win[col_e].astype(float).to_numpy(),
        "S":   df_win[col_s].astype(float).to_numpy(),
        "df":  df_win,
    }
