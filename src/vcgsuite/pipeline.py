# ══════════════════════════════════════════════════════════════════════════
#  PIPELINE  –  Ende-zu-Ende Orchestrator
#  EASI oder 12-Kanal ECG  →  df_analysis
#
#  Ersetzt Wrapper.py.
#
#  Hinweis Migration – behobene Code-Duplikation:
#  Das Original (`Wrapper.py::load_and_process`) implementierte das Laden
#  der Rohdatei (`pd.read_csv` + Zeitfenster-Schnitt) SELBST erneut, statt
#  die bereits vorhandenen `vcgsuite.io.easi.load_easi()` /
#  `vcgsuite.io.ecg12.load_ecg12()` zu nutzen — identische Lade-/
#  Fenster-Logik existierte damit zweifach. `load_and_process()` unten
#  ruft stattdessen `load_easi()`/`load_ecg12()` direkt auf.
#
#  Hinweis Migration – Moduswert:
#  Das Original nutzte `mode="ecg12"`. Diese Migration verwendet
#  `mode="12ch"` (siehe Bauauftrag), akzeptiert aber case-insensitiv nur
#  die exakten Werte "easi"/"12ch".
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Callable, Optional

import os
import numpy as np
import pandas as pd

from .config import FS, LEAD_ORDER
from .utils import parse_subject_id
from .io.easi import load_easi
from .io.ecg12 import load_ecg12
from .preprocessing.filter import filter_pipeline
from .transform.easi2frank import easi_to_frank_xyz
from .transform.ecg12_to_frank import ecg12_to_frank_xyz


def load_and_process(
    # ── Pflichtparameter
    filepath: str,
    mode: str,                                   # "easi" | "12ch"

    # ── Zeitfenster
    start_sec: float = 0,
    duration: Optional[float] = None,

    # ── Signal
    fs: int = FS,
    sep: str = ";",

    # ── EASI-spezifisch
    col_i: str = "R",
    col_e: str = "M",
    col_s: str = "L",

    # ── 12-Kanal-spezifisch
    lead_order: list = LEAD_ORDER,
    transform: str = "IDT",                       # "IDT"|"KORS"|"QLSV"|"PLSV"

    # ── Filterparameter
    use_zapline: bool = True,
    fline: float = 50.0,
    nremove_zap: int = 1,
    hp_cutoff: float = 1.5,
    hp_taps: int = 201,
    lp_cutoff: float = 37.5,
    lp_taps: int = 21,

    # ── Proband-ID-Extraktion
    subject_id_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Ende-zu-Ende Pipeline für die VCG-Vorverarbeitung: Rohdatei laden
    (`vcgsuite.io.easi.load_easi()` bzw. `vcgsuite.io.ecg12.load_ecg12()`)
    → filtern (`vcgsuite.preprocessing.filter.filter_pipeline()`:
    ZapLine → FIR-Hochpass → FIR-Tiefpass) → VCG-Transformation
    (`vcgsuite.transform.easi2frank.easi_to_frank_xyz()` bzw.
    `vcgsuite.transform.ecg12_to_frank.ecg12_to_frank_xyz()`) →
    zusammengesetztes `df_analysis` mit gesetzten `attrs` (u. a. `fs`, das
    von den `vcgsuite.features.loop`-Funktionen für die Ableitung der
    tatsächlichen Abtastrate genutzt wird statt eines hartcodierten
    Default-Werts).

    Parameters
    ----------
    filepath : str
        Pfad zur Datendatei auf dem Laptop.
    mode : str
        "easi" – EASI 3-Elektroden-Aufnahme
        "12ch" – Standard 12-Kanal ECG
    start_sec : float
        Schnittstart in Sekunden (0 = von Anfang).
    duration : float | None
        Analysedauer in Sekunden ab start_sec.
        None = bis zum Ende der Aufnahme.
    fs : int
        Abtastrate in Hz.
    sep : str
        CSV-Trennzeichen.
    col_i / col_e / col_s : str
        Spaltennamen der EASI-Elektroden I, E, S.
    lead_order : list[str]
        Reihenfolge der 8 unabhängigen Leads für 12-Kanal.
    transform : str
        Inversmatrix für 12-Kanal → VCG: "IDT" | "KORS" | "QLSV" | "PLSV"
    use_zapline : bool
        True (Default) = ZapLine läuft vor dem FIR-Bandpass. False =
        ZapLine wird übersprungen, es läuft nur der FIR-Bandpass.
    fline : float
        Netzfrequenz für ZapLine in Hz (nur relevant wenn use_zapline=True).
    nremove_zap : int
        DSS-Komponenten für ZapLine (nur relevant wenn use_zapline=True).
    hp_cutoff / lp_cutoff : float
        Grenzfrequenzen FIR-Hochpass / Tiefpass in Hz.
    hp_taps / lp_taps : int
        Filter-Taps für FIR-Hochpass / Tiefpass.
    subject_id_fn : callable | None
        Optionale Funktion  filepath → (proband_id, subject_id)
        um die Standard-Regex-Logik (`parse_subject_id`) zu überschreiben.
        Beispiel: lambda p: ("exp1", "S_exp1")
        None = Standard parse_subject_id() wird verwendet (das passiert
        bereits intern in load_easi()/load_ecg12(); subject_id_fn
        überschreibt hier nur das Ergebnis).

    Returns
    -------
    df_analysis : pd.DataFrame
        Spalten (EASI):   Time, X, Y, Z, V_IS, V_ES, V_AS
        Spalten (12-CH):  Time, X, Y, Z, I, II, V1, V2, V3, V4, V5, V6
        Metadaten:        df_analysis.attrs["proband_id"]
                          df_analysis.attrs["subject_id"]
                          df_analysis.attrs["mode"]
                          df_analysis.attrs["fs"]
                          df_analysis.attrs["transform"]  (nur 12ch)
    """
    # ── Validierung
    mode = mode.lower()
    if mode not in ("easi", "12ch"):
        raise ValueError(f"mode muss 'easi' oder '12ch' sein, nicht '{mode}'.")

    print(f"\n{'=' * 60}")
    print(f"  Pipeline : {mode.upper()}  |  {os.path.basename(filepath)}")
    print(f"{'=' * 60}")

    # ── Proband-ID (Standard: bereits in load_easi()/load_ecg12() via
    #    parse_subject_id() bestimmt; subject_id_fn überschreibt bei Bedarf)
    if subject_id_fn is not None:
        proband_id, subject_id = subject_id_fn(filepath)
    else:
        proband_id, subject_id = parse_subject_id(filepath)
    print(f"  Proband  : {proband_id}  →  {subject_id}")

    # ─────────────────────────────────────────────
    #  1. LADEN  (nutzt vcgsuite.io statt eigenem pd.read_csv)
    # ─────────────────────────────────────────────
    if mode == "easi":
        loaded = load_easi(filepath, sep=sep, col_i=col_i, col_e=col_e, col_s=col_s,
                           start_sec=start_sec, duration=duration, fs=fs)
        raw_signals = [loaded["I"], loaded["E"], loaded["S"]]
    else:
        loaded = load_ecg12(filepath, sep=sep, start_sec=start_sec, duration=duration,
                            lead_order=lead_order, fs=fs)
        raw_signals = [loaded["leads"][lead] for lead in lead_order]

    t = loaded["t"]

    # ─────────────────────────────────────────────
    #  2. FILTERN  –  ZapLine → FIR HP → FIR LP
    # ─────────────────────────────────────────────
    if use_zapline:
        print(f"  [Filter] ZapLine {fline} Hz  →  "
              f"FIR HP {hp_cutoff} Hz / LP {lp_cutoff} Hz")
    else:
        print(f"  [Filter] ZapLine übersprungen (use_zapline=False)  →  "
              f"FIR HP {hp_cutoff} Hz / LP {lp_cutoff} Hz")

    signals = filter_pipeline(
        raw_signals,
        fs=fs,
        fline=fline,
        nremove_zap=nremove_zap,
        hp_cutoff=hp_cutoff,
        hp_taps=hp_taps,
        lp_cutoff=lp_cutoff,
        lp_taps=lp_taps,
        use_zapline=use_zapline,
    )

    # ─────────────────────────────────────────────
    #  3. VCG-TRANSFORMATION
    # ─────────────────────────────────────────────
    if mode == "easi":
        v_is, v_es, v_as = signals[0], signals[1], signals[2]
        X, Y, Z = easi_to_frank_xyz(v_is, v_es, v_as)
        print("  [VCG]    EASI → Frank XYZ  (W · T)")
    else:
        ecg_filt_dict = {lead: signals[i] for i, lead in enumerate(lead_order)}
        X, Y, Z = ecg12_to_frank_xyz(ecg_filt_dict, method=transform,
                                      lead_order=lead_order)

    # ─────────────────────────────────────────────
    #  4. df_analysis ZUSAMMENSTELLEN
    # ─────────────────────────────────────────────
    if mode == "easi":
        signal_cols = {"V_IS": signals[0], "V_ES": signals[1], "V_AS": signals[2]}
    else:
        signal_cols = {lead: signals[i] for i, lead in enumerate(lead_order)}

    df_analysis = pd.DataFrame({"Time": t, "X": X, "Y": Y, "Z": Z, **signal_cols})

    df_analysis.attrs["proband_id"] = proband_id
    df_analysis.attrs["subject_id"] = subject_id
    df_analysis.attrs["mode"] = mode
    df_analysis.attrs["fs"] = fs
    if mode == "12ch":
        df_analysis.attrs["transform"] = transform

    print(f"\n  df_analysis: {df_analysis.shape}  |  Proband: {subject_id}")
    print(f"{'=' * 60}\n")

    return df_analysis
