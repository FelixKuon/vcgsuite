"""
Gemeinsame Hilfsfunktionen fuer die LUDB-Notebooks.

Wortgleiche Kopie der in ludb_validation.ipynb (Schritt 1-4) entwickelten
und dort bereits vom Nutzer laufen gelassenen/validierten Funktionen.
ludb_validation.ipynb selbst wird bewusst NICHT auf dieses Modul
umgestellt, um am bereits geprueften Notebook nichts nachtraeglich zu
riskieren. Neue Notebooks (ab ludb_tuning.ipynb) importieren von hier,
statt die Funktionen erneut zu duplizieren -- das ist der "aufgeraeumtere"
Teil, um den es bei den neuen Notebooks gehen soll.

Erwartet, wie ludb_validation.ipynb, dass der Jupyter-Kernel im
notebooks/-Ordner laeuft (Standard bei `jupyter lab notebooks/`).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

import vcgsuite as ecg

LUDB_DIR = Path("../../lobachevsky-university-electrocardiography-database-1.0.1")
DATA_DIR = LUDB_DIR / "data"

LEAD_ORDER_8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]  # == vcgsuite.config.LEAD_ORDER

KEY_META = {
    "P_on": ("P", "on"), "P_peak": ("P", "peak"), "P_off": ("P", "off"),
    "QRS_on": ("QRS", "on"), "R_peak": ("QRS", "peak"), "QRS_off": ("QRS", "off"),
    "T_on": ("T", "on"), "T_peak": ("T", "peak"), "T_off": ("T", "off"),
}
GROUP_COLOR = {"P": "#f4d35e", "QRS": "#ee6c4d", "T": "#5fa8d3"}
ROLE_SYMBOL = {"on": "triangle-left", "peak": "circle", "off": "triangle-right"}

TOLERANCE_S = 0.075  # 150 ms Fenster => +/- 75 ms um jede Annotation (ANSI/AAMI-EC57)

CSE_2SIGMA_MS = {  # Tabelle II, Emrich et al. -- nur fuer Wellengrenzen definiert, nicht fuer Peaks
    "P_on": 10.2, "P_peak": np.nan, "P_off": 12.7,
    "QRS_on": 6.5, "R_peak": np.nan, "QRS_off": 11.6,
    "T_on": np.nan, "T_peak": np.nan, "T_off": 30.6,
}

# Suchradius je Wellentyp fuer per_lead_times_for_beat()/cluster_beats_from_rpeaks()
# (aus ludb_lead_spread_analysis.ipynb uebernommen -- dort lokal definiert und
# bereits gelaufen/validiert; hier als wortgleiche Kopie fuer neue Notebooks,
# aus demselben Grund wie beim Rest dieses Moduls: das validierte Notebook
# selbst wird nicht nachtraeglich auf diese Datei umgestellt).
SEARCH_RADIUS_MS = {
    "P_on": 300, "P_peak": 300, "P_off": 300,
    "QRS_on": 150, "R_peak": 100, "QRS_off": 150,
    "T_on": 450, "T_peak": 450, "T_off": 500,
}


def parse_ludb_annotations(record_id: int, lead: str, data_dir: Path = DATA_DIR) -> dict[str, list[int]]:
    """Liest die WFDB-Annotation eines LUDB-Records/Leads und ordnet die
    Symbole den neun Fiducial-Point-Typen zu (WFDB-Konvention: '(' Onset,
    ')' Offset, 'p'/'N'/'t' Peak fuer P-/QRS-/T-Welle). Siehe
    ludb_validation.ipynb Schritt 2 fuer Herleitung/Validierung (Parser
    stimmt exakt mit den LUDB-README-Gesamtzahlen ueberein).
    """
    ann = wfdb.rdann(str(data_dir / str(record_id)), extension=lead)
    symbols, samples = ann.symbol, ann.sample

    keys = ["P_on", "P_peak", "P_off", "QRS_on", "R_peak", "QRS_off", "T_on", "T_peak", "T_off"]
    result: dict[str, list[int]] = {k: [] for k in keys}
    wave_of = {"p": "P", "N": "QRS", "t": "T"}
    peak_key_of = {"P": "P_peak", "QRS": "R_peak", "T": "T_peak"}
    on_key_of = {"P": "P_on", "QRS": "QRS_on", "T": "T_on"}
    off_key_of = {"P": "P_off", "QRS": "QRS_off", "T": "T_off"}

    for i, sym in enumerate(symbols):
        if sym not in wave_of:
            continue
        wave = wave_of[sym]
        result[peak_key_of[wave]].append(int(samples[i]))
        if i - 1 >= 0 and symbols[i - 1] == "(":
            result[on_key_of[wave]].append(int(samples[i - 1]))
        if i + 1 < len(symbols) and symbols[i + 1] == ")":
            result[off_key_of[wave]].append(int(samples[i + 1]))

    return {k: sorted(v) for k, v in result.items()}


def cluster_beats_from_rpeaks(gt_by_lead: dict[str, dict[str, list[int]]],
                              fs: float = 500.0, cluster_tol_s: float = 0.1) -> list[float]:
    """Poolt die R_peak-Samples aller Leads eines Records, sortiert sie und
    gruppiert benachbarte Samples innerhalb `cluster_tol_s` zu einem Beat.
    Rueckgabe: kanonische Referenzzeit (Median der Samples im Cluster) je
    Beat, in Sekunden, aufsteigend sortiert. Wortgleiche Kopie der Funktion
    aus ludb_lead_spread_analysis.ipynb (dort bereits gelaufen/validiert).
    """
    all_samples = sorted(s for lead in gt_by_lead for s in gt_by_lead[lead]["R_peak"])
    if not all_samples:
        return []
    clusters: list[list[int]] = [[all_samples[0]]]
    for s in all_samples[1:]:
        if (s - clusters[-1][-1]) / fs <= cluster_tol_s:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    return [float(np.median(c)) / fs for c in clusters]


def per_lead_times_for_beat(gt_by_lead: dict[str, dict[str, list[int]]], beat_ref_t: float,
                            wave: str, fs: float = 500.0) -> list[float]:
    """Fuer einen Beat (Referenzzeit `beat_ref_t`, aus cluster_beats_from_rpeaks)
    und einen Wellentyp: die naechstgelegene Annotation je Lead innerhalb
    SEARCH_RADIUS_MS[wave], falls vorhanden. Rueckgabe: Liste der gefundenen
    Zeiten in Sekunden (ein Wert je Lead, das etwas gefunden hat -- bis zu 12).
    Wortgleiche Kopie der Funktion aus ludb_lead_spread_analysis.ipynb.
    """
    radius_s = SEARCH_RADIUS_MS[wave] / 1000.0
    times = []
    for lead in gt_by_lead:
        samples = gt_by_lead[lead][wave]
        if not samples:
            continue
        arr = np.asarray(samples, dtype=float) / fs
        idx = int(np.argmin(np.abs(arr - beat_ref_t)))
        if abs(arr[idx] - beat_ref_t) <= radius_s:
            times.append(float(arr[idx]))
    return times


def build_pooled_consensus_gt(record_id: int, leads: list[str], data_dir: Path = DATA_DIR,
                              fs: float = 500.0) -> dict[str, list[float]]:
    """Baut je Wellentyp EINE gepoolte Konsens-Ground-Truth pro Beat (Median
    ueber alle Leads, die diesen Beat fuer diesen Wellentyp annotiert haben),
    statt der bis zu 12 einzelnen Lead-Werte, gegen die evaluate_detections()
    matcht. Nutzt dieselbe Beat-Clusterung (via R_peak) wie
    ludb_lead_spread_analysis.ipynb. Rueckgabe: wave -> sortierte Liste von
    Konsens-Zeitpunkten in Sekunden (nur Beats mit >=1 Lead-Treffer)."""
    gt_by_lead = {lead: parse_ludb_annotations(record_id, lead, data_dir) for lead in leads}
    beat_refs = cluster_beats_from_rpeaks(gt_by_lead, fs=fs)
    pooled: dict[str, list[float]] = {w: [] for w in KEY_META}
    for beat_ref_t in beat_refs:
        for wave in KEY_META:
            times = per_lead_times_for_beat(gt_by_lead, beat_ref_t, wave, fs=fs)
            if times:
                pooled[wave].append(float(np.median(times)))
    return {w: sorted(v) for w, v in pooled.items()}


def _aggregate_wave_stats(per_wave_totals: dict[str, dict[str, int]],
                          per_wave_tp_errors_by_record: dict[str, dict[int, list[float]]]) -> pd.DataFrame:
    """Faktorisierter Kern von evaluate_detections(): baut aus TP/FN/FP-Summen
    und den Pro-Record-Fehlerlisten die Se/PPV/F1/m/sigma-Tabelle. Von
    evaluate_detections() UND evaluate_detections_pooled() genutzt, damit
    beide exakt dieselbe Metrik-Definition (inkl. sigma = Mittel der
    Pro-Record-SDs) verwenden."""
    rows = []
    for wave in KEY_META:
        tp = per_wave_totals[wave]["TP"]
        fn = per_wave_totals[wave]["FN"]
        fp = per_wave_totals[wave]["FP"]
        se = tp / (tp + fn) * 100 if (tp + fn) > 0 else np.nan
        ppv = tp / (tp + fp) * 100 if (tp + fp) > 0 else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else np.nan

        all_errors = [e for errs in per_wave_tp_errors_by_record[wave].values() for e in errs]
        m = float(np.mean(all_errors)) if all_errors else np.nan

        record_sds = [np.std(errs, ddof=1) for errs in per_wave_tp_errors_by_record[wave].values() if len(errs) >= 2]
        sigma = float(np.mean(record_sds)) if record_sds else np.nan

        cse = CSE_2SIGMA_MS[wave]
        rows.append({
            "Wave": wave, "TP": tp, "FN": fn, "FP": fp,
            "Se (%)": se, "PPV (%)": ppv, "F1 (%)": f1,
            "m (ms)": m, "sigma (ms)": sigma,
            "2*sigma_CSE (ms)": cse,
            "sigma < 2*sigma_CSE": (sigma < cse) if not np.isnan(cse) else np.nan,
        })
    return pd.DataFrame(rows).set_index("Wave").loc[list(KEY_META)]


def evaluate_detections_pooled(all_detections: dict[int, dict[str, list[int]]],
                               leads: list[str], fs: float = 500.0) -> pd.DataFrame:
    """Wie evaluate_detections(), aber gegen die GEPOOLTE Konsens-GT
    (build_pooled_consensus_gt) statt gegen 12 einzelne Lead-GTs. Jeder Beat
    zaehlt hier genau EINMAL pro Wellentyp (statt bis zu 12x, einmal je
    Lead) -- testet, ob die Per-Lead-Metrik aus Schritt 4 die Werte
    (v.a. P/T) strukturell wegen Lead-zu-Lead-Uneinigkeit in der GT drueckt,
    unabhaengig von der eigentlichen algorithmischen Praezision."""
    per_wave_tp_errors_by_record: dict[str, dict[int, list[float]]] = {w: {} for w in KEY_META}
    per_wave_totals = {w: {"TP": 0, "FN": 0, "FP": 0} for w in KEY_META}

    for rid in sorted(all_detections.keys()):
        det = all_detections[rid]
        pooled_gt = build_pooled_consensus_gt(rid, leads, fs=fs)
        for wave in KEY_META:
            gt_samples = [int(round(t * fs)) for t in pooled_gt[wave]]
            errors, n_fn, n_fp = match_one_to_one(gt_samples, det[wave], fs)
            per_wave_totals[wave]["TP"] += len(errors)
            per_wave_totals[wave]["FN"] += n_fn
            per_wave_totals[wave]["FP"] += n_fp
            per_wave_tp_errors_by_record[wave][rid] = errors

    return _aggregate_wave_stats(per_wave_totals, per_wave_tp_errors_by_record)


# Zuordnung KEY_META-Wellentyp -> Spalte in df_beats (Rueckgabe von
# run_tuned_on_record()/run_hybrid_on_record(), hierarchical.py-Namensschema).
# T_peak hat KEINE eigene Spalte (Sonderfall, siehe per_beat_diagnostics()).
_KEY_TO_BEATCOL = {
    "P_on": "t_P_on", "P_peak": "t_P_peak", "P_off": "t_P_off",
    "QRS_on": "t_Q_on", "R_peak": "t_R_peak+", "QRS_off": "t_S_off",
    "T_on": "t_T_on", "T_off": "t_T_off",
}


def per_beat_diagnostics(record_id: int, df_beats: pd.DataFrame, leads: list[str],
                         fs: float = 500.0, df_analysis: pd.DataFrame | None = None) -> list[dict]:
    """Pro Beat UND Wellentyp: Inter-Lead-Spread der GT (SD ueber die Leads,
    die annotiert haben, in ms) UND der Fehler der EIGENEN Detektion dieses
    Beats zur gepoolten Konsens-GT (in ms, absolut UND vorzeichenbehaftet).
    Grundlage fuer die Spread-vs-Fehler-Korrelation: wenn Beats mit hohem
    Lead-Spread systematisch auch hohen Detektionsfehler zeigen, spricht das
    dafuer, dass GT-Unsicherheit (nicht algorithmische Ungenauigkeit) die
    F1-Schwaeche bei P/T erklaert. Nur Beats mit >=2 Lead-Treffern (sonst ist
    kein Spread definiert) und nur wenn die EIGENE Detektion des Beats
    existiert (nicht NaN) und innerhalb SEARCH_RADIUS_MS[wave] liegt.

    WICHTIG -- Unterschied zur ersten Version dieser Funktion: nimmt jetzt
    `df_beats` (eine Zeile je Beat, Spalten t_<marker>, Rueckgabe von
    run_tuned_on_record()/run_hybrid_on_record()) statt einer flachen
    Sample-Liste pro Record. Grund: bei der flachen Liste wurde jedem Beat
    schlicht die NAECHSTGELEGENE verfuegbare Detektion im ganzen Record
    zugeordnet -- wenn die eigene Detektion eines Beats fehlgeschlagen war
    (NaN, durch beats_to_sample_dict()/dropna() aus der flachen Liste
    entfernt), "erbte" der Beat faelschlich die Detektion eines NACHBAR-Beats.
    Das erzeugte irrefuehrende Duplikate (mehrere verschiedene Beats mit
    exakt demselben geliehenen Fehler) gerade bei den grössten Fehlern -- also
    genau dort, wo die Beispiel-Selektion (groesster abs_err_ms) hinschaute.
    Jetzt: eigene Detektion fehlt -> Beat wird uebersprungen, keine Ersatzwerte.

    Zusaetzlich `in_envelope`: liegt die Detektion innerhalb [min, max] der
    tatsaechlich beobachteten Lead-Zeiten (nicht nur nahe am Median)? Das ist
    ein schaerferer Test als die reine Spread-Korrelation -- ein Fehler
    ausserhalb der von den 12 Leads selbst aufgespannten Bandbreite kann NICHT
    durch Lead-Unsicherheit erklaert werden (dann ist es eher ein eigener
    algorithmischer Fehler statt eine plausible 13. "Lead-Meinung")."""
    gt_by_lead = {lead: parse_ludb_annotations(record_id, lead) for lead in leads}
    rows = []
    for _, brow in df_beats.iterrows():
        r_peak_t = float(brow["t_R_peak+"])
        for wave in KEY_META:
            if wave == "T_peak":
                t1 = float(brow.get("t_T_turn1", np.nan))
                t2 = float(brow.get("t_T_turn2", np.nan))
                cands = [t for t in (t1, t2) if not np.isnan(t)]
                if not cands:
                    det_t = np.nan
                elif len(cands) == 1:
                    det_t = cands[0]
                elif df_analysis is not None:
                    r1, r2 = _nearest_r(df_analysis, t1, fs), _nearest_r(df_analysis, t2, fs)
                    det_t = t1 if r1 >= r2 else t2
                else:
                    det_t = t2  # ohne df_analysis: konservativer Fallback (spaeterer Turn)
            else:
                det_t = float(brow.get(_KEY_TO_BEATCOL[wave], np.nan))
            if np.isnan(det_t):
                continue  # eigene Detektion fehlgeschlagen -- NICHT durch fremde ersetzen

            times = per_lead_times_for_beat(gt_by_lead, r_peak_t, wave, fs=fs)
            if len(times) < 2:
                continue
            spread_ms = float(np.std(np.asarray(times) * 1000.0, ddof=1))
            consensus_t = float(np.median(times))
            gt_lo, gt_hi = float(min(times)), float(max(times))
            dist_s = abs(det_t - consensus_t)
            if dist_s > SEARCH_RADIUS_MS[wave] / 1000.0:
                continue

            rows.append({
                "record_id": record_id, "beat_id": int(brow.get("beat_id", -1)),
                "wave": wave, "beat_t": r_peak_t,
                "n_leads": len(times), "spread_ms": spread_ms,
                "err_ms": (det_t - consensus_t) * 1000.0,
                "abs_err_ms": dist_s * 1000.0,
                "gt_range_ms": (gt_hi - gt_lo) * 1000.0,
                "in_envelope": bool(gt_lo <= det_t <= gt_hi),
            })
    return rows


def _nearest_r(df_analysis: pd.DataFrame, t_val: float, fs: float) -> float:
    """Liest den VCG-Zeigerradius r an der Zeit `t_val` (naechstgelegener Sample-Index)."""
    if np.isnan(t_val):
        return np.nan
    idx = int(round(t_val * fs))
    idx = min(max(idx, 0), len(df_analysis) - 1)
    return float(df_analysis["r"].values[idx])


def beats_to_sample_dict(df_beats: pd.DataFrame, df_analysis: pd.DataFrame, fs: float) -> dict[str, list[int]]:
    """Normiert annotate_all_beats()-Output auf das 9-Punkte-Schema. Siehe
    ludb_validation.ipynb Schritt 3 fuer die vollstaendige Begruendung der
    Mapping-Entscheidungen (R_peak <- t_R_peak+, T_peak <- staerkerer von
    T_turn1/T_turn2)."""
    direct_map = {
        "P_on": "t_P_on", "P_peak": "t_P_peak", "P_off": "t_P_off",
        "QRS_on": "t_Q_on", "QRS_off": "t_S_off",
        "T_on": "t_T_on", "T_off": "t_T_off",
    }
    keys = ["P_on", "P_peak", "P_off", "QRS_on", "R_peak", "QRS_off", "T_on", "T_peak", "T_off"]
    result: dict[str, list[int]] = {k: [] for k in keys}

    for key, col in direct_map.items():
        result[key] = sorted(int(round(v * fs)) for v in df_beats[col].dropna().values)

    result["R_peak"] = sorted(int(round(v * fs)) for v in df_beats["t_R_peak+"].dropna().values)

    t_peaks = []
    for _, row in df_beats.iterrows():
        t1, t2 = row["t_T_turn1"], row["t_T_turn2"]
        cands = [t for t in (t1, t2) if not np.isnan(t)]
        if len(cands) == 1:
            t_peaks.append(cands[0])
        elif len(cands) == 2:
            r1, r2 = _nearest_r(df_analysis, t1, fs), _nearest_r(df_analysis, t2, fs)
            t_peaks.append(t1 if r1 >= r2 else t2)
    result["T_peak"] = sorted(int(round(t * fs)) for t in t_peaks)

    return result


def run_delineation_on_ludb_record(record_id: int, data_dir: Path = DATA_DIR,
                                   transform: str = "IDT", use_zapline: bool = True,
                                   verbose: bool = False):
    """Fuehrt die vcgsuite-Pipeline (Filter -> Frank-XYZ -> Kinematik ->
    R-Peak/-Turn -> hierarchische Annotation) auf einem LUDB-Record aus.
    Reine Integrationsschicht, ruft nur unveraenderte vcgsuite-Funktionen
    auf. Siehe ludb_validation.ipynb Schritt 3.

    Anders als in ludb_validation.ipynb ist der Default hier verbose=False
    (im Tuning-Notebook wird diese Funktion sehr oft aufgerufen)."""
    rec = wfdb.rdrecord(str(data_dir / str(record_id)))
    fs = float(rec.fs)
    lead_idx = {name: i for i, name in enumerate(rec.sig_name)}
    ecg_raw = {lead: rec.p_signal[:, lead_idx[lead.lower()]] for lead in LEAD_ORDER_8}

    def _run():
        filtered = ecg.filter_pipeline(
            [ecg_raw[lead] for lead in LEAD_ORDER_8], fs=fs, use_zapline=use_zapline,
        )
        ecg_filt = {lead: filtered[i] for i, lead in enumerate(LEAD_ORDER_8)}
        X, Y, Z = ecg.ecg12_to_frank_xyz(ecg_filt, method=transform, lead_order=LEAD_ORDER_8)
        t_ax = np.arange(rec.sig_len) / fs
        df_a = pd.DataFrame({"Time": t_ax, "X": X, "Y": Y, "Z": Z, **ecg_filt})
        df_a.attrs["fs"] = fs
        df_a.attrs["transform"] = transform
        df_a, _dt = ecg.compute_vcg_kinematics(df_a)
        r_peak_times, _ = ecg.detect_r_peaks(df_a)
        r_turn_times, _ = ecg.detect_r_turn(df_a, r_peak_times)
        df_b = ecg.annotate_all_beats(df_a, r_peak_times, r_turn_times)
        return df_a, df_b

    if verbose:
        df_analysis, df_beats = _run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            df_analysis, df_beats = _run()

    detections = beats_to_sample_dict(df_beats, df_analysis, fs)
    return detections, df_beats, df_analysis


def match_one_to_one(gt_samples: list[int], det_samples: list[int], fs: float,
                     tol_s: float = TOLERANCE_S) -> tuple[list[float], int, int]:
    """Greedy 1:1-Zuordnung zwischen Ground-Truth- und Detektions-Samples
    innerhalb eines Toleranzfensters. Siehe ludb_validation.ipynb Schritt 4.
    """
    if not gt_samples and not det_samples:
        return [], 0, 0
    gt_t = sorted(gt_samples)
    det_t = sorted(det_samples)
    used_det = [False] * len(det_t)
    tp_errors_ms = []
    n_fn = 0

    for g in gt_t:
        best_j, best_dist = -1, None
        for j, d in enumerate(det_t):
            if used_det[j]:
                continue
            dist = abs(d - g) / fs
            if dist <= tol_s and (best_dist is None or dist < best_dist):
                best_dist, best_j = dist, j
        if best_j >= 0:
            used_det[best_j] = True
            tp_errors_ms.append((det_t[best_j] - g) / fs * 1000.0)
        else:
            n_fn += 1

    n_fp = used_det.count(False)
    return tp_errors_ms, n_fn, n_fp


def evaluate_detections(all_detections: dict[int, dict[str, list[int]]],
                        leads: list[str], fs: float = 500.0) -> pd.DataFrame:
    """Baut die Schritt-4-Ergebnistabelle (Se/PPV/F1/m/sigma pro Wellentyp)
    fuer ein gegebenes all_detections-Dict (record_id -> 9-Punkte-Schema).
    Nutzbar sowohl fuer die volle Baseline als auch fuer Train/Test-Splits
    im Tuning-Notebook -- identische Metrik wie ludb_validation.ipynb
    Schritt 4, hier als wiederverwendbare Funktion statt Notebook-Zellen.
    """
    per_wave_tp_errors_by_record: dict[str, dict[int, list[float]]] = {w: {} for w in KEY_META}
    per_wave_totals = {w: {"TP": 0, "FN": 0, "FP": 0} for w in KEY_META}

    for rid in sorted(all_detections.keys()):
        det = all_detections[rid]
        record_errors = {w: [] for w in KEY_META}
        for lead in leads:
            gt_lead = parse_ludb_annotations(rid, lead)
            for wave in KEY_META:
                errors, n_fn, n_fp = match_one_to_one(gt_lead[wave], det[wave], fs)
                per_wave_totals[wave]["TP"] += len(errors)
                per_wave_totals[wave]["FN"] += n_fn
                per_wave_totals[wave]["FP"] += n_fp
                record_errors[wave].extend(errors)
        for wave in KEY_META:
            per_wave_tp_errors_by_record[wave][rid] = record_errors[wave]

    return _aggregate_wave_stats(per_wave_totals, per_wave_tp_errors_by_record)
