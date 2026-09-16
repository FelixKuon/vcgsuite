# ══════════════════════════════════════════════════════════════════════════
#  ZYKLUS-FEATURES  (ST-Strecke, Basislinie, QRS-Fragmentierung, Zyklusanteile)
#
#  Ergänzt die P-/QRS-/T-Loop-Features (die jede Welle isoliert als
#  geschlossene Schleife betrachten, siehe p_wave.py/qrs_complex.py/t_wave.py)
#  um Merkmale, die BEZIEHUNGEN zwischen den Wellen und der isoelektrischen
#  Basislinie erfassen — allen voran die ST-Strecke (klassischer
#  Infarkt-Marker, in den reinen Wellen-Loop-Features nicht abgedeckt, da die
#  den Bereich ZWISCHEN QRS-Ende und T-Beginn gar nicht betrachten) und
#  QRS-Fragmentierung/Kerbung (klassischer Marker für
#  Erregungsleitungsstörungen).
#
#  Design-Entscheidung: für jede Kernkennzahl gibt es eine rotationsinvariante
#  3D-/Achsen-Variante (Hauptmerkmal, passt zur "globale 3D-Geometrie statt
#  Einzel-Lead"-These des Projekts) UND eine ergänzende Pro-Lead-Variante
#  (Spaltenname endet auf den Lead-Namen, z. B. `ST_V1_J60`). Die Pro-Lead-
#  Spalten dienen ausdrücklich dem späteren Ablationsvergleich "hilft die
#  3D-Geometrie allein schon, oder braucht es zusätzlich Einzel-Lead-
#  Information" (siehe docs/ptbxl_classification_findings.md, Abschnitt 6b)
#  — und sind direkt relevant für die EASI-Reduced-Lead-Fragestellung: wenn
#  die Pro-Lead-Spalten in der Ablation kaum zusätzlichen Nutzen zeigen,
#  spricht das dafür, dass eine aus wenigen Elektroden rekonstruierte VCG
#  (statt allen 12 Kanälen) für die Klassifikation ausreicht.
#
#  Bewusst KEINE TP-Segment-Basislinie (nur PQ-Segment) und KEIN
#  Nachbar-Beat-Zugriff in dieser ersten Version — hält die Implementierung
#  einfach und beat-lokal, analog zu p_wave.py/qrs_complex.py/t_wave.py.
#
#  v4-Erweiterung: nutzt zwei bis dahin in der GESAMTEN Pipeline komplett
#  ungenutzte, aber echte (datengetriebene) Annotations-Marker:
#    - S_on   (Übergang R-Zacken-Kuppe → S-Zacke, HIERARCHICAL_WINDOWS:
#              R_peak+20..100ms, also zwischen R_turn und S_off) — zerlegt
#              den bisher als EIN Loop behandelten QRS-Komplex in
#              Q-/R-/S-Anteil. Ermöglicht insbesondere ein R/S-Amplituden-
#              verhältnis pro Lead — die klassische Hypertrophie-/RBBB-
#              Spannungsdiagnostik (z. B. Sokolow-Lyon-Index), die in den
#              reinen Loop-Geometrie-Features nicht abgebildet war.
#    - T_turn2 (zweiter, späterer Wendepunkt in der T-Welle, HIERARCHICAL_
#              WINDOWS: T_turn1+60..130ms) — ermöglicht eine Kerben-/
#              Biphasie-Erkennung der T-Welle (T_turn1↔T_turn2-Abstand und
#              -Tiefe) statt nur der bisherigen Zweiteilung (Anstieg/Abfall
#              um T_turn1). Bonus: da unser T_off-Suchfenster laut LUDB-
#              Untersuchung (docs/ludb_projection_blindness_findings.md)
#              vermutlich zu oft eine Struktur HINTER T_turn2 statt des
#              echten T-Endes markiert, dient `QT_alt_ms` (Q_on→T_turn2)
#              als T_off-unabhängige Alternative zu QT_ms/QTc_Bazett.
#  (Q_off wurde bewusst NICHT genutzt — kein echter Marker, sondern laut
#  `annotation/hierarchical.py` ein fixer Versatz R_peak−20ms, keine
#  eigene Information.)
#
#  v5-Erweiterung (nach der N=1200-Auswertung von v3+v4, Abschnitte 10c/10d
#  in ptbxl_classification_pilot.ipynb — siehe
#  docs/ptbxl_classification_findings.md, Abschnitt 6f/6g für die
#  Herleitung): reizt den "Engineering-Deckel" der bestehenden
#  Punkt-/Skalar-Features weiter aus, bevor der volle Datensatz prozessiert
#  wird.
#    - **Achsen-Vorzeichen-Fix** (`_canonicalize_axis`): `QRS_axis_svd`/
#      `T_axis_svd` haben eine beat-abhängige, arbiträre Vorzeichen-
#      Mehrdeutigkeit (SVD legt sich nicht fest) — betraf bisher JEDE
#      vorzeichenbehaftete Projektion auf diese Achsen (`ST_axis_J*`,
#      `ST_slope_3d`, `Q_depth_3d`, `R_amp_3d`, `S_depth_3d`,
#      `T_notch_depth_3d`, `QRST_transition_angle_QRSaxis`) unbemerkt.
#      Jetzt beat-lokal kanonisiert (QRS-Achse zeigt Richtung R_turn,
#      T-Achse Richtung T_turn1). **Ändert die numerischen Werte dieser
#      bereits existierenden v3/v4-Spalten** — nicht mehr direkt vergleichbar
#      mit den v3/v4-Zahlen aus Abschnitt 6c/6d/6e/6f.
#    - **ST-/Q-Richtungsvektoren** (`ST_dx/dy/dz_J60`, `Q_dx/dy/dz`): rohe
#      XYZ-Komponenten (festes Aufnahme-Koordinatensystem, keine
#      Vorzeichen-Mehrdeutigkeit) statt nur Betrag+eine Skalar-Projektion —
#      Motivation: Abschnitt 6f zeigte, dass MI von den Pro-Lead-ST-Spalten
#      weit mehr profitiert als jede andere Klasse, vermutlich weil sie
#      (umständlich, über 8 Einzelspalten) Richtungsinformation tragen, die
#      `ST_mag`/`ST_axis` als reine Skalare nicht abbilden. `ST_angle_
#      QRSaxis_J60` ergänzt den vollen Winkel zur (jetzt kanonisierten)
#      QRS-Achse.
#    - **Transmuraler Repolarisationsgradient** (Nutzer-Idee):
#      `T_peak_end_ms` (klassischer Tpeak-Tend-Proxy, T_turn1→T_off) +
#      `T_terminal_slope_3d` (Steigung des terminalen Abfalls) + bei
#      vorhandenem T_turn2 aufgeteilt in `T_early_fall_slope_3d`/
#      `T_late_fall_slope_3d` — nutzt die zwei T-Wellen-Wendepunkte, um die
#      Repolarisations-RATE statt nur Zeitanteile zu erfassen.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import (
    safe, t2idx, loop_tortuosity, count_local_extrema, project_onto_axis,
    angle_between,
)
from ...config import LEAD_ORDER as _DEFAULT_LEADS

# Klinisch übliche ST-Messpunkte relativ zum J-Punkt (QRS-Ende).
ST_OFFSETS_MS = (0.0, 60.0, 80.0)
# Fenster für die ST-Steigung (up-/downsloping vs. horizontal).
ST_SLOPE_WINDOW_MS = (0.0, 80.0)
# Referenzpunkt für die (bewusst auf einen Punkt reduzierte) Pro-Lead-ST-Kennzahl
# UND für die rohen ST-Richtungskomponenten (v5, siehe unten).
PER_LEAD_ST_OFFSET_MS = 60.0


def _canonicalize_axis(axis: np.ndarray, point: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Orientiert eine SVD-Achse (`QRS_axis_svd`/`T_axis_svd`) konsistent.

    `np.linalg.svd` legt sich nicht auf eine Vorzeichen-Richtung für die
    Singulärvektoren fest -- das Vorzeichen kann von Beat zu Beat durch
    numerisches Rauschen kippen. Ohne Kanonisierung ist jede VORZEICHEN-
    BEHAFTETE Projektion auf diese Achse (`ST_axis_J*`, `ST_slope_3d`,
    `Q_depth_3d`, `R_amp_3d`, `S_depth_3d`, `T_notch_depth_3d`,
    `QRST_transition_angle_QRSaxis`, ...) zwischen Beats NICHT konsistent
    interpretierbar -- z. B. würde `Q_depth_3d` bei zufällig geflipptem
    Vorzeichen Minimum/Maximum der Projektion vertauschen statt nur das
    Vorzeichen zu wechseln (verfälscht den Wert, nicht nur dessen
    Vorzeichen). Fix: die Achse wird so orientiert, dass sie in Richtung
    `point - baseline` zeigt -- für die QRS-Achse `point = R_turn`
    (robustester, praktisch immer vorhandener QRS-Landmark), für die
    T-Achse `point = T_turn1` (Haupt-T-Wellen-Peak).

    Betrifft nur diese Datei (`cycle.py`) -- dieselbe Mehrdeutigkeit steckt
    vermutlich auch in `QRS_axis_svd`/`T_axis_svd`, wie sie in
    `qrs_complex.py`/`t_wave.py` exportiert werden (z. B. `theta_QRS_T_svd`),
    dort aber bewusst NICHT angefasst, um bereits erhobene/verglichene
    Ergebnisse nicht rückwirkend zu verändern -- siehe
    docs/ptbxl_classification_findings.md für die Einordnung.
    """
    if not (np.all(np.isfinite(axis)) and np.all(np.isfinite(point)) and np.all(np.isfinite(baseline))):
        return axis
    ref = point - baseline
    if np.dot(axis, ref) < 0:
        return -axis
    return axis


def compute_cycle_features(df_analysis: pd.DataFrame,
                           df_annotations: pd.DataFrame,
                           df_r: pd.DataFrame,
                           df_qrs: pd.DataFrame,
                           df_t: pd.DataFrame,
                           lead_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Berechnet Zyklus-/Zwischen-Wellen-Features pro Beat: ST-Streckenlage
    relativ zur PQ-Basislinie (3D + pro Lead + rohe Richtung, v5), ST-
    Steigung, QRS→T-Übergangsvektor, QRS-Fragmentierung (Kerbenzahl +
    Pfad-Tortuosität), pathologische-Q-Zacke-Proxy (+ Richtung, v5),
    RR-normierte Zyklusanteile (PR/QRS/ST), R-/S-Zacken-Zerlegung +
    R/S-Amplitudenverhältnis (v4), T-Wellen-Kerbe über T_turn1/T_turn2 (v4),
    transmuraler Repolarisationsgradient über Tpeak-Tend + terminale
    T-Steigung(en) (v5). Kanonisiert seit v5 außerdem das Vorzeichen von
    `QRS_axis_svd`/`T_axis_svd` vor jeder vorzeichenbehafteten Projektion
    (siehe `_canonicalize_axis`).

    Parameters
    ----------
    df_analysis    : pd.DataFrame  –  Spalten Time, X, Y, Z, optional
                                      Roh-Lead-Spalten (siehe `lead_cols`).
    df_annotations : pd.DataFrame  –  Ausgabe von `annotate_all_beats()`.
    df_r           : pd.DataFrame  –  Ausgabe von `compute_beat_rotation()`,
                                      muss 'beat_id', 'RR_ms' enthalten.
    df_qrs         : pd.DataFrame  –  Ausgabe von `compute_qrs_features()`
                                      (für `QRS_axis_svd_{x,y,z}` — die
                                      rotationsinvariante QRS-Hauptachse).
    df_t           : pd.DataFrame  –  Ausgabe von `compute_t_wave_features()`
                                      (für `T_axis_svd_{x,y,z}` — die
                                      T-Wellen-Hauptachse, für die
                                      T-Kerben-Tiefe, v4).
    lead_cols      : list[str] | None  –  Namen der Roh-Lead-Spalten in
                                      `df_analysis`, falls vorhanden (Default:
                                      `vcgsuite.config.LEAD_ORDER`; nur die
                                      tatsächlich vorhandenen Spalten werden
                                      genutzt — fehlt `df_analysis` jede
                                      Lead-Spalte, werden einfach nur die
                                      3D-Features berechnet).

    Returns
    -------
    df_cycle : pd.DataFrame  –  eine Zeile pro Beat.
    """
    vcg = df_analysis[['X', 'Y', 'Z']].to_numpy()
    t_vcg = df_analysis['Time'].to_numpy()

    leads = [c for c in (lead_cols or _DEFAULT_LEADS) if c in df_analysis.columns]
    lead_arr = {lead: df_analysis[lead].to_numpy(dtype=float) for lead in leads}

    df_merged = df_annotations.merge(df_r[['beat_id', 'RR_ms']], on='beat_id', how='left')
    df_merged = df_merged.merge(
        df_qrs[['beat_id', 'QRS_axis_svd_x', 'QRS_axis_svd_y', 'QRS_axis_svd_z']],
        on='beat_id', how='left',
    )
    df_merged = df_merged.merge(
        df_t[['beat_id', 'T_axis_svd_x', 'T_axis_svd_y', 'T_axis_svd_z']],
        on='beat_id', how='left',
    )
    df_merged = df_merged.sort_values('beat_id').reset_index(drop=True)

    records = []

    for _, row in df_merged.iterrows():
        bid = int(row['beat_id'])
        t_pon = safe(row, 't_P_on')
        t_qon = safe(row, 't_Q_on')
        t_rturn = safe(row, 't_R_turn')
        t_son = safe(row, 't_S_on')
        t_soff = safe(row, 't_S_off')
        t_qoff = safe(row, 't_Q_off')
        t_ton = safe(row, 't_T_on')
        t_t1 = safe(row, 't_T_turn1')
        t_t2 = safe(row, 't_T_turn2')
        t_toff = safe(row, 't_T_off')
        rr = safe(row, 'RR_ms')

        t_qrs_end = t_soff if np.isfinite(t_soff) else t_qoff
        qrs_axis = np.array([row.get('QRS_axis_svd_x', np.nan),
                              row.get('QRS_axis_svd_y', np.nan),
                              row.get('QRS_axis_svd_z', np.nan)], dtype=float)
        has_qrs_axis = np.all(np.isfinite(qrs_axis))
        t_axis = np.array([row.get('T_axis_svd_x', np.nan),
                            row.get('T_axis_svd_y', np.nan),
                            row.get('T_axis_svd_z', np.nan)], dtype=float)
        has_t_axis = np.all(np.isfinite(t_axis))

        rec = {'beat_id': bid, 'RR_ms': rr}

        # ── PQ-Basislinie (isoelektrische Referenz) ─────────────────────────
        baseline_xyz = np.array([np.nan, np.nan, np.nan])
        baseline_leads = {lead: np.nan for lead in leads}
        pq_baseline_std = np.nan
        if np.isfinite(t_pon) and np.isfinite(t_qon) and t_qon > t_pon:
            i0, i1 = t2idx(t_pon, t_vcg), t2idx(t_qon, t_vcg)
            if i1 - i0 >= 2:
                seg_pq = vcg[i0:i1 + 1]
                baseline_xyz = seg_pq.mean(axis=0)
                pq_baseline_std = float(np.linalg.norm(seg_pq - baseline_xyz, axis=1).std())
                for lead in leads:
                    baseline_leads[lead] = float(lead_arr[lead][i0:i1 + 1].mean())
        rec['PQ_baseline_std'] = pq_baseline_std
        has_baseline = bool(np.all(np.isfinite(baseline_xyz)))

        # ── v5: Achsen-Vorzeichen kanonisieren (siehe _canonicalize_axis) ────
        if has_baseline and has_qrs_axis and np.isfinite(t_rturn):
            qrs_axis = _canonicalize_axis(qrs_axis, vcg[t2idx(t_rturn, t_vcg)], baseline_xyz)
        if has_baseline and has_t_axis and np.isfinite(t_t1):
            t_axis = _canonicalize_axis(t_axis, vcg[t2idx(t_t1, t_vcg)], baseline_xyz)

        # ── ST-Streckenlage: 3D-Magnitude + Achsen-Projektion, mehrere Offsets ──
        # v5: an EINEM Referenzpunkt (J+60ms, wie die Pro-Lead-ST-Spalte)
        # zusätzlich die ROHEN Richtungskomponenten (festes X/Y/Z-Koordinaten-
        # system der Aufnahme, keine Vorzeichen-Mehrdeutigkeit wie bei den
        # SVD-Achsen) + den vollen Winkel zur QRS-Achse. Motivation: `ST_mag`
        # (Betrag) und `ST_axis` (eine Skalar-Projektion) allein können die
        # RICHTUNG der ST-Verschiebung nicht eindeutig festlegen -- genau
        # diese Richtung unterscheidet klinisch z. B. anteriore von
        # inferiorer Infarktlokalisation (siehe docs/ptbxl_classification_
        # findings.md, Abschnitt 6f: Pro-Lead-Spalten halfen MI am meisten
        # von allen Klassen -- das ist ein indirekter Beleg dafür, dass die
        # Richtungsinformation fehlte und offenbar nützlich ist).
        rec['ST_dx_J60'] = rec['ST_dy_J60'] = rec['ST_dz_J60'] = np.nan
        rec['ST_angle_QRSaxis_J60'] = np.nan
        for off in ST_OFFSETS_MS:
            key = f'{off:.0f}'
            mag_col, axis_col = f'ST_mag_J{key}', f'ST_axis_J{key}'
            rec[mag_col] = rec[axis_col] = np.nan
            if has_baseline and np.isfinite(t_qrs_end):
                t_st = t_qrs_end + off / 1000.0
                if not np.isfinite(t_ton) or t_st < t_ton:  # nicht in die T-Welle hineinsampeln
                    idx = t2idx(t_st, t_vcg)
                    st_vec = vcg[idx] - baseline_xyz
                    rec[mag_col] = float(np.linalg.norm(st_vec))
                    if has_qrs_axis:
                        rec[axis_col] = float(np.dot(st_vec, qrs_axis))
                    if off == PER_LEAD_ST_OFFSET_MS:
                        rec['ST_dx_J60'], rec['ST_dy_J60'], rec['ST_dz_J60'] = (
                            float(st_vec[0]), float(st_vec[1]), float(st_vec[2]))
                        if has_qrs_axis:
                            rec['ST_angle_QRSaxis_J60'] = angle_between(st_vec, qrs_axis)

        # Pro-Lead-ST (bewusst nur EIN Referenzpunkt, siehe Modul-Docstring)
        for lead in leads:
            col = f'ST_{lead}_J{PER_LEAD_ST_OFFSET_MS:.0f}'
            rec[col] = np.nan
            if np.isfinite(baseline_leads.get(lead, np.nan)) and np.isfinite(t_qrs_end):
                t_st = t_qrs_end + PER_LEAD_ST_OFFSET_MS / 1000.0
                if not np.isfinite(t_ton) or t_st < t_ton:
                    idx = t2idx(t_st, t_vcg)
                    rec[col] = float(lead_arr[lead][idx] - baseline_leads[lead])

        # ── ST-Steigung (3D, über QRS-Achse projiziert) ─────────────────────
        rec['ST_slope_3d'] = np.nan
        if has_qrs_axis and np.isfinite(t_qrs_end):
            t0 = t_qrs_end + ST_SLOPE_WINDOW_MS[0] / 1000.0
            t1 = t_qrs_end + ST_SLOPE_WINDOW_MS[1] / 1000.0
            if not np.isfinite(t_ton) or t1 < t_ton:
                i0, i1 = t2idx(t0, t_vcg), t2idx(t1, t_vcg)
                if i1 - i0 >= 3:
                    seg_st = vcg[i0:i1 + 1]
                    proj = project_onto_axis(seg_st, qrs_axis)
                    tt = t_vcg[i0:i1 + 1] - t_vcg[i0]
                    if np.all(np.isfinite(proj)):
                        rec['ST_slope_3d'] = float(np.polyfit(tt, proj, 1)[0])

        # ── QRS→T-Übergangsvektor ────────────────────────────────────────────
        rec['QRST_transition_dist'] = rec['QRST_transition_angle_QRSaxis'] = np.nan
        if np.isfinite(t_qrs_end) and np.isfinite(t_ton):
            i_end, i_ton = t2idx(t_qrs_end, t_vcg), t2idx(t_ton, t_vcg)
            trans_vec = vcg[i_ton] - vcg[i_end]
            trans_norm = float(np.linalg.norm(trans_vec))
            rec['QRST_transition_dist'] = trans_norm
            if has_qrs_axis and trans_norm > 1e-12:
                cos_a = np.clip(np.dot(trans_vec, qrs_axis) / trans_norm, -1, 1)
                rec['QRST_transition_angle_QRSaxis'] = float(np.degrees(np.arccos(cos_a)))

        # ── QRS-Fragmentierung: Kerbenzahl (3D + pro Lead) + Pfad-Tortuosität ──
        rec['QRS_frag_count_3d'] = np.nan
        rec['QRS_path_tortuosity'] = np.nan
        for lead in leads:
            rec[f'QRS_frag_count_{lead}'] = np.nan
        if np.isfinite(t_qon) and np.isfinite(t_qrs_end) and t_qrs_end > t_qon:
            i0, i1 = t2idx(t_qon, t_vcg), t2idx(t_qrs_end, t_vcg)
            if i1 - i0 >= 4:
                seg_qrs = vcg[i0:i1 + 1]
                rec['QRS_path_tortuosity'] = loop_tortuosity(seg_qrs)
                if has_qrs_axis:
                    proj_qrs = project_onto_axis(seg_qrs, qrs_axis)
                    if np.all(np.isfinite(proj_qrs)):
                        rec['QRS_frag_count_3d'] = float(count_local_extrema(proj_qrs))
                for lead in leads:
                    rec[f'QRS_frag_count_{lead}'] = float(
                        count_local_extrema(lead_arr[lead][i0:i1 + 1]))

        # ── Pathologische-Q-Zacke-Proxy (Q_on → R_turn) ─────────────────────
        rec['Q_depth_3d'] = rec['Q_dur_ms'] = np.nan
        rec['Q_dx'] = rec['Q_dy'] = rec['Q_dz'] = np.nan
        for lead in leads:
            rec[f'Q_depth_{lead}'] = np.nan
        if np.isfinite(t_qon) and np.isfinite(t_rturn) and t_rturn > t_qon:
            i0, i1 = t2idx(t_qon, t_vcg), t2idx(t_rturn, t_vcg)
            rec['Q_dur_ms'] = (t_rturn - t_qon) * 1000
            if i1 - i0 >= 2:
                if has_baseline and has_qrs_axis:
                    seg_q = vcg[i0:i1 + 1]
                    proj_q = project_onto_axis(seg_q, qrs_axis)
                    baseline_proj = project_onto_axis(baseline_xyz.reshape(1, 3), qrs_axis)[0]
                    if np.isfinite(baseline_proj) and np.all(np.isfinite(proj_q)):
                        rec['Q_depth_3d'] = float(baseline_proj - proj_q.min())
                        # v5: rohe Richtung der Q-Zacken-Talsohle (fixes X/Y/Z,
                        # keine Achsen-Vorzeichen-Mehrdeutigkeit) -- Pendant zu
                        # ST_dx/dy/dz, relevant fuer die Lokalisation von
                        # ALTEN/pathologischen Q-Zacken statt nur akuter
                        # ST-Veraenderungen.
                        i_qmin = i0 + int(np.argmin(proj_q))
                        q_vec = vcg[i_qmin] - baseline_xyz
                        rec['Q_dx'], rec['Q_dy'], rec['Q_dz'] = (
                            float(q_vec[0]), float(q_vec[1]), float(q_vec[2]))
                for lead in leads:
                    if np.isfinite(baseline_leads.get(lead, np.nan)):
                        rec[f'Q_depth_{lead}'] = float(
                            baseline_leads[lead] - lead_arr[lead][i0:i1 + 1].min())

        # ── Zyklus-Anteile (RR-normiert) ────────────────────────────────────
        rec['PR_ratio'] = rec['QRS_ratio'] = rec['ST_dur_ms'] = rec['ST_ratio'] = np.nan
        if np.isfinite(rr) and rr > 0:
            if np.isfinite(t_qon) and np.isfinite(t_pon):
                rec['PR_ratio'] = ((t_qon - t_pon) * 1000) / rr
            if np.isfinite(t_qrs_end) and np.isfinite(t_qon):
                rec['QRS_ratio'] = ((t_qrs_end - t_qon) * 1000) / rr
            if np.isfinite(t_ton) and np.isfinite(t_qrs_end):
                st_dur = (t_ton - t_qrs_end) * 1000
                rec['ST_dur_ms'] = st_dur
                rec['ST_ratio'] = st_dur / rr

        # ── v4: R-Zacken-Plateau + S-Zacke (Q_on → R_turn → S_on → S_off) ────
        rec['R_plateau_dur_ms'] = rec['S_dur_ms'] = np.nan
        rec['R_amp_3d'] = rec['S_depth_3d'] = rec['RS_ratio_3d'] = np.nan
        for lead in leads:
            rec[f'R_amp_{lead}'] = rec[f'S_depth_{lead}'] = rec[f'RS_ratio_{lead}'] = np.nan

        if np.isfinite(t_rturn) and np.isfinite(t_son) and t_son > t_rturn:
            rec['R_plateau_dur_ms'] = (t_son - t_rturn) * 1000
        if np.isfinite(t_son) and np.isfinite(t_soff) and t_soff > t_son:
            rec['S_dur_ms'] = (t_soff - t_son) * 1000

        r_amp_3d = np.nan
        r_amp_lead = {lead: np.nan for lead in leads}
        if np.isfinite(t_rturn):
            idx_r = t2idx(t_rturn, t_vcg)
            if has_baseline and has_qrs_axis:
                baseline_proj_qrs = project_onto_axis(baseline_xyz.reshape(1, 3), qrs_axis)[0]
                r_proj = project_onto_axis(vcg[idx_r].reshape(1, 3), qrs_axis)[0]
                if np.isfinite(baseline_proj_qrs) and np.isfinite(r_proj):
                    r_amp_3d = float(r_proj - baseline_proj_qrs)
            for lead in leads:
                if np.isfinite(baseline_leads.get(lead, np.nan)):
                    r_amp_lead[lead] = float(lead_arr[lead][idx_r] - baseline_leads[lead])
        rec['R_amp_3d'] = r_amp_3d
        for lead in leads:
            rec[f'R_amp_{lead}'] = r_amp_lead[lead]

        s_depth_3d = np.nan
        s_depth_lead = {lead: np.nan for lead in leads}
        if np.isfinite(t_son) and np.isfinite(t_soff) and t_soff > t_son:
            i0, i1 = t2idx(t_son, t_vcg), t2idx(t_soff, t_vcg)
            if i1 - i0 >= 1:
                if has_baseline and has_qrs_axis:
                    baseline_proj_qrs = project_onto_axis(baseline_xyz.reshape(1, 3), qrs_axis)[0]
                    proj_s = project_onto_axis(vcg[i0:i1 + 1], qrs_axis)
                    if np.isfinite(baseline_proj_qrs) and np.all(np.isfinite(proj_s)):
                        s_depth_3d = float(baseline_proj_qrs - proj_s.min())
                for lead in leads:
                    if np.isfinite(baseline_leads.get(lead, np.nan)):
                        s_depth_lead[lead] = float(
                            baseline_leads[lead] - lead_arr[lead][i0:i1 + 1].min())
        rec['S_depth_3d'] = s_depth_3d
        for lead in leads:
            rec[f'S_depth_{lead}'] = s_depth_lead[lead]

        if np.isfinite(r_amp_3d) and np.isfinite(s_depth_3d):
            rec['RS_ratio_3d'] = float(abs(r_amp_3d) / (abs(s_depth_3d) + 1e-9))
        for lead in leads:
            ra, sd = r_amp_lead[lead], s_depth_lead[lead]
            if np.isfinite(ra) and np.isfinite(sd):
                rec[f'RS_ratio_{lead}'] = float(abs(ra) / (abs(sd) + 1e-9))

        # QRS-Formsignatur: Anteile Q-/R-/S-Segment an der QRS-Gesamtdauer
        rec['QRS_shape_Q_frac'] = rec['QRS_shape_R_frac'] = rec['QRS_shape_S_frac'] = np.nan
        if (np.isfinite(t_qon) and np.isfinite(t_rturn) and np.isfinite(t_son)
                and np.isfinite(t_qrs_end)):
            total = t_qrs_end - t_qon
            if total > 0 and t_qon <= t_rturn <= t_son <= t_qrs_end:
                rec['QRS_shape_Q_frac'] = (t_rturn - t_qon) / total
                rec['QRS_shape_R_frac'] = (t_son - t_rturn) / total
                rec['QRS_shape_S_frac'] = (t_qrs_end - t_son) / total

        # ── v4: T-Wellen-Kerbe (T_turn1 ↔ T_turn2) + T_turn2-basierte QT-Alternative ──
        rec['T_notch_gap_ms'] = rec['T_notch_depth_3d'] = np.nan
        if np.isfinite(t_t1) and np.isfinite(t_t2) and t_t2 > t_t1:
            rec['T_notch_gap_ms'] = (t_t2 - t_t1) * 1000
            if has_t_axis:
                i0, i1 = t2idx(t_t1, t_vcg), t2idx(t_t2, t_vcg)
                if i1 - i0 >= 2:
                    proj_notch = project_onto_axis(vcg[i0:i1 + 1], t_axis)
                    if np.all(np.isfinite(proj_notch)):
                        shoulder = min(proj_notch[0], proj_notch[-1])
                        rec['T_notch_depth_3d'] = float(shoulder - proj_notch.min())

        rec['QT_alt_ms'] = np.nan
        if np.isfinite(t_qon) and np.isfinite(t_t2):
            rec['QT_alt_ms'] = (t_t2 - t_qon) * 1000

        # ── v5: transmuraler Repolarisationsgradient (Nutzer-Idee) ───────────
        # Klassischer Tpeak-Tend-Proxy (T_turn1 -> T_off, ein etablierter
        # Marker für die transmurale Dispersion der Repolarisation, TDR) +
        # die STEIGUNG des terminalen Abfalls -- direktere Kennzahl für die
        # Repolarisations-RATE als die reinen Zeitanteile aus
        # T_shape_{rise,mid,fall}_frac. Mit T_turn2 (falls als echter zweiter
        # Wendepunkt vorhanden) lässt sich der Abfall zusätzlich in einen
        # frühen und einen späten Teilabschnitt aufteilen -- steilt der
        # Abfall zum Ende hin (T_late < T_early) oder flacht er ab, ist
        # potenziell selbst schon eine Aussage über die Form des
        # zugrundeliegenden Gradienten, nicht nur über seine mittlere Rate.
        rec['T_peak_end_ms'] = np.nan
        rec['T_terminal_slope_3d'] = np.nan
        rec['T_early_fall_slope_3d'] = np.nan
        rec['T_late_fall_slope_3d'] = np.nan
        if np.isfinite(t_t1) and np.isfinite(t_toff) and t_toff > t_t1:
            rec['T_peak_end_ms'] = (t_toff - t_t1) * 1000
            if has_t_axis:
                i_t1, i_toff = t2idx(t_t1, t_vcg), t2idx(t_toff, t_vcg)
                if i_toff - i_t1 >= 3:
                    proj_term = project_onto_axis(vcg[i_t1:i_toff + 1], t_axis)
                    tt_term = t_vcg[i_t1:i_toff + 1] - t_vcg[i_t1]
                    if np.all(np.isfinite(proj_term)):
                        rec['T_terminal_slope_3d'] = float(np.polyfit(tt_term, proj_term, 1)[0])

                if np.isfinite(t_t2) and t_t1 < t_t2 < t_toff:
                    i_t2 = t2idx(t_t2, t_vcg)
                    if i_t2 - i_t1 >= 2:
                        proj_early = project_onto_axis(vcg[i_t1:i_t2 + 1], t_axis)
                        tt_early = t_vcg[i_t1:i_t2 + 1] - t_vcg[i_t1]
                        if np.all(np.isfinite(proj_early)):
                            rec['T_early_fall_slope_3d'] = float(np.polyfit(tt_early, proj_early, 1)[0])
                    if i_toff - i_t2 >= 2:
                        proj_late = project_onto_axis(vcg[i_t2:i_toff + 1], t_axis)
                        tt_late = t_vcg[i_t2:i_toff + 1] - t_vcg[i_t2]
                        if np.all(np.isfinite(proj_late)):
                            rec['T_late_fall_slope_3d'] = float(np.polyfit(tt_late, proj_late, 1)[0])

        # T-Formsignatur: Anteile Anstieg/Mitte(Kerbe)/Abfall an der T-Gesamtdauer
        rec['T_shape_rise_frac'] = rec['T_shape_mid_frac'] = rec['T_shape_fall_frac'] = np.nan
        if np.isfinite(t_ton) and np.isfinite(t_t1) and np.isfinite(t_t2) and np.isfinite(t_toff):
            total = t_toff - t_ton
            if total > 0 and t_ton <= t_t1 <= t_t2 <= t_toff:
                rec['T_shape_rise_frac'] = (t_t1 - t_ton) / total
                rec['T_shape_mid_frac'] = (t_t2 - t_t1) / total
                rec['T_shape_fall_frac'] = (t_toff - t_t2) / total

        records.append(rec)

    df_cycle = pd.DataFrame(records)

    feat_cols = [c for c in df_cycle.columns if c not in ('beat_id', 'RR_ms')]
    print(f"df_cycle: {len(df_cycle)} Beats, {df_cycle.shape[1]} Spalten "
          f"({len(leads)} Lead-Spalten erkannt: {leads})")
    print(f"\nFehlende Werte (%), Top 15:\n"
          f"{(df_cycle[feat_cols].isna().mean() * 100).round(1).sort_values(ascending=False).head(15)}")

    return df_cycle
