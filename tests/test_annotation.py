"""Hierarchische Beat-Annotation (vcgsuite.annotation)."""

import numpy as np

import vcgsuite.annotation.features as feat_mod
from vcgsuite.annotation.annotate import ALL_MARKERS
from vcgsuite.annotation.features import detect_consensus, detect_in_window


def test_annotations_row_per_beat(df_annotations, r_peaks):
    r_peak_times, _ = r_peaks
    assert len(df_annotations) == len(r_peak_times)


def test_annotations_have_all_marker_columns(df_annotations):
    assert "beat_id" in df_annotations.columns
    assert "t_R_peak+" in df_annotations.columns
    for marker in ALL_MARKERS:
        assert f"t_{marker}" in df_annotations.columns


def test_most_beats_have_qrs_markers_detected(df_annotations):
    # R_turn/S_on/S_off sind die "einfachsten" Marker (nahe am R-Peak) und
    # sollten auf einer realen Aufnahme in der klaren Mehrheit der Beats
    # gefunden werden. Kein Anspruch auf 100 % (physiologische Varianz,
    # Rand-Beats am Anfang/Ende des Fensters).
    n = len(df_annotations)
    for marker in ("R_turn", "S_on", "S_off"):
        found = df_annotations[f"t_{marker}"].notna().sum()
        assert found / n > 0.5, f"Zu wenige '{marker}' gefunden: {found}/{n}"


# ══════════════════════════════════════════════════════════════════════════
#  detect_consensus(): Performance-Fix (extract_features() einmal statt
#  einmal pro Strategie berechnet) -- Korrektheit + die Redundanz selbst
#  als Regressionstest, damit sie nicht versehentlich wieder eingefuehrt wird.
# ══════════════════════════════════════════════════════════════════════════

_STRATEGIES = [("Curvature_lmax", "MAX"), ("A_abs_lmax", "MAX"), ("r_lmax", "MAX")]
_LO_MS, _HI_MS = -50.0, 50.0


def _mid_r_peak_time(r_peaks):
    r_peak_times, _ = r_peaks
    return float(r_peak_times[len(r_peak_times) // 2])


def test_detect_consensus_matches_manual_median(df_kinematics, r_peaks):
    anchor_t = _mid_r_peak_time(r_peaks)

    manual_votes = [
        detect_in_window(df_kinematics, anchor_t, _LO_MS, _HI_MS, fcol, op)
        for fcol, op in _STRATEGIES
    ]
    manual_votes = [v for v in manual_votes if not np.isnan(v)]

    got_t, got_spread = detect_consensus(df_kinematics, anchor_t, _LO_MS, _HI_MS, _STRATEGIES)

    if not manual_votes:
        assert np.isnan(got_t)
    else:
        expected_t = float(np.median(manual_votes))
        expected_spread = (max(manual_votes) - min(manual_votes)) * 1000 if len(manual_votes) > 1 else 0.0
        assert got_t == expected_t
        assert got_spread == expected_spread


def test_detect_consensus_calls_extract_features_once_per_window(df_kinematics, r_peaks, monkeypatch):
    """Regressionstest gegen die urspruengliche Performance-Redundanz:
    detect_consensus() darf extract_features() nur EINMAL fuer alle
    Strategien zusammen aufrufen (sie teilen sich dasselbe Fenster), nicht
    einmal pro Strategie."""
    anchor_t = _mid_r_peak_time(r_peaks)

    call_count = {"n": 0}
    orig = feat_mod.extract_features

    def _counting(*args, **kwargs):
        call_count["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(feat_mod, "extract_features", _counting)
    detect_consensus(df_kinematics, anchor_t, _LO_MS, _HI_MS, _STRATEGIES)

    assert call_count["n"] == 1, (
        f"extract_features() wurde {call_count['n']}x aufgerufen fuer "
        f"{len(_STRATEGIES)} Strategien mit demselben Fenster -- erwartet 1x."
    )
