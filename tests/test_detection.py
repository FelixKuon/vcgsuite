"""R-Peak- und R-Turn-Detektion (vcgsuite.detection)."""

import numpy as np


def test_r_peaks_found_in_plausible_physiological_range(r_peaks, df_kinematics):
    r_peak_times, r_peak_values = r_peaks
    duration_s = df_kinematics["Time"].iloc[-1] - df_kinematics["Time"].iloc[0]

    assert len(r_peak_times) == len(r_peak_values)
    # Großzügiger Bereich (30-220 bpm), um auf jeder halbwegs plausiblen
    # Ruhe-/Belastungs-EKG-Aufnahme robust zu bleiben, ohne die Detektion
    # selbst zu überprüfen (das ist kein Algorithmus-Genauigkeitstest).
    bpm = len(r_peak_times) / (duration_s / 60.0)
    assert 30 <= bpm <= 220, f"Unrealistische Herzfrequenz: {bpm:.1f} bpm"


def test_r_peak_times_are_monotonic(r_peaks):
    r_peak_times, _ = r_peaks
    assert np.all(np.diff(r_peak_times) > 0)


def test_r_turn_follows_r_peak(r_peaks, r_turn):
    r_peak_times, _ = r_peaks
    r_turn_times, _ = r_turn
    assert len(r_turn_times) == len(r_peak_times)

    valid = ~np.isnan(r_turn_times)
    assert valid.sum() > 0, "Kein einziger R_turn gefunden"
    # R_turn liegt per Definition 5-13 ms nach R_peak (Default-Suchfenster).
    delta_ms = (r_turn_times[valid] - r_peak_times[valid]) * 1000
    assert np.all(delta_ms >= 4.9)
    assert np.all(delta_ms <= 13.1)
