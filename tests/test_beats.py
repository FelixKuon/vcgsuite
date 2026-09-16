"""Beat-zu-Beat-Rotation, komplexe Amplitude, VLS (vcgsuite.beats)."""

import numpy as np

from vcgsuite.beats.complex_amplitude import compute_rpeak_ca
from vcgsuite.beats.vls import compute_vls


def test_df_r_has_expected_columns_and_length(df_r, df_annotations):
    for col in ("beat_id", "t_R_peak+", "Rpeak_r", "Rpeak_ca",
                "Rturn_r", "Rturn_ca", "RR_ms"):
        assert col in df_r.columns
    assert len(df_r) == len(df_annotations)


def test_rr_ms_is_positive_except_first_beat(df_r):
    rr = df_r.sort_values("beat_id")["RR_ms"].to_numpy()
    assert np.isnan(rr[0])
    assert np.all(rr[1:] > 0)


def test_compute_rpeak_ca_returns_expected_shape(df_kinematics, r_peaks):
    r_peak_times, _ = r_peaks
    df_beat = compute_rpeak_ca(df_kinematics, r_peak_times)
    assert len(df_beat) == len(r_peak_times)
    for col in ("beat_id", "t_rpeak", "r_rpeak", "ca_rpeak", "RR_ms"):
        assert col in df_beat.columns
    assert np.isnan(df_beat["RR_ms"].iloc[0])


def test_compute_vls_no_side_effect_on_input(df_kinematics, r_peaks):
    r_peak_times, _ = r_peaks
    cols_before = set(df_kinematics.columns)
    vls = compute_vls(df_kinematics, r_peak_times)

    assert len(vls) == len(df_kinematics)
    # Behobener Seiteneffekt aus dem Original: compute_vls() darf df_analysis
    # NICHT mehr selbst mutieren, die Zuweisung obliegt dem Aufrufer.
    assert set(df_kinematics.columns) == cols_before
    assert "VLS" not in df_kinematics.columns
