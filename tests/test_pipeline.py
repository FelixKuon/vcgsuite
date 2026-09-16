"""End-to-End: Laden -> Filtern -> VCG-Transformation (vcgsuite.pipeline.load_and_process)."""

import numpy as np


def test_df_analysis_has_expected_columns(df_analysis):
    for col in ("Time", "X", "Y", "Z", "V_IS", "V_ES", "V_AS"):
        assert col in df_analysis.columns


def test_df_analysis_time_is_monotonic_and_starts_at_zero(df_analysis):
    t = df_analysis["Time"].to_numpy()
    assert t[0] == 0.0
    assert np.all(np.diff(t) > 0)


def test_df_analysis_duration_matches_requested_window(df_analysis):
    # duration=90s wurde in der Fixture angefragt; die tatsächliche Länge
    # darf wegen Sample-Rundung leicht abweichen, muss aber nahe dran sein.
    fs = df_analysis.attrs["fs"]
    expected_samples = int(90.0 * fs)
    assert abs(len(df_analysis) - expected_samples) <= 1


def test_df_analysis_attrs_are_set(df_analysis):
    assert df_analysis.attrs["mode"] == "easi"
    assert df_analysis.attrs["fs"] > 0
    assert "proband_id" in df_analysis.attrs
    assert "subject_id" in df_analysis.attrs


def test_df_analysis_signals_are_finite(df_analysis):
    for col in ("X", "Y", "Z"):
        assert np.all(np.isfinite(df_analysis[col].to_numpy()))
