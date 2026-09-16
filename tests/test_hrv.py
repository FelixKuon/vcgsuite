"""HRV-Aufbereitung und Kennwerte (vcgsuite.hrv)."""

import numpy as np

from vcgsuite.hrv.prep import build_rr_dataframe
from vcgsuite.hrv.helpers import compute_hrv_full


def test_build_rr_dataframe_has_expected_columns(df_r):
    df_rr = build_rr_dataframe(df_r)
    for col in ("Time", "RR_interval", "Rpeak_ca", "Rpeak_r", "RR_derived"):
        assert col in df_rr.columns
    assert len(df_rr) == len(df_r)
    assert not df_rr["RR_interval"].isna().any(), "RR_ms wird interpoliert/ge-ffill/bfill"


def test_compute_hrv_full_returns_plausible_values(df_r):
    df_rr = build_rr_dataframe(df_r)
    hrv = compute_hrv_full(df_rr["RR_interval"].to_numpy())

    for key in ("mean_rr", "mean_hr", "sdnn", "rmssd", "pnn50",
                "SD1", "SD2", "SI", "PNS_index", "SNS_index"):
        assert key in hrv

    assert 30 <= hrv["mean_hr"] <= 220
    assert hrv["sdnn"] >= 0
    assert hrv["rmssd"] >= 0
    assert np.isfinite(hrv["SI"])
