"""P-/QRS-/T-Loop-Features + Merge (vcgsuite.features.loop)."""

import numpy as np


def test_df_p_has_beat_id_and_is_not_empty(loop_features):
    df_p = loop_features["df_p"]
    assert "beat_id" in df_p.columns
    assert len(df_p) > 0


def test_df_qrs_has_beat_id_and_is_not_empty(loop_features):
    df_qrs = loop_features["df_qrs"]
    assert "beat_id" in df_qrs.columns
    assert len(df_qrs) > 0


def test_df_t_has_beat_id_and_is_not_empty(loop_features):
    df_t = loop_features["df_t"]
    assert "beat_id" in df_t.columns
    assert len(df_t) > 0


def test_merged_df_full_covers_all_beats(loop_features, df_annotations):
    df_full = loop_features["df_full"]
    # merge_loop_features nutzt outer joins auf 'beat_id' -> mindestens so
    # viele Zeilen wie die größte Einzeltabelle, i.d.R. = Anzahl Beats.
    assert len(df_full) >= len(df_annotations) * 0.9
    assert df_full["beat_id"].is_monotonic_increasing


def test_qrs_area_is_non_negative_where_present(loop_features):
    df_qrs = loop_features["df_qrs"]
    area = df_qrs["QRS_area"].dropna().to_numpy()
    assert len(area) > 0, "Keine einzige QRS_area berechnet"
    assert np.all(area >= 0)
