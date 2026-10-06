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


def test_theta_qt_tp_is_valid_angle_and_mostly_available(loop_features):
    th = loop_features["df_t"]["theta_QT_tp"]
    assert "theta_QT_tp" in loop_features["df_full"].columns
    v = th.dropna().to_numpy()
    assert len(v) > 0.5 * len(th), "TP-Grundlinie fehlt bei mehr als der Haelfte der Beats"
    assert np.all((v >= 0) & (v <= 180))


def test_theta_qt_tp_is_invariant_to_dc_offset(df_kinematics, df_annotations):
    """Der Sinn der Variante: ein konstanter Versatz im Signal darf den Winkel nicht aendern,
    `theta_QT_deg` (ab Hochpass-Nullpunkt) dagegen schon."""
    from vcgsuite.features.loop.t_wave import build_vagus_features
    shifted = df_kinematics.copy()
    shifted[["X", "Y", "Z"]] += np.array([0.3, -0.2, 0.5])
    a = build_vagus_features(df_kinematics, df_annotations)
    b = build_vagus_features(shifted, df_annotations)
    both = a.theta_QT_tp.notna() & b.theta_QT_tp.notna()
    assert both.sum() > 10
    assert np.allclose(a.theta_QT_tp[both], b.theta_QT_tp[both], atol=1e-6)
    assert not np.allclose(a.theta_QT_deg.dropna(), b.theta_QT_deg.dropna(), atol=1e-3)
