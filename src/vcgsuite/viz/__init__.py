"""
vcgsuite.viz – Reine Plotly-Visualisierungen (keine Berechnungslogik).

Migriert aus Plot_VCG_annotated.py, plot_r_peak.py, plot_loop_features.py
und HRV_Dashboard.py.
"""

from .vcg_annotated import (
    GT_MARKERS, QRS_MARKERS, MARKER_SEGMENT, SEGMENT_COLORS,
    MARKER_SIZES, SEGMENT_EDGES,
    to_df_markers, to_df_segments, build_plot_data,
    select_time_window, plot_signal_segments_and_markers, plot_vcg_3d,
)
from .r_peaks import plot_r_peak_detection
from .loop_features import (
    PANELS_P, PANELS_QRS, PANELS_T,
    plot_loop_feature_timeseries,
    plot_p_loop_features, plot_qrs_loop_features, plot_t_loop_features,
)
from .hrv_dashboard import (
    plot_kubios_ans_balance, build_edr_dashboard, build_edr_summary_html,
)

__all__ = [
    "GT_MARKERS", "QRS_MARKERS", "MARKER_SEGMENT", "SEGMENT_COLORS",
    "MARKER_SIZES", "SEGMENT_EDGES",
    "to_df_markers", "to_df_segments", "build_plot_data",
    "select_time_window", "plot_signal_segments_and_markers", "plot_vcg_3d",
    "plot_r_peak_detection",
    "PANELS_P", "PANELS_QRS", "PANELS_T",
    "plot_loop_feature_timeseries",
    "plot_p_loop_features", "plot_qrs_loop_features", "plot_t_loop_features",
    "plot_kubios_ans_balance", "build_edr_dashboard", "build_edr_summary_html",
]
