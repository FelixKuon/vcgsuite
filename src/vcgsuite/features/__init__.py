"""
vcgsuite.features – Beat-für-Beat Feature-Extraktion (P-/QRS-/T-Loop).

Migriert aus P-loop_features.py, QRS-loop_features.py, T-loop_features.py
und merge_features.py (siehe `vcgsuite.features.loop`-Untermodule).
"""

from .loop import (
    compute_p_wave_features,
    build_vagus_features, compute_t_wave_features,
    compute_qrs_features,
    merge_loop_features,
)

__all__ = [
    "compute_p_wave_features",
    "build_vagus_features", "compute_t_wave_features",
    "compute_qrs_features",
    "merge_loop_features",
]
