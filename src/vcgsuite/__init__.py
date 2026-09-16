"""
vcgsuite – Python-Bibliothek zur EKG/VCG-Signalverarbeitung.

Migriert aus einer Sammlung von Notebook-Export-Skripten (siehe
Modul-Docstrings der Untermodule für Herkunft/Original-Quelle).

Re-exportiert die wichtigsten Einstiegspunkte der einzelnen Untermodule,
sodass z. B. `from vcgsuite import load_easi` funktioniert.
"""

from .utils import parse_subject_id
from .io import load_easi, load_ecg12
from .preprocessing import filter_pipeline
from .transform import easi_to_frank_xyz, ecg12_to_frank_xyz
from .kinematics import compute_vcg_kinematics
from .detection import detect_r_peaks, detect_r_turn
from .annotation import annotate_all_beats
from .beats import compute_beat_rotation, compute_rpeak_ca, compute_vls

# ── Teil 2: Features (P-/QRS-/T-Loop), HRV, Pipeline ──
from .features import merge_loop_features
from .hrv import compute_hrv_full, build_rr_dataframe
from .pipeline import load_and_process

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "parse_subject_id",
    "load_easi",
    "load_ecg12",
    "filter_pipeline",
    "easi_to_frank_xyz",
    "ecg12_to_frank_xyz",
    "compute_vcg_kinematics",
    "detect_r_peaks",
    "detect_r_turn",
    "annotate_all_beats",
    "compute_beat_rotation",
    "compute_rpeak_ca",
    "compute_vls",
    "merge_loop_features",
    "compute_hrv_full",
    "build_rr_dataframe",
    "load_and_process",
]
