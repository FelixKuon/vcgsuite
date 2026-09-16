"""
vcgsuite.hrv – Herzfrequenzvariabilität (HRV) und EDR (ECG-Derived Respiration).

Migriert aus HRV_helpers.py, HRV_prep.py, HRV_Analysis.py.
Das Dashboard (HRV_Dashboard.py) ist als reine Visualisierung in
`vcgsuite.viz.hrv_dashboard` migriert.
"""

from .helpers import (
    PNS_REF, PNS_STD, SNS_REF, SNS_STD, BANDS,
    compute_sd1_sd2, compute_baevsky_stress_index,
    compute_psd_welch, band_power,
    compute_hrv_full, compute_rsa_amplitude, compute_rsa_hf_power,
)
from .prep import build_rr_dataframe
from .analysis import (
    FS_INTERP, EDR_CONFIG, RESP_COL,
    extract_edr_signal, compute_rsa_coherence,
)

__all__ = [
    "PNS_REF", "PNS_STD", "SNS_REF", "SNS_STD", "BANDS",
    "compute_sd1_sd2", "compute_baevsky_stress_index",
    "compute_psd_welch", "band_power",
    "compute_hrv_full", "compute_rsa_amplitude", "compute_rsa_hf_power",
    "build_rr_dataframe",
    "FS_INTERP", "EDR_CONFIG", "RESP_COL",
    "extract_edr_signal", "compute_rsa_coherence",
]
