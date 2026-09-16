# ============================================================
#  VCG-TRANSFORMATION  –  12-Kanal ECG → Frank XYZ
#  Eingangs-Reihenfolge: [I, II, V1, V2, V3, V4, V5, V6]
#  Ausgabe:  Zeile 0 = X  |  Zeile 1 = Y  |  Zeile 2 = Z
# ============================================================

import numpy as np

from ..config import LEAD_ORDER

MATRICES = {

    # Edenbrandt & Pahlm, J Electrocardiol 1988;21:361-367
    "IDT": np.array([
        [ 0.156, -0.010, -0.172, -0.074, -0.122,  0.231,  0.239,  0.194],  # X
        [ 0.227,  0.887, -0.057, -0.019,  0.106,  0.022, -0.041,  0.048],  # Y
        [-0.022, -0.102,  0.229,  0.310,  0.246, -0.063, -0.055, -0.108],  # Z
    ]),

    # Kors et al., Eur Heart J 1990;11:1083-1092
    "KORS": np.array([
        [ 0.380,  0.070, -0.130, -0.050, -0.010,  0.140,  0.060,  0.540],  # X
        [-0.070,  0.930, -0.060, -0.020,  0.050, -0.060, -0.170,  0.130],  # Y
        [-0.110, -0.230,  0.430,  0.060,  0.140,  0.200,  0.110, -0.310],  # Z
    ]),

    # Guillem et al., Computers in Cardiology 2006  –  QRS-optimiert
    "QLSV": np.array([
        [ 0.370,  0.154, -0.266, -0.027, -0.065,  0.131,  0.203,  0.220],  # X
        [ 0.131,  0.717, -0.088, -0.088, -0.003, -0.042, -0.047,  0.067],  # Y
        [-0.184, -0.114,  0.319,  0.198,  0.167,  0.099,  0.009, -0.060],  # Z
    ]),

    # Guillem et al., Computers in Cardiology 2006  –  P-Wellen-optimiert
    "PLSV": np.array([
        [ 0.199,  0.018, -0.147, -0.058, -0.037,  0.139,  0.232,  0.226],  # X
        [ 0.164,  0.503, -0.023, -0.085, -0.003, -0.033, -0.060,  0.104],  # Y
        [-0.085, -0.130,  0.184,  0.163,  0.190,  0.119,  0.023, -0.043],  # Z
    ]),
}


def ecg12_to_frank_xyz(ecg_leads: dict,
                       method: str = "IDT",
                       lead_order: list[str] = LEAD_ORDER
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    8 unabhängige ECG-Leads → Frank XYZ via Inversmatrix.

    Parameters
    ----------
    ecg_leads  : dict  Keys nach lead_order, je np.ndarray shape (N,)
    method     : str   "IDT" | "KORS" | "QLSV" | "PLSV"

    Returns
    -------
    X, Y, Z : np.ndarray  je shape (N,)
    """
    if method not in MATRICES:
        raise ValueError(f"Unbekannte Methode '{method}'. "
                         f"Wähle aus: {list(MATRICES.keys())}")
    M   = MATRICES[method]
    E   = np.vstack([ecg_leads[lead] for lead in lead_order])  # 8 × N
    VCG = M @ E                                                 # 3 × N
    print(f"✅ 12-Kanal → Frank XYZ  (Methode: {method})")
    return VCG[0], VCG[1], VCG[2]
