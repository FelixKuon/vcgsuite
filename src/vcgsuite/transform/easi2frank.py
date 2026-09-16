# ============================================================
#  VCG-TRANSFORMATION  –  EASI → Frank XYZ
#
#  Pipeline (siehe Diagramm):
#    [I, E, S]  →  Matrix W (3×3)  →  Quasi-orthogonale Leads
#               →  Matrix T (3×3)  →  Frank XYZ
# ============================================================

import numpy as np

#                ai       ei       si
W = np.array([[ 0.610,  0.171,  0.000],   # X'
              [ 0.354,  0.000, -1.000],   # Y'
              [ 0.869, -0.605,  0.000]])  # Z'

T = np.array([[ 1.118,  0.000, -0.109],
              [-0.051,  0.933, -0.087],
              [-1.108,  0.000,  0.772]])

# Dower-Matrix D (EASI → 8 Standard-Leads, für spätere Nutzung)
D = np.array([[ 0.156, -0.010, -0.147],
              [ 0.058,  0.060, -0.022],
              [-0.102,  0.065,  0.059],
              [-0.088,  0.108,  0.022],
              [-0.019,  0.106, -0.041],
              [ 0.061,  0.097, -0.063],
              [ 0.128,  0.074, -0.129],
              [ 0.128,  0.022, -0.159]])


def easi_to_frank_xyz(v_is: np.ndarray,
                      v_es: np.ndarray,
                      v_as: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    EASI-Elektroden → Frank XYZ via Matrizen W und T.

    Differenzsignale:
        ei = V_ES - V_IS   (E minus I)
        ai = V_AS - V_IS   (A minus I)
        si = -V_IS

    Returns
    -------
    X, Y, Z : np.ndarray  je shape (N,)
    """
    ei = v_es - v_is
    ai = v_as - v_is
    si = -v_is

    quasi_ortho = W @ np.vstack((ai, ei, si))   # 3 × N
    frank       = T @ quasi_ortho               # 3 × N
    return frank[0], frank[1], frank[2]
