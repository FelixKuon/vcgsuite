# ══════════════════════════════════════════════════════════════════════════
#  KONSTANTEN  –  Kinematik-Features & hierarchische Beat-Annotation
# ══════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'rel_t_ms',
    'Curvature_raw',  'A_abs_raw',       'V_abs_raw',    'Torsion_abs_raw',
    'r_raw',          'dr_dt_raw',        'dtheta_dt_raw','dphi_dt_raw',
    'Curvature_sm3',  'A_abs_sm3',        'V_abs_sm3',    'r_sm3',
    'dCurvature_dt',  'dA_abs_dt',        'dV_abs_dt',
    'Curvature_lmax', 'A_abs_lmax',       'V_abs_lmax',   'r_lmax',
    'Curvature_lstd', 'A_abs_lstd',
    'Curvature_rank', 'A_abs_rank',       'V_abs_rank',   'r_rank',
    'CurvRadius_raw', 'CurvRadius_sm3',
    'r_rel',          'r_normalized',     'r_slope',
]


# ─────────────────────────────────────────────────────────────────────────
#  HIERARCHICAL_WINDOWS
#
#  Ersetzt die alte, nicht mehr synchron gehaltene WINDOWS_LOCAL-Konstante
#  aus const_annontation.py — diese wich von den tatsächlich verwendeten
#  Fenstergrenzen ab und wurde deshalb konsolidiert.
#
#  Diese Werte wurden direkt aus den `detect_consensus(...)`-Aufrufen in
#  annontate_beat_hierarchical.py extrahiert (nicht aus der verwaisten
#  WINDOWS_LOCAL-Konstante, die davon abwich).
#
#  Jeder Eintrag beschreibt für einen Marker das Suchfenster, in dem
#  `detect_consensus()` nach ihm sucht:
#    anchor : Name des Referenzmarkers, relativ zu dem gesucht wird.
#             "R_peak" bezeichnet den R_peak+ Zeitpunkt selbst (Eingabe-
#             parameter r_peak_t von annotate_beat_hierarchical()).
#    lo_ms, hi_ms : Suchfenstergrenzen relativ zum Anker-Zeitpunkt [ms].
# ─────────────────────────────────────────────────────────────────────────
HIERARCHICAL_WINDOWS = {
    "R_turn":  {"anchor": "R_peak",  "lo_ms":   3, "hi_ms":  28},
    "S_on":    {"anchor": "R_peak",  "lo_ms":  20, "hi_ms": 100},
    "S_off":   {"anchor": "R_peak",  "lo_ms":  40, "hi_ms": 130},
    "Q_on":    {"anchor": "R_peak",  "lo_ms": -80, "hi_ms": -20},
    "P_peak":  {"anchor": "R_peak",  "lo_ms": -145, "hi_ms": -95},
    "P_on":    {"anchor": "P_peak",  "lo_ms": -90, "hi_ms": -10},
    "P_off":   {"anchor": "P_peak",  "lo_ms":   5, "hi_ms":  95},
    "T_on":    {"anchor": "S_off",   "lo_ms":  70, "hi_ms": 140},
    "T_turn1": {"anchor": "T_on",    "lo_ms":  80, "hi_ms": 140},
    "T_turn2": {"anchor": "T_turn1", "lo_ms":  60, "hi_ms": 130},
    "T_off":   {"anchor": "T_turn2", "lo_ms":  60, "hi_ms": 140},
}

# Q_off wird nicht per Konsens-Fenster bestimmt, sondern als fixer Offset
# relativ zum R_peak+ Zeitpunkt berechnet (Q_off = R_peak+ + Q_OFF_OFFSET_S),
# siehe annotation/hierarchical.py.
Q_OFF_OFFSET_S = -0.020
