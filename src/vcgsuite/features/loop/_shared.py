# ══════════════════════════════════════════════════════════════════════════
#  GEMEINSAME LOOP-HELPER  (P-/QRS-/T-Loop-Features)
#
#  Konsolidiert acht Hilfsfunktionen, die im ursprünglichen Notebook-Export
#  in P-loop_features.py, QRS-loop_features.py UND T-loop_features.py
#  (dort sogar zweifach) nahezu wortgleich dupliziert waren:
#    safe(), t2idx(), loop_area_3d(), loop_svd(), loop_dipol_norm(),
#    loop_asym(), loop_roundness_fwhm(), angle_between()
#
#  Als Quelle diente P-loop_features.py — die dort ausformulierteste und am
#  ausführlichsten kommentierte Variante (QRS-/T-loop_features.py enthielten
#  dieselbe Logik nur kompakter/einzeilig geschrieben, ohne inhaltliche
#  Abweichung).
#
#  WICHTIGE ÄNDERUNG ggü. Original:
#  Im Original war `fs = 250.0` (und daraus `dt = 1.0/fs`) in allen drei
#  Dateien fest einprogrammiert und wurde implizit über Modul-globale
#  Variablen an loop_dipol_norm()/t2idx() durchgereicht. Das stand im
#  Widerspruch dazu, dass df_analysis.attrs["fs"] (siehe pipeline.py) die
#  tatsächliche Abtastrate der jeweiligen Aufnahme trägt. Hier werden `dt`
#  (für loop_dipol_norm) und `t_vcg` (für t2idx) deshalb als explizite
#  Parameter verlangt, nicht mehr als Closure-/Modul-Globale — siehe
#  `resolve_fs()` unten, die von den aufrufenden Feature-Modulen
#  (p_wave.py, qrs_complex.py, t_wave.py) genutzt wird, um `fs` aus
#  `df_analysis.attrs.get("fs", vcgsuite.config.FS)` abzuleiten, falls kein
#  expliziter fs-Parameter übergeben wurde.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import FS as _DEFAULT_FS


def resolve_fs(df_analysis: pd.DataFrame, fs: float | None = None) -> float:
    """
    Ermittelt die Abtastrate nach Priorität:
      1. explizit übergebenes `fs`
      2. `df_analysis.attrs["fs"]`  (von `vcgsuite.pipeline.load_and_process` gesetzt)
      3. `vcgsuite.config.FS`  (globaler Fallback-Default)

    Behebt hartcodiertes fs=250.0 aus dem Original, das im Widerspruch zu
    df_analysis.attrs['fs'] stand.
    """
    if fs is not None:
        return float(fs)
    return float(df_analysis.attrs.get("fs", _DEFAULT_FS))


def safe(row, col):
    """Robuster Zugriff auf row[col] → float oder NaN (statt Exception)."""
    try:
        v = row[col]
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan


def t2idx(t_sec: float, t_vcg: np.ndarray) -> int:
    """Zeitstempel [s] → nächster Sample-Index via Zeitachse `t_vcg`."""
    return int(np.argmin(np.abs(t_vcg - t_sec)))


def loop_area_3d(seg: np.ndarray) -> float:
    """Fläche der 3D-Loop-Trajektorie via Kreuzprodukt-Summe (Stokes)."""
    if len(seg) < 3:
        return np.nan
    crosses = np.cross(seg[:-1], np.diff(seg, axis=0))
    return 0.5 * np.linalg.norm(crosses.sum(axis=0))


def loop_svd(seg: np.ndarray):
    """SVD der um den Schwerpunkt zentrierten Loop-Trajektorie → (S, Vt)."""
    centered = seg - seg.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return S, Vt


def loop_dipol_norm(seg: np.ndarray, dt: float) -> float:
    """
    Norm des zeitlich integrierten Dipolvektors ‖∫v dt‖.

    `dt` (Abtastintervall in Sekunden) muss explizit übergeben werden
    (siehe Modul-Docstring: ersetzt das hartcodierte `dt = 1/250.0` aus
    dem Original).
    """
    integral = np.trapz(seg, dx=dt, axis=0)
    return np.linalg.norm(integral)


def loop_asym(seg: np.ndarray) -> float:
    """(A_late - A_early) / (A_late + A_early) → +1 = spät, -1 = früh."""
    h = len(seg) // 2
    a_e = loop_area_3d(seg[:h])
    a_l = loop_area_3d(seg[h:])
    denom = a_e + a_l
    return (a_l - a_e) / denom if (denom and np.isfinite(denom) and denom > 0) else np.nan


def loop_roundness_fwhm(seg: np.ndarray) -> float:
    """FWHM der Geschwindigkeitsnorm, normalisiert auf Segmentlänge."""
    speed = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    peak = speed.max()
    if peak == 0:
        return np.nan
    fwhm_samples = (speed >= peak / 2).sum()
    return fwhm_samples / len(speed)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Winkel in Grad zwischen zwei Vektoren."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_a))


# ══════════════════════════════════════════════════════════════════════════
#  Zusätzliche Helper für Zyklus-/Zwischen-Wellen-Features (features/loop/cycle.py)
# ══════════════════════════════════════════════════════════════════════════

def loop_tortuosity(seg: np.ndarray) -> float:
    """Pfadlänge / Sehnenlänge (chord) einer 3D-Trajektorie — Maß für
    geometrische Komplexität/Umwegigkeit (1.0 = perfekt gerade, höhere Werte
    = gewundenerer Pfad, z. B. durch QRS-Fragmentierung/Kerbung)."""
    seg = np.asarray(seg, dtype=float)
    if len(seg) < 2:
        return np.nan
    path_len = np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()
    chord_len = np.linalg.norm(seg[-1] - seg[0])
    return float(path_len / chord_len) if chord_len > 1e-12 else np.nan


def count_local_extrema(sig: np.ndarray) -> int:
    """Anzahl lokaler Extrema (Vorzeichenwechsel der ersten Ableitung) in
    einem 1D-Signal — Proxy für "Kerbung"/Fragmentierung einer Wellenform.
    Nullen in der Ableitung (exakt konstante Nachbarwerte) werden ignoriert,
    nicht als eigener Vorzeichenwechsel gezählt."""
    sig = np.asarray(sig, dtype=float)
    if len(sig) < 3:
        return 0
    d = np.diff(sig)
    d = d[d != 0]
    if len(d) < 2:
        return 0
    signs = np.sign(d)
    return int(np.sum(signs[:-1] != signs[1:]))


def project_onto_axis(seg: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Projiziert eine 3D-Trajektorie (Shape (n, 3)) auf einen (nicht
    notwendig normierten) Achsenvektor → 1D-Signal der Projektionslängen.
    Gibt ein NaN-Array zurück, falls `axis` degeneriert (Norm ≈ 0 oder
    nicht-endliche Werte) ist."""
    seg = np.asarray(seg, dtype=float)
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12 or not np.all(np.isfinite(axis)):
        return np.full(len(seg), np.nan)
    axis_hat = axis / n
    return seg @ axis_hat
