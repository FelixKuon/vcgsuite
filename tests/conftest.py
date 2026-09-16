# ══════════════════════════════════════════════════════════════════════════
#  PYTEST FIXTURES – gemeinsame Test-Infrastruktur
#
#  Die End-to-End-Fixtures nutzen eine echte EASI-Beispielaufnahme, die NICHT
#  im Repository versioniert ist (siehe .gitignore: sample_data/*.txt bleibt
#  lokal beim Nutzer). Ist die Datei nicht vorhanden (z. B. in CI auf GitHub),
#  werden alle davon abhängigen Tests automatisch übersprungen
#  (pytest.skip), statt fehlzuschlagen.
#
#  Nur ein kurzes Zeitfenster (`DURATION_S`) wird verarbeitet, damit die
#  Test-Suite schnell bleibt (ZapLine/DSS auf 6 Minuten Rohdaten wäre für
#  einen Unit-Test unnötig langsam).
# ══════════════════════════════════════════════════════════════════════════

from pathlib import Path

import pytest

import vcgsuite as ecg
from vcgsuite.kinematics.frenet_serret import compute_vcg_kinematics
from vcgsuite.detection.r_peaks import detect_r_peaks
from vcgsuite.detection.r_turn import detect_r_turn
from vcgsuite.annotation.annotate import annotate_all_beats
from vcgsuite.beats.rotation import compute_beat_rotation
from vcgsuite.features.loop.p_wave import compute_p_wave_features
from vcgsuite.features.loop.t_wave import build_vagus_features, compute_t_wave_features
from vcgsuite.features.loop.qrs_complex import compute_qrs_features
from vcgsuite.features.loop.merge import merge_loop_features

# Repo-Layout: <ECG_VCG_Suite>/vcgsuite/tests/conftest.py
#           -> <ECG_VCG_Suite>/sample_data/<file>.txt
SAMPLE_DATA_DIR = Path(__file__).resolve().parents[2] / "sample_data"
SAMPLE_EASI_FILE = SAMPLE_DATA_DIR / "-I-2025-4-6_6min_2brust_2bauchOhne_2bauchmit.txt"

DURATION_S = 90.0  # nur die ersten 90 s verarbeiten (Tests bleiben schnell)


@pytest.fixture(scope="session")
def sample_easi_path() -> Path:
    if not SAMPLE_EASI_FILE.exists():
        pytest.skip(
            f"Beispieldaten nicht gefunden ({SAMPLE_EASI_FILE}). "
            "End-to-End-Tests werden übersprungen (z. B. in CI erwartet, "
            "da Rohdaten nicht versioniert werden)."
        )
    return SAMPLE_EASI_FILE


@pytest.fixture(scope="session")
def df_analysis(sample_easi_path: Path):
    """df_analysis (Time, X, Y, Z, V_IS, V_ES, V_AS) nach Laden+Filtern+VCG-Transform."""
    return ecg.load_and_process(
        str(sample_easi_path),
        mode="easi",
        duration=DURATION_S,
    )


@pytest.fixture(scope="session")
def df_kinematics(df_analysis):
    """df_analysis ergänzt um V/A/Jerk/Curvature/Torsion/r/theta/phi."""
    df, dt = compute_vcg_kinematics(df_analysis.copy())
    df.attrs["fs"] = df_analysis.attrs["fs"]
    return df


@pytest.fixture(scope="session")
def r_peaks(df_kinematics):
    return detect_r_peaks(df_kinematics)


@pytest.fixture(scope="session")
def r_turn(df_kinematics, r_peaks):
    r_peak_times, _ = r_peaks
    return detect_r_turn(df_kinematics, r_peak_times)


@pytest.fixture(scope="session")
def df_annotations(df_kinematics, r_peaks, r_turn):
    r_peak_times, _ = r_peaks
    r_turn_times, _ = r_turn
    return annotate_all_beats(df_kinematics, r_peak_times, r_turn_times)


@pytest.fixture(scope="session")
def df_r(df_annotations, df_kinematics):
    return compute_beat_rotation(df_annotations, df_kinematics)


@pytest.fixture(scope="session")
def loop_features(df_kinematics, df_annotations, df_r):
    """Baut df_p/df_vagus/df_qrs/df_t/df_full in der vorgeschriebenen Reihenfolge."""
    df_p = compute_p_wave_features(df_kinematics, df_annotations, df_r)
    df_vagus = build_vagus_features(df_kinematics, df_annotations)
    df_qrs = compute_qrs_features(df_kinematics, df_annotations, df_vagus, df_p, df_r)
    df_t = compute_t_wave_features(df_kinematics, df_annotations, df_r, df_p, df_vagus=df_vagus)
    df_full = merge_loop_features(df_p, df_qrs, df_t, df_r)
    return {"df_p": df_p, "df_vagus": df_vagus, "df_qrs": df_qrs, "df_t": df_t, "df_full": df_full}
