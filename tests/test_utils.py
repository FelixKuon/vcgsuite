"""Unit-Tests für vcgsuite.utils.parse_subject_id — benötigen keine echten Daten."""

from vcgsuite.utils import parse_subject_id


def test_parse_subject_id_cpt_pattern():
    proband_id, subject_id = parse_subject_id("/some/path/-cpt204-2024-01-01.txt")
    assert proband_id == "cpt204"
    assert subject_id == "Scpt204"


def test_parse_subject_id_numeric_pattern():
    proband_id, subject_id = parse_subject_id("-76-2024-07-25-10-32-13.txt")
    assert proband_id == "76"
    assert subject_id == "S76"


def test_parse_subject_id_skips_non_numeric_leading_token():
    # "m100" enthält Buchstaben, matcht die "-(\\d+)-"-Gruppe nicht selbst,
    # aber re.search() findet den NÄCHSTEN rein numerischen Block im
    # Dateinamen ("-2024-") — das ist kein Fallback, sondern das reguläre
    # Verhalten von re.search() (kein Anker an den Stringanfang).
    proband_id, subject_id = parse_subject_id("-m100-2024-06-26-13-10-29.txt")
    assert proband_id == "2024"
    assert subject_id == "S2024"


def test_parse_subject_id_fallback_no_dash_number():
    # Kein "-cptNNN-" und kein "-NNN-"-Muster irgendwo im Dateinamen
    # (keine Bindestriche) -> Fallback auf Dateiname ohne Endung.
    proband_id, subject_id = parse_subject_id("signal_easi_recording.txt")
    assert proband_id == "signal_easi_recording"
    assert subject_id == "Ssignal_easi_recording"


def test_parse_subject_id_sample_data_filename():
    # Dokumentiert das tatsächliche (etwas überraschende) Verhalten für die
    # im Repo genutzte Beispieldatei: "-I-2025-4-6_6min_..." enthält den
    # ersten "-Ziffern-"-Block als "-2025-", nicht die führende ID "I".
    proband_id, subject_id = parse_subject_id(
        "-I-2025-4-6_6min_2brust_2bauchOhne_2bauchmit.txt"
    )
    assert proband_id == "2025"
    assert subject_id == "S2025"
