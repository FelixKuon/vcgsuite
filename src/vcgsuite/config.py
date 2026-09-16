# ============================================================
#  KONFIGURATION  –  zentrale, nicht-personenbezogene Parameter
#
#  WICHTIG:
#  Der ursprüngliche Notebook-Export (Config.py) enthielt hier zusätzlich
#  hartcodierte, absolute Dateipfade zu individuellen Proband:innen-
#  Rohdaten (EASI_FILEPATH, ECG12_FILEPATH) auf der Festplatte des Autors.
#  Das gehört nicht in eine wiederverwendbare/öffentliche Bibliothek und
#  wurde deshalb ersatzlos entfernt. Stattdessen übergibt man den Pfad zur
#  Datendatei explizit als Argument beim Aufruf von
#  `vcgsuite.io.easi.load_easi()` bzw. `vcgsuite.io.ecg12.load_ecg12()`.
#
#  Die *_START_SEC / *_DURATION-Werte unten sind lediglich generische
#  Default-Werte (0 = "vom Anfang" / None = "bis zum Ende") und werden als
#  Funktionsparameter mit Default weitergereicht – nicht als globale
#  Konstanten, die versehentlich personenbezogene Werte tragen könnten.
# ============================================================

# ── Abtastrate
FS = 250  # Hz

# ── EASI-Datei (Format, keine Pfade)
EASI_SEP = ";"
EASI_COL_I = "R"    # Elektrode I
EASI_COL_E = "M"    # Elektrode E (Stern)
EASI_COL_S = "L"    # Elektrode S
EASI_START_SEC = 0      # Default-Schnittstart in Sekunden  (0 = von Anfang)
EASI_DURATION = None   # Default-Dauer in Sekunden         (None = bis zum Ende)

# ── 12-Kanal-Datei (Format, keine Pfade)
ECG12_SEP = ";"
ECG12_START_SEC = 0      # Default-Schnittstart in Sekunden
ECG12_DURATION = None   # Default-Dauer in Sekunden         (None = bis zum Ende)

# ── Leads (nur die 8 linear-unabhängigen)
LEAD_ORDER = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# ── Transformationsmethode  →  wird erst beim Aufruf der
#    Transformationsfunktion übergeben, nicht hier global gesetzt.
#    Verfügbare Optionen: "IDT" | "KORS" | "QLSV" | "PLSV"
