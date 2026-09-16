
# VCG Analysis Pipeline

Modulare Python-Pipeline zur Berechnung des Vektorkardiogramms (VCG / Frank XYZ)
aus EASI- oder Standard-12-Kanal-EKG-Daten.

> **Migrationshinweis:** Dieses Dokument stammt ursprünglich aus dem
> Notebook-Export und wurde beim Umbau in das `vcgsuite`-Paket aktualisiert.
> Der `zapline_auto()`/`zap_auto`-Mechanismus, der hier früher als
> Pipeline-Schritt 4 und als Parameter (`zap_auto`, `zap_f_min`, `zap_sigma`)
> dokumentiert war, wurde **entfernt** — er war nie implementiert (weder im
> `.py`-Export noch in den Original-Notebooks), nur in dieser Doku
> beschrieben. Die tatsächliche Filterpipeline besteht aus genau zwei
> Schritten: ZapLine + FIR-Bandpass.

---

## Überblick

```
Rohdaten (.txt / .csv)
        │
        ▼
┌───────────────┐     ┌─────────────────┐
│  EASI (3-El.) │     │ 12-Kanal ECG    │
│  I, E, S      │     │ I,II,V1–V6      │
└──────┬────────┘     └────────┬────────┘
       │                       │
       ▼                       ▼
  load_and_process()  ←  mode="easi" / "12ch"
       │
       ├─ 1. Laden & Zeitfenster schneiden   (vcgsuite.io)
       ├─ 2. ZapLine  (50 Hz + Harmonische)   (vcgsuite.preprocessing)
       ├─ 3. FIR Hochpass + Tiefpass          (vcgsuite.preprocessing)
       ├─ 4. VCG-Transformation               (vcgsuite.transform)
       │       EASI  →  W · T  →  Frank XYZ
       │       12-CH →  IDT / KORS / QLSV / PLSV
       │
       ▼
   df_analysis   ←── Analyse startet hier
```

---

## Modulstruktur (vcgsuite)

| Modul | Inhalt |
|-------|--------|
| `vcgsuite.config` | Globale Konfiguration (Abtastrate, Spaltennamen, Lead-Reihenfolge — **keine** Dateipfade) |
| `vcgsuite.utils` | `parse_subject_id()` |
| `vcgsuite.io.easi` | `load_easi()` |
| `vcgsuite.io.ecg12` | `load_ecg12()` |
| `vcgsuite.preprocessing.filter` | `zapline()`, `fir_bandpass()`, `filter_pipeline()` |
| `vcgsuite.transform.easi2frank` | `easi_to_frank_xyz()` + Matrizen W, T, D |
| `vcgsuite.transform.ecg12_to_frank` | `ecg12_to_frank_xyz()` + `MATRICES`-Dict |
| `vcgsuite.pipeline` | Ende-zu-Ende-Orchestrator `load_and_process()` |

---

## Schnellstart

```python
import vcgsuite as ecg

# EASI-Aufnahme laden und verarbeiten
df_analysis = ecg.load_and_process(
    filepath  = "/pfad/zur/datei.txt",
    mode      = "easi",
    start_sec = 5,       # erste 5 s abschneiden
    duration  = 300,     # 5 Minuten analysieren
)
# 12-Kanal-Aufnahme mit KORS-Matrix
df_analysis = ecg.load_and_process(
    filepath  = "/pfad/zur/datei.txt",
    mode      = "12ch",
    transform = "KORS",
)
# Analyse-Variablen entpacken
t = df_analysis["Time"].to_numpy()
X = df_analysis["X"].to_numpy()
Y = df_analysis["Y"].to_numpy()
Z = df_analysis["Z"].to_numpy()
```

---

## Parameter-Referenz

### `load_and_process()` — Pflichtparameter

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `filepath` | `str` | Pfad zur Datendatei |
| `mode` | `str` | `"easi"` oder `"12ch"` |

### Zeitfenster

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `start_sec` | `0` | Schnittstart in Sekunden |
| `duration` | `None` | Dauer in Sekunden — `None` = bis zum Ende |
| `fs` | `250` | Abtastrate in Hz |

### EASI-spezifisch

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `col_i` | `"R"` | Spaltenname Elektrode I |
| `col_e` | `"M"` | Spaltenname Elektrode E |
| `col_s` | `"L"` | Spaltenname Elektrode S |

### 12-Kanal-spezifisch

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `transform` | `"IDT"` | Transformationsmatrix: `"IDT"` · `"KORS"` · `"QLSV"` · `"PLSV"` |

### Filterparameter

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `use_zapline` | `True` | `False` überspringt ZapLine komplett, es läuft nur noch der FIR-Bandpass |
| `fline` | `50.0` | Netzfrequenz in Hz (nur bei `use_zapline=True`) |
| `nremove_zap` | `1` | DSS-Komponenten für ZapLine (nur bei `use_zapline=True`) |
| `hp_cutoff` | `1.5` | FIR-Hochpass-Grenzfrequenz in Hz |
| `hp_taps` | `201` | FIR-Taps Hochpass |
| `lp_cutoff` | `37.5` | FIR-Tiefpass-Grenzfrequenz in Hz |
| `lp_taps` | `21` | FIR-Taps Tiefpass |

### Proband-ID

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `subject_id_fn` | `None` | Eigene Funktion `filepath → (id, subject_id)` — `None` = Standard-Regex |

Standard-Regex (`vcgsuite.utils.parse_subject_id`) erkennt:
- `-cpt204-` → `"cpt204"`
- `-57-` → `"57"`
- Ersten `-Ziffern-`-Block irgendwo im Dateinamen (nicht nur am Anfang)
- Fallback: Dateiname ohne Endung

---

## df_analysis — Struktur

```python
df_analysis.columns
# EASI:   ["Time", "X", "Y", "Z", "V_IS", "V_ES", "V_AS"]
# 12-CH:  ["Time", "X", "Y", "Z", "I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
df_analysis.attrs
# {"proband_id": "57", "subject_id": "S57", "mode": "12ch",
#  "fs": 250, "transform": "IDT"}
```

`df_analysis.attrs["fs"]` wird von nachgelagerten Modulen (u. a.
`vcgsuite.features.loop`) als Quelle der tatsächlichen Abtastrate genutzt,
statt eines hartcodierten Default-Werts.

---

## Transformationsmatrizen (12-Kanal)

| Kürzel | Quelle | Optimiert für |
|--------|--------|---------------|
| `IDT` | Edenbrandt & Pahlm 1988 | Allgemein |
| `KORS` | Kors et al. 1990 | Allgemein |
| `QLSV` | Guillem et al. 2006 | QRS-Komplex |
| `PLSV` | Guillem et al. 2006 | P-Welle |

---

## Status der ursprünglich "geplanten Erweiterungen"

Diese standen im Original-Notebook als offene TODOs — Stand nach dem Umbau
in `vcgsuite`:

- [x] HRV-Analyse auf Basis der VCG-Zeitreihe → `vcgsuite.hrv`
- [x] VCG-Delineation (P, QRS, T) → `vcgsuite.annotation`, siehe
      [`vcg_beat_annotation.md`](vcg_beat_annotation.md)
- [x] 3D-Darstellung der Herzachse → `vcgsuite.viz.vcg_annotated.plot_vcg_3d`
- [x] Auslagerung in `.py`-Bibliothek → dieses Repository (`vcgsuite`)
