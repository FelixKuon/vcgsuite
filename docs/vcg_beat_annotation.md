
# VCG Beat Annotation via 3D-Trajektoriendynamik

> **Migrationshinweis:** Ursprünglich `VCG_Beat_Annontation.md` (Tippfehler
> im Dateinamen behoben). Die Tabelle unter
> ["Feature-Fenster"](#feature-fenster-hierarchical_windows) wurde
> korrigiert: Die alte `WINDOWS_LOCAL`-Konstante war im Original-Code
> verwaist (nirgends tatsächlich referenziert) und wich von den real in
> `annotate_beat_hierarchical()` verwendeten Fenstergrenzen ab. Die Tabelle
> unten listet jetzt `vcgsuite.kinematics.constants.HIERARCHICAL_WINDOWS` —
> die tatsächlich genutzten Werte, jeweils mit ihrem Anker-Zeitpunkt (nicht
> alle Fenster sind relativ zu R_peak, siehe Spalte "Anker").

## Übersicht

Dieses Modul implementiert einen **beat-basierten Annotationsalgorithmus für Vektorkardiogramme (VCG)**, der charakteristische Punkte des kardialen Erregungszyklus (P, Q, R, S, T) nicht anhand einzelner Ableitungen, sondern anhand der **geometrisch-kinematischen Eigenschaften der 3D-Raumkurve** des VCG-Vektorlooops erkennt.

Der zentrale Unterschied zu klassischen Ansätzen: Die **Orientierung des Vektors im Raum wird herausnormiert** – stattdessen wird analysiert, *wie sich die Trajektorie bewegt* (Krümmung, Torsion, Geschwindigkeit), unabhängig davon, *wohin sie zeigt*. Das ermöglicht eine perspektivisch robuste Markierung von Ereignissen im 3D-Raum, die in einzelnen 2D-Projektionen (z. B. Frontalebene, transversale Ebene) unsichtbar sein können.

---

## Motivation & Innovation

Klassische EKG-Delineationsalgorithmen arbeiten entweder auf einzelnen Ableitungen (z. B. Ableitung II für P-Wellen, V5 für T-Wellen) oder auf 2D-VCG-Projektionen. Beide Ansätze haben eine fundamentale Schwäche: bestimmte **Bewegungen der Herzachse** verlaufen in der betrachteten Projektion nahezu senkrecht zur Beobachtungsachse und erscheinen daher flach oder gar nicht sichtbar.

Dieser Algorithmus adressiert das Problem durch:

- **Transformation in die Frenet-Serret-Geometrie der 3D-Kurve**: Die Trajektorie wird nicht als Zeitreihe von Vektoren, sondern als parametrisierte Raumkurve behandelt.
- **Orientierungs-unabhängige Features**: Krümmung κ, Torsion τ, Betrag der Geschwindigkeit |v| und Beschleunigung |a| sind invariant gegenüber Rotation des Koordinatensystems.
- **Sphärische Koordinaten als Ergänzung**: Radius r, Azimutwinkel θ und Polarwinkel φ beschreiben die Zeigergeometrie; ihre zeitlichen Ableitungen (dr/dt, dθ/dt, dφ/dt) quantifizieren Orientierungsänderung getrennt von Amplitudenänderung.
- **Hierarchische Annotationsstrategie**: Landmarks werden kaskadiert detektiert — von robusten Ankerpunkten (R-Peak) zu feineren Wellenstrukturen (P, T).

---

## Theoretische Grundlage: Frenet-Serret

Sei **r**(t) = (X(t), Y(t), Z(t)) der VCG-Vektor. Dann gilt:

**Geschwindigkeit:**
$$\mathbf{v} = \frac{d\mathbf{r}}{dt}, \quad |\mathbf{v}| = \sqrt{\dot{X}^2 + \dot{Y}^2 + \dot{Z}^2}$$

**Beschleunigung:**
$$\mathbf{a} = \frac{d\mathbf{v}}{dt}$$

**Krümmung (κ)** — misst, wie stark sich die Richtung der Kurve ändert:
$$\kappa = \frac{|\mathbf{v} \times \mathbf{a}|}{|\mathbf{v}|^3}$$

**Torsion (τ)** — misst, wie stark die Kurve aus ihrer momentanen Ebene herausdreht:
$$\tau = \frac{(\mathbf{v} \times \mathbf{a}) \cdot \mathbf{j}}{|\mathbf{v} \times \mathbf{a}|^2}, \quad \mathbf{j} = \frac{d\mathbf{a}}{dt}$$

Diese Größen sind **rotationsinvariant**: Sie ändern sich nicht, wenn das Herz im Körper anders orientiert ist oder wenn das Messsystem gedreht wird. Das macht sie besonders robust für interindividuelle Vergleiche.

---

## Modulstruktur (vcgsuite)

### 1. `vcgsuite.kinematics.frenet_serret.compute_vcg_kinematics(df)`
Berechnet aus den XYZ-Rohdaten des VCG alle kinematischen Größen:
- 1., 2. und 3. Ableitung (Geschwindigkeit, Beschleunigung, Jerk)
- Krümmung κ und Torsion τ nach Frenet-Serret
- Sphärische Koordinaten: r, θ (Azimut), φ (Polar)

### 2. `vcgsuite.annotation.features.extract_features(df, twin, r_peak_t)`
Extrahiert für ein Beat-Zeitfenster eine **Feature-Matrix** mit ca. 30 kinematischen Merkmalen (siehe `vcgsuite.kinematics.constants.FEATURE_COLS`):

| Feature-Gruppe       | Beispiele                                      | Bedeutung                                      |
|----------------------|------------------------------------------------|------------------------------------------------|
| Rohdaten             | `Curvature_raw`, `V_abs_raw`, `A_abs_raw`      | Unmittelbare kinematische Signale              |
| Geglättet (σ=3)      | `Curvature_sm3`, `V_abs_sm3`                   | Rauschreduzierte Varianten (Gaußfilter)        |
| Zeitableitungen      | `dCurvature_dt`, `dA_abs_dt`                   | Änderungsrate kinematischer Größen             |
| Lokale Statistiken   | `Curvature_lmax`, `A_abs_lstd`                 | Lokales Maximum / Standardabweichung ±20 ms   |
| Rang-Normierungen    | `Curvature_rank`, `V_abs_rank`                 | Rangnormiert auf [0,1], ausreißerrobust        |
| Sphärische Ableitungen | `dr_dt_raw`, `dtheta_dt_raw`, `dphi_dt_raw`  | Getrennte Analyse von Amplitude und Richtung  |
| r-Metriken           | `r_rel`, `r_normalized`, `r_slope`             | Zeigerabstand vom Ursprung (normiert/relativ)  |

### 3. `vcgsuite.annotation.features.detect_in_window(df, anchor_t, lo_ms, hi_ms, fcol, op)`
Bestimmt den Zeitpunkt des Maximums oder Minimums eines Features in einem definierten Zeitfenster relativ zu einem Ankerpunkt.

### 4. `vcgsuite.annotation.features.detect_consensus(df, anchor_t, lo_ms, hi_ms, strategies)`
Führt **Multi-Feature-Voting** durch: Mehrere `(Feature, MAX/MIN)`-Strategien liefern je einen Zeitpunkt-Kandidaten. Der **Median aller Votes** wird als finaler Zeitpunkt zurückgegeben, zusammen mit dem Spread (Streuung der Kandidaten in ms) als Qualitätsmaß.

### 5. `vcgsuite.detection.r_peaks.detect_r_peaks(df, ...)`
Detektiert R-Peaks im `A_abs`-Signal (Betrag der Beschleunigung) mit:
- **Adaptiver, fensterbasierter Schwelle** (gleitendes Quantil, Standard: 95. Perzentil in 10-s-Fenstern)
- Mindestabstand zwischen Peaks (Standard: 400 ms ≙ max. 150 bpm)
- Ausgabe: Zeitpunkte, Herzfrequenz-Statistiken

### 6. `vcgsuite.detection.r_turn.detect_r_turn(df, r_peak_times)`
Sucht direkt nach dem R-Peak das erste lokale Minimum in `A_abs` (Fenster: +5 bis +13 ms). Entspricht dem Moment, in dem die Depolarisationsfront ihre Spitzengeschwindigkeit überschritten hat und die Richtungsänderung abklingt — der **Wendepunkt des QRS-Komplexes**.

### 7. `vcgsuite.annotation.hierarchical.annotate_beat_hierarchical(df, r_peak_t, r_turn_t)`
Kernfunktion. Annotiert alle Landmarks eines einzelnen Herzschlags in **drei hierarchischen Stufen**:

**Stufe 1 — Absoluter Anker:**
- `R_peak+`: extern übergeben (höchste Beschleunigung im QRS)

**Stufe 2 — Relativ zu R_peak:**
- `R_turn`: Wendepunkt des QRS (deterministisch oder via dθ/dt)
- `Q_on`: Beginn der Q-Zacke (Krümmungsmaximum, Geschwindigkeitsminimum)
- `Q_off`: fixer Offset `R_peak+ − 20 ms` (kein Konsens-Fenster)
- `S_on`: Beginn der S-Zacke (Krümmungsminimum, r-slope Minimum)
- `S_off`: Ende der S-Zacke (lokale Variabilitätsmaxima)

**Stufe 3a — P-Komplex** (relativ zu `P_peak`, der relativ zu `R_peak` liegt):
- `P_peak`: Maximaler Zeigerradius r im Fenster −145 bis −95 ms relativ zu R_peak
- `P_on`: Onset via Beschleunigungsanstieg (dA/dt MAX), relativ zu `P_peak`
- `P_off`: Ende via Krümmungsmaximum, relativ zu `P_peak`

**Stufe 3b — T-Welle** (kaskadierende Kette: `S_off` → `T_on` → `T_turn1` → `T_turn2` → `T_off`, jeder Marker relativ zum vorherigen):
- `T_on`: Kleiner Krümmungsradius (Einbiegen der T-Welle), relativ zu `S_off`
- `T_turn1`, `T_turn2`: Zwei Wendepunkte der T-Welle via r-Maximum (ermöglicht biphasische T-Wellen), jeweils relativ zum vorherigen Marker
- `T_off`: Abklingen via dA/dt und dV/dt Minima, relativ zu `T_turn2`

### 8. `vcgsuite.annotation.annotate.annotate_all_beats(df, r_peak_times, r_turn_times)`
Iteriert über alle detektierten R-Peaks und erstellt einen **DataFrame `df_annotations`** mit einer Zeile pro Herzschlag und Zeitstempeln aller 12 Landmarks als Spalten.

---

## Feature-Fenster (HIERARCHICAL_WINDOWS)

Die Suche nach jedem Landmark ist auf physiologisch plausible Zeitfenster
relativ zu einem **Anker-Zeitpunkt** beschränkt — dieser ist bei den meisten
Markern `R_peak`, bei der T-Welle jedoch der jeweils vorherige Marker in der
Kette (siehe Spalte "Anker"). Quelle:
`vcgsuite.kinematics.constants.HIERARCHICAL_WINDOWS`.

| Landmark  | Anker      | Fenster relativ zum Anker |
|-----------|------------|----------------------------|
| R_turn    | R_peak     | +3 bis +28 ms              |
| Q_on      | R_peak     | −80 bis −20 ms             |
| Q_off     | R_peak     | fixer Offset: −20 ms       |
| S_on      | R_peak     | +20 bis +100 ms            |
| S_off     | R_peak     | +40 bis +130 ms            |
| P_peak    | R_peak     | −145 bis −95 ms            |
| P_on      | P_peak     | −90 bis −10 ms             |
| P_off     | P_peak     | +5 bis +95 ms              |
| T_on      | S_off      | +70 bis +140 ms            |
| T_turn1   | T_on       | +80 bis +140 ms            |
| T_turn2   | T_turn1    | +60 bis +130 ms            |
| T_off     | T_turn2    | +60 bis +140 ms            |

---

## Ausgabe

`df_annotations`: DataFrame mit `beat_id`, `t_R_peak+` und Zeitstempeln aller 12 Landmarks als absolute Zeit [s]. NaN für nicht detektierbare Landmarks.

Konsolenausgabe (Beispiel):
```
Marker        gefunden    Ø rel. Zeit
----------------------------------------
  R_turn            42      +12.3 ms
  S_on              41      +28.7 ms
  P_peak            40     -121.4 ms
  T_turn1           39     +248.1 ms
  ...
```

---

## Abhängigkeiten

```
numpy, scipy, pandas
```

Kein Machine-Learning-Modell notwendig — der Algorithmus ist vollständig **regelbasiert und interpretierbar**.

---

## Hinweise

- Sampling-Rate: die Fenstergrenzen oben sind für ~250 Hz VCG-Daten kalibriert (Millisekunden-Angaben werden intern anhand der tatsächlichen `df.attrs["fs"]` in Samples umgerechnet).
- Das Verfahren ist unabhängig von der Frank-Lead-Konfiguration (funktioniert mit beliebigen orthogonalen XYZ-Leads).
- Die Torsion ist numerisch instabil bei sehr kleinen Geschwindigkeiten (nahe Isoelektrikum) — entsprechende Episoden sollten vor der Analyse maskiert werden.
