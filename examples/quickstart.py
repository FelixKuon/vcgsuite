#!/usr/bin/env python3
"""
Quickstart: kompletter vcgsuite-Durchlauf auf einer eigenen Aufnahme.

Beispiel für den in README.md und docs/vcg_analysis_pipeline.md
beschriebenen Ablauf: Laden -> Filtern -> VCG-Transformation -> Kinematik
-> R-Peak-Erkennung -> Beat-Annotation -> Beat-Rotation -> Loop-Features
-> HRV.

Aufruf:
    python examples/quickstart.py /pfad/zur/aufnahme.txt --mode easi
    python examples/quickstart.py /pfad/zur/aufnahme.txt --mode 12ch --duration 120

Ersetzt die alten Notebook-Beispielaufrufe (Wrapper_bsp.py).
"""

from __future__ import annotations

import argparse

import vcgsuite as ecg
from vcgsuite.hrv.prep import build_rr_dataframe
from vcgsuite.hrv.helpers import compute_hrv_full
from vcgsuite.features.loop.p_wave import compute_p_wave_features
from vcgsuite.features.loop.t_wave import build_vagus_features, compute_t_wave_features
from vcgsuite.features.loop.qrs_complex import compute_qrs_features
from vcgsuite.features.loop.merge import merge_loop_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", help="Pfad zur EASI- oder 12-Kanal-Rohdatendatei")
    parser.add_argument("--mode", choices=["easi", "12ch"], default="easi")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None,
                        help="Sekunden ab --start-sec (Default: bis zum Ende)")
    args = parser.parse_args()

    # 1-4: Laden -> Filtern -> VCG-Transformation
    df_analysis = ecg.load_and_process(
        args.filepath, mode=args.mode,
        start_sec=args.start_sec, duration=args.duration,
    )

    # 5: Frenet-Serret-Kinematik
    df_analysis, dt = ecg.compute_vcg_kinematics(df_analysis)
    df_analysis.attrs.setdefault("fs", 1.0 / dt)

    # 6: R-Peak- und R-Turn-Detektion
    r_peak_times, _ = ecg.detect_r_peaks(df_analysis)
    r_turn_times, _ = ecg.detect_r_turn(df_analysis, r_peak_times)
    print(f"\n{len(r_peak_times)} Beats erkannt.")

    # 7: Hierarchische Beat-Annotation
    df_annotations = ecg.annotate_all_beats(df_analysis, r_peak_times, r_turn_times)

    # 8: Beat-zu-Beat-Rotation (Grundlage für Loop-Features + HRV)
    df_r = ecg.compute_beat_rotation(df_annotations, df_analysis)

    # 9: P-/QRS-/T-Loop-Features
    df_p = compute_p_wave_features(df_analysis, df_annotations, df_r)
    df_vagus = build_vagus_features(df_analysis, df_annotations)
    df_qrs = compute_qrs_features(df_analysis, df_annotations, df_vagus, df_p, df_r)
    df_t = compute_t_wave_features(df_analysis, df_annotations, df_r, df_p, df_vagus=df_vagus)
    df_full = merge_loop_features(df_p, df_qrs, df_t, df_r)
    print(f"\nLoop-Features: {df_full.shape[0]} Beats x {df_full.shape[1]} Spalten")

    # 10: HRV
    df_rr = build_rr_dataframe(df_r)
    hrv = compute_hrv_full(df_rr["RR_interval"].to_numpy())
    print("\nHRV-Kennwerte:")
    for key in ("mean_hr", "sdnn", "rmssd", "pnn50", "SI"):
        print(f"  {key:10s} = {hrv[key]:.2f}")


if __name__ == "__main__":
    main()
