# ══════════════════════════════════════════════════════════════════════════
#  HRV / EDR DASHBOARD  –  Plot-Teil
#
#  Migriert aus HRV_Dashboard.py ("BLOCK HRV_DASHBOARD").
#
#  Hinweis Migration – deutliche Abweichung vom Original:
#  Das Original war eine interaktive Notebook-Zelle: `build_edr_dashboard()`
#  griff über Python-Closures auf Notebook-globale Variablen zu (df_rr,
#  t_edr, edr_signal, f_resp_edr, freqs_cwt_full, ..., das ipywidgets-
#  Dropdown `resp_dd`) und berechnete HRV-Metriken/RSA/PSD SELBST inline,
#  bevor geplottet wurde. Das widerspricht der Vorgabe für dieses
#  viz-Paket ("keine Berechnungslogik hier, nur Visualisierung"). Deshalb:
#    - Alle Berechnungen (compute_hrv_full, compute_rsa_amplitude,
#      compute_rsa_hf_power, compute_psd_welch, band_power) MÜSSEN vom
#      Aufrufer vorher über `vcgsuite.hrv.helpers` ausgeführt und als
#      Parameter übergeben werden.
#    - Die ipywidgets-Slider/Dropdown-Steuerung (t_sl9, w_sl9, resp_dd,
#      _update9) ist NICHT migriert — das ist reine Notebook-UI-Glue-Code,
#      kein Plot. Wiederholte Aufrufe mit anderen Parametern übernehmen
#      diese Rolle in einer Bibliothek.
#    - `fig.show()` / `IPython.display.display(HTML(...))` wurden durch
#      Rückgabewerte ersetzt (`go.Figure` bzw. HTML-String).
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm as sp_norm

from ..hrv.analysis import EDR_CONFIG, RESP_COL
from ..hrv.helpers import BANDS


def plot_kubios_ans_balance(hrv: dict) -> go.Figure:
    """
    Kubios-Style PNS/SNS-Verteilungsplot.

    Parameters
    ----------
    hrv : dict  –  Ausgabe von `vcgsuite.hrv.helpers.compute_hrv_full()`.
    """
    zmin, zmax = -5, 5
    x = np.linspace(zmin, zmax, 400)
    y_norm = sp_norm.pdf(x, 0, 1)
    y_max = y_norm.max() * 1.25

    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Parasympathetic (PNS)", "Sympathetic (SNS)"],
        horizontal_spacing=0.10,
    )

    configs = [
        (1, "PNS", hrv["PNS_index"], hrv["pns_zs"],
         ["Mean RR", "RMSSD", "SD1"], "#45CDFF",
         ["navy", "royalblue", "deepskyblue"]),
        (2, "SNS", hrv["SNS_index"], hrv["sns_zs"],
         ["Mean HR", "Stress Index", "SD2"], "#FF8E43",
         ["#DB6223", "#F4B183", "#F9A26C"]),
    ]

    custom_cs = [
        [0.0, "rgb(255,140,0)"],
        [0.5, "rgb(255,255,255)"],
        [1.0, "rgb(37,150,190)"],
    ]

    for col_i, name, idx_val, zs, labels, lc, bar_cols in configs:
        z_grad = np.tile(x, (80, 1))
        y_grid = np.linspace(0, y_max, 80)
        fig2.add_trace(go.Heatmap(
            x=x, y=y_grid, z=z_grad,
            colorscale=custom_cs, opacity=0.28,
            showscale=False, zmin=zmin, zmax=zmax,
        ), row=1, col=col_i)

        fig2.add_trace(go.Scatter(
            x=x, y=y_norm, mode="lines",
            line=dict(color="#888", width=2), showlegend=False,
        ), row=1, col=col_i)

        heights = y_max * np.array([0.10, 0.38, 0.66])
        bh = y_max * 0.19

        for i, (z, lbl, bc) in enumerate(zip(zs, labels, bar_cols)):
            zc = np.clip(z, zmin, zmax)
            fig2.add_trace(go.Scatter(
                x=[zmin, zc, zc, zmin, zmin],
                y=[heights[i] - bh / 2, heights[i] - bh / 2,
                   heights[i] + bh / 2, heights[i] + bh / 2, heights[i] - bh / 2],
                fill="toself", fillcolor=bc, mode="lines",
                line=dict(color="black", width=1.8),
                opacity=0.70, showlegend=False,
            ), row=1, col=col_i)

            fig2.add_annotation(
                x=(zmin + zc) / 2, y=heights[i],
                text=f"<b>{lbl}</b>", showarrow=False,
                font=dict(color="white", size=13, family="Arial"),
                row=1, col=col_i,
            )

        xref = "x" if col_i == 1 else "x2"
        yref = "y" if col_i == 1 else "y2"
        for xv, lw, ld, clr in [
            (idx_val, 3.5, "solid", lc),
            (-1, 1.0, "dash", "#888"), (1, 1.0, "dash", "#888"),
            (-2, 1.0, "dot", "#888"), (2, 1.0, "dot", "#888"),
        ]:
            fig2.add_shape(type="line",
                x0=xv, x1=xv, y0=0, y1=y_max,
                line=dict(color=clr, width=lw, dash=ld),
                xref=xref, yref=yref,
            )

        fig2.add_annotation(
            x=np.clip(idx_val, zmin + 0.4, zmax - 0.4), y=y_max * 0.90,
            text=f"<b>{name} = {idx_val:+.2f}</b>",
            showarrow=False,
            font=dict(color=lc, size=12, family="Arial"),
            bgcolor="white", opacity=0.85,
            row=1, col=col_i,
        )

    fig2.update_layout(
        height=290,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(t=45, b=30, l=20, r=20),
        title=dict(text="ANS-Balance (Kubios-Style)", font=dict(size=13)),
    )
    for c in [1, 2]:
        fig2.update_xaxes(range=[zmin, zmax], title_text="z-score",
                          gridcolor="#eee", row=1, col=c)
        fig2.update_yaxes(range=[0, y_max * 1.05], showticklabels=False,
                          showgrid=False, row=1, col=c)
    return fig2


def build_edr_dashboard(
    df_rr: pd.DataFrame,
    t_start: float, t_end: float,
    hrv: dict,
    t_edr: np.ndarray, edr_signal: np.ndarray,
    freqs_cwt_full: np.ndarray, power_cwt_full: np.ndarray, ridge_cwt_full: np.ndarray,
    f_resp_win: float | None, rpm_win: float | None,
    rsa_amp: float | None,
    freqs_psd: np.ndarray, psd: np.ndarray,
    lf_pow: float | None, hf_pow: float | None, lf_hf: float | None,
    resp_col: str = RESP_COL,
) -> go.Figure:
    """
    Baut das HRV/EDR-Dashboard (Tachogramm + EDR, CWT-Scalogram, PSD,
    Poincaré) für ein Analysefenster [t_start, t_end].

    Alle HRV-/RSA-/PSD-Kennwerte müssen vom Aufrufer VORHER berechnet
    werden (z. B. via `vcgsuite.hrv.helpers.compute_hrv_full()`,
    `compute_rsa_amplitude()`, `compute_psd_welch()`, `band_power()`) —
    diese Funktion ist reine Visualisierung.

    Parameters
    ----------
    df_rr          : pd.DataFrame  –  Ausgabe von `vcgsuite.hrv.build_rr_dataframe()`.
    t_start, t_end : float          –  Analysefenster [s].
    hrv            : dict           –  Ausgabe von `compute_hrv_full(rr)` für
                                       das Fenster.
    t_edr, edr_signal, freqs_cwt_full, power_cwt_full, ridge_cwt_full :
                     Ausgabe von `extract_edr_signal()` / `_compute_cwt()`
                     (über das GESAMTE Signal, nicht nur das Fenster).
    f_resp_win, rpm_win : float | None  –  Lokale Atemfrequenz im Fenster
                                       (z. B. Median der CWT-Ridge im
                                       Fenster).
    rsa_amp        : float | None   –  Ausgabe von `compute_rsa_amplitude()`.
    freqs_psd, psd : np.ndarray     –  Ausgabe von `compute_psd_welch()` für
                                       das Fenster.
    lf_pow, hf_pow, lf_hf : float | None  –  Bandleistungen (via
                                       `band_power()`) für das Fenster.
    resp_col       : str            –  aktiver EDR-Kanal (Schlüssel in
                                       `vcgsuite.hrv.analysis.EDR_CONFIG`).

    Returns
    -------
    go.Figure
    """
    mask = (df_rr["Time"] >= t_start) & (df_rr["Time"] <= t_end)
    df_w = df_rr[mask]

    rr = df_w["RR_interval"].values
    rr_t = df_w["Time"].values

    cfg_active = EDR_CONFIG[resp_col]
    active_color = cfg_active['color']
    active_label = cfg_active['label']

    has_psd = len(freqs_psd) > 0

    # ── Subplot-Layout ─────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "RR-Tachogramm",
            "Heartmovement Scalogram mit EDR-Ridge",
            "PSD des Analysefensters",
            "Poincaré-Plot",
        ],
        specs=[
            [{"colspan": 2, "secondary_y": True}, None],
            [{"colspan": 2}, None],
            [{}, {}],
        ],
        row_heights=[0.25, 0.42, 0.33],
        vertical_spacing=0.10,
        horizontal_spacing=0.22,
    )

    # ── Row 1: RR-Tachogramm ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df_rr["Time"], y=df_rr["RR_interval"] * 1000,
        mode="lines", line=dict(color="#cccccc", width=1),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df_w["Time"], y=df_w["RR_interval"] * 1000,
        mode="lines+markers", name="RR – Analysefenster",
        line=dict(color="#2596be", width=2),
        marker=dict(size=5, color="#2596be"),
        hovertemplate="t=%{x:.1f}s | RR=%{y:.0f}ms<extra>RR</extra>",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=t_edr, y=edr_signal,
        mode="lines",
        name=f"EDR ({active_label.split('—')[0].strip()})",
        line=dict(color=active_color, width=1.2, dash="dot"),
        opacity=0.7,
        hovertemplate="t=%{x:.1f}s | EDR=%{y:.4f}<extra>EDR</extra>",
    ), row=1, col=1, secondary_y=True)

    fig.add_vrect(
        x0=t_start, x1=t_end,
        fillcolor="rgba(37,150,190,0.10)",
        line=dict(color="#2596be", dash="dot", width=1.5),
        row=1, col=1,
    )

    # ── Row 2: CWT-Scalogram ───────────────────────────────────────────────
    fig.add_trace(go.Heatmap(
        x=t_edr,
        y=freqs_cwt_full,
        z=power_cwt_full,
        colorscale="Jet",
        zmin=np.percentile(power_cwt_full, 2),
        zmax=np.percentile(power_cwt_full, 98),
        showscale=True,
        colorbar=dict(
            title=dict(text="Power", side="right"),
            len=0.38, y=0.555, x=1.01,
            tickfont=dict(size=10),
        ),
        hovertemplate="t=%{x:.1f}s | f=%{y:.3f}Hz<extra>CWT</extra>",
        name="",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t_edr, y=ridge_cwt_full,
        mode="lines", name="EDR-Ridge (instantane Atemfreq.)",
        line=dict(color="#00ff88", width=2),
        hovertemplate="t=%{x:.1f}s | f=%{y:.3f}Hz<extra>Ridge</extra>",
    ), row=2, col=1)

    fig.add_vrect(
        x0=t_start, x1=t_end,
        fillcolor="rgba(255,255,255,0.10)",
        line=dict(color="white", dash="dot", width=1.5),
        row=2, col=1,
    )

    for f_mark, lbl in [
        (0.05, "0.05 Hz"),
        (0.15, "LF | HF"),
        (0.40, "0.4 Hz"),
    ]:
        fig.add_hline(
            y=f_mark, line_dash="dash", line_color="white", line_width=1.0,
            annotation_text=lbl,
            annotation_font=dict(color="white", size=9),
            annotation_position="right",
            row=2, col=1,
        )

    if f_resp_win:
        fig.add_hline(
            y=f_resp_win, line_dash="solid",
            line_color="#00ff88", line_width=1.5,
            annotation_text=f"⌀ {rpm_win:.1f} /min",
            annotation_font=dict(color="#00ff88", size=10),
            annotation_position="left",
            row=2, col=1,
        )

    # ── Row 3, Col 1: PSD ─────────────────────────────────────────────────
    if has_psd:
        for band_name, (flo, fhi, fill, lc) in list(BANDS.items())[:3]:
            bm = (freqs_psd >= flo) & (freqs_psd <= fhi)
            if not bm.any():
                continue
            fx = np.concatenate([[flo], freqs_psd[bm], [fhi]])
            fy = np.concatenate([[0], psd[bm], [0]])
            fig.add_trace(go.Scatter(
                x=fx, y=fy, fill="tozeroy",
                fillcolor=fill, line=dict(color=lc, width=0.8),
                name=band_name, legendgroup="bands",
            ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=freqs_psd, y=psd, mode="lines", name="PSD (RR)",
            line=dict(color="#111", width=1.5),
        ), row=3, col=1)

        if f_resp_win:
            fig.add_vline(
                x=f_resp_win, line_dash="dash", line_color="#00cc66",
                annotation_text=f"💨 {rpm_win:.1f}/min",
                annotation_font=dict(color="#00cc66", size=10),
                annotation_position="top right",
                row=3, col=1,
            )

    # ── Row 3, Col 2: Poincaré ────────────────────────────────────────────
    SD1, SD2 = hrv["SD1"], hrv["SD2"]
    rr_mean_ms = np.mean(rr) * 1000
    rr_x = rr[:-1] * 1000 - rr_mean_ms
    rr_y = rr[1:] * 1000 - rr_mean_ms

    fig.add_trace(go.Scatter(
        x=rr_x, y=rr_y, mode="markers",
        marker=dict(color="#2596be", size=6, opacity=0.55),
        name="Poincaré",
        hovertemplate="RRₙ=%{x:.0f}ms | RRₙ₊₁=%{y:.0f}ms<extra></extra>",
    ), row=3, col=2)

    ax = np.max(np.abs(np.concatenate([rr_x, rr_y]))) * 1.15
    fig.add_trace(go.Scatter(
        x=[-ax, ax], y=[-ax, ax], mode="lines",
        line=dict(color="#aaa", dash="dash", width=1),
        showlegend=False,
    ), row=3, col=2)

    theta_e = np.linspace(0, 2 * np.pi, 300)
    ex = (SD2 * 1000 * np.cos(theta_e) - SD1 * 1000 * np.sin(theta_e)) / np.sqrt(2)
    ey = (SD2 * 1000 * np.cos(theta_e) + SD1 * 1000 * np.sin(theta_e)) / np.sqrt(2)
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color="#333", width=2.5),
        showlegend=False, name="SD1/SD2-Ellipse",
    ), row=3, col=2)

    for (px, py, lbl, clr) in [
        (-SD1 / np.sqrt(2) * 1000, SD1 / np.sqrt(2) * 1000,
         f"SD1 {SD1 * 1000:.1f}ms", "#1a8aaa"),
        (SD2 / np.sqrt(2) * 1000, SD2 / np.sqrt(2) * 1000,
         f"SD2 {SD2 * 1000:.1f}ms", "#c06820"),
    ]:
        fig.add_annotation(
            x=px, y=py, ax=0, ay=0,
            xref="x4", yref="y5", axref="x4", ayref="y5",
            text=f"<b>{lbl}</b>",
            arrowhead=4, arrowwidth=2.5, arrowcolor=clr,
            font=dict(color=clr, size=11, family="Arial"),
            showarrow=True,
        )

    # ── Achsenbeschriftungen ───────────────────────────────────────────────
    fig.update_yaxes(title_text="RR-Intervall [ms]",
                     gridcolor="#eee", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="EDR [a.u.]",
                     showgrid=False, row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Zeit [s]", gridcolor="#eee", row=1, col=1)

    fig.update_yaxes(title_text="Frequenz [Hz]",
                     gridcolor="rgba(255,255,255,0.1)",
                     range=[0.01, 0.5], row=2, col=1)
    fig.update_xaxes(title_text="Zeit [s]",
                     gridcolor="rgba(255,255,255,0.1)", row=2, col=1)

    fig.update_yaxes(title_text="PSD [s²/Hz]",
                     gridcolor="#eee", row=3, col=1)
    fig.update_xaxes(title_text="Frequenz [Hz]", gridcolor="#eee",
                     range=[0, 0.5], row=3, col=1)

    fig.update_yaxes(title_text="RRₙ₊₁ − R̄R [ms]",
                     gridcolor="#eee", row=3, col=2)
    fig.update_xaxes(title_text="RRₙ − R̄R [ms]",
                     gridcolor="#eee", row=3, col=2)

    # ── Titel + Layout ────────────────────────────────────────────────────
    rsa_str = f"{rsa_amp:.1f} ms" if rsa_amp is not None else "–"
    rsa_qual = ("stark" if rsa_amp and rsa_amp > 30
                else "mittel" if rsa_amp and rsa_amp > 15
                else "schwach")

    fig.update_layout(
        height=900,
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(
            text=(f"HRV Dashboard  |  "
                  f"t = {t_start:.0f}–{t_end:.0f} s  |  "
                  f"N = {len(rr)} Schläge  |  "
                  f"∅ HR = {hrv['mean_hr']:.0f} bpm  |  "
                  f"RSA = {rsa_str} ({rsa_qual})  |  "
                  f"EDR: {active_label.split('—')[0].strip()}  "
                  f"[HP {cfg_active['fc']} Hz]"),
            font=dict(size=13, family="Arial"),
        ),
        legend=dict(orientation="h", y=-0.04, x=0, font=dict(size=11)),
        hovermode="closest",
        font=dict(family="Arial", size=12),
        margin=dict(t=70, b=60, r=100),
    )

    return fig


def build_edr_summary_html(hrv: dict, resp_col: str, f_resp_win: float | None,
                           rpm_win: float | None, rsa_amp: float | None,
                           lf_pow: float | None, hf_pow: float | None,
                           lf_hf: float | None) -> str:
    """
    Baut die HTML-Parametertabelle des Dashboards (Zeit-/Frequenzbereich,
    EDR/RSA, PNS-/SNS-Index) als eigenständigen HTML-String, statt (wie im
    Original) direkt per `IPython.display.display(HTML(...))` auszugeben.
    """
    cfg_active = EDR_CONFIG[resp_col]
    active_label = cfg_active['label']
    band_info = (f"HP {cfg_active['fc']} Hz  Ord.{cfg_active['order']}  "
                 f"({'Trend-Entfernung' if cfg_active['fc'] <= 0.05 else 'LF-Unterdrückung'})")
    resp_src = (f"{f_resp_win:.3f} Hz = {rpm_win:.1f} Atemz./min"
                if f_resp_win else "–")
    rsa_str = f"{rsa_amp:.1f} ms" if rsa_amp is not None else "–"
    rsa_qual = ("stark" if rsa_amp and rsa_amp > 30
                else "mittel" if rsa_amp and rsa_amp > 15
                else "schwach")

    return f"""
    <style>
      .hrv9 {{ border-collapse:collapse;width:100%;font-family:Arial;
               font-size:13px;margin-top:8px }}
      .hrv9 th {{ background:#2596be;color:white;padding:6px 12px;text-align:left }}
      .hrv9 td {{ padding:5px 12px;border-bottom:1px solid #eee }}
      .hrv9 tr:hover td {{ background:#f0f8ff }}
      .badge {{ padding:2px 9px;border-radius:10px;font-weight:bold }}
    </style>
    <table class="hrv9">
      <tr>
        <th colspan="2">Zeitbereich</th>
        <th colspan="2">Frequenzbereich</th>
        <th colspan="2">EDR / RSA</th>
      </tr>
      <tr>
        <td><b>Mean RR</b></td><td>{hrv['mean_rr'] * 1000:.1f} ms</td>
        <td><b>LF Power</b></td>
        <td>{f"{lf_pow * 1e6:.1f} ms²" if lf_pow else "–"}</td>
        <td><b>Atemfreq. (EDR)</b></td><td>{resp_src}</td>
      </tr>
      <tr>
        <td><b>RMSSD</b></td><td>{hrv['rmssd'] * 1000:.1f} ms</td>
        <td><b>HF Power</b></td>
        <td>{f"{hf_pow * 1e6:.1f} ms²" if hf_pow else "–"}</td>
        <td><b>RSA-Amplitude</b></td>
        <td><span class="badge"
          style="background:{'#e8f8e8' if rsa_amp and rsa_amp > 30 else '#fff8e0' if rsa_amp and rsa_amp > 15 else '#fde8e8'};
                 color:{'green' if rsa_amp and rsa_amp > 30 else '#b07000' if rsa_amp and rsa_amp > 15 else 'red'}">
          {rsa_str} ({rsa_qual})
        </span></td>
      </tr>
      <tr>
        <td><b>SD1</b></td><td>{hrv['SD1'] * 1000:.1f} ms</td>
        <td><b>LF/HF</b></td><td>{f"{lf_hf:.2f}" if lf_hf else "–"}</td>
        <td><b>PNS-Index</b></td>
        <td><span class="badge" style="background:#1a8aaa22;color:#1a8aaa">
            {hrv['PNS_index']:+.2f}</span></td>
      </tr>
      <tr>
        <td><b>SD2</b></td><td>{hrv['SD2'] * 1000:.1f} ms</td>
        <td><b>SDNN</b></td><td>{hrv['sdnn'] * 1000:.1f} ms</td>
        <td><b>SNS-Index</b></td>
        <td><span class="badge" style="background:#c0682022;color:#c06820">
            {hrv['SNS_index']:+.2f}</span></td>
      </tr>
      <tr>
        <td colspan="6" style="color:#777;font-size:11px;padding-top:6px">
          <b>EDR-Kanal:</b> {active_label}  |  Bandpass: {band_info}<br>
          <b>RSA-Amplitude</b> = Halbamplitude des RR-Signals, schmalbandig gefiltert
          um EDR-Atemfrequenz (±0.025 Hz). Direkt proportional zur Vagusaktivität.
          SD1 ≈ RSA-Amplitude im Poincaré-Diagramm.
        </td>
      </tr>
    </table>
    """
