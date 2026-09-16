"""vcgsuite.preprocessing.filter — insbesondere den use_zapline-Schalter.

Rein synthetisch, benötigt keine echten Aufnahmedaten und läuft daher
immer (auch in CI).
"""

from unittest.mock import patch

import numpy as np

from vcgsuite.preprocessing.filter import filter_pipeline


def _make_signals(fs=250.0, n_seconds=10.0, n_channels=2):
    # n_seconds bewusst > 4s (zapline()'s nfft=4*fs) gewaehlt, damit die
    # echte DSS-ZapLine im dritten Test nicht an zu wenig Samples scheitert.
    n = int(fs * n_seconds)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    signals = []
    for ch in range(n_channels):
        sig = (
            0.5 * np.sin(2 * np.pi * 1.2 * t)          # "Herzschlag"-artige Komponente
            + 0.1 * np.sin(2 * np.pi * 50.0 * t)         # Netzbrumm
            + 0.02 * rng.standard_normal(n)              # Rauschen
        )
        signals.append(sig)
    return signals, fs


def test_filter_pipeline_skips_zapline_when_disabled():
    signals, fs = _make_signals()

    with patch("vcgsuite.preprocessing.filter.zapline") as mock_zapline:
        filter_pipeline(signals, fs=fs, use_zapline=False)
        mock_zapline.assert_not_called()


def test_filter_pipeline_calls_zapline_by_default():
    signals, fs = _make_signals()

    with patch(
        "vcgsuite.preprocessing.filter.zapline", wraps=_identity_zapline
    ) as mock_zapline:
        filter_pipeline(signals, fs=fs)
        mock_zapline.assert_called_once()


def _identity_zapline(signals, fs=None, fline=None, nremove=None):
    """Ersetzt die echte (langsamere) DSS-ZapLine für den reinen Aufruf-Test."""
    return signals


def test_filter_pipeline_output_shape_matches_input_with_and_without_zapline():
    signals, fs = _make_signals()

    out_with = filter_pipeline(signals, fs=fs, use_zapline=True)
    out_without = filter_pipeline(signals, fs=fs, use_zapline=False)

    assert len(out_with) == len(signals)
    assert len(out_without) == len(signals)
    for s_in, s_out in zip(signals, out_without):
        assert len(s_out) == len(s_in)
