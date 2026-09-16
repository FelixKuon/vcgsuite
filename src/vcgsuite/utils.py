"""
Allgemeine Hilfsfunktionen, die von mehreren Untermodulen (z. B. vcgsuite.io)
genutzt werden.
"""

import os
import re


def parse_subject_id(filepath: str) -> tuple[str, str]:
    """
    Extrahiert Proband-ID und Subject-ID aus dem Dateinamen.

    Priorität:
      1. -cpt204-  ->  "cpt204"
      2. -55-      ->  "55"
      3. Fallback: Dateiname ohne Endung

    Returns
    -------
    proband_id : str   z. B. "57" oder "cpt204"
    subject_id : str   z. B. "S57"
    """
    fname = os.path.basename(filepath)
    m_cpt = re.match(r"-?(cpt\d+)-", fname)
    m_num = re.search(r"-(\d+)-", fname)

    if m_cpt:
        proband_id = m_cpt.group(1)
    elif m_num:
        proband_id = m_num.group(1)
    else:
        proband_id = os.path.splitext(fname)[0]

    return proband_id, f"S{proband_id}"
