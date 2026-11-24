"""Manifest de sessões e rotulagem por fase."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import DATA, MANIFEST

LABELS = ('EMPTY', 'STATIC', 'MOVING')
LABEL_ID = {name: i for i, name in enumerate(LABELS)}


@dataclass
class Session:
    id: str
    path: Path
    role: str
    calibration_start_s: float
    phases: list = field(default_factory=list)  # (name, label, start, end)
    notes: str = ''
    geometry: str = ''


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sessions(role: str | None = None, path: Path = MANIFEST) -> list[Session]:
    man = load_manifest(path)
    out = []
    for s in man['sessions']:
        if role is not None and s['role'] != role:
            continue
        out.append(Session(id=s['id'], path=DATA / s['file'], role=s['role'],
                           calibration_start_s=float(s['calibration_start_s']),
                           phases=[tuple(p) for p in s['phases']],
                           notes=s.get('notes', ''), geometry=s.get('geometry', '')))
    return out


def guards(path: Path = MANIFEST) -> tuple[float, float]:
    pol = load_manifest(path)['label_policy']
    return float(pol['guard_start_s']), float(pol['guard_end_s'])


def label_times(t: np.ndarray, phases, guard_start: float, guard_end: float):
    """Retorna (label_str, phase_name, scored) por instante t.

    label_str: EMPTY/STATIC/MOVING/UNKNOWN/'' (fora de qualquer fase)
    scored: True se a janela está no núcleo seguro de uma fase rotulada (não UNKNOWN).
    """
    n = len(t)
    label = np.full(n, '', dtype=object)
    phase = np.full(n, '', dtype=object)
    scored = np.zeros(n, dtype=bool)
    for name, lab, start, end in phases:
        inside = (t >= start) & (t < end)
        label[inside] = lab
        phase[inside] = name
        core = (t >= start + guard_start) & (t < end - guard_end)
        if lab in LABELS:
            scored |= core
    scored &= np.isin(label, LABELS)
    return label, phase, scored
