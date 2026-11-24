"""Estágio 2 — movimento (MOVING vs NÃO-MOVING) com as features Core7.

O ESP32 calcula 12 features temporais por janela de 50x51 (src/rx/csi_features.cpp).
O Core7 usa 7 delas, que medem variação entre pacotes (não a forma média):
  f3 carrier_change_mean, f4 carrier_change_median, f5 carrier_change_p90,
  f7 frame_change_std,    f8 frame_change_p90,
  f9 shape_change_mean,   f10 shape_change_p90
Calibração: ref[j] = mediana das 20 janelas EMPTY; x[j] = log(max(f,1e-9)/max(ref,1e-9)).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .labels import load_manifest
from .paths import DATA

FEATURE_NAMES_12 = (
    'carrier_std_mean', 'carrier_std_median', 'carrier_std_p90',
    'carrier_change_mean', 'carrier_change_median', 'carrier_change_p90',
    'frame_change_mean', 'frame_change_std', 'frame_change_p90',
    'shape_change_mean', 'shape_change_p90', 'frame_spread_mean')
CORE7 = (3, 4, 5, 7, 8, 9, 10)
FEATURE_NAMES = tuple(FEATURE_NAMES_12[i] for i in CORE7)
CONDITIONS = ('empty', 'static', 'moving')
CAL_WINDOWS = 20
EPS = 1e-9


def log_ratio(f7: np.ndarray, ref7: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(f7, EPS) / np.maximum(ref7, EPS))


def _load_csv(path: Path) -> np.ndarray:
    with path.open(newline='', encoding='utf-8-sig') as fh:
        rows = [[float(r[f'f{i}']) for i in range(12)] for r in csv.DictReader(fh)]
    x = np.asarray(rows, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError(f'{path}: CSV inválido')
    return x


def load_setup(name: str, campaign: str, folder: Path) -> dict:
    raw = {c: _load_csv(folder / f'{c}.csv') for c in CONDITIONS}
    ref = np.median(raw['empty'][:CAL_WINDOWS], axis=0)
    # As janelas de calibração não entram em treino/avaliação.
    raw['empty'] = raw['empty'][CAL_WINDOWS:]
    x = {c: log_ratio(v[:, CORE7], ref[list(CORE7)]) for c, v in raw.items()}
    return dict(name=name, campaign=campaign, ref12=ref, raw=raw, x=x)


def setups(group: str = 'train') -> list[dict]:
    man = load_manifest()['motion_feature_setups']
    return [load_setup(n, c, DATA / f) for n, c, f, *_ in man[group]]


def stack(sets: list[dict]):
    """X (n,7), y (1=moving), cells (setup:condition), setup names, condition."""
    xs, ys, cells, names, conds = [], [], [], [], []
    for s in sets:
        for c in CONDITIONS:
            v = s['x'][c]
            xs.append(v)
            ys.append(np.full(len(v), int(c == 'moving')))
            cells.append(np.full(len(v), f"{s['name']}:{c}", dtype=object))
            names.append(np.full(len(v), s['name'], dtype=object))
            conds.append(np.full(len(v), c, dtype=object))
    return (np.vstack(xs), np.concatenate(ys), np.concatenate(cells),
            np.concatenate(names), np.concatenate(conds))
