"""Estágio 1 — presença (EMPTY vs OCUPADO) a partir da assinatura espectral CSI.

Cada janela CSI (50 pacotes reamostrados a 25 Hz = 2 s, salto de 480 ms) gera uma
assinatura s[51]: amplitude média de cada subportadora LLTF (pacotes já normalizados
pela mediana no ESP32), quantizada em Q12. A presença de uma pessoa parada muda o
padrão de multipercurso, isto é, a FORMA do perfil s[51] em relação ao ambiente vazio.

Calibração (igual ao firmware, src/rx/csi_presence.h):
  p      = s / mean(s)                       perfil de forma (invariante a ganho)
  ref    = mediana por subportadora de p nas 20 janelas EMPTY de calibração
  rssi0  = mediana do RSSI nessas janelas
  scale  = max(mediana de cada feature bruta nas 20 janelas, piso)   (ruído do ambiente)

Features brutas por janela (7):
  shape_rms     sqrt(mean((p-ref)^2))
  shape_mae     mean(|p-ref|)
  shape_p90     percentil 90 de |p-ref|
  shape_max     max |p-ref|
  corr_loss     1 - Pearson(p, ref)
  rssi_shift    |rssi - rssi0|
  temporal_delta  sqrt(mean((p_t - p_{t-5})^2))   (mudança em ~2,4 s)

Entrada do modelo: x = log(clip(F/scale, 0.01, 1e4)) -> StandardScaler -> MLP.
Dividir pelo ruído da própria calibração torna as entradas comparáveis entre salas/geometrias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import Session, guards, label_times
from .logs import QCOLS, parse_signatures

FEATURE_NAMES = ('shape_rms', 'shape_mae', 'shape_p90', 'shape_max',
                 'corr_loss', 'rssi_shift', 'temporal_delta')
FLOORS = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-5, 0.25, 1e-3])
CAL_WINDOWS = 20
LAG = 5
CLIP_LO, CLIP_HI = 0.01, 1e4


def shape_profiles(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / q.mean(axis=1, keepdims=True)


def lagged(p: np.ndarray, lag: int = LAG) -> np.ndarray:
    """p[t-lag], usando o perfil mais antigo disponível no início (igual ao ring buffer do ESP32)."""
    idx = np.maximum(np.arange(len(p)) - lag, 0)
    return p[idx]


def raw_features(p: np.ndarray, p_lag: np.ndarray, rssi: np.ndarray,
                 ref: np.ndarray, rssi_ref: float) -> np.ndarray:
    d = p - ref
    a = np.abs(d)
    rms = np.sqrt(np.mean(d ** 2, axis=1))
    mae = a.mean(axis=1)
    # percentil 90 com interpolação linear (mesma regra do C++: idx=(n-1)*0.9)
    p90 = np.quantile(a, 0.9, axis=1, method='linear')
    mx = a.max(axis=1)
    pc = p - p.mean(axis=1, keepdims=True)
    rc = ref - ref.mean()
    den = np.sqrt(np.sum(pc ** 2, axis=1) * np.sum(rc ** 2))
    corr = np.clip(pc @ rc / np.maximum(den, 1e-12), -1.0, 1.0)
    rs = np.abs(rssi - rssi_ref)
    td = np.sqrt(np.mean((p - p_lag) ** 2, axis=1))
    return np.column_stack([rms, mae, p90, mx, 1.0 - corr, rs, td])


def calibrate(p: np.ndarray, p_lag: np.ndarray, rssi: np.ndarray, idx: np.ndarray) -> dict:
    if len(idx) < CAL_WINDOWS:
        raise ValueError(f'calibração requer {CAL_WINDOWS} janelas, há {len(idx)}')
    idx = idx[:CAL_WINDOWS]
    ref = np.median(p[idx], axis=0)
    rssi_ref = float(np.median(rssi[idx]))
    f_cal = raw_features(p[idx], p_lag[idx], rssi[idx], ref, rssi_ref)
    scale = np.maximum(np.median(f_cal, axis=0), FLOORS)
    return dict(ref=ref, rssi_ref=rssi_ref, scale=scale, idx=idx)


def model_inputs(f_raw: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.log(np.clip(f_raw / scale, CLIP_LO, CLIP_HI)).astype(np.float32)


def session_table(s: Session, guard: tuple[float, float] | None = None) -> pd.DataFrame:
    """Tabela por janela: tempo, rótulos, flags de calibração/score, features brutas e entradas."""
    gs, ge = guard if guard else guards()
    df = parse_signatures(s.path)
    t = df['t_s'].to_numpy()
    q = df[QCOLS].to_numpy()
    rssi = df['rssi'].to_numpy()
    p = shape_profiles(q)
    p_lag = lagged(p)
    cal_idx = np.where(t >= s.calibration_start_s)[0]
    cal = calibrate(p, p_lag, rssi, cal_idx)
    f_raw = raw_features(p, p_lag, rssi, cal['ref'], cal['rssi_ref'])
    x = model_inputs(f_raw, cal['scale'])
    label, phase, scored = label_times(t, s.phases, gs, ge)
    is_cal = np.zeros(len(t), dtype=bool)
    is_cal[cal['idx']] = True
    t_cal_end = t[cal['idx']].max()
    operational = t > t_cal_end
    out = pd.DataFrame({'session': s.id, 'role': s.role, 'window': df['window'], 't_s': t,
                        'label': label, 'phase': phase,
                        'scored': scored & operational, 'calibration': is_cal,
                        'operational': operational, 'rssi': rssi, 'raw_med': df['raw_med'].to_numpy()})
    for i, name in enumerate(FEATURE_NAMES):
        out[name] = f_raw[:, i]
    for i, name in enumerate(FEATURE_NAMES):
        out['x_' + name] = x[:, i]
    out.attrs['calibration'] = cal
    out.attrs['profiles'] = p
    return out


X_COLS = ['x_' + n for n in FEATURE_NAMES]
