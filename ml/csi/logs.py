"""Parsers dos logs seriais do RX.

Linhas usadas:
  CSI_SIG     assinatura espectral por janela (51 subportadoras LLTF, Q12 hex) + RSSI + mediana bruta
  CSI_FRAME   saída do detector de movimento do firmware antigo (estado estável Core7 MLP/logística)
  CSI_PARITY  features Core7 brutas por janela + referências EMPTY (logs de paridade)
  CSI_STATE   saída do firmware novo de 3 estados (este projeto)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

N_CARRIERS = 51
Q12 = 4096.0

SIG_RE = re.compile(
    r"CSI_SIG: v=1 window=(\d+) t_ms=(\d+) rssi=(-?[\d.]+) "
    r"raw_med=([\d.]+) clip=(\d+) q12=([0-9A-Fa-f]+)")
LOG_MS_RE = re.compile(r"^[IWE] \((\d+)\) ([A-Z0-9_]+): (.*)$")
FIELD_RE = re.compile(r"(\w+)=([^\s]+)")
QCOLS = [f'q{i:02d}' for i in range(N_CARRIERS)]


def _read(path: Path) -> str:
    return Path(path).read_text(encoding='utf-8', errors='replace')


def parse_signatures(path: Path) -> pd.DataFrame:
    """Uma linha por janela CSI (~480 ms). Colunas: window, t_s, rssi, raw_med, clip, q00..q50.

    q?? é a amplitude média da subportadora na janela de 2 s (cada pacote já normalizado
    pela mediana das 51 subportadoras no ESP32), decodificada de Q12 para float.
    """
    rows = []
    for m in SIG_RE.finditer(_read(path)):
        window, ms, rssi, raw_med, clip, hexdump = m.groups()
        if len(hexdump) != 4 * N_CARRIERS:
            continue  # linha truncada na serial
        q = [int(hexdump[i:i + 4], 16) / Q12 for i in range(0, 4 * N_CARRIERS, 4)]
        rows.append((int(window), int(ms) / 1000.0, float(rssi), float(raw_med), int(clip), *q))
    df = pd.DataFrame(rows, columns=['window', 't_s', 'rssi', 'raw_med', 'clip', *QCOLS])
    if df.empty:
        raise ValueError(f'{path}: nenhuma linha CSI_SIG válida')
    if (np.diff(df['t_s'].to_numpy()) < 0).any():
        raise ValueError(f'{path}: t_ms reinicia (mais de um boot); separe em arquivos')
    return df.reset_index(drop=True)


def parse_frame_states(path: Path) -> pd.DataFrame:
    """Linhas CSI_FRAME do firmware binário (estágios 0005-0017C).

    O firmware imprime CSI_FRAME em toda TRANSIÇÃO de estado estável (MLP ou logística)
    e a cada 10 janelas. Logo o estado estável é reconstruível como degrau por janela.
    """
    rows = []
    for line in _read(path).splitlines():
        m = LOG_MS_RE.match(line.strip())
        if not m or m.group(2) != 'CSI_FRAME' or 'window=' not in m.group(3):
            continue
        f = dict(FIELD_RE.findall(m.group(3)))
        if 'window' not in f:
            continue  # linha de configuração (every_window=...)
        rows.append(dict(window=int(f['window']), log_ms=int(m.group(1)),
                         f9_ratio=float(f.get('f9_ratio', 'nan')),
                         core7_p=float(f.get('core7_p', 'nan')),
                         mlp_stable=f.get('stable', ''),
                         logistic_p=float(f.get('logistic_p', 'nan')),
                         logistic_stable=f.get('logistic_stable', '')))
    return pd.DataFrame(rows)


def motion_state_per_window(frames: pd.DataFrame, windows: np.ndarray, column: str) -> np.ndarray:
    """Estado estável (True=MOVING) por janela, propagando o último valor logado.

    Válido porque o firmware loga toda mudança de estado. Janelas antes do primeiro
    registro (ou em WARMUP) ficam False (o LED também fica apagado nessas condições).
    """
    out = np.zeros(len(windows), dtype=bool)
    if frames.empty:
        return out
    fr = frames.sort_values('window')
    fw = fr['window'].to_numpy()
    fs = (fr[column].to_numpy() == 'MOVING')
    idx = np.searchsorted(fw, windows, side='right') - 1
    ok = idx >= 0
    out[ok] = fs[idx[ok]]
    return out


def _floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split(',')], dtype=np.float64)


def parse_parity(path: Path) -> dict:
    """Log CSI_PARITY (firmware 0006-0010): referência EMPTY + 7 features Core7 por janela."""
    refs, config, rows = None, {}, []
    for line in _read(path).splitlines():
        m = LOG_MS_RE.match(line.strip())
        if not m or m.group(2) != 'CSI_PARITY':
            continue
        ms, body = int(m.group(1)), m.group(3)
        f = dict(FIELD_RE.findall(body))
        if body.startswith('config'):
            config = {k: float(v) for k, v in f.items()}
        elif 'refs' in f:
            refs = _floats(f['refs'])
        elif 'window' in f and 'f' in f:
            rows.append(dict(window=int(f['window']), t_s=ms / 1000.0, f=_floats(f['f']),
                             out_q=int(f['out_q']) if 'out_q' in f else None))
    if refs is None or not rows:
        raise ValueError(f'{path}: sem refs/janelas CSI_PARITY')
    return dict(refs=refs, config=config, rows=rows,
                t_s=np.array([r['t_s'] for r in rows]),
                window=np.array([r['window'] for r in rows]),
                f=np.vstack([r['f'] for r in rows]))


def parse_state_lines(path: Path) -> pd.DataFrame:
    """Linhas CSI_STATE do firmware de 3 estados deste projeto."""
    rows = []
    for line in _read(path).splitlines():
        m = LOG_MS_RE.match(line.strip())
        if not m or m.group(2) != 'CSI_STATE' or 'window=' not in m.group(3):
            continue
        f = dict(FIELD_RE.findall(m.group(3)))
        if 'window' not in f:
            continue
        rows.append({k: (float(v) if re.fullmatch(r'-?[\d.]+(e-?\d+)?', v) else v) for k, v in f.items()})
    return pd.DataFrame(rows)


REF_RE = re.compile(r'CSI_REF v=2 origin=(\S+) fp=([0-9A-Fa-f]+) rssi_ref=(\S+) core7=(\S+) scale=(\S+) ref=(\S+)')
F12_RE = re.compile(r'CSI_F12: window=(\d+) f=(\S+)')
VEC_RE = re.compile(r'CSI_VEC: window=(\d+) qm=(\S+) om=(-?\d+) qp=(\S+) op=(-?\d+)')


def parse_device_log(path: Path) -> dict:
    """Log do firmware de 3 estados no modo VALIDATION (ou CAPTURE).

    ref    última linha CSI_REF: referência EMPTY usada pela placa (core7, rssi_ref, scale, ref[51])
    sig    DataFrame CSI_SIG;  f12  {window: 12 features};  vec  DataFrame CSI_VEC;  state  CSI_STATE
    """
    text = _read(path)
    refs = REF_RE.findall(text)
    ref = None
    if refs:
        origin, fp, rssi_ref, core7, scale, prof = refs[-1]
        ref = dict(origin=origin, fingerprint=int(fp, 16), rssi_ref=float(rssi_ref),
                   core7=_floats(core7), scale=_floats(scale), ref=_floats(prof))
    f12 = {int(w): _floats(v) for w, v in F12_RE.findall(text)}
    vec = pd.DataFrame([dict(window=int(w), qm=[int(v) for v in qm.split(',')], om=int(om),
                             qp=[int(v) for v in qp.split(',')], op=int(op))
                        for w, qm, om, qp, op in VEC_RE.findall(text)])
    return dict(ref=ref, sig=parse_signatures(path), f12=f12, vec=vec, state=parse_state_lines(path))
