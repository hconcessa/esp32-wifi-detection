#!/usr/bin/env python3
"""Analisa um log NOVO do RX (gravado com tools/capture_serial.py ou pio device monitor).

* Recalcula P(presença) no PC com o modelo INT8 final, calibrando nas 20 primeiras
  assinaturas a partir de --cal-start (precisa ser um trecho comprovadamente VAZIO).
* Se o log vier do firmware de 3 estados, sobrepõe o estado que o ESP32 decidiu
  (linhas CSI_STATE) e a referência que ele usou (linha CSI_REF).
* Se houver marcações MARK (capture_serial.py), desenha as fases.

Uso:
    python ml/05_analyze_log.py data/dataset/presence/G.log --cal-start 20 --out docs/figures/F.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from csi import presence as P
from csi.export import Int8Runner
from csi.labels import Session
from csi.logs import parse_state_lines
from csi.paths import MODELS
from csi.plots import STATE_COLORS, plt, save
from csi.temporal import run_filter

MARK_RE = re.compile(r'^MARK t_s=([\d.]+) name=(\S+)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', type=Path)
    ap.add_argument('--cal-start', type=float, default=20.0, help='início (s de boot) do trecho EMPTY de calibração')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    s = Session(id=args.log.stem, path=args.log, role='new', calibration_start_s=args.cal_start)
    tb = P.session_table(s)
    pre = json.loads((MODELS / 'presence' / 'preprocessing.json').read_text(encoding='utf-8'))
    z = ((tb[P.X_COLS].to_numpy() - np.array(pre['scaler_mean'])) / np.array(pre['scaler_std'])).astype(np.float32)
    p = Int8Runner((MODELS / 'presence' / 'presence_int8.tflite').read_bytes())(z)
    active, mean = run_filter(p)
    t = tb['t_s'].to_numpy()
    marks = [(float(m.group(1)), m.group(2)) for m in
             (MARK_RE.match(line) for line in args.log.read_text(errors='replace').splitlines()) if m]
    dev = parse_state_lines(args.log)

    rows = 3 if len(dev) else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 1.9 * rows), sharex=True)
    axes[0].plot(t, p, lw=0.6, color='#6baed6', label='P(presença) recalculada no PC')
    axes[0].plot(t, mean, lw=1.0, color='#08306b', label='média de 5')
    cal = tb['calibration'].to_numpy()
    axes[0].axvspan(t[cal].min(), t[cal].max(), color='#31a354', alpha=0.35, lw=0, label='calibração')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, loc='upper left')
    axes[1].fill_between(t, 0, 1, where=active, step='post', color=STATE_COLORS[1], lw=0)
    axes[1].set_yticks([])
    axes[1].set_ylabel('presença\n(PC)', fontsize=8)
    if len(dev):
        st = dev['state'].astype(str).to_numpy()
        td = dev['t_ms'].astype(float).to_numpy() / 1000
        for k, name in enumerate(('EMPTY', 'STATIC', 'MOVING')):
            axes[2].fill_between(td, 0, 1, where=st == name, step='post', color=STATE_COLORS[k], lw=0)
        axes[2].set_yticks([])
        axes[2].set_ylabel('estado\nno ESP32', fontsize=8)
    for ax in axes:
        for tm, name in marks:
            ax.axvline(tm, color='k', lw=0.6)
            if ax is axes[0]:
                ax.text(tm, 1.02, name, fontsize=7, va='bottom')
    axes[-1].set_xlabel('tempo desde o boot do RX (s)')
    out = args.out or args.log.with_suffix('.png')
    save(fig, out)
    print(f'{len(t)} janelas, presença (PC) em {100 * active[~cal].mean():.1f}% do tempo após a calibração')
    print(f'figura: {out}')


if __name__ == '__main__':
    main()
