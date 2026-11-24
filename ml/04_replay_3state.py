#!/usr/bin/env python3
"""Etapa 4 — replay do sistema completo de 3 estados nos logs gravados.

Para cada sessão CSI_SIG:
  * presença:   probabilidades FORA-DA-DOBRA da etapa 2 (cada sessão prevista por um modelo
                que não a viu no treino); sessões 'holdout' usam o modelo final INT8;
  * movimento:  estado estável MOVING do detector Core7 que estava gravado no ESP32 durante
                a captura (reconstruído das linhas CSI_FRAME). A etapa 1 mostra que a MLP
                retreinada reproduz esse detector (concordância >= 99,7 %);
  * fusão:      MOVING > STATIC > EMPTY  ->  LED aceso / piscando / apagado.
Métricas: presença binária (EMPTY x OCUPADO) da saída final e matriz 3x3.

Uso:  python ml/04_replay_3state.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from csi import presence as P
from csi.export import Int8Runner
from csi.labels import LABEL_ID, LABELS, sessions
from csi.logs import motion_state_per_window, parse_frame_states
from csi.metrics import binary_metrics, confusion
from csi.paths import FIGURES, MODELS, RESULTS
from csi.plots import COLORS, STATE_COLORS, plt, save, shade_phases
from csi.temporal import STATE_NAMES, fuse_states, run_filter


def main():
    pre = json.loads((MODELS / 'presence' / 'preprocessing.json').read_text(encoding='utf-8'))
    mean, std = np.array(pre['scaler_mean']), np.array(pre['scaler_std'])
    runner = Int8Runner((MODELS / 'presence' / 'presence_int8.tflite').read_bytes())
    oof = pd.read_csv(RESULTS / 'presence_oof.csv')

    replays, rows, conf = [], [], {}
    for s in sessions():
        tb = P.session_table(s)
        if s.role == 'dataset':
            o = oof[oof.session == s.id]
            p = o['p_mlp'].to_numpy()
            source = 'fora-da-dobra (float)'
        else:
            z = ((tb[P.X_COLS].to_numpy() - mean) / std).astype(np.float32)
            p = runner(z)
            source = 'modelo final INT8'
        pres_active, _ = run_filter(p)
        frames = parse_frame_states(s.path)
        moving = motion_state_per_window(frames, tb['window'].to_numpy(), 'mlp_stable')
        # Nas capturas, o detector de movimento do ESP32 usava a referência EMPTY gravada na NVS
        # em OUTRO momento. Se ele marcava MOVING durante o próprio intervalo EMPTY de calibração,
        # a referência era de outra geometria (inválida) e o estágio de movimento é descartado
        # nessa sessão. O firmware novo calibra os dois estágios juntos, o que evita esse caso.
        motion_valid = bool(np.mean(moving[tb['calibration'].to_numpy()]) < 0.5)
        if not motion_valid:
            moving = np.zeros_like(moving)
        state = fuse_states(pres_active, moving)
        replays.append(dict(s=s, tb=tb, p=p, pres=pres_active, moving=moving, state=state, source=source,
                            motion_valid=motion_valid))
        sc = tb['scored'].to_numpy()
        y = (tb['label'] != 'EMPTY').to_numpy()
        for name, pred in (('apenas presença', pres_active), ('3 estados (presença OU movimento)', state > 0)):
            m = binary_metrics(y[sc], pred[sc])
            rows.append(dict(session=s.id, role=s.role, output=name, motion_reference_valid=motion_valid,
                             n=m['n'], recall=m['recall'],
                             specificity=m['specificity'], balanced_accuracy=m['balanced_accuracy']))
        yt = np.array([LABEL_ID[v] for v in tb.loc[sc, 'label']])
        conf.setdefault(s.role, np.zeros((3, 3), dtype=int))
        conf[s.role] += confusion(yt, state[sc])
        conf[s.id] = confusion(yt, state[sc])

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / 'replay_3state.csv', index=False, float_format='%.4f')
    summary = {'presence_binary': res.to_dict(orient='records'),
               'confusion_3x3': {k: v.tolist() for k, v in conf.items()}, 'labels': list(LABELS)}
    (RESULTS / 'replay_3state.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # ---------------- figura: linhas do tempo do LED
    fig, axes = plt.subplots(len(replays), 1, figsize=(12, 1.55 * len(replays)))
    for ax, r in zip(axes, replays):
        t = r['tb']['t_s'].to_numpy()
        shade_phases(ax, r['s'].phases, ymax=2.05)
        ax.plot(t, 1.05 + 0.9 * r['p'], color='#08306b', lw=0.6)
        for k, c in enumerate(STATE_COLORS):
            ax.fill_between(t, 0.05, 0.9, where=r['state'] == k, step='post', color=c, lw=0)
        ax.set_ylim(0, 2.3)
        ax.set_yticks([0.47, 1.5])
        ax.set_yticklabels(['LED', 'P(pres.)'], fontsize=7)
        ax.set_xlim(t[0], t[-1])
        star = '*' if not r['motion_valid'] else ''
        ax.set_ylabel(f"{r['s'].id}{star}", fontsize=8, rotation=0, ha='right', va='center')
    from matplotlib.patches import Patch
    handles = [Patch(color=STATE_COLORS[0], label='saída EMPTY (LED apagado)'),
               Patch(color=STATE_COLORS[1], label='saída STATIC (LED pisca)'),
               Patch(color=STATE_COLORS[2], label='saída MOVING (LED aceso)')] + \
              [Patch(color=COLORS[k], alpha=0.55, label=f'verdade: {k}') for k in ('EMPTY', 'STATIC', 'MOVING')]
    axes[0].legend(handles=handles, fontsize=6.5, ncol=3, loc='upper left', bbox_to_anchor=(0, 2.0))
    axes[-1].set_xlabel('tempo desde o boot do RX (s)    '
                        '* estágio de movimento descartado: o ESP32 usava referência NVS de outra geometria')
    save(fig, FIGURES / 'replay_3state_timelines.png')

    # ---------------- figura: matriz de confusão
    keys = [k for k in ('dataset', 'holdout') if k in conf]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.6))
    axes = np.atleast_1d(axes)
    titles = {'dataset': '6 sessões (fora-da-dobra)', 'holdout': 'holdout (INT8)'}
    for ax, key in zip(axes, keys):
        m = conf[key]
        norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                if m[i].sum():
                    ax.text(j, i, f'{m[i, j]}\n{100 * norm[i, j]:.0f}%', ha='center', va='center', fontsize=8,
                            color='white' if norm[i, j] > 0.6 else 'black')
        ax.set_xticks(range(3))
        ax.set_xticklabels(STATE_NAMES, fontsize=8)
        ax.set_yticks(range(3))
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel('saída do sistema')
        ax.set_ylabel('verdade')
        ax.set_title(titles[key], fontsize=9)
    fig.suptitle('Sistema de 3 estados — janelas seguras', fontsize=10, y=1.02)
    save(fig, FIGURES / 'replay_confusion.png')

    print(res.round(3).to_string(index=False))
    for k in keys:
        print(f'\nConfusão 3x3 [{k}] (linhas=verdade EMPTY/STATIC/MOVING):\n{conf[k]}')


if __name__ == '__main__':
    main()
