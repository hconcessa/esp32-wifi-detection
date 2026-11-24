#!/usr/bin/env python3
"""Fase 7 (refinamento) — por que estas features, esta normalização e este modelo?

Reproduz, com a MESMA validação cruzada por sessão da etapa 2 (estado estável após o filtro),
as variantes comparadas antes de congelar o modelo de presença:

  normalização   'global'   : x = log(F) e z-score com média/desvio do treino
                 'calibração': x = log(F / ruído da própria calibração)   <- adotada
  features       forma (5) | forma+RSSI (6) | forma+RSSI+temporal (7) <- adotado | +mediana bruta (8)
  modelo         regressão logística | MLP 16-8 (embarcada)
  filtro         decisão por janela  x  estado estável (média de 5 + histerese)

Uso:  python ml/02b_presence_refinement.py      (~1 min)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from csi import presence as P
from csi.labels import sessions
from csi.metrics import binary_metrics
from csi.nets import cell_balanced_weights, predict_proba, train_binary, weighted_mean_std
from csi.paths import FIGURES, RESULTS, ensure_dirs
from csi.plots import plt, save
from csi.temporal import run_filter

HP = dict(hidden=(16, 8), epochs=300, lr=0.003, weight_decay=1e-3, seed=42)
FEATURE_SETS = {
    'forma (5)': [0, 1, 2, 3, 4],
    'forma + RSSI (6)': [0, 1, 2, 3, 4, 5],
    'forma + RSSI + temporal (7)': [0, 1, 2, 3, 4, 5, 6],
    '+ mediana bruta (8)': [0, 1, 2, 3, 4, 5, 6, 7],
}
RAWMED_FLOOR = 0.01


def session_matrices():
    out = []
    for s in sessions('dataset'):
        tb = P.session_table(s)
        cal = tb['calibration'].to_numpy()
        f7 = tb[list(P.FEATURE_NAMES)].to_numpy()
        rm = tb['raw_med'].to_numpy()
        rawmed = np.abs(np.log(rm / np.median(rm[cal])))[:, None]
        f8 = np.hstack([f7, rawmed])
        scale = np.maximum(np.median(f8[cal], axis=0), np.r_[P.FLOORS, RAWMED_FLOOR])
        out.append(dict(id=s.id, f=f8, scale=scale, y=(tb['label'] != 'EMPTY').to_numpy().astype(int),
                        scored=tb['scored'].to_numpy()))
    return out


def inputs(d, cols, norm):
    f = d['f'][:, cols]
    if norm == 'calibração':
        return np.log(np.clip(f / d['scale'][cols], P.CLIP_LO, P.CLIP_HI))
    return np.log(np.clip(f, 1e-6, None))


def loso(data, cols, norm, model):
    rows = []
    for i, test in enumerate(data):
        train = [d for j, d in enumerate(data) if j != i]
        x = np.vstack([inputs(d, cols, norm)[d['scored']] for d in train])
        y = np.concatenate([d['y'][d['scored']] for d in train])
        cells = np.concatenate([np.full(d['scored'].sum(), d['id'], dtype=object) for d in train])
        w = cell_balanced_weights(cells, y)
        mean, std = weighted_mean_std(x, w)
        z = ((x - mean) / std).astype(np.float32)
        zt = ((inputs(test, cols, norm) - mean) / std).astype(np.float32)
        if model == 'MLP':
            p = predict_proba(train_binary(z, y, w, **HP), zt)
        else:
            p = LogisticRegression(C=0.1, max_iter=3000).fit(z, y, sample_weight=w).predict_proba(zt)[:, 1]
        st, _ = run_filter(p)
        sc = test['scored']
        for output, pred in (('janela', p >= 0.5), ('estável', st)):
            m = binary_metrics(test['y'][sc], pred[sc])
            rows.append(dict(session=test['id'], output=output, balanced_accuracy=m['balanced_accuracy'],
                             recall=m['recall'], specificity=m['specificity']))
    return rows


def main():
    ensure_dirs()
    data = session_matrices()
    rows = []
    for norm in ('global', 'calibração'):
        for fname, cols in FEATURE_SETS.items():
            for model in ('logística', 'MLP'):
                for r in loso(data, cols, norm, model):
                    rows.append(dict(normalization=norm, features=fname, model=model, **r))
                print(f'  {norm:10s} {fname:28s} {model}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / 'presence_refinement_by_session.csv', index=False, float_format='%.4f')
    summ = (df.groupby(['normalization', 'features', 'model', 'output'], sort=False)
              .agg(mean_balanced_accuracy=('balanced_accuracy', 'mean'),
                   min_balanced_accuracy=('balanced_accuracy', 'min'),
                   mean_recall=('recall', 'mean'), mean_specificity=('specificity', 'mean')).reset_index())
    summ.to_csv(RESULTS / 'presence_refinement.csv', index=False, float_format='%.4f')

    st = summ[summ.output == 'estável']
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    labels = list(FEATURE_SETS)
    for ax, model in zip(axes, ('logística', 'MLP')):
        for k, (norm, color) in enumerate((('global', '#bdbdbd'), ('calibração', '#3182bd'))):
            g = st[(st.model == model) & (st.normalization == norm)].set_index('features').loc[labels]
            yy = np.arange(len(labels)) + (k - 0.5) * 0.36
            ax.barh(yy, g.mean_balanced_accuracy, 0.34, color=color, label=f'normalização {norm}')
            ax.plot(g.min_balanced_accuracy, yy, '|', color='k', ms=10)
            for v, yv in zip(g.mean_balanced_accuracy, yy):
                ax.text(v + 0.005, yv, f'{v:.3f}', va='center', fontsize=7)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0.5, 1.0)
        ax.set_title(f'{model}', fontsize=9)
        ax.set_xlabel('acurácia balanceada média por sessão (| = pior sessão)')
        ax.grid(axis='x', alpha=0.3)
    axes[0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Refinamento da presença — validação cruzada por sessão, estado estável', fontsize=10, y=1.02)
    save(fig, FIGURES / 'presence_refinement.png')
    print(summ.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
