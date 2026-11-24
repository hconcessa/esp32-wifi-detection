#!/usr/bin/env python3
"""Fase 5 (treino) — modelo de PRESENÇA (EMPTY vs OCUPADO) em PyTorch, sobre assinaturas CSI_SIG.

Protocolo:
  * Sessões role='dataset' (A, B, C, D, E, F): validação cruzada leave-one-session-out
    (a sessão de teste fica inteira fora do treino e do scaler). Cada dobra registra as curvas
    de perda/acurácia por época (treino x sessão de fora).
  * Modelo final treinado com todas as sessões -> models/presence/.
  * Sessões role='holdout' (nenhuma hoje): avaliadas uma única vez com o modelo final.
  Comparação: limiar não supervisionado (RMS > 3x ruído da calibração), regressão
  logística e MLP PyTorch (o modelo embarcado). Todas passam pelo mesmo filtro temporal.

Uso:  python ml/02_train_presence.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from csi import presence as P
from csi.labels import LABELS, sessions
from csi.metrics import binary_metrics, longest_run
from csi.nets import cell_balanced_weights, n_params, predict_proba, train_binary, weighted_mean_std
from csi.paths import FIGURES, MODELS, RESULTS, ensure_dirs
from csi.plots import COLORS, bar_groups, phase_legend, plot_curves, plt, save, shade_phases
from csi.temporal import run_filter

HP = dict(hidden=(16, 8), epochs=300, lr=0.003, weight_decay=1e-3, seed=42)
METHODS = ('threshold_3x', 'logistic', 'mlp')
HOP_S = 0.48


def matrix(tables):
    xs, ys, cells = [], [], []
    for tb in tables:
        m = tb['scored'].to_numpy()
        xs.append(tb.loc[m, P.X_COLS].to_numpy(np.float64))
        ys.append((tb.loc[m, 'label'] != 'EMPTY').to_numpy().astype(int))
        cells.append(np.full(m.sum(), tb['session'].iloc[0], dtype=object))
    return np.vstack(xs), np.concatenate(ys), np.concatenate(cells)


def fit(tables, held=None):
    x, y, cells = matrix(tables)
    w = cell_balanced_weights(cells, y)
    mean, std = weighted_mean_std(x, w)
    z = ((x - mean) / std).astype(np.float32)
    monitor = {'treino': (z, y, w)}
    if held:
        xh, yh, ch = matrix(held)
        monitor['validação'] = (((xh - mean) / std).astype(np.float32), yh, cell_balanced_weights(ch, yh))
    net, hist = train_binary(z, y, w, monitor=monitor, **HP)
    logit = LogisticRegression(C=0.1, max_iter=3000).fit(z, y, sample_weight=w)
    return dict(net=net, logit=logit, mean=mean, std=std, z=z, y=y, history=hist)


def predict(model, tb):
    z = ((tb[P.X_COLS].to_numpy(np.float64) - model['mean']) / model['std']).astype(np.float32)
    # regra não supervisionada: RMS da forma > 3x o ruído da calibração (log(3) na entrada)
    thr = (tb['x_shape_rms'].to_numpy() > np.log(3.0)).astype(float)
    return {'threshold_3x': thr, 'logistic': model['logit'].predict_proba(z)[:, 1],
            'mlp': predict_proba(model['net'], z)}


def evaluate(tb, probs, fold):
    rows, phases = [], []
    sc = tb['scored'].to_numpy()
    y = (tb['label'] != 'EMPTY').to_numpy()
    t = tb['t_s'].to_numpy()
    for name, p in probs.items():
        st, _ = run_filter(p)
        for kind, pred in (('janela', p >= 0.5), ('estável', st)):
            m = binary_metrics(y[sc], pred[sc])
            rows.append(dict(fold=fold, session=tb['session'].iloc[0], method=name, output=kind,
                             **{k: m[k] for k in ('n', 'TP', 'TN', 'FP', 'FN', 'recall', 'specificity',
                                                  'balanced_accuracy', 'f1')},
                             longest_false_presence_s=longest_run(pred & sc & ~y) * HOP_S))
        for ph in tb.loc[sc, 'phase'].unique():
            pm = sc & (tb['phase'] == ph).to_numpy()
            lab = tb.loc[pm, 'label'].iloc[0]
            rate = float(np.mean(st[pm]))
            phases.append(dict(fold=fold, session=tb['session'].iloc[0], method=name, phase=ph,
                               label=lab, n=int(pm.sum()), presence_rate=rate,
                               correct_rate=rate if lab != 'EMPTY' else 1 - rate,
                               t_start=float(t[pm].min()), t_end=float(t[pm].max())))
    return rows, phases


def timeline(ax, tb, p, title, phases):
    shade_phases(ax, phases, ymax=1.02)
    st, mean = run_filter(p)
    t = tb['t_s'].to_numpy()
    ax.plot(t, p, lw=0.6, color='#6baed6', label='P(presença) por janela')
    ax.plot(t, mean, lw=1.0, color='#08306b', label='média de 5 janelas')
    ax.fill_between(t, 0, 0.07, where=st, step='post', color='#3182bd', lw=0, label='estado estável OCUPADO')
    cal = tb['calibration'].to_numpy()
    if cal.any():
        ax.axvspan(t[cal].min(), t[cal].max(), color='#31a354', alpha=0.35, lw=0, label='calibração EMPTY (20 janelas)')
    ax.axhline(0.65, color='#444', lw=0.5, ls='--')
    ax.axhline(0.35, color='#444', lw=0.5, ls=':')
    ax.set_ylim(0, 1.12)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(title, fontsize=8)


def main():
    ensure_dirs()
    data = [P.session_table(s) for s in sessions('dataset')]
    hold = [P.session_table(s) for s in sessions('holdout')]
    meta = {s.id: s for s in sessions()}
    sid = lambda tb: tb['session'].iloc[0]  # noqa: E731

    # ---------------- validação cruzada leave-one-session-out
    rows, phase_rows, oof, histories = [], [], {}, []
    for i, tb in enumerate(data):
        model = fit([d for j, d in enumerate(data) if j != i], [tb])
        probs = predict(model, tb)
        oof[sid(tb)] = probs
        histories.append((sid(tb), model['history']))
        r, ph = evaluate(tb, probs, 'LOSO')
        rows += r
        phase_rows += ph

    # ---------------- modelo final (todas as sessões) — desempenho no próprio treino
    final = fit(data)
    for tb in data:
        r, _ = evaluate(tb, predict(final, tb), 'treino (in-sample)')
        rows += r
    for tb in hold:
        r, ph = evaluate(tb, predict(final, tb), 'HOLDOUT')
        rows += r
        phase_rows += ph
    res = pd.DataFrame(rows)
    phase_df = pd.DataFrame(phase_rows)
    res.to_csv(RESULTS / 'presence_validation.csv', index=False, float_format='%.4f')
    phase_df.to_csv(RESULTS / 'presence_phase_rates.csv', index=False, float_format='%.4f')
    pd.concat([pd.DataFrame({'session': tb['session'], 'window': tb['window'], 't_s': tb['t_s'],
                             **{f'p_{m}': oof[sid(tb)][m] for m in METHODS}})
               for tb in data]).to_csv(RESULTS / 'presence_oof.csv', index=False, float_format='%.5f')
    pd.DataFrame([dict(fold=f, set=k, epoch=i + 1, loss=l, balanced_accuracy=b)
                  for f, h in histories for k, v in h.items()
                  for i, (l, b) in enumerate(zip(v['loss'], v['balanced_accuracy']))]
                 ).to_csv(RESULTS / 'presence_training_curves.csv', index=False, float_format='%.5f')

    pooled = []
    for (fold, method, output), g in res.groupby(['fold', 'method', 'output'], sort=False):
        tp, tn, fp, fn = g.TP.sum(), g.TN.sum(), g.FP.sum(), g.FN.sum()
        pooled.append(dict(fold=fold, method=method, output=output, sessions=len(g),
                           mean_balanced_accuracy=g.balanced_accuracy.mean(),
                           min_balanced_accuracy=g.balanced_accuracy.min(),
                           pooled_recall=tp / (tp + fn), pooled_specificity=tn / (tn + fp)))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(RESULTS / 'presence_summary.csv', index=False, float_format='%.4f')

    ep = []
    for (fold, method), g in phase_df.groupby(['fold', 'method'], sort=False):
        occ, emp = g[g.label != 'EMPTY'], g[g.label == 'EMPTY']
        ep.append(dict(fold=fold, method=method,
                       occupied_episodes_detected=int((occ.presence_rate >= 0.5).sum()), occupied_episodes=len(occ),
                       empty_episodes_clean=int((emp.presence_rate < 0.5).sum()), empty_episodes=len(emp)))
    episodes = pd.DataFrame(ep)
    episodes.to_csv(RESULTS / 'presence_episodes.csv', index=False)

    # ---------------- salvar modelo
    out = MODELS / 'presence'
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(final['net'].state_dict(), out / 'presence_mlp.pt')
    np.save(out / 'representative_inputs.npy', final['z'])
    pre = dict(stage='presence', features=list(P.FEATURE_NAMES), floors=P.FLOORS.tolist(),
               calibration_windows=P.CAL_WINDOWS, lag_windows=P.LAG, clip=[P.CLIP_LO, P.CLIP_HI],
               transform='p=s/mean(s); F=features(p,ref,rssi); x=log(clip(F/scale,0.01,1e4)); z=(x-mean)/std',
               scaler_mean=final['mean'].astype(np.float32).tolist(),
               scaler_std=final['std'].astype(np.float32).tolist(),
               architecture=[len(P.FEATURE_NAMES), *HP['hidden'], 1], parameters=n_params(final['net']),
               hyperparameters=HP, training_sessions=[sid(d) for d in data],
               training_windows=int(len(final['y'])),
               final_training_loss=final['history']['treino']['loss'][-1],
               filter=dict(n=5, enter=0.65, exit=0.35))
    (out / 'preprocessing.json').write_text(json.dumps(pre, indent=2), encoding='utf-8')

    # ---------------- figuras
    plot_curves(histories, final['history'], FIGURES / 'presence_training_curves.png',
                'Presença — curvas de treino (cada linha fina = uma dobra; validação = sessão fora do treino)')

    allx = pd.concat([pd.DataFrame(d.to_dict('list')) for d in data], ignore_index=True)
    allx = allx[allx['scored']]
    fig, axes = plt.subplots(1, 7, figsize=(13, 2.7))
    for j, (ax, col) in enumerate(zip(axes, P.X_COLS)):
        bp = ax.boxplot([allx.loc[allx.label == lab, col] for lab in LABELS], showfliers=False, patch_artist=True)
        for patch, lab in zip(bp['boxes'], LABELS):
            patch.set_facecolor(COLORS[lab])
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['E', 'S', 'M'], fontsize=7)
        ax.set_title(P.FEATURE_NAMES[j].replace('_', '\n'), fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel('x = log(F / ruído da calibração)', fontsize=7)
    fig.suptitle('Entradas do modelo de presença por classe (E=EMPTY, S=STATIC, M=MOVING)', fontsize=9, y=1.1)
    save(fig, FIGURES / 'presence_features.png')

    tb = data[[sid(d) for d in data].index('D')]
    prof, ref = tb.attrs['profiles'], tb.attrs['calibration']['ref']
    sc = tb['scored'].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    k = np.arange(51)
    for lab, color in (('EMPTY', '#636363'), ('STATIC', '#3182bd')):
        m = sc & (tb['label'] == lab).to_numpy()
        for i in np.where(m)[0][:: max(1, m.sum() // 40)]:
            axes[0].plot(k, prof[i], color=color, lw=0.4, alpha=0.35)
        axes[1].plot(k, np.median(prof[m] - ref, axis=0), color=color, lw=1.4, label=f'{lab} (mediana)')
        axes[1].fill_between(k, *np.percentile(prof[m] - ref, [10, 90], axis=0), color=color, alpha=0.2)
    axes[0].plot(k, ref, color='k', lw=2, label='referência EMPTY (calibração)')
    axes[0].set_title('Perfis de forma p = s/mean(s) — sessão D (cinza: EMPTY, azul: pessoa parada)', fontsize=8)
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, color='k', lw=0.6)
    axes[1].set_title('Desvio p - ref (mediana e faixa 10–90%)', fontsize=8)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel('índice da subportadora LLTF (51 úteis)')
        ax.grid(alpha=0.3)
    save(fig, FIGURES / 'presence_profiles.png')

    sess = [sid(d) for d in data]

    def ba(s, fold, method):
        g = res[(res.session == s) & (res.fold == fold) & (res.method == method) & (res.output == 'estável')]
        return float(g.balanced_accuracy.iloc[0])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    bar_groups(ax, sess, {m: [ba(s, 'LOSO', m) for s in sess] for m in METHODS},
               'acurácia balanceada (estado estável)', ylim=(0.4, 1.08))
    ax.set_title('Presença — validação cruzada: cada sessão prevista por um modelo que não a viu')
    save(fig, FIGURES / 'presence_validation.png')

    fig, axes = plt.subplots(len(data), 1, figsize=(12, 2.0 * len(data)))
    for ax, tb in zip(axes, data):
        timeline(ax, tb, oof[sid(tb)]['mlp'], sid(tb), meta[sid(tb)].phases)
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + phase_legend(axes[0]),
                   fontsize=6.5, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.75))
    axes[-1].set_xlabel('tempo desde o boot do RX (s)')
    save(fig, FIGURES / 'presence_loso_timelines.png')

    print('\n=== PRESENÇA: resumo ===')
    print(pooled.round(3).to_string(index=False))
    print('\n=== PRESENÇA: episódios (fases) ===')
    print(episodes.to_string(index=False))
    print('\n=== PRESENÇA: por sessão (estado estável) ===')
    r = res[res.output == 'estável']
    print(r[['fold', 'session', 'method', 'n', 'recall', 'specificity', 'balanced_accuracy',
             'longest_false_presence_s']].round(3).to_string(index=False))
    h = final['history']['treino']
    print(f"\nTreino final: perda {h['loss'][0]:.3f} -> {h['loss'][-1]:.3f} em {HP['epochs']} épocas")
    print(f'Modelo final: {out / "presence_mlp.pt"} ({n_params(final["net"])} parâmetros)')


if __name__ == '__main__':
    main()
