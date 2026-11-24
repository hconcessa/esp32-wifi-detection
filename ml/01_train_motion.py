#!/usr/bin/env python3
"""Fase 5 (treino) — modelo de MOVIMENTO (Core7) em PyTorch.

1. Validação cruzada deixando um SETUP inteiro fora (5 setups) e deixando uma CAMPANHA
   inteira fora (robustness x independent); compara a MLP com uma regressão logística.
   Em cada dobra registra as curvas de perda/acurácia por época (treino x setup de fora).
2. Treino final com os 5 setups -> models/motion/motion_mlp.pt + preprocessing.json.
3. Teste AO VIVO nos 3 logs de teste (data/tests/motion_live): features e referência EMPTY
   calculadas na própria placa; compara com o modelo que estava gravado no ESP32.

Uso:  python ml/01_train_motion.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from csi import motion as M
from csi.labels import guards, load_manifest
from csi.logs import parse_parity
from csi.metrics import binary_metrics
from csi.nets import cell_balanced_weights, n_params, predict_proba, train_binary, weighted_mean_std
from csi.paths import DATA, FIGURES, MODELS, RESULTS, ensure_dirs
from csi.plots import phase_legend, plot_curves, plt, save, shade_phases
from csi.temporal import run_filter

HP = dict(hidden=(16, 8), epochs=160, lr=0.002, weight_decay=1e-3, seed=42)


def standardize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def fit(train_sets, held_sets=None):
    x, y, cells, _, _ = M.stack(train_sets)
    w = cell_balanced_weights(cells, y)
    mean, std = weighted_mean_std(x, w)
    z = standardize(x, mean, std)
    monitor = {'treino': (z, y, w)}
    if held_sets:
        xh, yh, ch, _, _ = M.stack(held_sets)
        monitor['validação'] = (standardize(xh, mean, std), yh, cell_balanced_weights(ch, yh))
    net, hist = train_binary(z, y, w, monitor=monitor, **HP)
    logit = LogisticRegression(C=0.1, max_iter=3000).fit(z, y, sample_weight=w)
    return dict(net=net, logit=logit, mean=mean, std=std, history=hist)


def score_sets(model, sets, fold):
    rows = []
    for s in sets:
        for c in M.CONDITIONS:
            z = standardize(s['x'][c], model['mean'], model['std'])
            for name, p in (('mlp', predict_proba(model['net'], z)),
                            ('logistic', model['logit'].predict_proba(z)[:, 1])):
                stable, _ = run_filter(p)
                for kind, pred in (('raw', p >= 0.5), ('stable', stable)):
                    sl = slice(4, None) if kind == 'stable' else slice(None)  # aquecimento do filtro
                    rows.append(dict(fold=fold, setup=s['name'], condition=c, model=name, kind=kind,
                                     n=len(pred[sl]), moving_rate=float(np.mean(pred[sl])),
                                     truth=int(c == 'moving'), pred=pred[sl]))
    return rows


def summarize(rows):
    out = []
    df = pd.DataFrame(rows)
    for (fold, model, kind), g in df.groupby(['fold', 'model', 'kind'], sort=False):
        y = np.concatenate([np.full(r.n, r.truth) for r in g.itertuples()])
        p = np.concatenate(list(g['pred']))
        m = binary_metrics(y, p)
        static = g[g.condition == 'static']
        out.append(dict(fold=fold, model=model, kind=kind, **{k: m[k] for k in
                   ('n', 'balanced_accuracy', 'recall', 'specificity', 'FP', 'FN')},
                   static_false_moving_rate=float(np.average(static.moving_rate, weights=static.n))))
    return pd.DataFrame(out)


def live_tests(model):
    gs, ge = guards()
    rows, curves = [], []
    for item in load_manifest()['motion_live_logs']:
        log = parse_parity(DATA / item['file'])
        z = standardize(M.log_ratio(log['f'], log['refs']), model['mean'], model['std'])
        p = predict_proba(model['net'], z)
        # Saída do modelo que estava gravado no ESP32 (INT8): p = (out_q + 128) / 256
        p_dev = np.array([(r['out_q'] + 128) / 256.0 for r in log['rows']])
        st, _ = run_filter(p)
        st_dev, _ = run_filter(p_dev)
        curves.append(dict(id=item['id'], t=log['t_s'], p=p, p_dev=p_dev, st=st, phases=item['phases']))
        for ph_name, ph_lab, a, b in item['phases']:
            m = (log['t_s'] >= a + gs) & (log['t_s'] < b - ge)
            if not m.any():
                continue
            for name, pred in (('retrained_mlp_stable', st), ('device_mlp_stable', st_dev),
                               ('retrained_mlp_raw', p >= 0.5)):
                rows.append(dict(log=item['id'], phase=ph_name, label=ph_lab, model=name,
                                 n=int(m.sum()), moving_rate=float(np.mean(pred[m]))))
        rows.append(dict(log=item['id'], phase='ALL', label='-', model='agreement_retrained_vs_device',
                         n=len(p), moving_rate=float(np.mean((p >= 0.5) == (p_dev >= 0.5)))))
    return pd.DataFrame(rows), curves


def main():
    ensure_dirs()
    sets = M.setups('train')
    rows, histories = [], []
    for s in sets:
        model = fit([t for t in sets if t['name'] != s['name']], [s])
        rows += score_sets(model, [s], f'LOSO:{s["name"]}')
        histories.append((s['name'], model['history']))
    for camp in ('robustness', 'independent'):
        model = fit([t for t in sets if t['campaign'] != camp], [t for t in sets if t['campaign'] == camp])
        rows += score_sets(model, [t for t in sets if t['campaign'] == camp], f'campaign:{camp}')
    final = fit(sets)
    rows += score_sets(final, sets, 'treino (in-sample)')
    summary = summarize(rows)
    summary.to_csv(RESULTS / 'motion_validation.csv', index=False, float_format='%.4f')

    live, curves = live_tests(final)
    live.to_csv(RESULTS / 'motion_live_tests.csv', index=False, float_format='%.4f')

    # ---------------- curvas de treino
    pd.DataFrame([dict(fold=f, set=k, epoch=i + 1, loss=l, balanced_accuracy=b)
                  for f, h in histories for k, v in h.items()
                  for i, (l, b) in enumerate(zip(v['loss'], v['balanced_accuracy']))]
                 ).to_csv(RESULTS / 'motion_training_curves.csv', index=False, float_format='%.5f')
    plot_curves(histories, final['history'], FIGURES / 'motion_training_curves.png',
                'Movimento — curvas de treino (cada linha fina = uma dobra; validação = setup fora do treino)')

    # ---------------- salvar modelo final
    out = MODELS / 'motion'
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(final['net'].state_dict(), out / 'motion_mlp.pt')
    x, y, cells, _, cond = M.stack(sets)
    np.save(out / 'representative_inputs.npy', standardize(x, final['mean'], final['std']))
    pre = dict(stage='motion', feature_indices=list(M.CORE7), features=list(M.FEATURE_NAMES),
               calibration_windows=M.CAL_WINDOWS, eps=M.EPS,
               transform='x = log(max(f,1e-9)/max(ref,1e-9)); z = (x-mean)/std',
               scaler_mean=final['mean'].astype(np.float32).tolist(),
               scaler_std=final['std'].astype(np.float32).tolist(),
               architecture=[7, *HP['hidden'], 1], parameters=n_params(final['net']),
               hyperparameters={k: v for k, v in HP.items()},
               training_setups=[s['name'] for s in sets], training_windows=int(len(y)),
               final_training_loss=final['history']['treino']['loss'][-1],
               filter=dict(n=5, enter=0.65, exit=0.35))
    (out / 'preprocessing.json').write_text(json.dumps(pre, indent=2), encoding='utf-8')

    # ---------------- figuras
    fig, axes = plt.subplots(len(curves), 1, figsize=(11, 2.3 * len(curves)), sharex=False)
    for ax, c in zip(axes, curves):
        shade_phases(ax, c['phases'], ymax=1.02)
        ax.plot(c['t'], c['p'], lw=0.8, color='#08519c', label='P(movimento) — MLP retreinada')
        ax.plot(c['t'], c['p_dev'], lw=0.6, color='#999999', alpha=0.8, label='P(movimento) — MLP gravada no ESP32')
        ax.fill_between(c['t'], 0, 0.08, where=c['st'], step='post', color='#e6550d', lw=0,
                        label='estado estável MOVING')
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(c['id'])
        ax.set_xlim(c['t'][0], c['t'][-1])
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + phase_legend(axes[0], ('EMPTY', 'STATIC', 'MOVING', 'ARMS')),
                   fontsize=6.5, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.45))
    axes[-1].set_xlabel('tempo desde o boot do RX (s)')
    save(fig, FIGURES / 'motion_live_tests.png')

    fig, axes = plt.subplots(1, 7, figsize=(13, 2.6), sharey=False)
    for j, ax in enumerate(axes):
        ax.boxplot([x[cond == c, j] for c in M.CONDITIONS], showfliers=False)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['E', 'S', 'M'], fontsize=7)
        ax.set_title(M.FEATURE_NAMES[j].replace('_', '\n'), fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel('log(f / ref EMPTY)', fontsize=7)
    fig.suptitle('Features Core7 por condição (E=EMPTY, S=STATIC, M=MOVING)', fontsize=9, y=1.1)
    save(fig, FIGURES / 'motion_features.png')

    # ---------------- relatório
    print('\n=== MOVIMENTO: validação (estado estável) ===')
    print(summary[summary.kind == 'stable'][['fold', 'model', 'n', 'balanced_accuracy', 'recall',
                                             'specificity', 'static_false_moving_rate']].to_string(index=False))
    h = final['history']['treino']
    print(f"\nTreino final: perda {h['loss'][0]:.3f} -> {h['loss'][-1]:.3f} em {HP['epochs']} épocas")
    print('\n=== MOVIMENTO: testes ao vivo (taxa de MOVING por fase) ===')
    print(live.pivot_table(index=['log', 'phase', 'label'], columns='model', values='moving_rate',
                           sort=False).round(3).to_string())
    print(f'\nModelo final: {out / "motion_mlp.pt"} ({n_params(final["net"])} parâmetros)')


if __name__ == '__main__':
    main()
