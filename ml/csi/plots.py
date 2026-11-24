"""Figuras do README (matplotlib, sem estilo externo)."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {'EMPTY': '#d9d9d9', 'STATIC': '#9ecae1', 'MOVING': '#fdae6b',
          'ARMS': '#fdd0a2', 'UNKNOWN': '#f0f0f0'}
STATE_COLORS = ['#bdbdbd', '#3182bd', '#e6550d']  # EMPTY, STATIC, MOVING
DPI = 130


def shade_phases(ax, phases, ymax=1.0, label_text=True):
    for name, lab, start, end in phases:
        ax.axvspan(start, end, color=COLORS.get(lab, '#ffffff'), alpha=0.55, lw=0)
        if label_text:
            ax.text((start + end) / 2, ymax, name, ha='center', va='bottom', fontsize=6.5,
                    color='#444444', clip_on=True)


def phase_legend(ax, labels=('EMPTY', 'STATIC', 'MOVING')):
    from matplotlib.patches import Patch
    handles = [Patch(color=COLORS[k], alpha=0.55, label=f'verdade: {k}') for k in labels]
    return handles


def state_strip(ax, t, state, height=1.0):
    """Faixa colorida com o estado de saída (0/1/2) e o padrão do LED."""
    for k, c in enumerate(STATE_COLORS):
        m = state == k
        ax.fill_between(t, 0, height, where=m, step='post', color=c, lw=0)
    ax.set_yticks([])
    ax.set_ylim(0, height)


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def bar_groups(ax, groups, series: dict, ylabel: str, ylim=(0, 1.05)):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        b = ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=name)
        for rect, v in zip(b, vals):
            if np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=7, loc='lower right')


def plot_curves(fold_histories, final_history, path, title):
    """Curvas de perda e acurácia balanceada por época.

    fold_histories: [(nome_da_dobra, {'treino': {...}, 'validação': {...}}), ...]
    final_history:  {'treino': {...}} do modelo final (todas as sessões/setups).
    """
    colors = {'treino': '#3182bd', 'validação': '#e6550d'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, metric, label in ((axes[0], 'loss', 'perda BCE ponderada'),
                              (axes[1], 'balanced_accuracy', 'acurácia balanceada (janela, limiar 0,5)')):
        for split, color in colors.items():
            curves = [np.asarray(h[split][metric]) for _, h in fold_histories if split in h]
            if not curves:
                continue
            ep = np.arange(1, len(curves[0]) + 1)
            for c in curves:
                ax.plot(ep, c, color=color, lw=0.5, alpha=0.35)
            ax.plot(ep, np.mean(curves, axis=0), color=color, lw=2, label=f'{split} (média das dobras)')
        if final_history:
            f = np.asarray(final_history['treino'][metric])
            ax.plot(np.arange(1, len(f) + 1), f, color='k', lw=1.2, ls='--', label='modelo final (treino)')
        ax.set_xlabel('época')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1].set_ylim(0.4, 1.02)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=9.5)
    save(fig, path)
