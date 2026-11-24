"""MLPs PyTorch e treino determinístico (full-batch Adam, BCE ponderada)."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def mlp(in_dim: int, hidden: tuple[int, ...] = (16, 8)) -> nn.Sequential:
    """in_dim -> 16 -> ReLU -> 8 -> ReLU -> 1 (logit). A sigmoid fica fora (BCEWithLogits)."""
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def cell_balanced_weights(cells: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cada célula (ex.: sessão, ou setup+condição) recebe o mesmo peso total;
    depois as duas classes recebem o mesmo peso total. Média final = 1.

    Evita que sessões longas ou classes majoritárias dominem o ajuste.
    """
    w = np.zeros(len(y), dtype=np.float64)
    for c in np.unique(cells):
        m = cells == c
        w[m] = 1.0 / m.sum()
    for label in np.unique(y):
        m = y == label
        w[m] /= w[m].sum()
    return w * (len(w) / w.sum())


def weighted_mean_std(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """StandardScaler com pesos (idêntico ao sklearn StandardScaler(sample_weight=w))."""
    mean = np.average(x, axis=0, weights=w)
    var = np.average((x - mean) ** 2, axis=0, weights=w)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return mean, std


def train_binary(z: np.ndarray, y: np.ndarray, w: np.ndarray, *, hidden=(16, 8),
                 epochs: int = 200, lr: float = 0.002, weight_decay: float = 1e-3,
                 seed: int = 42, monitor: dict | None = None):
    """Treino com cronograma FIXO (sem early stopping nem escolha por dados de teste).

    monitor: {nome: (z, y, w)} — conjuntos apenas OBSERVADOS a cada época (perda BCE ponderada
    e acurácia balanceada com limiar 0,5), para as curvas de treino. Não influenciam o ajuste.
    Retorna a rede; com monitor, retorna (rede, histórico).
    """
    seed_everything(seed)
    net = mlp(z.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
    y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32).reshape(-1, 1))
    w_t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32).reshape(-1, 1))
    sets = {k: tuple(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32).reshape(len(a), -1))
                     for a in v) for k, v in (monitor or {}).items()}
    history = {k: {'loss': [], 'balanced_accuracy': []} for k in sets}

    def observe():
        net.eval()
        with torch.no_grad():
            for k, (xs, ys, ws) in sets.items():
                logit = net(xs)
                bce = nn.functional.binary_cross_entropy_with_logits(logit, ys, reduction='none')
                history[k]['loss'].append(float((ws * bce).sum() / ws.sum()))
                pred = (logit >= 0).float()
                tpr = float(((pred == 1) & (ys == 1)).sum() / max(float((ys == 1).sum()), 1.0))
                tnr = float(((pred == 0) & (ys == 0)).sum() / max(float((ys == 0).sum()), 1.0))
                history[k]['balanced_accuracy'].append((tpr + tnr) / 2)
        net.train()

    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(net(x_t), y_t, reduction='none')
        loss = (w_t * loss).sum() / w_t.sum()
        loss.backward()
        opt.step()
        if sets:
            observe()
    net.eval()
    return (net, history) if monitor is not None else net


@torch.no_grad()
def predict_proba(net: nn.Module, z: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
    return torch.sigmoid(net(x)).numpy().ravel()
