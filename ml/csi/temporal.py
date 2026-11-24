"""Filtro temporal e máquina de 3 estados — espelho exato de src/rx/csi_temporal_filter.h
e src/rx/csi_state_machine.h."""
from __future__ import annotations

import numpy as np

EMPTY, STATIC, MOVING = 0, 1, 2
STATE_NAMES = ('EMPTY', 'STATIC', 'MOVING')


class TemporalFilter:
    """Média causal das últimas N probabilidades + histerese (entra >= enter, sai <= exit)."""

    def __init__(self, n: int = 5, enter: float = 0.65, exit: float = 0.35):
        assert enter > exit
        self.n, self.enter, self.exit = n, enter, exit
        self.reset()

    def reset(self):
        self.values: list[float] = []
        self.active = False

    def observe(self, p: float) -> tuple[bool, bool, float]:
        """Retorna (ready, active, mean)."""
        self.values.append(float(p))
        if len(self.values) > self.n:
            self.values.pop(0)
        mean = float(np.mean(self.values))
        ready = len(self.values) == self.n
        if ready:
            if not self.active and mean >= self.enter:
                self.active = True
            elif self.active and mean <= self.exit:
                self.active = False
        return ready, self.active, mean


def run_filter(p: np.ndarray, n: int = 5, enter: float = 0.65, exit: float = 0.35):
    f = TemporalFilter(n, enter, exit)
    active = np.zeros(len(p), dtype=bool)
    mean = np.zeros(len(p))
    for i, v in enumerate(p):
        ready, a, m = f.observe(v)
        active[i] = ready and a
        mean[i] = m
    return active, mean


def fuse_states(presence_active: np.ndarray, moving_active: np.ndarray) -> np.ndarray:
    """Hierarquia do projeto: MOVING tem prioridade; senão presença => STATIC; senão EMPTY.

    Movimento implica presença (uma pessoa andando está presente mesmo que a assinatura
    espectral média ainda esteja perto da referência).
    """
    state = np.full(len(presence_active), EMPTY, dtype=np.int64)
    state[presence_active] = STATIC
    state[moving_active] = MOVING
    return state
