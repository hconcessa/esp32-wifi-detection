#pragma once

// Fusão dos dois estágios em 3 estados e padrão do LED (GPIO25).
// Espelho de ml/csi/temporal.py::fuse_states.
//
//   MOVING  se o filtro de movimento está ativo            -> LED aceso
//   STATIC  senão, se o filtro de presença está ativo      -> LED pisca 500 ms / 500 ms
//   EMPTY   caso contrário                                 -> LED apagado
//
// Movimento tem prioridade e implica presença. Sem dependências de ESP-IDF.

#include <cstdint>

enum class CsiState : int { Empty = 0, Static = 1, Moving = 2 };

inline const char *csi_state_name(CsiState s) {
    switch (s) {
    case CsiState::Moving: return "MOVING";
    case CsiState::Static: return "STATIC";
    default: return "EMPTY";
    }
}

inline CsiState csi_fuse(bool presence_active, bool moving_active) {
    if (moving_active) return CsiState::Moving;
    if (presence_active) return CsiState::Static;
    return CsiState::Empty;
}

// Nível do LED para o estado de detecção, dado o tempo em ms desde a entrada no estado.
inline bool csi_led_level(CsiState s, uint32_t elapsed_ms) {
    switch (s) {
    case CsiState::Moving: return true;
    case CsiState::Static: return (elapsed_ms % 1000u) < 500u;
    default: return false;
    }
}

// Padrões de serviço (têm prioridade sobre o estado de detecção).
enum class LedService : int {
    None = 0,         // segue o estado de detecção
    Calibrating = 1,  // flash curto a cada 1,6 s
    Saved = 2,        // curto-curto-longo x3 (~5 s), depois volta ao normal
    Error = 3,        // 4 flashes rápidos repetidos: sem referência / falha NVS
    Restored = 4,     // um flash curto no boot: referência NVS carregada
};

inline bool csi_service_level(LedService p, uint32_t elapsed) {
    switch (p) {
    case LedService::Calibrating: return (elapsed % 1600u) < 100u;
    case LedService::Saved: {
        if (elapsed >= 4950u) return false;
        const uint32_t t = elapsed % 1650u;
        return t < 150u || (t >= 300u && t < 450u) || (t >= 630u && t < 1230u);
    }
    case LedService::Error: {
        const uint32_t t = elapsed % 2200u;
        return t < 110u || (t >= 220u && t < 330u) || (t >= 440u && t < 550u) || (t >= 660u && t < 770u);
    }
    case LedService::Restored: return elapsed < 150u;
    default: return false;
    }
}

// Padrões temporários que terminam sozinhos.
inline bool csi_service_finished(LedService p, uint32_t elapsed) {
    return (p == LedService::Saved && elapsed >= 4950u) || (p == LedService::Restored && elapsed >= 350u);
}
