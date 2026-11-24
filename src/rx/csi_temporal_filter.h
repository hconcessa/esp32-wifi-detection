#pragma once

// Média causal das últimas N probabilidades + histerese.
// Entra no estado ativo com média >= kEnter, sai com média <= kExit; entre os dois mantém.
// Espelho de ml/csi/temporal.py::TemporalFilter. Sem alocação dinâmica nem ESP-IDF.

#include <cmath>

#include "model_params.h"

class CsiTemporalFilter {
public:
    static constexpr int kWindowCount = model_params::filter::kWindowCount;
    static constexpr float kEnter = model_params::filter::kEnter;
    static constexpr float kExit = model_params::filter::kExit;

    struct Result {
        bool valid;     // probabilidade aceita
        bool ready;     // já há kWindowCount probabilidades
        bool active;    // estado estável (MOVING ou OCUPADO)
        bool changed;   // houve transição nesta janela
        float mean;     // média das últimas kWindowCount probabilidades
    };

    void reset() {
        for (float &v : values_) v = 0.0f;
        next_ = 0;
        count_ = 0;
        sum_ = 0.0f;
        active_ = false;
    }

    Result observe(float p) {
        if (!std::isfinite(p) || p < 0.0f || p > 1.0f) {
            return {false, count_ == kWindowCount, active_, false, count_ ? sum_ / count_ : 0.0f};
        }
        if (count_ == kWindowCount) sum_ -= values_[next_];
        else ++count_;
        values_[next_] = p;
        sum_ += p;
        next_ = (next_ + 1) % kWindowCount;
        const float mean = sum_ / count_;
        const bool ready = count_ == kWindowCount;
        const bool before = active_;
        if (ready) {
            if (!active_ && mean >= kEnter) active_ = true;
            else if (active_ && mean <= kExit) active_ = false;
        }
        return {true, ready, ready && active_, ready && active_ != before, mean};
    }

    bool active() const { return active_ && count_ == kWindowCount; }

private:
    float values_[kWindowCount] = {};
    int next_ = 0;
    int count_ = 0;
    float sum_ = 0.0f;
    bool active_ = false;
};
