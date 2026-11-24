#pragma once

// Estágio 1 — pré-processamento de PRESENÇA a partir da assinatura espectral CSI.
// Espelho exato de ml/csi/presence.py (validado por tests_native/test_presence.cpp).
//
//   s[51]   assinatura da janela: média temporal da amplitude LLTF de cada subportadora
//           (pacotes normalizados pela mediana), quantizada em Q12 como no log CSI_SIG
//   p       = s / mean(s)                                   perfil de forma
//   ref     = mediana de p nas 20 janelas EMPTY de calibração
//   F[7]    = rms, mae, p90, max de |p-ref|; 1-Pearson(p,ref); |rssi-rssi0|; rms(p - p[t-5])
//   x       = log(clip(F/scale, 0.01, 1e4)),  scale = max(mediana de F na calibração, piso)
//   z       = (x - mean) / std                              entrada da MLP INT8
//
// Sem dependências de ESP-IDF/TFLM: compila no PC para os testes.

#include <algorithm>
#include <cmath>
#include <cstring>

#include "model_params.h"

class CsiPresence {
public:
    static constexpr int kCarriers = 51;
    static constexpr int kFeatures = model_params::presence::kInputs;
    static constexpr int kCalWindows = model_params::presence::kCalibrationWindows;
    static constexpr int kLag = model_params::presence::kLagWindows;

    struct Calibration {
        float ref[kCarriers];
        float rssi_ref;
        float scale[kFeatures];
    };

    // Q12 exatamente como no log CSI_SIG: round(média * 4096) / 4096.
    static float q12(float mean_amplitude) {
        if (!std::isfinite(mean_amplitude) || mean_amplitude <= 0.0f) return 0.0f;
        const float scaled = mean_amplitude * 4096.0f;
        if (scaled >= 65535.0f) return 65535.0f / 4096.0f;
        return static_cast<float>(std::lround(scaled)) / 4096.0f;
    }

    // Zera o histórico de perfis (ex.: após uma lacuna de CSI). Não mexe na calibração.
    void reset_history() {
        history_count_ = 0;
        history_next_ = 0;
    }

    // Recebe a assinatura da janela atual. Calcula p e p[t-kLag] (o mais antigo disponível
    // enquanto o histórico não está cheio) e atualiza o histórico. Deve ser chamada em TODA
    // janela, inclusive durante a espera e a calibração, como no treino.
    bool push(const float signature[kCarriers], float p_out[kCarriers], float p_lag_out[kCarriers]) {
        double sum = 0.0;
        for (int k = 0; k < kCarriers; ++k) {
            if (!std::isfinite(signature[k])) return false;
            sum += signature[k];
        }
        const float mean = static_cast<float>(sum / kCarriers);
        if (!(mean > 1e-6f)) return false;
        for (int k = 0; k < kCarriers; ++k) p_out[k] = signature[k] / mean;
        // histórico circular com kLag+1 perfis: o mais antigo é p[t-kLag]
        std::memcpy(history_[history_next_], p_out, sizeof(float) * kCarriers);
        history_next_ = (history_next_ + 1) % (kLag + 1);
        if (history_count_ < kLag + 1) ++history_count_;
        const int oldest = history_count_ < kLag + 1 ? 0 : history_next_;
        std::memcpy(p_lag_out, history_[oldest], sizeof(float) * kCarriers);
        return true;
    }

    // --- calibração (20 janelas EMPTY) ---
    void calibration_reset() { cal_count_ = 0; }
    int calibration_count() const { return cal_count_; }
    bool calibration_ready() const { return cal_count_ >= kCalWindows; }

    void calibration_add(const float p[kCarriers], const float p_lag[kCarriers], float rssi) {
        if (cal_count_ >= kCalWindows) return;
        std::memcpy(cal_p_[cal_count_], p, sizeof(float) * kCarriers);
        std::memcpy(cal_lag_[cal_count_], p_lag, sizeof(float) * kCarriers);
        cal_rssi_[cal_count_] = rssi;
        ++cal_count_;
    }

    bool calibration_finish(Calibration &c) const {
        if (cal_count_ < kCalWindows) return false;
        float col[kCalWindows];
        for (int k = 0; k < kCarriers; ++k) {
            for (int i = 0; i < kCalWindows; ++i) col[i] = cal_p_[i][k];
            c.ref[k] = median(col, kCalWindows);
        }
        c.rssi_ref = median(cal_rssi_, kCalWindows);
        float f[kCalWindows][kFeatures];
        for (int i = 0; i < kCalWindows; ++i) {
            if (!features(cal_p_[i], cal_lag_[i], cal_rssi_[i], c.ref, c.rssi_ref, f[i])) return false;
        }
        for (int j = 0; j < kFeatures; ++j) {
            for (int i = 0; i < kCalWindows; ++i) col[i] = f[i][j];
            c.scale[j] = std::max(median(col, kCalWindows), model_params::presence::kFloors[j]);
        }
        return valid(c);
    }

    static bool valid(const Calibration &c) {
        if (!std::isfinite(c.rssi_ref)) return false;
        for (float v : c.ref) if (!std::isfinite(v) || v <= 0.0f) return false;
        for (float v : c.scale) if (!std::isfinite(v) || v <= 0.0f) return false;
        return true;
    }

    // --- features brutas (7) ---
    static bool features(const float p[kCarriers], const float p_lag[kCarriers], float rssi,
                         const float ref[kCarriers], float rssi_ref, float f[kFeatures]) {
        float absdiff[kCarriers];
        double sq = 0.0, sa = 0.0, td = 0.0, pm = 0.0, rm = 0.0;
        float mx = 0.0f;
        for (int k = 0; k < kCarriers; ++k) {
            const double d = static_cast<double>(p[k]) - ref[k];
            sq += d * d;
            absdiff[k] = static_cast<float>(std::fabs(d));
            sa += absdiff[k];
            mx = std::max(mx, absdiff[k]);
            const double dl = static_cast<double>(p[k]) - p_lag[k];
            td += dl * dl;
            pm += p[k];
            rm += ref[k];
        }
        pm /= kCarriers;
        rm /= kCarriers;
        double num = 0.0, a = 0.0, b = 0.0;
        for (int k = 0; k < kCarriers; ++k) {
            const double x = p[k] - pm, y = ref[k] - rm;
            num += x * y;
            a += x * x;
            b += y * y;
        }
        double corr = num / std::max(std::sqrt(a * b), 1e-12);
        corr = std::max(-1.0, std::min(1.0, corr));
        std::sort(absdiff, absdiff + kCarriers);
        f[0] = static_cast<float>(std::sqrt(sq / kCarriers));
        f[1] = static_cast<float>(sa / kCarriers);
        f[2] = absdiff[45];  // percentil 90 de 51 valores: índice (51-1)*0.9 = 45
        f[3] = mx;
        f[4] = static_cast<float>(1.0 - corr);
        f[5] = std::fabs(rssi - rssi_ref);
        f[6] = static_cast<float>(std::sqrt(td / kCarriers));
        for (int j = 0; j < kFeatures; ++j) if (!std::isfinite(f[j])) return false;
        return true;
    }

    // --- entrada padronizada da MLP ---
    static bool inputs(const float f[kFeatures], const Calibration &c, float z[kFeatures]) {
        using namespace model_params::presence;
        for (int j = 0; j < kFeatures; ++j) {
            const float r = std::min(std::max(f[j] / c.scale[j], kClipLo), kClipHi);
            z[j] = (std::log(r) - kMean[j]) / kStd[j];
            if (!std::isfinite(z[j])) return false;
        }
        return true;
    }

private:
    static float median(const float *v, int n) {
        float s[kCalWindows > kCarriers ? kCalWindows : kCarriers];
        std::copy(v, v + n, s);
        std::sort(s, s + n);
        return (n % 2) ? s[n / 2] : 0.5f * (s[n / 2 - 1] + s[n / 2]);
    }

    float history_[kLag + 1][kCarriers] = {};
    int history_count_ = 0;
    int history_next_ = 0;
    float cal_p_[kCalWindows][kCarriers] = {};
    float cal_lag_[kCalWindows][kCarriers] = {};
    float cal_rssi_[kCalWindows] = {};
    int cal_count_ = 0;
};
