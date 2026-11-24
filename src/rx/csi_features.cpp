#include "csi_features.h"

#include <algorithm>
#include <cmath>

namespace {

float mean_of(const float* a, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += a[i];
    return static_cast<float>(sum / n);
}

float std_of(const float* a, int n)
{
    const float mean = mean_of(a, n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = static_cast<double>(a[i]) - mean;
        sum += diff * diff;
    }
    return static_cast<float>(std::sqrt(sum / n));
}

float percentile(const float* values, int n, float q)
{
    float sorted[CSI_LLTF_CARRIERS];
    for (int i = 0; i < n; ++i) sorted[i] = values[i];
    std::sort(sorted, sorted + n);
    const float idx = (n - 1) * q;
    const int lo = static_cast<int>(idx);
    const int hi = lo + 1 < n ? lo + 1 : lo;
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

} // namespace

bool csi_extract_features(const float w[CSI_WINDOW_FRAMES][CSI_LLTF_CARRIERS],
                          float out[CSI_FEATURE_COUNT])
{
    float carrier_std[CSI_LLTF_CARRIERS];
    float carrier_change[CSI_LLTF_CARRIERS];
    float frame_change[CSI_WINDOW_FRAMES - 1];
    float shape_change[CSI_WINDOW_FRAMES - 1];
    float frame_spread[CSI_WINDOW_FRAMES];
    float centered[CSI_LLTF_CARRIERS];
    float previous_centered[CSI_LLTF_CARRIERS];

    for (int t = 0; t < CSI_WINDOW_FRAMES; ++t) {
        for (int k = 0; k < CSI_LLTF_CARRIERS; ++k) {
            if (!std::isfinite(w[t][k])) return false;
        }
        frame_spread[t] = std_of(w[t], CSI_LLTF_CARRIERS);
    }

    for (int k = 0; k < CSI_LLTF_CARRIERS; ++k) {
        double sum = 0.0;
        for (int t = 0; t < CSI_WINDOW_FRAMES; ++t) sum += w[t][k];
        const double mean = sum / CSI_WINDOW_FRAMES;
        double variance = 0.0;
        double changes = 0.0;
        for (int t = 0; t < CSI_WINDOW_FRAMES; ++t) {
            const double d = w[t][k] - mean;
            variance += d * d;
            if (t > 0) changes += std::fabs(w[t][k] - w[t - 1][k]);
        }
        carrier_std[k] = static_cast<float>(std::sqrt(variance / CSI_WINDOW_FRAMES));
        carrier_change[k] = static_cast<float>(changes / (CSI_WINDOW_FRAMES - 1));
    }

    for (int t = 1; t < CSI_WINDOW_FRAMES; ++t) {
        double change = 0.0;
        for (int k = 0; k < CSI_LLTF_CARRIERS; ++k) {
            change += std::fabs(w[t][k] - w[t - 1][k]);
        }
        frame_change[t - 1] = static_cast<float>(change / CSI_LLTF_CARRIERS);

        const float prev_mean = mean_of(w[t - 1], CSI_LLTF_CARRIERS);
        const float curr_mean = mean_of(w[t], CSI_LLTF_CARRIERS);
        double numerator = 0.0;
        double a = 0.0;
        double b = 0.0;
        for (int k = 0; k < CSI_LLTF_CARRIERS; ++k) {
            previous_centered[k] = w[t - 1][k] - prev_mean;
            centered[k] = w[t][k] - curr_mean;
            numerator += static_cast<double>(centered[k]) * previous_centered[k];
            a += static_cast<double>(centered[k]) * centered[k];
            b += static_cast<double>(previous_centered[k]) * previous_centered[k];
        }
        const double denom = std::max(std::sqrt(a * b), 1e-9);
        const double corr = std::max(-1.0, std::min(1.0, numerator / denom));
        shape_change[t - 1] = static_cast<float>(1.0 - corr);
    }

    out[0] = mean_of(carrier_std, CSI_LLTF_CARRIERS);
    out[1] = percentile(carrier_std, CSI_LLTF_CARRIERS, 0.5f);
    out[2] = percentile(carrier_std, CSI_LLTF_CARRIERS, 0.9f);
    out[3] = mean_of(carrier_change, CSI_LLTF_CARRIERS);
    out[4] = percentile(carrier_change, CSI_LLTF_CARRIERS, 0.5f);
    out[5] = percentile(carrier_change, CSI_LLTF_CARRIERS, 0.9f);
    out[6] = mean_of(frame_change, CSI_WINDOW_FRAMES - 1);
    out[7] = std_of(frame_change, CSI_WINDOW_FRAMES - 1);
    out[8] = percentile(frame_change, CSI_WINDOW_FRAMES - 1, 0.9f);
    out[9] = mean_of(shape_change, CSI_WINDOW_FRAMES - 1);
    out[10] = percentile(shape_change, CSI_WINDOW_FRAMES - 1, 0.9f);
    out[11] = mean_of(frame_spread, CSI_WINDOW_FRAMES);
    for (int i = 0; i < CSI_FEATURE_COUNT; ++i) {
        if (!std::isfinite(out[i])) return false;
    }
    return true;
}
