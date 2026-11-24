#pragma once

// Calibrated Core7 feature preprocessing, matching the Python model training.
// The first 20 EMPTY windows provide per-feature medians. After calibration:
// x[j] = log(max(feature[index[j]], 1e-9) / max(reference[j], 1e-9))
// z[j] = (x[j] - training_mean[j]) / training_std[j]
// The calibration interval MUST be free of human motion.

#include <algorithm>
#include <cmath>

class CsiCore7Calibrator {
public:
    static constexpr int kInputs = 7;
    static constexpr int kCalibrationWindows = 20;
    static constexpr float kFloor = 1e-9f;

    enum class State { Collecting, Calibrated, Ready, Invalid };

    void reset()
    {
        count_ = 0;
        ready_ = false;
        for (int j = 0; j < kInputs; ++j) reference_[j] = 0.0f;
    }

    // Restore a previously saved EMPTY reference (without learning on boot).
    bool load_reference(const float refs[kInputs]) {
        if (!refs) return false;
        for (int i=0; i<kInputs; ++i) if (!std::isfinite(refs[i]) || refs[i] < 0.0f) return false;
        reset();
        for (int i=0; i<kInputs; ++i) reference_[i]=refs[i];
        count_=kCalibrationWindows;
        ready_=true;
        return true;
    }

    bool ready() const { return ready_; }
    int count() const { return count_; }
    float reference(int j) const { return reference_[j]; }

    static int feature_index(int j)
    {
        static const int indices[kInputs] = {3, 4, 5, 7, 8, 9, 10};
        return indices[j];
    }

    State observe(const float features[12])
    {
        if (!features) return State::Invalid;
        if (ready_) return State::Ready;
        for (int j = 0; j < kInputs; ++j) {
            const float value = features[feature_index(j)];
            if (!std::isfinite(value) || value < 0.0f) return State::Invalid;
        }
        for (int j = 0; j < kInputs; ++j) {
            samples_[count_][j] = features[feature_index(j)];
        }
        ++count_;
        if (count_ < kCalibrationWindows) return State::Collecting;

        for (int j = 0; j < kInputs; ++j) {
            float values[kCalibrationWindows];
            for (int i = 0; i < kCalibrationWindows; ++i) values[i] = samples_[i][j];
            std::sort(values, values + kCalibrationWindows);
            reference_[j] = 0.5f * (values[9] + values[10]);
            if (!std::isfinite(reference_[j]) || reference_[j] < 0.0f) {
                reset();
                return State::Invalid;
            }
        }
        ready_ = true;
        return State::Calibrated;
    }

    bool normalized(const float features[12], const float mean[kInputs],
                    const float stddev[kInputs], float z[kInputs]) const
    {
        if (!ready_ || !features || !mean || !stddev || !z) return false;
        for (int j = 0; j < kInputs; ++j) {
            const float value = features[feature_index(j)];
            if (!std::isfinite(value) || value < 0.0f ||
                !std::isfinite(mean[j]) || !std::isfinite(stddev[j]) ||
                !(stddev[j] > 0.0f)) return false;
            const float current = std::max(value, kFloor);
            const float baseline = std::max(reference_[j], kFloor);
            const float log_ratio = std::log(current / baseline);
            z[j] = (log_ratio - mean[j]) / stddev[j];
            if (!std::isfinite(z[j])) return false;
        }
        return true;
    }

private:
    float samples_[kCalibrationWindows][kInputs] = {};
    float reference_[kInputs] = {};
    int count_ = 0;
    bool ready_ = false;
};
