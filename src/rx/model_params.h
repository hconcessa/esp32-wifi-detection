// AUTO-GERADO por ml/03_export_firmware.py — não edite à mão.
// Constantes de pré-processamento que DEVEM ser idênticas às usadas no treino.
#pragma once
#include <cstdint>

namespace model_params {
// Muda sempre que um modelo ou scaler muda; invalida referências EMPTY antigas na NVS.
static constexpr uint32_t kFingerprint = 0x1D0D612Cu;

namespace motion {
static constexpr int kInputs = 7;
static constexpr int kFeatureIndex[kInputs] = {3, 4, 5, 7, 8, 9, 10};
static constexpr float kMean[7] = {0.341418177f, 0.325553209f, 0.431627393f, 0.905757427f, 0.496689349f, 0.527616858f, 0.726625502f};
static constexpr float kStd[7] = {0.338482171f, 0.325980693f, 0.418987811f, 0.806703746f, 0.477343589f, 0.825496435f, 1.00518596f};
// Quantização do tensor de entrada/saída (conferida em runtime contra o .tflite)
static constexpr float kInScale = 0.0220661256f;
static constexpr int kInZero = -44;
}  // namespace motion

namespace presence {
static constexpr int kInputs = 7;
static constexpr int kCalibrationWindows = 20;
static constexpr int kLagWindows = 5;
static constexpr float kClipLo = 0.00999999978f;
static constexpr float kClipHi = 10000.0f;
static constexpr float kFloors[7] = {0.00100000005f, 0.00100000005f, 0.00100000005f, 0.00100000005f, 9.99999975e-06f, 0.25f, 0.00100000005f};
static constexpr float kMean[7] = {1.74290109f, 1.808671f, 1.69547594f, 1.50728118f, 3.20368862f, 0.0746052861f, 0.577569366f};
static constexpr float kStd[7] = {1.37286282f, 1.41271234f, 1.33464277f, 1.22662199f, 2.68697691f, 2.45222497f, 1.23609996f};
static constexpr float kInScale = 0.0238644443f;
static constexpr int kInZero = -34;
}  // namespace presence

namespace filter {
static constexpr int kWindowCount = 5;
static constexpr float kEnter = 0.65f;
static constexpr float kExit = 0.35f;
}  // namespace filter
}  // namespace model_params
