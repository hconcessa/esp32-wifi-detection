#pragma once

// 12 features temporais por janela de 50x51 (amplitudes LLTF normalizadas, 25 Hz).
// São as colunas f0..f11 de data/dataset/motion/*/*.csv; o estágio de movimento usa
// os índices 3,4,5,7,8,9,10 (Core7). Ver ml/csi/motion.py.
constexpr int CSI_WINDOW_FRAMES = 50;
constexpr int CSI_LLTF_CARRIERS = 51;
constexpr int CSI_FEATURE_COUNT = 12;

bool csi_extract_features(const float window[CSI_WINDOW_FRAMES][CSI_LLTF_CARRIERS],
                          float features[CSI_FEATURE_COUNT]);
