#pragma once

// Modo do firmware RX. O mesmo código atende às três finalidades do projeto; muda só o que é
// registrado na serial (921600 baud). Selecione pelo ambiente do PlatformIO:
//
//   pio run -e rx            -t upload   -> DEPLOY      (demonstração / uso final)
//   pio run -e rx_capture    -t upload   -> CAPTURE     (gravar datasets)
//   pio run -e rx_validation -t upload   -> VALIDATION  (testar e validar o firmware contra o PC)
//
// ou editando o valor padrão abaixo (quando não houver -DCSI_FIRMWARE_MODE nas build_flags).
//
//                         | DEPLOY          | CAPTURE              | VALIDATION
// ------------------------+-----------------+----------------------+---------------------------
// LED de 3 estados        | sim             | sim (se calibrado)   | sim (se calibrado)
// CSI_STATE               | transições+4,8 s| transições+4,8 s     | TODA janela
// CSI_SIG (assinatura)    | não             | toda janela          | toda janela
// CSI_F12 (12 features)   | não             | toda janela          | toda janela
// CSI_VEC (int8 dos 2 MLP)| não             | não                  | toda janela
// CSI_REF (referência)    | no boot         | no boot              | no boot
//
// CAPTURE funciona mesmo SEM calibração (só registra os dados brutos de cada janela);
// VALIDATION exige referência na NVS para que os modelos rodem.

#define CSI_MODE_DEPLOY 0
#define CSI_MODE_CAPTURE 1
#define CSI_MODE_VALIDATION 2

#ifndef CSI_FIRMWARE_MODE
#define CSI_FIRMWARE_MODE CSI_MODE_DEPLOY
#endif

#if CSI_FIRMWARE_MODE != CSI_MODE_DEPLOY && CSI_FIRMWARE_MODE != CSI_MODE_CAPTURE && \
    CSI_FIRMWARE_MODE != CSI_MODE_VALIDATION
#error "CSI_FIRMWARE_MODE deve ser 0 (DEPLOY), 1 (CAPTURE) ou 2 (VALIDATION)"
#endif

namespace firmware_config {
constexpr int kMode = CSI_FIRMWARE_MODE;
constexpr const char *kModeName = kMode == CSI_MODE_CAPTURE      ? "CAPTURE"
                                  : kMode == CSI_MODE_VALIDATION ? "VALIDATION"
                                                                 : "DEPLOY";
constexpr bool kLogSignature = kMode != CSI_MODE_DEPLOY;      // CSI_SIG: dataset de presença
constexpr bool kLogFeatures = kMode != CSI_MODE_DEPLOY;       // CSI_F12: dataset de movimento
constexpr bool kLogVectors = kMode == CSI_MODE_VALIDATION;    // CSI_VEC: paridade com o PC
constexpr bool kLogStateEveryWindow = kMode == CSI_MODE_VALIDATION;
}  // namespace firmware_config
