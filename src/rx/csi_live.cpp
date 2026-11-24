// Pipeline embarcado de 3 estados (EMPTY / STATIC / MOVING) — ESP32-WROOM RX.
//
// Por pacote CSI (≈40 Hz, HT20, canal 11, ESP-NOW do TX):
//   1. valida PHY (HT20, 256 bytes), extrai 51 amplitudes LLTF e normaliza pela mediana
//   2. reamostra por interpolação linear para uma grade de 25 Hz
// Por janela (50 amostras = 2 s, salto de 12 amostras = 480 ms):
//   3. 12 features temporais -> Core7 (7) -> log-razão vs. referência EMPTY -> MLP INT8 -> P(movimento)
//   4. assinatura s[51] (média da janela, Q12) -> 7 features de forma vs. referência EMPTY
//      -> MLP INT8 -> P(presença)
//   5. dois filtros temporais (média de 5 + histerese 0,65/0,35) -> fusão em 3 estados -> LED GPIO25
// Calibração EMPTY: jumper GPIO26-3V3 no boot -> 12 s de espera -> 20 janelas -> NVS.

#include "csi_live.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "csi_calibration_nvs.h"
#include "csi_core7_calibrator.h"
#include "csi_features.h"
#include "csi_presence.h"
#include "csi_state_machine.h"
#include "csi_temporal_filter.h"
#include "firmware_config.h"
#include "model_motion_data.h"
#include "model_params.h"
#include "model_presence_data.h"

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
// ----------------------------------------------------------------- configuração
constexpr char TAG[] = "CSI_LIVE";
constexpr char CAL_TAG[] = "CSI_CAL";
constexpr char STATE_TAG[] = "CSI_STATE";

constexpr gpio_num_t kLedPin = GPIO_NUM_25;        // LED + resistor 330 ohm -> GND
constexpr gpio_num_t kCalJumperPin = GPIO_NUM_26;  // jumper para 3V3 no boot = calibrar EMPTY

// O que vai para a serial depende do modo (firmware_config.h): DEPLOY, CAPTURE ou VALIDATION.
constexpr bool kLogSignature = firmware_config::kLogSignature;  // CSI_SIG: dataset de presença
constexpr bool kLogFeatures = firmware_config::kLogFeatures;    // CSI_F12: dataset de movimento
constexpr bool kLogVectors = firmware_config::kLogVectors;      // CSI_VEC: int8 de entrada/saída
constexpr uint32_t kHeartbeatWindows = firmware_config::kLogStateEveryWindow ? 1 : 10;  // CSI_STATE

constexpr uint32_t kGridStepUs = 40000;      // 25 Hz, igual ao treino
constexpr int kHopFrames = 12;               // 480 ms entre janelas
constexpr uint32_t kMaxGapUs = 250000;       // lacuna maior que 250 ms reinicia a janela
constexpr int64_t kGraceUs = 12LL * 1000000; // tempo para sair da sala antes da calibração
constexpr int64_t kNoDataUs = 5LL * 1000000; // sem janelas por 5 s -> LED apagado
constexpr size_t kArenaSize = 8 * 1024;
constexpr int kLLTFComplexCount = 64;
constexpr int kExpectedCsiBytes = 256;
constexpr int kInputs = 7;

static_assert(model_params::motion::kInputs == CsiCore7Calibrator::kInputs, "Core7 inputs");
static_assert(model_params::presence::kInputs == CsiPresence::kFeatures, "presence inputs");

// ----------------------------------------------------------------- estado
struct TinyModel {
    const char *name;
    const unsigned char *data;
    unsigned int len;
    tflite::MicroInterpreter *interp;
    TfLiteTensor *in;
    TfLiteTensor *out;
};

alignas(16) uint8_t s_motion_arena[kArenaSize];
alignas(16) uint8_t s_presence_arena[kArenaSize];
TinyModel s_motion{"motion", g_motion_model, g_motion_model_len, nullptr, nullptr, nullptr};
TinyModel s_presence_model{"presence", g_presence_model, g_presence_model_len, nullptr, nullptr, nullptr};

float s_window[CSI_WINDOW_FRAMES][CSI_LLTF_CARRIERS];
float s_raw_median_window[CSI_WINDOW_FRAMES];
float s_rssi_window[CSI_WINDOW_FRAMES];
float s_previous[CSI_LLTF_CARRIERS];
float s_current[CSI_LLTF_CARRIERS];
float s_previous_raw_median = 0.0f, s_current_raw_median = 0.0f;
float s_previous_rssi = 0.0f, s_current_rssi = 0.0f;
uint32_t s_previous_ts = 0;
uint64_t s_previous_time = 0, s_next_grid_time = 0;
int s_window_count = 0;
bool s_have_previous = false;

uint32_t s_windows = 0, s_rejected = 0, s_gap_resets = 0, s_failed = 0, s_since_log = 0;
int64_t s_last_reject_report_us = 0;

CsiCore7Calibrator s_core7;
CsiPresence s_presence;
CsiPresence::Calibration s_presence_cal{};
CsiTemporalFilter s_motion_filter;
CsiTemporalFilter s_presence_filter;

enum class Mode { Missing, Learning, Ready };
Mode s_mode = Mode::Missing;
int64_t s_boot_us = 0;
bool s_cal_announced = false;

// LED: lido pelo timer (tarefa esp_timer), escrito pela tarefa CSI.
std::atomic<int> s_led_state{static_cast<int>(CsiState::Empty)};
std::atomic<uint32_t> s_led_state_since_ms{0};
std::atomic<int> s_led_service{static_cast<int>(LedService::None)};
std::atomic<uint32_t> s_led_service_since_ms{0};
std::atomic<int64_t> s_last_window_us{0};
CsiState s_state = CsiState::Empty;

uint32_t now_ms() { return static_cast<uint32_t>(esp_timer_get_time() / 1000); }

void led_service(LedService p) {
    s_led_service_since_ms.store(now_ms(), std::memory_order_relaxed);
    s_led_service.store(static_cast<int>(p), std::memory_order_release);
}

void set_state(CsiState s) {
    if (s == s_state) return;
    s_state = s;
    s_led_state_since_ms.store(now_ms(), std::memory_order_relaxed);
    s_led_state.store(static_cast<int>(s), std::memory_order_release);
}

void led_timer_cb(void *) {
    const uint32_t now = now_ms();
    auto service = static_cast<LedService>(s_led_service.load(std::memory_order_acquire));
    const uint32_t service_elapsed = now - s_led_service_since_ms.load(std::memory_order_relaxed);
    if (csi_service_finished(service, service_elapsed)) {
        s_led_service.store(static_cast<int>(LedService::None), std::memory_order_release);
        service = LedService::None;
    }
    bool level;
    if (service != LedService::None) {
        level = csi_service_level(service, service_elapsed);
    } else {
        const bool stale = esp_timer_get_time() - s_last_window_us.load(std::memory_order_relaxed) > kNoDataUs;
        const auto st = static_cast<CsiState>(s_led_state.load(std::memory_order_acquire));
        level = !stale && csi_led_level(st, now - s_led_state_since_ms.load(std::memory_order_relaxed));
    }
    static int last = -1;
    if (last != static_cast<int>(level)) {
        gpio_set_level(kLedPin, level ? 1 : 0);
        last = static_cast<int>(level);
    }
}

bool led_init() {
    gpio_config_t cfg{};
    cfg.pin_bit_mask = 1ULL << static_cast<unsigned>(kLedPin);
    cfg.mode = GPIO_MODE_OUTPUT;
    if (gpio_config(&cfg) != ESP_OK || gpio_set_level(kLedPin, 0) != ESP_OK) return false;
    esp_timer_create_args_t t{};
    t.callback = led_timer_cb;
    t.dispatch_method = ESP_TIMER_TASK;
    t.name = "csi_led";
    esp_timer_handle_t h = nullptr;
    return esp_timer_create(&t, &h) == ESP_OK && esp_timer_start_periodic(h, 50000) == ESP_OK;
}

bool jumper_closed() {
    gpio_config_t cfg{};
    cfg.pin_bit_mask = 1ULL << static_cast<unsigned>(kCalJumperPin);
    cfg.mode = GPIO_MODE_INPUT;
    cfg.pull_down_en = GPIO_PULLDOWN_ENABLE;
    if (gpio_config(&cfg) != ESP_OK) return false;
    return gpio_get_level(kCalJumperPin) == 1;
}

// ----------------------------------------------------------------- TFLite Micro
bool model_init(TinyModel &m, uint8_t *arena, float expected_in_scale, int expected_in_zero) {
    static tflite::MicroMutableOpResolver<2> resolver;
    static bool resolver_ready = false;
    if (!resolver_ready) {
        if (resolver.AddFullyConnected() != kTfLiteOk || resolver.AddLogistic() != kTfLiteOk) return false;
        resolver_ready = true;
    }
    const tflite::Model *model = tflite::GetModel(m.data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "%s: versão de schema TFLite incompatível", m.name);
        return false;
    }
    // new de posicionamento em memória estática: sem heap, vida útil = aplicação.
    alignas(tflite::MicroInterpreter) static uint8_t storage[2][sizeof(tflite::MicroInterpreter)];
    static int used = 0;
    if (used >= 2) return false;
    m.interp = new (storage[used++]) tflite::MicroInterpreter(model, resolver, arena, kArenaSize);
    if (m.interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "%s: AllocateTensors falhou", m.name);
        return false;
    }
    m.in = m.interp->input(0);
    m.out = m.interp->output(0);
    const bool shape_ok = m.in && m.out && m.in->type == kTfLiteInt8 && m.out->type == kTfLiteInt8 &&
                          m.in->dims->size == 2 && m.in->dims->data[1] == kInputs &&
                          m.out->dims->size == 2 && m.out->dims->data[1] == 1;
    // Garante que model_params.h e o .tflite vieram da mesma exportação.
    const bool quant_ok = shape_ok && std::fabs(m.in->params.scale - expected_in_scale) < 1e-7f &&
                          m.in->params.zero_point == expected_in_zero;
    if (!quant_ok) {
        ESP_LOGE(TAG, "%s: tensor/quantização inesperados; regenere com ml/03_export_firmware.py", m.name);
        return false;
    }
    ESP_LOGI(TAG, "%s: %u B, arena usada %u B", m.name, m.len, static_cast<unsigned>(m.interp->arena_used_bytes()));
    return true;
}

// Quantiza z como no treino (round-half-even, satura em int8), invoca e devolve P em [0,1].
// q_in/q_out recebem a entrada e a saída int8 (cópia feita ANTES do Invoke, pois o TFLM pode
// reutilizar a memória do tensor de entrada durante a inferência).
float model_run(TinyModel &m, const float z[kInputs], int8_t q_in[kInputs], int8_t &q_out, uint32_t &clipped) {
    for (int i = 0; i < kInputs; ++i) {
        const float raw = z[i] / m.in->params.scale + static_cast<float>(m.in->params.zero_point);
        if (raw < -128.0f || raw > 127.0f) ++clipped;
        const int q = std::max(-128, std::min(127, static_cast<int>(std::nearbyint(raw))));
        m.in->data.int8[i] = static_cast<int8_t>(q);
        q_in[i] = static_cast<int8_t>(q);
    }
    if (m.interp->Invoke() != kTfLiteOk) return NAN;
    q_out = m.out->data.int8[0];
    const float p = (static_cast<int>(q_out) - m.out->params.zero_point) * m.out->params.scale;
    return std::max(0.0f, std::min(1.0f, p));
}

// ----------------------------------------------------------------- logs
void hex_q12(const float *values, int n, char *out) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    for (int k = 0; k < n; ++k) {
        const int v = static_cast<int>(std::lround(values[k] * 4096.0f));
        for (int d = 0; d < 4; ++d) out[4 * k + d] = kHex[(v >> (12 - 4 * d)) & 0xF];
    }
    out[4 * n] = '\0';
}

void log_reference(const char *origin) {
    // Referência completa (float32 com 9 dígitos) para o PC reproduzir as entradas exatas:
    // ml/06_verify_device_log.py usa esta linha.
    char ref[CsiPresence::kCarriers * 18];
    int used = 0;
    for (int k = 0; k < CsiPresence::kCarriers && used < static_cast<int>(sizeof(ref)); ++k) {
        used += std::snprintf(ref + used, sizeof(ref) - used, k ? ",%.9g" : "%.9g",
                              static_cast<double>(s_presence_cal.ref[k]));
    }
    const float *s = s_presence_cal.scale;
    ESP_LOGI(CAL_TAG, "CSI_REF v=2 origin=%s fp=%08lX rssi_ref=%.9g core7=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g "
             "scale=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g ref=%s",
             origin, static_cast<unsigned long>(model_params::kFingerprint),
             static_cast<double>(s_presence_cal.rssi_ref),
             static_cast<double>(s_core7.reference(0)), static_cast<double>(s_core7.reference(1)),
             static_cast<double>(s_core7.reference(2)), static_cast<double>(s_core7.reference(3)),
             static_cast<double>(s_core7.reference(4)), static_cast<double>(s_core7.reference(5)),
             static_cast<double>(s_core7.reference(6)),
             static_cast<double>(s[0]), static_cast<double>(s[1]), static_cast<double>(s[2]),
             static_cast<double>(s[3]), static_cast<double>(s[4]), static_cast<double>(s[5]),
             static_cast<double>(s[6]), ref);
}

// ----------------------------------------------------------------- pré-processamento por pacote
float median51(const float *values) {
    float sorted[CSI_LLTF_CARRIERS];
    std::memcpy(sorted, values, sizeof(sorted));
    std::sort(sorted, sorted + CSI_LLTF_CARRIERS);
    return sorted[CSI_LLTF_CARRIERS / 2];
}

// 51 subportadoras LLTF úteis (2..26 e 38..63); formato do buffer: [imag, real] int8.
bool decode_lltf(const int8_t *iq, float *amplitude, float *raw_median) {
    int k = 0;
    for (int sc = 2; sc <= 26; ++sc) amplitude[k++] = std::hypot(static_cast<float>(iq[2 * sc]), static_cast<float>(iq[2 * sc + 1]));
    for (int sc = 38; sc < kLLTFComplexCount; ++sc) amplitude[k++] = std::hypot(static_cast<float>(iq[2 * sc]), static_cast<float>(iq[2 * sc + 1]));
    const float median = median51(amplitude);
    if (!(median > 1e-6f) || !std::isfinite(median)) return false;
    *raw_median = median;
    for (int i = 0; i < CSI_LLTF_CARRIERS; ++i) amplitude[i] /= median;
    return true;
}

// ----------------------------------------------------------------- por janela
void learn_window(const float f12[CSI_FEATURE_COUNT], const float p[], const float p_lag[], float rssi) {
    if (esp_timer_get_time() - s_boot_us < kGraceUs) return;
    if (!s_cal_announced) {
        s_cal_announced = true;
        ESP_LOGI(CAL_TAG, "Calibrando EMPTY: mantenha a área vazia por %d janelas (~10 s)",
                 CsiPresence::kCalWindows);
    }
    if (s_core7.observe(f12) == CsiCore7Calibrator::State::Invalid) {
        ESP_LOGW(CAL_TAG, "Janela Core7 inválida na calibração; ignorada");
        return;
    }
    s_presence.calibration_add(p, p_lag, rssi);
    if (!(s_core7.ready() && s_presence.calibration_ready())) return;

    if (!s_presence.calibration_finish(s_presence_cal)) {
        ESP_LOGE(CAL_TAG, "Calibração inválida; reiniciando coleta");
        s_core7.reset();
        s_presence.calibration_reset();
        return;
    }
    float refs[7];
    for (int i = 0; i < 7; ++i) refs[i] = s_core7.reference(i);
    const bool saved = csi_calibration_nvs::save(model_params::kFingerprint, refs, s_presence_cal);
    log_reference("calibration");
    s_mode = Mode::Ready;
    s_motion_filter.reset();
    s_presence_filter.reset();
    led_service(saved ? LedService::Saved : LedService::Error);
    if (saved) ESP_LOGI(CAL_TAG, "Referência EMPTY salva na NVS (LED: curto-curto-longo x3)");
    else ESP_LOGE(CAL_TAG, "Falha ao gravar NVS (LED: 4 flashes); detecção segue até desligar");
}

void infer_window(const float f12[CSI_FEATURE_COUNT], const float p[], const float p_lag[], float rssi) {
    float zm[kInputs], fp[kInputs], zp[kInputs];
    if (!s_core7.normalized(f12, model_params::motion::kMean, model_params::motion::kStd, zm) ||
        !CsiPresence::features(p, p_lag, rssi, s_presence_cal.ref, s_presence_cal.rssi_ref, fp) ||
        !CsiPresence::inputs(fp, s_presence_cal, zp)) {
        ++s_failed;
        return;
    }
    uint32_t clipped = 0;
    int8_t qm[kInputs], qp[kInputs], om = 0, op = 0;
    const int64_t t0 = esp_timer_get_time();
    const float p_move = model_run(s_motion, zm, qm, om, clipped);
    const float p_pres = model_run(s_presence_model, zp, qp, op, clipped);
    const int64_t infer_us = esp_timer_get_time() - t0;
    if (kLogVectors) {
        ESP_LOGI("CSI_VEC", "window=%lu qm=%d,%d,%d,%d,%d,%d,%d om=%d qp=%d,%d,%d,%d,%d,%d,%d op=%d",
                 static_cast<unsigned long>(s_windows), qm[0], qm[1], qm[2], qm[3], qm[4], qm[5], qm[6], om,
                 qp[0], qp[1], qp[2], qp[3], qp[4], qp[5], qp[6], op);
    }
    const auto rm = s_motion_filter.observe(p_move);
    const auto rp = s_presence_filter.observe(p_pres);
    if (!rm.valid || !rp.valid) {
        ++s_failed;
        return;
    }
    const CsiState before = s_state;
    set_state(csi_fuse(rp.active, rm.active));
    const bool changed = s_state != before;
    if (changed || ++s_since_log >= kHeartbeatWindows) {
        s_since_log = 0;
        ESP_LOGI(STATE_TAG, "window=%lu t_ms=%lld state=%s event=%s p_move=%.4f move5=%.4f moving=%d "
                 "p_pres=%.4f pres5=%.4f present=%d clip=%lu infer_us=%lld",
                 static_cast<unsigned long>(s_windows), static_cast<long long>(esp_timer_get_time() / 1000),
                 csi_state_name(s_state), changed ? csi_state_name(s_state) : "NONE",
                 static_cast<double>(p_move), static_cast<double>(rm.mean), rm.active ? 1 : 0,
                 static_cast<double>(p_pres), static_cast<double>(rp.mean), rp.active ? 1 : 0,
                 static_cast<unsigned long>(clipped), static_cast<long long>(infer_us));
    }
}

void process_window() {
    float f12[CSI_FEATURE_COUNT];
    if (!csi_extract_features(s_window, f12)) {
        ++s_failed;
        return;
    }
    ++s_windows;
    s_last_window_us.store(esp_timer_get_time(), std::memory_order_relaxed);

    // Assinatura da janela (idêntica ao log CSI_SIG usado no treino).
    float sig[CsiPresence::kCarriers];
    uint32_t clipped = 0;
    for (int k = 0; k < CsiPresence::kCarriers; ++k) {
        float sum = 0.0f;
        for (int t = 0; t < CSI_WINDOW_FRAMES; ++t) sum += s_window[t][k];
        sig[k] = CsiPresence::q12(sum / CSI_WINDOW_FRAMES);
        if (sig[k] * 4096.0f >= 65535.0f) ++clipped;
    }
    float rssi_sum = 0.0f, med_sum = 0.0f;
    for (int t = 0; t < CSI_WINDOW_FRAMES; ++t) {
        rssi_sum += s_rssi_window[t];
        med_sum += s_raw_median_window[t];
    }
    // RSSI arredondado a 0,01 dB, como aparece no log de treino.
    const float rssi = std::round(rssi_sum / CSI_WINDOW_FRAMES * 100.0f) / 100.0f;
    if (kLogSignature) {
        char hex[4 * CsiPresence::kCarriers + 1];
        hex_q12(sig, CsiPresence::kCarriers, hex);
        ESP_LOGI("CSI_SIG", "v=1 window=%lu t_ms=%lld rssi=%.2f raw_med=%.4f clip=%lu q12=%s",
                 static_cast<unsigned long>(s_windows), static_cast<long long>(esp_timer_get_time() / 1000),
                 static_cast<double>(rssi), static_cast<double>(med_sum / CSI_WINDOW_FRAMES),
                 static_cast<unsigned long>(clipped), hex);
    }
    if (kLogFeatures) {
        ESP_LOGI("CSI_F12", "window=%lu f=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g",
                 static_cast<unsigned long>(s_windows),
                 static_cast<double>(f12[0]), static_cast<double>(f12[1]), static_cast<double>(f12[2]),
                 static_cast<double>(f12[3]), static_cast<double>(f12[4]), static_cast<double>(f12[5]),
                 static_cast<double>(f12[6]), static_cast<double>(f12[7]), static_cast<double>(f12[8]),
                 static_cast<double>(f12[9]), static_cast<double>(f12[10]), static_cast<double>(f12[11]));
    }

    float p[CsiPresence::kCarriers], p_lag[CsiPresence::kCarriers];
    if (!s_presence.push(sig, p, p_lag)) {
        ++s_failed;
        return;
    }
    switch (s_mode) {
    case Mode::Learning: learn_window(f12, p, p_lag, rssi); break;
    case Mode::Ready: infer_window(f12, p, p_lag, rssi); break;
    case Mode::Missing: break;  // sem referência: não classifica
    }
}

void reset_resampler(uint32_t timestamp) {
    std::memcpy(s_previous, s_current, sizeof(s_previous));
    s_previous_raw_median = s_current_raw_median;
    s_previous_rssi = s_current_rssi;
    s_previous_ts = timestamp;
    s_previous_time = 0;
    s_next_grid_time = kGridStepUs;
    s_window_count = 0;
    s_have_previous = true;
}
}  // namespace

// ================================================================= API C
extern "C" bool csi_live_init(void)
{
    s_boot_us = esp_timer_get_time();
    if (!led_init()) return false;
    if (!model_init(s_motion, s_motion_arena, model_params::motion::kInScale, model_params::motion::kInZero) ||
        !model_init(s_presence_model, s_presence_arena, model_params::presence::kInScale,
                    model_params::presence::kInZero)) {
        led_service(LedService::Error);
        return false;
    }
    s_core7.reset();
    s_presence.reset_history();
    s_presence.calibration_reset();
    s_motion_filter.reset();
    s_presence_filter.reset();

    if (jumper_closed()) {
        s_mode = Mode::Learning;
        led_service(LedService::Calibrating);
        ESP_LOGI(CAL_TAG, "GPIO26 em 3V3: espera de 12 s e calibração EMPTY de 20 janelas; grava NVS");
    } else {
        csi_calibration_nvs::Record r{};
        if (csi_calibration_nvs::load(model_params::kFingerprint, r) && s_core7.load_reference(r.core7)) {
            s_presence_cal = r.presence;
            s_mode = Mode::Ready;
            led_service(LedService::Restored);
            log_reference("nvs");
            ESP_LOGI(CAL_TAG, "Referência EMPTY carregada da NVS; detecção imediata");
        } else {
            s_mode = Mode::Missing;
            led_service(LedService::Error);
            ESP_LOGE(CAL_TAG, "Sem referência compatível na NVS (fp=%08lX). Feche GPIO26-3V3 e reinicie com a sala vazia",
                     static_cast<unsigned long>(model_params::kFingerprint));
        }
    }
    ESP_LOGI(TAG, "Pronto (modo %s): 3 estados (EMPTY/STATIC/MOVING), janela 50x51 @25 Hz, salto 480 ms, "
             "filtro %d janelas (entra %.2f / sai %.2f)", firmware_config::kModeName, CsiTemporalFilter::kWindowCount,
             static_cast<double>(CsiTemporalFilter::kEnter), static_cast<double>(CsiTemporalFilter::kExit));
    return true;
}

extern "C" bool csi_live_process(const wifi_pkt_rx_ctrl_t *ctrl, const int8_t *iq, uint16_t len,
                                 bool first_word_invalid)
{
    (void)first_word_invalid;  // a máscara LLTF fixa exclui as duas primeiras amostras complexas
    if (!ctrl || !iq) return false;
    if (len != kExpectedCsiBytes || ctrl->sig_mode != 1 || ctrl->cwb != 0 ||
        ctrl->secondary_channel != WIFI_SECOND_CHAN_NONE || ctrl->stbc != 0) {
        ++s_rejected;
        const int64_t now = esp_timer_get_time();
        if (now - s_last_reject_report_us > 5000000) {
            s_last_reject_report_us = now;
            ESP_LOGW(TAG, "PHY inesperado: len=%u sig_mode=%u cwb=%u secondary=%u stbc=%u (esperado HT20, 256 B)",
                     static_cast<unsigned>(len), static_cast<unsigned>(ctrl->sig_mode),
                     static_cast<unsigned>(ctrl->cwb), static_cast<unsigned>(ctrl->secondary_channel),
                     static_cast<unsigned>(ctrl->stbc));
        }
        return false;
    }
    if (!decode_lltf(iq, s_current, &s_current_raw_median)) {
        ++s_rejected;
        return false;
    }
    s_current_rssi = static_cast<float>(ctrl->rssi);
    if (!s_have_previous) {
        reset_resampler(ctrl->timestamp);
        return true;
    }
    const uint32_t delta_us = static_cast<uint32_t>(ctrl->timestamp - s_previous_ts);
    if (delta_us == 0 || delta_us > kMaxGapUs) {
        // Nunca misturar janelas/probabilidades através de uma lacuna de CSI.
        ++s_gap_resets;
        s_motion_filter.reset();
        s_presence_filter.reset();
        s_presence.reset_history();
        set_state(CsiState::Empty);
        if (s_mode == Mode::Learning && (s_core7.count() > 0 || s_presence.calibration_count() > 0) &&
            !s_core7.ready()) {
            s_core7.reset();
            s_presence.calibration_reset();
            ESP_LOGW(CAL_TAG, "Lacuna de CSI durante a calibração; reiniciando as 20 janelas");
        }
        reset_resampler(ctrl->timestamp);
        return true;
    }
    const uint64_t current_time = s_previous_time + delta_us;
    while (s_next_grid_time <= current_time) {
        const float r = static_cast<float>(s_next_grid_time - s_previous_time) / static_cast<float>(delta_us);
        for (int i = 0; i < CSI_LLTF_CARRIERS; ++i) {
            s_window[s_window_count][i] = s_previous[i] + (s_current[i] - s_previous[i]) * r;
        }
        s_raw_median_window[s_window_count] = s_previous_raw_median + (s_current_raw_median - s_previous_raw_median) * r;
        s_rssi_window[s_window_count] = s_previous_rssi + (s_current_rssi - s_previous_rssi) * r;
        if (++s_window_count == CSI_WINDOW_FRAMES) {
            process_window();
            const int keep = CSI_WINDOW_FRAMES - kHopFrames;
            std::memmove(s_window, s_window + kHopFrames, keep * sizeof(s_window[0]));
            std::memmove(s_raw_median_window, s_raw_median_window + kHopFrames, keep * sizeof(float));
            std::memmove(s_rssi_window, s_rssi_window + kHopFrames, keep * sizeof(float));
            s_window_count = keep;
        }
        s_next_grid_time += kGridStepUs;
    }
    std::memcpy(s_previous, s_current, sizeof(s_previous));
    s_previous_raw_median = s_current_raw_median;
    s_previous_rssi = s_current_rssi;
    s_previous_ts = ctrl->timestamp;
    s_previous_time = current_time;
    return true;
}
