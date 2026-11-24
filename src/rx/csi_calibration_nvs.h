#pragma once

// Persistência da referência EMPTY na NVS (sobrevive a reset e falta de energia).
// Guarda SOMENTE a calibração do ambiente; pesos e scalers ficam no firmware.
// O fingerprint (model_params::kFingerprint) invalida referências de outra versão de modelo.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "csi_presence.h"
#include "nvs.h"

namespace csi_calibration_nvs {
constexpr uint32_t kMagic = 0x43534933u;  // "CSI3"
constexpr uint16_t kVersion = 2;
constexpr char kNamespace[] = "csi_ref";
constexpr char kKey[] = "baseline_v2";

struct Record {
    uint32_t magic;
    uint16_t version;
    uint16_t reserved;
    uint32_t fingerprint;
    float core7[7];                         // referência do estágio de movimento
    CsiPresence::Calibration presence;      // referência espectral + ruído
    uint32_t checksum;
};

inline uint32_t fnv1a(uint32_t h, const void *data, size_t size) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) h = (h ^ p[i]) * 16777619u;
    return h;
}

inline bool valid(const Record &r, uint32_t fingerprint) {
    if (r.magic != kMagic || r.version != kVersion || r.reserved != 0 || r.fingerprint != fingerprint ||
        r.checksum != fnv1a(2166136261u, &r, offsetof(Record, checksum))) return false;
    for (float f : r.core7) if (!std::isfinite(f) || f < 0.0f) return false;
    return CsiPresence::valid(r.presence);
}

inline bool load(uint32_t fingerprint, Record &out) {
    nvs_handle_t h;
    if (nvs_open(kNamespace, NVS_READONLY, &h) != ESP_OK) return false;
    Record r{};
    size_t length = sizeof(r);
    const esp_err_t rc = nvs_get_blob(h, kKey, &r, &length);
    nvs_close(h);
    if (rc != ESP_OK || length != sizeof(r) || !valid(r, fingerprint)) return false;
    out = r;
    return true;
}

inline bool save(uint32_t fingerprint, const float core7[7], const CsiPresence::Calibration &presence) {
    Record r{};
    r.magic = kMagic;
    r.version = kVersion;
    r.fingerprint = fingerprint;
    for (int i = 0; i < 7; ++i) r.core7[i] = core7[i];
    r.presence = presence;
    r.checksum = fnv1a(2166136261u, &r, offsetof(Record, checksum));
    if (!valid(r, fingerprint)) return false;
    nvs_handle_t h;
    if (nvs_open(kNamespace, NVS_READWRITE, &h) != ESP_OK) return false;
    esp_err_t rc = nvs_set_blob(h, kKey, &r, sizeof(r));
    if (rc == ESP_OK) rc = nvs_commit(h);
    nvs_close(h);
    return rc == ESP_OK;
}
}  // namespace csi_calibration_nvs
