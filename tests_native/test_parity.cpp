// Testes nativos (PC) do pré-processamento embarcado contra os vetores gerados pelo Python.
// Garante que o C++ do firmware produz as MESMAS entradas INT8 que o modelo viu no treino.
//
//   g++ -std=c++17 -O2 -I../src/rx test_parity.cpp -o test_parity && ./test_parity vectors
//   (ou: bash tests_native/run.sh)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "csi_core7_calibrator.h"
#include "csi_presence.h"
#include "csi_state_machine.h"
#include "csi_temporal_filter.h"
#include "model_params.h"

static int g_fail = 0;
#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("  FALHA %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++g_fail;                                        \
        }                                                    \
    } while (0)

static std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) out.push_back(item);
    return out;
}

static std::vector<double> nums(const std::string &s) {
    std::vector<double> v;
    std::stringstream ss(s);
    double x;
    while (ss >> x) v.push_back(x);
    return v;
}

static int quantize(float z, float scale, int zero) {
    const float raw = z / scale + static_cast<float>(zero);
    return std::max(-128, std::min(127, static_cast<int>(std::nearbyint(raw))));
}

// ------------------------------------------------------------------ presença
static void test_presence(const std::string &dir) {
    std::printf("[presença] vetores %s/presence_D.txt\n", dir.c_str());
    std::ifstream in(dir + "/presence_D.txt");
    CHECK(in.good(), "arquivo de vetores ausente");
    std::string line;
    int cal_start = -1;
    struct Row { float rssi; float s[51]; double f[7], z[7]; int q[7]; };
    std::vector<Row> rows;
    while (std::getline(in, line)) {
        if (line.rfind("# calibration_start_index", 0) == 0) { cal_start = std::atoi(line.substr(26).c_str()); continue; }
        if (line.empty() || line[0] == '#') continue;
        auto parts = split(line, '|');
        auto a = nums(parts[0]), f = nums(parts[1]), z = nums(parts[2]), q = nums(parts[3]);
        Row r{};
        r.rssi = static_cast<float>(a[0]);
        for (int k = 0; k < 51; ++k) r.s[k] = static_cast<float>(a[1 + k]) / 4096.0f;
        for (int j = 0; j < 7; ++j) { r.f[j] = f[j]; r.z[j] = z[j]; r.q[j] = static_cast<int>(q[j]); }
        rows.push_back(r);
    }
    CHECK(cal_start >= 0 && rows.size() > static_cast<size_t>(cal_start + 20), "vetores incompletos");

    // 1ª passada: histórico + calibração exatamente como o firmware (push em toda janela)
    static CsiPresence pres;
    pres.reset_history();
    pres.calibration_reset();
    std::vector<std::vector<float>> P(rows.size(), std::vector<float>(51)), L(rows.size(), std::vector<float>(51));
    for (size_t i = 0; i < rows.size(); ++i) {
        CHECK(pres.push(rows[i].s, P[i].data(), L[i].data()), "push falhou em %zu", i);
        if (static_cast<int>(i) >= cal_start && !pres.calibration_ready()) pres.calibration_add(P[i].data(), L[i].data(), rows[i].rssi);
    }
    CsiPresence::Calibration cal{};
    CHECK(pres.calibration_finish(cal), "calibração falhou");

    double max_rel_f = 0, max_abs_z = 0;
    int q_mismatch = 0, q_off_by_more = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        float f[7], z[7];
        CHECK(CsiPresence::features(P[i].data(), L[i].data(), rows[i].rssi, cal.ref, cal.rssi_ref, f), "features %zu", i);
        CHECK(CsiPresence::inputs(f, cal, z), "inputs %zu", i);
        for (int j = 0; j < 7; ++j) {
            const double denom = std::max(std::fabs(rows[i].f[j]), 1e-4);
            max_rel_f = std::max(max_rel_f, std::fabs(f[j] - rows[i].f[j]) / denom);
            max_abs_z = std::max(max_abs_z, std::fabs(z[j] - rows[i].z[j]));
            const int q = quantize(z[j], model_params::presence::kInScale, model_params::presence::kInZero);
            if (q != rows[i].q[j]) ++q_mismatch;
            if (std::abs(q - rows[i].q[j]) > 1) ++q_off_by_more;
        }
    }
    const int total = static_cast<int>(rows.size()) * 7;
    std::printf("  janelas=%zu  max erro relativo features=%.2e  max |dz|=%.2e  int8 diferentes=%d/%d (>1 LSB: %d)\n",
                rows.size(), max_rel_f, max_abs_z, q_mismatch, total, q_off_by_more);
    CHECK(max_abs_z < 2e-3, "z diverge do Python (%.3e)", max_abs_z);
    CHECK(q_off_by_more == 0, "entrada INT8 difere em mais de 1 LSB");
    CHECK(q_mismatch <= total / 200, "muitas entradas INT8 diferentes (%d)", q_mismatch);
}

// ------------------------------------------------------------------ movimento
static void test_motion(const std::string &dir) {
    std::printf("[movimento] vetores %s/motion_live_B.txt\n", dir.c_str());
    std::ifstream in(dir + "/motion_live_B.txt");
    CHECK(in.good(), "arquivo de vetores ausente");
    std::string line;
    std::vector<double> refs;
    CsiCore7Calibrator cal;
    int n = 0, q_mismatch = 0;
    double max_abs_z = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (refs.empty()) {
            refs = nums(line);
            float r[7];
            for (int j = 0; j < 7; ++j) r[j] = static_cast<float>(refs[j]);
            CHECK(cal.load_reference(r), "load_reference");
            continue;
        }
        auto parts = split(line, '|');
        auto f7 = nums(parts[0]), z = nums(parts[1]), q = nums(parts[2]);
        float f12[12] = {};
        for (int j = 0; j < 7; ++j) f12[CsiCore7Calibrator::feature_index(j)] = static_cast<float>(f7[j]);
        float zz[7];
        CHECK(cal.normalized(f12, model_params::motion::kMean, model_params::motion::kStd, zz), "normalized");
        for (int j = 0; j < 7; ++j) {
            max_abs_z = std::max(max_abs_z, std::fabs(zz[j] - z[j]));
            if (quantize(zz[j], model_params::motion::kInScale, model_params::motion::kInZero) != static_cast<int>(q[j])) ++q_mismatch;
        }
        ++n;
    }
    std::printf("  janelas=%d  max |dz|=%.2e  int8 diferentes=%d/%d\n", n, max_abs_z, q_mismatch, n * 7);
    CHECK(max_abs_z < 1e-4, "z diverge (%.3e)", max_abs_z);
    CHECK(q_mismatch == 0, "entrada INT8 difere");
}

// ------------------------------------------------------------------ filtro e 3 estados
static void test_filter_and_states() {
    std::printf("[filtro + 3 estados]\n");
    CsiTemporalFilter f;
    f.reset();
    for (int i = 0; i < 4; ++i) CHECK(!f.observe(0.9f).active, "não pode ativar antes de 5 janelas");
    CHECK(f.observe(0.9f).active, "deve ativar com média >= 0,65");
    CHECK(f.observe(0.5f).active, "histerese: 0,5 mantém ativo");
    for (int i = 0; i < 4; ++i) f.observe(0.0f);
    CHECK(!f.observe(0.0f).active, "deve desativar com média <= 0,35");
    CHECK(!f.observe(NAN).valid, "NaN rejeitado");
    f.reset();
    for (int i = 0; i < 5; ++i) f.observe(0.6f);
    CHECK(!f.active(), "0,6 (entre os limiares) a partir de inativo permanece inativo");

    CHECK(csi_fuse(false, false) == CsiState::Empty, "EMPTY");
    CHECK(csi_fuse(true, false) == CsiState::Static, "STATIC");
    CHECK(csi_fuse(true, true) == CsiState::Moving, "MOVING");
    CHECK(csi_fuse(false, true) == CsiState::Moving, "movimento implica presença");
    CHECK(!csi_led_level(CsiState::Empty, 123), "EMPTY apagado");
    CHECK(csi_led_level(CsiState::Moving, 777), "MOVING aceso");
    CHECK(csi_led_level(CsiState::Static, 100) && !csi_led_level(CsiState::Static, 600) &&
          csi_led_level(CsiState::Static, 1100), "STATIC pisca 500/500 ms");
    CHECK(csi_service_finished(LedService::Saved, 5000) && !csi_service_finished(LedService::Error, 100000),
          "padrões de serviço");
}

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "vectors";
    test_presence(dir);
    test_motion(dir);
    test_filter_and_states();
    std::printf(g_fail ? "RESULTADO: %d falha(s)\n" : "RESULTADO: todos os testes passaram\n", g_fail);
    return g_fail ? 1 : 0;
}
