#!/usr/bin/env python3
"""Etapa 3 — conversão para o ESP32: PyTorch -> Keras -> TFLite INT8 -> C.

Para cada estágio (motion, presence):
  1. Reconstrói a MLP salva (.pt) e copia os pesos para um tf.keras.Sequential idêntico.
  2. Converte para TFLite com quantização INT8 completa (entrada/saída int8), usando
     as entradas padronizadas de treino como conjunto representativo.
  3. Mede a paridade float PyTorch x float Keras x INT8 (TFLite Interpreter) e a
     concordância de decisão nos dados de teste (logs ao vivo e holdout).
  4. Gera os fontes do firmware:
        src/rx/model_motion_data.{h,cpp}     bytes do .tflite de movimento
        src/rx/model_presence_data.{h,cpp}   bytes do .tflite de presença
        src/rx/model_params.h                scalers, pisos, filtros, fingerprint
  5. Gera vetores de teste para os testes nativos C++ (tests_native/vectors/).

Uso:  python ml/03_export_firmware.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import torch

from csi import motion as M
from csi import presence as P
from csi.export import Int8Runner, c_float_array, cf, export_model, write_model_source
from csi.labels import load_manifest, sessions
from csi.logs import parse_parity, parse_signatures
from csi.metrics import binary_metrics
from csi.nets import mlp, predict_proba
from csi.paths import DATA, FIRMWARE_RX, MODELS, RESULTS, ROOT, ensure_dirs
from csi.temporal import run_filter


def load_stage(name: str):
    d = MODELS / name
    pre = json.loads((d / 'preprocessing.json').read_text(encoding='utf-8'))
    arch = pre['architecture']
    net = mlp(arch[0], tuple(arch[1:-1]))
    net.load_state_dict(torch.load(d / f'{name}_mlp.pt', map_location='cpu', weights_only=True))
    net.eval()
    z = np.load(d / 'representative_inputs.npy')
    return net, pre, z


def fnv1a(h: int, data: bytes) -> int:
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def main():
    ensure_dirs()
    report = {}
    blobs, pres = {}, {}
    for stage in ('motion', 'presence'):
        net, pre, z = load_stage(stage)
        rep = export_model(net, z, MODELS / stage, f'{stage}_int8')
        blobs[stage] = (MODELS / stage / f'{stage}_int8.tflite').read_bytes()
        pres[stage] = (net, pre)
        report[stage] = rep
        print(f'{stage}: {rep["bytes"]} B, ops={rep["ops"]}, quant={rep["quantization"]}')
        print(f'   float torch x keras max |dp| = {rep["float_torch_vs_keras_max_abs"]:.2e}; '
              f'INT8 x float max |dp| = {rep["int8_vs_float_max_abs"]:.4f}, '
              f'decisão igual em {100 * rep["int8_vs_float_decision_agreement"]:.2f}% (treino)')

    # ---------------- paridade INT8 em dados de TESTE
    net_m, pre_m = pres['motion']
    run_m = Int8Runner(blobs['motion'])
    mean_m, std_m = np.array(pre_m['scaler_mean']), np.array(pre_m['scaler_std'])
    live_rows, scatter = [], {'motion': ([], []), 'presence': ([], [])}
    for item in load_manifest()['motion_live_logs']:
        log = parse_parity(DATA / item['file'])
        z = ((M.log_ratio(log['f'], log['refs']) - mean_m) / std_m).astype(np.float32)
        pf, pq = predict_proba(net_m, z), run_m(z)
        sf, _ = run_filter(pf)
        sq, _ = run_filter(pq)
        scatter['motion'][0].append(pf)
        scatter['motion'][1].append(pq)
        live_rows.append(dict(log=item['id'], n=len(z), raw_decision_agreement=float(np.mean((pf >= .5) == (pq >= .5))),
                              stable_state_agreement=float(np.mean(sf == sq)),
                              max_abs_dp=float(np.max(np.abs(pf - pq)))))
    report['motion_int8_live'] = live_rows

    # ---------------- paridade PC x ESP32 do runtime INT8 (TFLite x TFLite Micro)
    # Os logs CSI_PARITY trazem a entrada int8 (q=) e a saída int8 (out_q=) calculadas NO ESP32
    # pelo modelo Core7 que estava gravado (models/legacy/). Reexecutamos no PC.
    legacy = Int8Runner((MODELS / 'legacy' / 'core7_mlp_int8_gravado_no_esp32.tflite').read_bytes())
    n_tot = n_eq = max_diff = 0
    for item in load_manifest()['motion_live_logs']:
        for line in (DATA / item['file']).read_text(errors='replace').splitlines():
            m = re.search(r'CSI_PARITY: window=\d+ .* q=([-\d,]+) out_q=(-?\d+)', line)
            if m:
                out = legacy.invoke_q([int(v) for v in m.group(1).split(',')])
                n_tot += 1
                n_eq += out == int(m.group(2))
                max_diff = max(max_diff, abs(out - int(m.group(2))))
    report['pc_vs_esp32_tflm'] = dict(windows=n_tot, identical_int8_outputs=n_eq, max_abs_diff_lsb=max_diff)
    print(f'TFLite (PC) x TFLite Micro (ESP32): saída int8 idêntica em {n_eq}/{n_tot} janelas ao vivo '
          f'(maior diferença {max_diff} LSB = {max_diff / 256:.4f} em probabilidade)')

    net_p, pre_p = pres['presence']
    run_p = Int8Runner(blobs['presence'])
    mean_p, std_p = np.array(pre_p['scaler_mean']), np.array(pre_p['scaler_std'])
    pres_rows = []
    for s in sessions():
        tb = P.session_table(s)
        z = ((tb[P.X_COLS].to_numpy() - mean_p) / std_p).astype(np.float32)
        pf, pq = predict_proba(net_p, z), run_p(z)
        sf, _ = run_filter(pf)
        sq, _ = run_filter(pq)
        if s.role in ('dataset', 'holdout'):
            scatter['presence'][0].append(pf)
            scatter['presence'][1].append(pq)
        row = dict(session=s.id, role=s.role, n=len(z),
                   raw_decision_agreement=float(np.mean((pf >= .5) == (pq >= .5))),
                   stable_state_agreement=float(np.mean(sf == sq)), max_abs_dp=float(np.max(np.abs(pf - pq))))
        if s.role == 'holdout':
            sc = tb['scored'].to_numpy()
            y = (tb['label'] != 'EMPTY').to_numpy()
            row['holdout_balanced_accuracy_float'] = binary_metrics(y[sc], sf[sc])['balanced_accuracy']
            row['holdout_balanced_accuracy_int8'] = binary_metrics(y[sc], sq[sc])['balanced_accuracy']
        pres_rows.append(row)
    report['presence_int8_sessions'] = pres_rows
    # tempo de inferência medido NA PLACA (campo infer_us dos logs de teste; MLP Core7 INT8,
    # mesma arquitetura 7-16-8-1 dos dois modelos atuais, ESP32 a 240 MHz)
    us = []
    for item in load_manifest()['motion_live_logs']:
        us += [int(v) for v in re.findall(r'infer_us=(\d+)', (DATA / item['file']).read_text(errors='replace'))][1:]
    us = np.array(us)
    report['device_inference_us'] = dict(n=int(len(us)), median=float(np.median(us)),
                                         p10=float(np.percentile(us, 10)), p90=float(np.percentile(us, 90)))
    report['optimization'] = {stage: dict(parameters=report[stage]['parameters'],
                                          weights_float32_bytes=4 * report[stage]['parameters'],
                                          tflite_float32_bytes=report[stage]['float32_tflite_bytes'],
                                          tflite_int8_bytes=report[stage]['bytes'])
                              for stage in ('motion', 'presence')}
    print(f"Inferência medida no ESP32 (MLP 7-16-8-1 INT8): mediana {np.median(us):.0f} µs em {len(us)} janelas")
    for stage, o in report['optimization'].items():
        print(f"{stage}: {o['parameters']} parâmetros; TFLite float32 {o['tflite_float32_bytes']} B -> INT8 {o['tflite_int8_bytes']} B")
    (RESULTS / 'export_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    from csi.paths import FIGURES
    from csi.plots import plt, save
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (stage, (pf, pq)), title in zip(axes, scatter.items(),
                                            ('Movimento — 3 logs de teste ao vivo', 'Presença — 6 sessões do dataset')):
        pf, pq = np.concatenate(pf), np.concatenate(pq)
        ax.scatter(pf, pq, s=3, alpha=0.35, color='#08519c')
        ax.plot([0, 1], [0, 1], color='k', lw=0.6)
        ax.axhline(0.5, color='#999', lw=0.5, ls=':')
        ax.axvline(0.5, color='#999', lw=0.5, ls=':')
        agree = 100 * np.mean((pf >= .5) == (pq >= .5))
        ax.set_title(f'{title}\n{len(pf)} janelas; decisão igual em {agree:.1f}%', fontsize=8.5)
        ax.set_xlabel('P float32 (PyTorch)')
        ax.set_ylabel('P INT8 (TFLite)')
        ax.grid(alpha=0.3)
    fig.suptitle(f"Quantização INT8: modelos de {report['motion']['bytes']} B e {report['presence']['bytes']} B", fontsize=10, y=1.04)
    save(fig, FIGURES / 'export_int8_parity.png')
    for r in live_rows:
        print(f"motion INT8 x float [{r['log']}]: decisão {100*r['raw_decision_agreement']:.1f}%, "
              f"estado estável {100*r['stable_state_agreement']:.1f}%")
    for r in pres_rows:
        extra = (f", BA holdout float={r['holdout_balanced_accuracy_float']:.3f} "
                 f"int8={r['holdout_balanced_accuracy_int8']:.3f}") if 'holdout_balanced_accuracy_int8' in r else ''
        print(f"presence INT8 x float [{r['session']}]: decisão {100*r['raw_decision_agreement']:.1f}%, "
              f"estado estável {100*r['stable_state_agreement']:.1f}%{extra}")

    # ---------------- fontes do firmware
    f32 = lambda a: np.asarray(a, dtype=np.float32).tobytes()  # noqa: E731
    fp = 2166136261
    for part in (blobs['motion'], blobs['presence'], f32(pre_m['scaler_mean']), f32(pre_m['scaler_std']),
                 f32(pre_p['scaler_mean']), f32(pre_p['scaler_std']), f32(pre_p['floors'])):
        fp = fnv1a(fp, part)
    write_model_source(blobs['motion'], 'g_motion_model', FIRMWARE_RX / 'model_motion_data.h',
                       FIRMWARE_RX / 'model_motion_data.cpp',
                       f'Movimento Core7: MLP 7-16-8-1 INT8, {len(blobs["motion"])} bytes')
    write_model_source(blobs['presence'], 'g_presence_model', FIRMWARE_RX / 'model_presence_data.h',
                       FIRMWARE_RX / 'model_presence_data.cpp',
                       f'Presença espectral: MLP 7-16-8-1 INT8, {len(blobs["presence"])} bytes')
    qm, qp = report['motion']['quantization'], report['presence']['quantization']
    params = f"""// AUTO-GERADO por ml/03_export_firmware.py — não edite à mão.
// Constantes de pré-processamento que DEVEM ser idênticas às usadas no treino.
#pragma once
#include <cstdint>

namespace model_params {{
// Muda sempre que um modelo ou scaler muda; invalida referências EMPTY antigas na NVS.
static constexpr uint32_t kFingerprint = 0x{fp:08X}u;

namespace motion {{
static constexpr int kInputs = 7;
static constexpr int kFeatureIndex[kInputs] = {{{', '.join(str(i) for i in pre_m['feature_indices'])}}};
{c_float_array('kMean', pre_m['scaler_mean'])}
{c_float_array('kStd', pre_m['scaler_std'])}
// Quantização do tensor de entrada/saída (conferida em runtime contra o .tflite)
static constexpr float kInScale = {cf(qm['in_scale'])};
static constexpr int kInZero = {qm['in_zero']};
}}  // namespace motion

namespace presence {{
static constexpr int kInputs = 7;
static constexpr int kCalibrationWindows = {pre_p['calibration_windows']};
static constexpr int kLagWindows = {pre_p['lag_windows']};
static constexpr float kClipLo = {cf(pre_p['clip'][0])};
static constexpr float kClipHi = {cf(pre_p['clip'][1])};
{c_float_array('kFloors', pre_p['floors'])}
{c_float_array('kMean', pre_p['scaler_mean'])}
{c_float_array('kStd', pre_p['scaler_std'])}
static constexpr float kInScale = {cf(qp['in_scale'])};
static constexpr int kInZero = {qp['in_zero']};
}}  // namespace presence

namespace filter {{
static constexpr int kWindowCount = 5;
static constexpr float kEnter = 0.65f;
static constexpr float kExit = 0.35f;
}}  // namespace filter
}}  // namespace model_params
"""
    (FIRMWARE_RX / 'model_params.h').write_text(params, encoding='utf-8')
    print(f'\nFirmware: model_motion_data.cpp, model_presence_data.cpp, model_params.h '
          f'(fingerprint 0x{fp:08X}) em {FIRMWARE_RX}')

    # ---------------- vetores para os testes nativos C++
    vec = ROOT / 'tests_native' / 'vectors'
    vec.mkdir(parents=True, exist_ok=True)
    s = [x for x in sessions('dataset') if x.id == 'D'][0]
    df = parse_signatures(s.path).iloc[:300]
    tb = P.session_table(s).iloc[:300]
    cal_start = int(tb.index[tb['t_s'] >= s.calibration_start_s][0])
    z = ((tb[P.X_COLS].to_numpy() - mean_p) / std_p).astype(np.float32)
    q_in = run_p.quantize(z)
    with (vec / 'presence_D.txt').open('w') as fh:
        fh.write(f'# calibration_start_index {cal_start}\n')
        fh.write('# rssi q12[51] | expected raw features[7] | expected z[7] | expected int8[7]\n')
        q12 = np.rint(df[[f'q{i:02d}' for i in range(51)]].to_numpy() * 4096).astype(int)
        for i in range(len(df)):
            fh.write(' '.join([f'{df.rssi.iloc[i]:.2f}', *map(str, q12[i])]) + ' | '
                     + ' '.join(f'{v:.9g}' for v in tb[list(P.FEATURE_NAMES)].iloc[i]) + ' | '
                     + ' '.join(f'{v:.9g}' for v in z[i]) + ' | ' + ' '.join(map(str, q_in[i])) + '\n')
    live = {t['id']: t for t in load_manifest()['motion_live_logs']}
    log = parse_parity(DATA / live['live_B']['file'])
    zm = ((M.log_ratio(log['f'], log['refs']) - mean_m) / std_m).astype(np.float32)
    with (vec / 'motion_live_B.txt').open('w') as fh:
        fh.write('# refs[7]\n' + ' '.join(f'{v:.9g}' for v in log['refs']) + '\n')
        fh.write('# f7[7] | expected z[7] | expected int8[7]\n')
        for f7, zz, qq in zip(log['f'][:200], zm[:200], run_m.quantize(zm[:200])):
            fh.write(' '.join(f'{v:.9g}' for v in f7) + ' | ' + ' '.join(f'{v:.9g}' for v in zz)
                     + ' | ' + ' '.join(map(str, qq)) + '\n')
    print(f'Vetores de teste C++: {vec}')


if __name__ == '__main__':
    main()
