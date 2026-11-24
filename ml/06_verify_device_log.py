#!/usr/bin/env python3
"""Fase 8/9 — valida o firmware NA PLACA contra o PC, a partir de um log do modo VALIDATION.

Para cada janela em que a placa registrou CSI_VEC, recalcula no PC — com a MESMA referência
EMPTY que a placa usou (linha CSI_REF) — as entradas int8 dos dois modelos a partir das linhas
CSI_SIG (presença) e CSI_F12 (movimento), executa o .tflite no PC sobre as entradas int8 da
placa e refaz os filtros e a fusão. Compara:

  1. entradas int8 (pré-processamento C++ x Python)
  2. saídas int8   (TFLite Micro no ESP32 x TFLite no PC)
  3. estado final  (CSI_STATE x filtros + fusão no PC)

Uso:
    python ml/06_verify_device_log.py data/tests/firmware/validation_01.log
    python ml/06_verify_device_log.py --self-test     # gera um log sintético e verifica a ferramenta
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from csi import motion as M
from csi import presence as P
from csi.export import Int8Runner
from csi.labels import load_manifest, sessions
from csi.logs import QCOLS, parse_device_log, parse_parity, parse_signatures
from csi.paths import DATA, MODELS
from csi.temporal import STATE_NAMES, fuse_states, run_filter


def load_stage(stage):
    pre = json.loads((MODELS / stage / 'preprocessing.json').read_text(encoding='utf-8'))
    run = Int8Runner((MODELS / stage / f'{stage}_int8.tflite').read_bytes())
    return pre, run


def pc_presence_q(sig, ref, pre, run):
    p = P.shape_profiles(sig[QCOLS].to_numpy())
    f = P.raw_features(p, P.lagged(p), sig['rssi'].to_numpy(), ref['ref'], ref['rssi_ref'])
    x = np.log(np.clip(f / ref['scale'], P.CLIP_LO, P.CLIP_HI))
    z = ((x - np.array(pre['scaler_mean'])) / np.array(pre['scaler_std'])).astype(np.float32)
    return dict(zip(sig['window'].to_numpy(), run.quantize(z)))


def pc_motion_q(f12, ref, pre, run):
    out = {}
    for w, f in f12.items():
        z = ((M.log_ratio(f[list(M.CORE7)], ref['core7']) - np.array(pre['scaler_mean']))
             / np.array(pre['scaler_std'])).astype(np.float32)
        out[w] = run.quantize(z[None, :])[0]
    return out


def compare(name, pc, dev):
    d = np.abs(np.asarray(pc, int) - np.asarray(dev, int))
    exact = int(np.sum(d.max(axis=-1) == 0)) if d.ndim > 1 else int(np.sum(d == 0))
    n = len(d)
    worst = int(d.max()) if n else 0
    print(f'  {name:<42s} idênticas {exact}/{n}  ({100 * exact / max(n, 1):.2f} %), maior diferença {worst} LSB')
    return dict(n=n, identical=exact, max_diff_lsb=worst)


def verify(path: Path) -> bool:
    log = parse_device_log(path)
    if log['ref'] is None or log['vec'].empty:
        sys.exit('O log precisa de CSI_REF e CSI_VEC (firmware no modo VALIDATION e calibrado).')
    pre_p, run_p = load_stage('presence')
    pre_m, run_m = load_stage('motion')
    vec = log['vec'].sort_values('window')
    win = vec['window'].to_numpy()
    print(f'{path}: {len(log["sig"])} CSI_SIG, {len(log["f12"])} CSI_F12, {len(vec)} CSI_VEC, '
          f'referência origin={log["ref"]["origin"]} fp={log["ref"]["fingerprint"]:08X}')
    report = {}

    qp_pc = pc_presence_q(log['sig'], log['ref'], pre_p, run_p)
    qm_pc = pc_motion_q(log['f12'], log['ref'], pre_m, run_m)
    ok = [w in qp_pc and w in qm_pc for w in win]
    v = vec[ok]
    print('Pré-processamento (C++ no ESP32 x Python):')
    report['presence_inputs'] = compare('entradas int8 da presença', [qp_pc[w] for w in v.window], list(v.qp))
    report['motion_inputs'] = compare('entradas int8 do movimento', [qm_pc[w] for w in v.window], list(v.qm))
    print('Inferência (TFLite Micro no ESP32 x TFLite no PC, mesmas entradas int8):')
    report['presence_outputs'] = compare('saída int8 da presença', [run_p.invoke_q(q) for q in vec.qp], list(vec.op))
    report['motion_outputs'] = compare('saída int8 do movimento', [run_m.invoke_q(q) for q in vec.qm], list(vec.om))

    st = log['state']
    if not st.empty and 'state' in st:
        pm = (vec.om.to_numpy() + 128) / 256.0
        pp = (vec.op.to_numpy() + 128) / 256.0
        mv, _ = run_filter(pm)
        pr, _ = run_filter(pp)
        pc_state = dict(zip(win, (STATE_NAMES[k] for k in fuse_states(pr, mv))))
        dev_state = dict(zip(st['window'].astype(int), st['state'].astype(str)))
        common = [w for w in win if w in dev_state]
        same = sum(pc_state[w] == dev_state[w] for w in common)
        print('Decisão final:')
        print(f'  {"estado (filtros + fusão)":<42s} idênticos {same}/{len(common)}')
        report['state'] = dict(n=len(common), identical=same)
    good = (report['presence_inputs']['max_diff_lsb'] <= 1 and report['motion_inputs']['max_diff_lsb'] <= 1
            and report['presence_outputs']['max_diff_lsb'] <= 1 and report['motion_outputs']['max_diff_lsb'] <= 1)
    print('RESULTADO:', 'firmware equivalente ao PC (diferenças <= 1 LSB)' if good else 'DIVERGÊNCIA — investigar')
    out = path.with_suffix('.verify.json')
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return good


def synthetic_log(dest: Path) -> None:
    """Emula no PC o que o firmware VALIDATION imprimiria, usando a sessão D e o teste live_B."""
    pre_p, run_p = load_stage('presence')
    pre_m, run_m = load_stage('motion')
    s = [x for x in sessions('dataset') if x.id == 'D'][0]
    full = P.session_table(s)
    cal = full.attrs['calibration']
    tb = full.iloc[:260]
    sig = parse_signatures(s.path).iloc[:260]
    q12 = np.rint(sig[QCOLS].to_numpy() * 4096).astype(int)
    mot = parse_parity(DATA / {t['id']: t for t in load_manifest()['motion_live_logs']}['live_B']['file'])
    lines = [f'I (500) CSI_CAL: CSI_REF v=2 origin=calibration fp=1D0D612C rssi_ref={cal["rssi_ref"]:.9g} '
             f'core7={",".join(f"{v:.9g}" for v in mot["refs"])} '
             f'scale={",".join(f"{v:.9g}" for v in cal["scale"])} ref={",".join(f"{v:.9g}" for v in cal["ref"])}']
    ref = dict(ref=cal['ref'], rssi_ref=cal['rssi_ref'], scale=cal['scale'], core7=mot['refs'])
    qp = pc_presence_q(sig, ref, pre_p, run_p)
    f12 = {}
    for i, w in enumerate(sig['window']):
        f = np.zeros(12)
        f[list(M.CORE7)] = mot['f'][i % len(mot['f'])]
        f12[int(w)] = f
    qm = pc_motion_q(f12, ref, pre_m, run_m)
    pm_list, pp_list, ws = [], [], []
    start = int(np.where(tb['calibration'])[0].max()) + 1
    for i, row in enumerate(sig.itertuples()):
        w, ms = int(row.window), int(round(row.t_s * 1000))
        lines.append(f'I ({ms}) CSI_SIG: v=1 window={w} t_ms={ms} rssi={row.rssi:.2f} raw_med={row.raw_med:.4f} '
                     f'clip=0 q12={"".join(f"{v:04X}" for v in q12[i])}')
        lines.append(f'I ({ms}) CSI_F12: window={w} f={",".join(f"{v:.9g}" for v in f12[w])}')
        if i < start:
            continue
        om, op = run_m.invoke_q(qm[w]), run_p.invoke_q(qp[w])
        lines.append(f'I ({ms}) CSI_VEC: window={w} qm={",".join(map(str, qm[w]))} om={om} '
                     f'qp={",".join(map(str, qp[w]))} op={op}')
        pm_list.append((om + 128) / 256)
        pp_list.append((op + 128) / 256)
        ws.append((w, ms))
    mv, _ = run_filter(np.array(pm_list))
    pr, _ = run_filter(np.array(pp_list))
    states = fuse_states(pr, mv)
    state_lines = [f'I ({ms}) CSI_STATE: window={w} t_ms={ms} state={STATE_NAMES[k]} event=NONE'
                   for (w, ms), k in zip(ws, states)]
    dest.write_text('\n'.join(lines + state_lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', nargs='?', type=Path)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'synthetic_validation.log'
            synthetic_log(path)
            print('Autoteste: log sintético no formato do modo VALIDATION (sessão D + teste live_B)')
            ok = verify(path)
        sys.exit(0 if ok else 1)
    if not args.log:
        ap.error('informe o log ou --self-test')
    sys.exit(0 if verify(args.log) else 1)


if __name__ == '__main__':
    main()
