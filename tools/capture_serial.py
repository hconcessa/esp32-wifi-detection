#!/usr/bin/env python3
"""Grava o log serial do RX (firmware rx_capture ou rx_validation), com marcação de fases pelo teclado.

Uso:
    python tools/capture_serial.py --port COM6 --out data/dataset/presence/G.log

Durante a captura, digite o nome da fase e Enter (ex.: EMPTY, P1, EMPTY, WALK...).
Cada marcação vira uma linha "MARK t_s=<tempo do ESP32> name=<fase>" no log, usando o
t_ms da última linha CSI_SIG recebida — o MESMO relógio usado no manifest.json.
Ctrl+C encerra. Depois, copie as marcações para data/manifest.json (fase = [nome, rótulo, início, fim]).

Observação: abrir a porta pode resetar o ESP32 (DTR/RTS). Isso é esperado: o log começa no boot.
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path

T_RE = re.compile(r'CSI_SIG: v=1 window=\d+ t_ms=(\d+)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', required=True, help='ex.: COM6 ou /dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    try:
        import serial
    except ImportError:
        sys.exit('pip install pyserial')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    last_t = [0.0]
    lock = threading.Lock()
    marks = []
    with serial.Serial(args.port, args.baud, timeout=0.2) as ser, args.out.open('w', encoding='utf-8') as out:
        def keyboard():
            for line in sys.stdin:
                name = line.strip()
                if not name:
                    continue
                with lock:
                    t = last_t[0]
                    out.write(f'MARK t_s={t:.3f} name={name}\n')
                    out.flush()
                    marks.append((t, name))
                print(f'  >> marca {name} em t={t:.1f} s (relógio do ESP32)')
        threading.Thread(target=keyboard, daemon=True).start()
        print(f'Gravando {args.port} -> {args.out}. Digite a fase + Enter para marcar; Ctrl+C para sair.')
        n_sig, t0 = 0, time.time()
        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                text = raw.decode('utf-8', errors='replace').rstrip('\r\n')
                m = T_RE.search(text)
                with lock:
                    if m:
                        last_t[0] = int(m.group(1)) / 1000.0
                        n_sig += 1
                    out.write(text + '\n')
                if 'CSI_STATE' in text or 'CSI_CAL' in text:
                    print(text)
                if m and n_sig % 50 == 0:
                    print(f'  {n_sig} assinaturas, t_ESP32={last_t[0]:.1f} s, {time.time() - t0:.0f} s de captura')
        except KeyboardInterrupt:
            pass
    print(f'\n{n_sig} linhas CSI_SIG salvas em {args.out}')
    if marks:
        print('Fases marcadas (início):')
        for t, name in marks:
            print(f'  {t:8.1f} s  {name}')


if __name__ == '__main__':
    main()
