#!/usr/bin/env python3
"""Reproduz TODO o pipeline do README, na ordem das fases de desenvolvimento:

  0. testes unitários Python
  1. treino + validação do modelo de movimento ........ ml/01_train_motion.py        (fases 5 e 9)
  2. treino + validação do modelo de presença ......... ml/02_train_presence.py      (fases 5 e 9)
  3. refinamento: variantes de features/normalização .. ml/02b_presence_refinement.py (fase 7)
  4. otimização + conversão TFLite INT8 + headers C ... ml/03_export_firmware.py     (fases 6 e 8)
  5. replay do sistema completo de 3 estados .......... ml/04_replay_3state.py       (fase 9)
  6. autoteste da ferramenta de validação do firmware . ml/06_verify_device_log.py --self-test
  7. testes nativos C++ x Python ...................... tests_native/run.sh (se houver g++/clang++)

Uso (na raiz do projeto):  python ml/run_all.py
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='3', TF_ENABLE_ONEDNN_OPTS='0', PYTHONHASHSEED='0')


def run(title, cmd):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}', flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)
    print(f'[ok em {time.time() - t0:.0f} s]', flush=True)


def main():
    py = sys.executable
    run('0/7 testes unitários Python', [py, '-m', 'unittest', 'discover', '-s', 'ml/tests'])
    run('1/7 movimento: validação cruzada + treino final + testes ao vivo', [py, 'ml/01_train_motion.py'])
    run('2/7 presença: validação cruzada + treino final', [py, 'ml/02_train_presence.py'])
    run('3/7 presença: refinamento (variantes)', [py, 'ml/02b_presence_refinement.py'])
    run('4/7 otimização e conversão TFLite INT8 + firmware', [py, 'ml/03_export_firmware.py'])
    run('5/7 replay do sistema de 3 estados', [py, 'ml/04_replay_3state.py'])
    run('6/7 autoteste da validação de logs do firmware', [py, 'ml/06_verify_device_log.py', '--self-test'])
    cxx = shutil.which('g++') or shutil.which('clang++')
    if cxx and shutil.which('bash'):
        run('7/7 testes nativos C++ (paridade com o Python)', ['bash', 'tests_native/run.sh'])
    else:
        print('\n7/7 testes nativos C++: pulado (g++/clang++ ou bash não encontrado)')
    print('\nConcluído. Métricas em results/, figuras em docs/figures/, firmware em src/rx/.')


if __name__ == '__main__':
    main()
