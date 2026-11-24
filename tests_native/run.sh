#!/usr/bin/env bash
# Compila e roda os testes nativos de paridade C++ x Python (requer g++ ou clang++).
# Os vetores são gerados por: python ml/03_export_firmware.py
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
"$CXX" -std=c++17 -O2 -Wall -Wextra -I../src/rx test_parity.cpp -o test_parity
./test_parity vectors
