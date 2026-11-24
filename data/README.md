# Dados

Todos os arquivos são **logs seriais brutos do ESP32 RX** (texto, 921600 baud) ou tabelas de
features calculadas **no próprio ESP32**. O conteúdo não foi editado (só os nomes dos arquivos
foram padronizados; o nome original de cada um está em `original_file` no manifest). Os rótulos
ficam separados em [`manifest.json`](manifest.json).

```text
data/
├── manifest.json            rótulos (fases com início/fim em segundos de boot do RX)
├── dataset/                 DADOS DE TREINO — entram no treino e na validação cruzada
│   ├── presence/            sessões A … F com assinaturas espectrais (CSI_SIG)
│   └── motion/              setups M1 … M5 com 12 features por janela
└── tests/                   DADOS DE TESTE — nunca entram no treino
    └── motion_live/         testes ao vivo live_A … live_C do detector de movimento (CSI_PARITY)
```

## Padrão de nomes

| conjunto | nomes | ordem |
|---|---|---|
| sessões de presença | `A` … `F` (`dataset/presence/<id>.log`) | cronológica |
| setups de movimento | `M1` … `M5` (`dataset/motion/<id>/{empty,static,moving}.csv`) | cronológica |
| testes ao vivo | `live_A` … `live_C` (`tests/motion_live/<id>.log`) | cronológica |
| fases | `E0, E1…` vazio · `P1, P2…` pessoa parada em posições diferentes · `P1+P2` duas pessoas · `WALK` caminhada · `ARMS` parado mexendo os braços | ordem no tempo |

## dataset/presence — assinaturas espectrais

Uma linha `CSI_SIG` por janela de 2 s (nova janela a cada 480 ms), gravada com o firmware de
**captura** (hoje: `pio run -e rx_capture`):

```
I (23720) CSI_SIG: v=1 window=45 t_ms=23720 rssi=-48.50 raw_med=24.1200 clip=0 q12=<204 hex>
```

| campo | significado |
|---|---|
| `window` | índice da janela desde o boot |
| `t_ms` | tempo do ESP32 desde o boot (relógio dos rótulos) |
| `rssi` | RSSI médio da janela (dBm) |
| `raw_med` | mediana bruta da amplitude LLTF antes da normalização (diagnóstico de ganho) |
| `clip` | subportadoras que saturaram o formato Q12 |
| `q12` | 51 amplitudes médias (uma por subportadora LLTF útil), `uint16` Q12 em 4 dígitos hex |

Os mesmos logs trazem `CSI_FRAME`: a saída do detector de movimento que estava gravado na placa
durante a captura (usado no replay do sistema completo).

| sessão | arquivo original | data | duração | conteúdo |
|---|---|---|---:|---|
| A | `csi_signature_A.txt` | 18/11 | 400 s | P1, P2 parado + caminhada |
| B | `csi_signature_B.txt` | 18/11 | 429 s | idem, TX deslocado; E3 UNKNOWN (possível 2ª pessoa no cômodo vizinho) |
| C | `csi_signature_C.txt` | 18/11 | 330 s | P1 dentro, P2 na porta, P3 **atrás da parede**, caminhada |
| D | `csi_signature_D2.txt` | 21/11 | 692 s | P1, P2, P3, **P1+P2 (duas pessoas)**, P2 repetida |
| E | `csi_signature_D0020_pt1.txt` | 21/11 | 591 s | P1, P2, P3, P1+P2 |
| F | `csi_signature_D0020_pt2.log` | 21/11 | 440 s | P1, P2 (UNKNOWN: gato junto), P3 |

A, B e C foram gravadas originalmente como **testes** da assinatura espectral (etapa 0017). Como
têm protocolo e horários registrados, passaram a fazer parte do dataset.

## dataset/motion — 12 features por janela

Gravações separadas por condição (`empty.csv`, `static.csv`, `moving.csv`), feitas com o firmware
que registrava as 12 features calculadas na placa (hoje: linhas `CSI_F12` do `rx_capture`).

| setup | pasta original | campanha | janelas por condição | observação |
|---|---|---|---:|---|
| M1 | `robustness/baseline` | robustness | 79 | referência |
| M2 | `robustness/rx_restart` | robustness | 79 | RX reiniciado |
| M3 | `robustness/tx_restart` | robustness | 79 | TX reiniciado |
| M4 | `independent_validation/reference` | independent | 152 | nova sessão |
| M5 | `independent_validation/changed` | independent | 152 | geometria alterada |

Colunas `f0..f11` = saída de `src/rx/csi_features.cpp` (as colunas de texto internas mantêm os
nomes originais, ex. `movement`). As 20 primeiras janelas EMPTY de cada setup são a calibração e
não entram em treino/avaliação.

## tests/motion_live — teste ao vivo

| teste | arquivo original | data | protocolo |
|---|---|---|---|
| live_A | `core7_shadow_0010.txt` | 13/11 | E0 22–75 s · P1 75–110 · WALK 110–145 · ARMS 145–180 · E1 180–387 |
| live_B | `core7_test_A.txt` | 15/11 | E0 22–80 s · P1 80–120 · WALK 120–165 · ARMS 165–205 · E1 205–300 |
| live_C | `core7_test_B.txt` | 15/11 | idem live_B, outra geometria |

Logs do firmware de **validação** da época (`CSI_PARITY`): as 7 features Core7 de cada janela, a
referência EMPTY calculada na placa e a entrada/saída INT8 do TFLite Micro. Servem para medir o
modelo no domínio real da placa e para comparar TFLite (PC) × TFLite Micro (ESP32). Hoje o
equivalente é o `rx_validation` (linhas `CSI_VEC`, `CSI_F12`, `CSI_REF`).

## Rótulos

Os tempos de fase vêm do protocolo de cada captura (cronômetro humano alinhado ao boot do RX). As
métricas ignoram 7 s no início (5 s de incerteza + 2 s da janela CSI) e 5 s no fim de cada fase.
`UNKNOWN` nunca entra em treino nem em métrica.

## Capturas descartadas

| captura | motivo |
|---|---|
| `teste_E.log` (22/11) | trecho vazio final sem rótulo confiável |
| `csi_signature_D.txt` (19/11) | sem horários de fase registrados |
| `changed_obstructed` | sem protocolo registrado; o manifesto aponta para os arquivos de `changed` |
| CSV bruto inicial (`CSI_DATA`) | HT40 a 100 Hz, formato diferente e perda de pacotes |

## Novas capturas

1. Grave o RX com `pio run -e rx_capture -t upload`.
2. `python tools/capture_serial.py --port COM6 --out data/dataset/presence/G.log` (próxima letra),
   digitando o nome de cada fase + Enter no momento em que ela começa.
3. Acrescente a sessão em `manifest.json` (`role: "dataset"` para treinar, `"holdout"` para teste final).
4. `python ml/run_all.py`.
