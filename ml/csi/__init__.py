"""Pipeline de ML para detecção de presença/movimento com Wi-Fi CSI (ESP32).

Módulos:
    paths      caminhos do projeto
    logs       parsers dos logs seriais do RX (CSI_SIG, CSI_FRAME, CSI_PARITY, CSI_STATE)
    labels     manifest de sessões e rotulagem por fase com margens de guarda
    motion     estágio 2: features Core7 (movimento) e conjuntos de treino
    presence   estágio 1: features espectrais relativas à referência EMPTY
    nets       MLPs PyTorch e rotina de treino determinística
    temporal   filtro temporal (média de 5 + histerese) e máquina de 3 estados
    metrics    métricas de classificação e de eventos
    export     PyTorch -> Keras -> TFLite INT8 -> arrays C
"""
