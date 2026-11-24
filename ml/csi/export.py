"""PyTorch -> Keras -> TensorFlow Lite INT8 -> arrays C para TFLite Micro.

Por que via Keras: a MLP é pequena (Dense+ReLU), então copiar os pesos para um
tf.keras.Sequential equivalente é exato e evita a cadeia ONNX. O conversor TFLite
faz quantização INT8 completa (pesos, ativações, entrada e saída) usando um
conjunto representativo (as próprias entradas padronizadas de treino).
"""
from __future__ import annotations

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')     # silencia logs do TensorFlow
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')    # resultados determinísticos em CPU

import hashlib
import json
from pathlib import Path

import numpy as np
import torch.nn as nn

from .nets import predict_proba


def torch_to_keras(net: nn.Sequential, in_dim: int):
    import tensorflow as tf
    linears = [m for m in net if isinstance(m, nn.Linear)]
    layers = [tf.keras.Input(shape=(in_dim,), dtype=tf.float32)]
    for i, lin in enumerate(linears):
        last = i == len(linears) - 1
        layers.append(tf.keras.layers.Dense(lin.out_features,
                                            activation='sigmoid' if last else 'relu',
                                            name=f'dense_{i}'))
    keras = tf.keras.Sequential(layers)
    for i, lin in enumerate(linears):
        keras.get_layer(f'dense_{i}').set_weights([
            lin.weight.detach().numpy().T.astype(np.float32),
            lin.bias.detach().numpy().astype(np.float32)])
    return keras


def convert_float32(keras) -> bytes:
    """Conversão sem quantização (só para comparar tamanho/precisão na fase de otimização)."""
    import tensorflow as tf
    return tf.lite.TFLiteConverter.from_keras_model(keras).convert()


def convert_int8(keras, representative: np.ndarray) -> bytes:
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(keras)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    rep = np.ascontiguousarray(representative, dtype=np.float32)

    def gen():
        for row in rep:
            yield [row.reshape(1, -1)]
    conv.representative_dataset = gen
    return conv.convert()


class Int8Runner:
    """Executa o .tflite exatamente como o firmware: quantiza a entrada com scale/zero_point
    do tensor, arredonda para o inteiro mais próximo, satura em [-128,127]."""

    def __init__(self, blob: bytes):
        import tensorflow as tf
        # Kernels builtin sem delegates (XNNPACK): o mais próximo do TFLite Micro.
        self.interp = tf.lite.Interpreter(
            model_content=blob, num_threads=1,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale, self.in_zero = self.inp['quantization']
        self.out_scale, self.out_zero = self.out['quantization']

    def quantize(self, z: np.ndarray) -> np.ndarray:
        q = np.asarray(z, np.float32) / np.float32(self.in_scale) + np.float32(self.in_zero)
        return np.clip(np.rint(q), -128, 127).astype(np.int8)

    def invoke_q(self, q: np.ndarray) -> int:
        """Executa com uma entrada já quantizada (int8) e devolve a saída int8 bruta."""
        self.interp.set_tensor(self.inp['index'], np.asarray(q, np.int8).reshape(1, -1))
        self.interp.invoke()
        return int(self.interp.get_tensor(self.out['index'])[0, 0])

    def __call__(self, z: np.ndarray) -> np.ndarray:
        out = np.empty(len(z), dtype=np.float32)
        for i, row in enumerate(self.quantize(z)):
            self.interp.set_tensor(self.inp['index'], row.reshape(1, -1))
            self.interp.invoke()
            q = int(self.interp.get_tensor(self.out['index'])[0, 0])
            out[i] = (q - self.out_zero) * self.out_scale
        return np.clip(out, 0.0, 1.0)

    def ops(self) -> list[str]:
        return sorted({d['op_name'] for d in self.interp._get_ops_details()})

    def quant_params(self) -> dict:
        return dict(in_scale=float(self.in_scale), in_zero=int(self.in_zero),
                    out_scale=float(self.out_scale), out_zero=int(self.out_zero))


def export_model(net: nn.Sequential, z_train: np.ndarray, out_dir: Path, name: str) -> dict:
    """Gera <name>.tflite e métricas de paridade float(PyTorch) vs float(Keras) vs INT8."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keras = torch_to_keras(net, z_train.shape[1])
    p_torch = predict_proba(net, z_train)
    p_keras = keras(z_train.astype(np.float32), training=False).numpy().ravel()
    blob = convert_int8(keras, z_train)
    (out_dir / f'{name}.tflite').write_bytes(blob)
    blob_f32 = convert_float32(keras)
    runner = Int8Runner(blob)
    p_int8 = runner(z_train)
    report = dict(
        model=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
        float32_tflite_bytes=len(blob_f32),
        parameters=int(sum(p.numel() for p in net.parameters())),
        ops=runner.ops(), quantization=runner.quant_params(),
        input_shape=[int(v) for v in runner.inp['shape']], output_shape=[int(v) for v in runner.out['shape']],
        float_torch_vs_keras_max_abs=float(np.max(np.abs(p_torch - p_keras))),
        int8_vs_float_max_abs=float(np.max(np.abs(p_int8 - p_torch))),
        int8_vs_float_mean_abs=float(np.mean(np.abs(p_int8 - p_torch))),
        int8_vs_float_decision_agreement=float(np.mean((p_int8 >= 0.5) == (p_torch >= 0.5))),
        input_clip_fraction=float(np.mean(np.abs(runner.quantize(z_train).astype(int)) >= 127)),
        n_representative=int(len(z_train)))
    (out_dir / f'{name}_export.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


# ---------------------------------------------------------------- C code generation
def cf(v: float) -> str:
    """Literal float C++ com 9 dígitos significativos (ida e volta exata em float32)."""
    text = f'{float(np.float32(v)):.9g}'
    if not any(c in text for c in '.en'):
        text += '.0'
    return text + 'f'


def c_float_array(name: str, values) -> str:
    vals = ', '.join(cf(v) for v in np.ravel(values))
    return f'static constexpr float {name}[{len(np.ravel(values))}] = {{{vals}}};'


def write_model_source(blob: bytes, symbol: str, header: Path, source: Path, comment: str) -> None:
    guard = symbol.upper() + '_H'
    header.write_text(
        f'// AUTO-GERADO por ml/03_export_firmware.py — não edite à mão.\n'
        f'// {comment}\n#pragma once\n#include <cstdint>\n\n'
        f'extern const unsigned char {symbol}[];\nextern const unsigned int {symbol}_len;\n',
        encoding='utf-8')
    lines = []
    for i in range(0, len(blob), 12):
        lines.append('    ' + ', '.join(f'0x{b:02x}' for b in blob[i:i + 12]) + ',')
    source.write_text(
        f'// AUTO-GERADO por ml/03_export_firmware.py — não edite à mão.\n'
        f'// {comment}\n// sha256={hashlib.sha256(blob).hexdigest()}\n'
        f'#include "{header.name}"\n\n'
        f'alignas(16) const unsigned char {symbol}[] = {{\n' + '\n'.join(lines) + '\n};\n'
        f'const unsigned int {symbol}_len = {len(blob)};\n', encoding='utf-8')
