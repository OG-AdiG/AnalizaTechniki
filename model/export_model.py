"""
=======================================================
  Eksport modelu techniki (TCN) PyTorch → TFLite (LiteRT)
  Google AI Edge Torch — BEZPOŚREDNIO, bez ONNX
=======================================================

⚡ URUCHAMIAĆ NA GOOGLE COLAB (Linux) — nie na Windowsie!
  litert-torch wymaga Linux. Colab daje Ci go za darmo.

Instrukcja Colab:
  1. Wrzuć na Colab: model.py, config.py, export_model.py
  2. Wrzuć checkpoint: saved_models/pushup_best.pt
  3. Uruchom w komórce:
       !pip install litert-torch
       !python -m model.export_model --exercise pushup --fp16 --test
  4. Pobierz: saved_models/pushup_best_fp16.tflite

Pipeline: PyTorch (.pt) → litert_torch.convert() → TFLite (.tflite)
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import TemporalCNN
from model.config import (
    MODELS_DIR, ACTIVE_EXERCISE, EXERCISE_CLASSES,
    INPUT_CHANNELS, SEQUENCE_LENGTH,
)


def load_trained_model(checkpoint_path: str) -> tuple:
    """
    Ładuje wytrenowany model z checkpointu.

    Returns:
        (model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes = checkpoint.get("num_classes", 5)
    input_channels = checkpoint.get("input_channels", INPUT_CHANNELS)

    model = TemporalCNN(
        input_channels=input_channels,
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Załadowano model z: {checkpoint_path}")
    print(f"   Ćwiczenie: {checkpoint.get('exercise', 'unknown')}")
    print(f"   Epoka: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.1%}")
    print(f"   Input: ({input_channels}, {checkpoint.get('sequence_length', SEQUENCE_LENGTH)})")
    print(f"   Klasy: {num_classes}")

    return model, checkpoint


def convert_to_tflite(model: TemporalCNN, tflite_path: str,
                      input_channels: int = INPUT_CHANNELS,
                      sequence_length: int = SEQUENCE_LENGTH,
                      fp16: bool = False):
    """
    Eksport PyTorch → TFLite przy użyciu litert-torch.

    Pipeline:
      PyTorch model (eval)
      → litert_torch.convert()    (torch.export → TFLite)
      → float16 kwantyzacja       (opcjonalnie, 2x mniejszy)
      → .export()                 (zapis .tflite)
    """
    import litert_torch
    import tensorflow as tf

    precision = "float16" if fp16 else "float32"
    print(f"🔄 Konwersja PyTorch → TFLite (litert-torch, {precision})...")

    # ===== 1. WALIDACJA PYTORCH =====
    sample_input = torch.randn(1, input_channels, sequence_length)
    with torch.no_grad():
        pytorch_output = model(sample_input)
    print(f"   PyTorch output shape: {pytorch_output.shape}")

    # ===== 2. KONWERSJA → TFLITE =====
    # TCN nie ma konwolucji 2D (brak NCHW/NHWC),
    # więc NIE potrzebujemy to_channel_last_io().
    # Input to (1, channels, sequence) — Conv1d — bez transpozycji.

    convert_kwargs = {}

    if fp16:
        # Dokładnie ten sam sposób co w export_tflite.py (pose estimation)
        # który działa idealnie na telefonie:
        #   - Wagi przechowywane jako float16 (2x mniejszy plik)
        #   - GPU na telefonie liczy natywnie w FP16
        #   - Zerowa utrata precyzji dla klasyfikacji
        tfl_converter_flags = {
            'optimizations': [tf.lite.Optimize.DEFAULT],
            'target_spec': {
                'supported_types': [tf.float16]
            }
        }
        convert_kwargs['_ai_edge_converter_flags'] = tfl_converter_flags
        print(f"   Kwantyzacja: float16 (via _ai_edge_converter_flags)")

    edge_model = litert_torch.convert(
        model.eval(),
        (sample_input,),
        **convert_kwargs,
    )

    # ===== 3. SMOKE TEST =====
    edge_output = edge_model(sample_input)
    if isinstance(edge_output, (tuple, list)):
        edge_np = edge_output[0] if isinstance(edge_output[0], np.ndarray) else edge_output[0].detach().numpy()
    else:
        edge_np = edge_output if isinstance(edge_output, np.ndarray) else edge_output.detach().numpy()

    ref_np = pytorch_output.detach().numpy()
    max_diff = np.max(np.abs(ref_np - edge_np))
    status = "✅" if max_diff < 0.05 else "⚠️"
    print(f"   Smoke test: max_diff={max_diff:.6f} {status}")

    # ===== 4. ZAPIS =====
    edge_model.export(tflite_path)

    file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  SUKCES! Model TFLite gotowy: {tflite_path}")
    print(f"  Rozmiar:      {file_size_mb:.2f} MB ({precision})")
    print(f"  Wejście:      [1, {input_channels}, {sequence_length}]")
    print(f"  Wyjście:      [1, num_classes]")
    print(f"  Kwantyzacja:  {precision}")
    print(f"{'='*60}")


def verify_tflite(tflite_path: str, pytorch_model: TemporalCNN):
    """Szczegółowa weryfikacja modelu TFLite + benchmark."""
    import tensorflow as tf
    import time

    print(f"\n--- Szczegółowa weryfikacja TFLite ---")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    print(f"Input:")
    for inp in input_details:
        print(f"  {inp['name']}: shape={inp['shape']} dtype={inp['dtype']}")
    print(f"Outputs:")
    for out in output_details:
        print(f"  {out['name']}: shape={out['shape']} dtype={out['dtype']}")

    # Porównanie numeryczne
    test_input = np.random.randn(1, INPUT_CHANNELS, SEQUENCE_LENGTH).astype(np.float32)
    expected_shape = input_details[0]["shape"]
    tflite_input = test_input

    if expected_shape[1] == SEQUENCE_LENGTH and expected_shape[2] == INPUT_CHANNELS:
        print(f"  ℹ️ Transponowanie wejścia: (1, {INPUT_CHANNELS}, {SEQUENCE_LENGTH}) → (1, {SEQUENCE_LENGTH}, {INPUT_CHANNELS})")
        tflite_input = np.transpose(test_input, (0, 2, 1))

    if input_dtype == np.float16:
        tflite_input = tflite_input.astype(np.float16)

    interpreter.set_tensor(input_details[0]["index"], tflite_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]["index"]).astype(np.float32)

    pytorch_model.eval()
    with torch.no_grad():
        pytorch_input = torch.FloatTensor(test_input)
        pytorch_output = pytorch_model(pytorch_input).numpy()

    max_diff = np.max(np.abs(tflite_output - pytorch_output))
    print(f"\n  Max różnica TFLite vs PyTorch: {max_diff:.6f}")

    if max_diff < 1e-2:
        print("  ✅ Wyniki zgodne!")
    elif max_diff < 1e-1:
        print("  ⚠️  Niewielka różnica (akceptowalna dla FP16)")
    else:
        print("  ❌ Duża różnica — sprawdź konwersję!")

    # Benchmark
    inp_shape = input_details[0]['shape']
    bench_input = np.random.randn(*inp_shape).astype(np.float32)
    if input_dtype == np.float16:
        bench_input = bench_input.astype(np.float16)
    interpreter.set_tensor(input_details[0]['index'], bench_input)

    for _ in range(5):
        interpreter.invoke()

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"\n  Benchmark (CPU, 50 runs):")
    print(f"    Mean: {avg:.1f}ms | P50: {p50:.1f}ms | P95: {p95:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Eksport modelu PyTorch → TFLite (LiteRT)")
    parser.add_argument("--exercise", type=str, default=ACTIVE_EXERCISE,
                        choices=list(EXERCISE_CLASSES.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fp16", action="store_true",
                        help="Eksportuj w float16 (~50%% mniejszy model)")
    parser.add_argument("--test", action="store_true",
                        help="Weryfikuj wyniki TFLite vs PyTorch + benchmark")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(MODELS_DIR, f"{args.exercise}_best.pt")

    base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    suffix = "_fp16" if args.fp16 else ""
    tflite_path = os.path.join(MODELS_DIR, f"{base_name}{suffix}.tflite")

    model, ckpt = load_trained_model(args.checkpoint)

    input_ch = ckpt.get("input_channels", INPUT_CHANNELS)
    seq_len = ckpt.get("sequence_length", SEQUENCE_LENGTH)

    convert_to_tflite(model, tflite_path, input_ch, seq_len, fp16=args.fp16)

    if args.test:
        verify_tflite(tflite_path, model)

    print(f"\n🎉 Eksport zakończony!")
    print(f"   TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
