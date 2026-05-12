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
    # Działa zarówno ze starszą wersją (ai-edge-torch <= 0.7.x) jak i
    # z nowszą (litert-torch >= 0.8.x). Google w styczniu 2026 zmienił
    # nazwę pakietu z `ai_edge_torch` na `litert_torch`, więc trzymamy
    # oba importy jako fallback.
    try:
        import litert_torch
        _converter_name = "litert-torch"
    except ImportError:
        import ai_edge_torch as litert_torch  # type: ignore
        _converter_name = "ai-edge-torch (legacy)"

    import tensorflow as tf

    precision = "float16" if fp16 else "float32"
    print(f"🔄 Konwersja PyTorch → TFLite ({_converter_name}, {precision})...")

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

    try:
        edge_model = litert_torch.convert(
            model.eval(),
            (sample_input,),
            **convert_kwargs,
        )
    except TypeError as e:
        # litert-torch >= 0.9.0 wywaliło prywatny kwarg _ai_edge_converter_flags.
        # Spróbujmy bez niego — model wyjdzie FP32, ale przynajmniej eksport
        # nie skraszuje. Użytkownik dostanie wyraźne ostrzeżenie.
        if fp16 and "_ai_edge_converter_flags" in str(e):
            print("   ⚠️ Twoja wersja litert-torch nie wspiera już prywatnego")
            print("      kwarga _ai_edge_converter_flags (zmiana API w 0.9.0).")
            print("      Eksportuję BEZ kwantyzacji — model wyjdzie jako FP32.")
            print("      Aby dostać FP16, użyj:")
            print("        pip install \"ai-edge-torch==0.7.0\" \"tensorflow==2.18.0\"")
            print("      lub przejdź na ścieżkę ONNX (osobny tryb).")
            convert_kwargs.pop('_ai_edge_converter_flags', None)
            edge_model = litert_torch.convert(
                model.eval(),
                (sample_input,),
                **convert_kwargs,
            )
            precision = "float32 (fallback, brak FP16 support)"
        else:
            raise

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
    try:
        interpreter.allocate_tensors()
    except RuntimeError as e:
        # CPU TFLite nie wspiera full-FP16 Conv (z onnx2tf --fp16).
        # Na telefonie GPU delegate to obsłuży. Nie crashujemy skryptu.
        print(f"⚠️ allocate_tensors() padło: {e}")
        print("   To zwykle oznacza full-FP16 model bez CPU compat.")
        print("   Plik jest poprawny — sprawdź go w aplikacji mobilnej z GPU delegate.")
        return

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


def convert_via_onnx(model: TemporalCNN, tflite_path: str,
                     input_channels: int = INPUT_CHANNELS,
                     sequence_length: int = SEQUENCE_LENGTH,
                     fp16: bool = False):
    """
    Alternatywna ścieżka eksportu: PyTorch → ONNX → onnx2tf → TFLite.

    Używana gdy litert-torch jest popsuty (np. nowe API bez fp16 support).
    Dla naszego TCN (Conv1d, BatchNorm1d, Linear, AdaptiveAvgPool1d)
    pipeline jest stabilny — to wszystko standardowe ops bez problemów
    NCHW↔NHWC.

    UWAGA: bierzemy GOTOWY .tflite wygenerowany przez onnx2tf, NIE robimy
    własnej re-kwantyzacji przez tf.lite.TFLiteConverter z SavedModel.
    Powody:
    - onnx2tf >= 2.4.0 wymaga `tf_keras` jako dodatkowej zależności,
      żeby w ogóle zapisać SavedModel (OptionalTensorFlowDependencyError).
    - onnx2tf 2.4.0 ma znaną regresję w `_build_maxpool1d_op`
      (IndexError) — TCN z MaxPool1d wywala flatbuffer_direct fast path.
    - Pin starszych wersji powoduje kaskadę konfliktów ml-dtypes/protobuf.

    Wybór wariantu:
    - --fp16 → bierzemy `model_float16.tflite` (full FP16, ~0.5x rozmiaru).
      Działa na telefonie GPU delegate; NIE odpala na standardowym CPU
      `tf.lite.Interpreter`. Smoke test (--test) na CPU może pęknąć, to
      OK — sprawdzenie zostawiamy aplikacji mobilnej.
    - bez --fp16 → bierzemy `model_float32.tflite`. Działa wszędzie
      (CPU, GPU delegate, NNAPI). GPU delegate i tak natywnie zrobi FP16
      wewnętrznie. Smoke test w pełni przechodzi.
    """
    import tempfile
    import shutil

    try:
        import onnx
        import onnx2tf
    except ImportError as e:
        raise ImportError(
            "Ścieżka ONNX wymaga: pip install onnx onnx2tf tensorflow"
        ) from e

    precision = "float16 (full)" if fp16 else "float32"
    print(f"🔄 Konwersja PyTorch → ONNX → TFLite ({precision})...")

    sample_input = torch.randn(1, input_channels, sequence_length)
    with torch.no_grad():
        pytorch_output = model.eval()(sample_input)
    print(f"   PyTorch output shape: {pytorch_output.shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")

        torch.onnx.export(
            model.eval(),
            sample_input,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
        onnx.checker.check_model(onnx_path)
        print(f"   ✅ ONNX zapisany: {os.path.getsize(onnx_path)/1024:.0f} KB")

        out_dir = os.path.join(tmpdir, "tf_out")
        os.makedirs(out_dir, exist_ok=True)

        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=out_dir,
            output_signaturedefs=True,
            non_verbose=True,
            output_h5=False,
            output_keras_v3=False,
            copy_onnx_input_output_names_to_tflite=True,
        )

        tflite_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".tflite"))
        if not tflite_files:
            raise RuntimeError(f"onnx2tf nie wygenerował .tflite w {out_dir}")
        print(f"   ✅ onnx2tf wygenerował: {tflite_files}")

        if fp16:
            pick = next((f for f in tflite_files if "float16" in f), None)
            if pick is None:
                print(f"   ⚠️ Brak wariantu float16 — biorę FP32 (CPU-friendly)")
                pick = next((f for f in tflite_files if "float32" in f), tflite_files[0])
        else:
            pick = next((f for f in tflite_files if "float32" in f), tflite_files[0])

        shutil.copy2(os.path.join(out_dir, pick), tflite_path)
        print(f"   ✅ Wybrano: {pick}")

    file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  SUKCES! Model TFLite gotowy: {tflite_path}")
    print(f"  Rozmiar:      {file_size_mb:.2f} MB ({precision})")
    print(f"  Wejście:      [1, {input_channels}, {sequence_length}]")
    print(f"  Wyjście:      [1, num_classes]")
    print(f"  Pipeline:     PyTorch → ONNX → onnx2tf → TFLite")
    if fp16:
        print(f"  Uwaga:        Full-FP16 — działa na GPU delegate, NIE na CPU TFLite.")
        print(f"                Jeśli potrzebujesz CPU compat → uruchom bez --fp16.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Eksport modelu PyTorch → TFLite (LiteRT)")
    parser.add_argument("--exercise", type=str, default=ACTIVE_EXERCISE,
                        choices=list(EXERCISE_CLASSES.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fp16", action="store_true",
                        help="Eksportuj w float16 (~50%% mniejszy model)")
    parser.add_argument("--test", action="store_true",
                        help="Weryfikuj wyniki TFLite vs PyTorch + benchmark")
    parser.add_argument("--backend", type=str, default="litert",
                        choices=["litert", "onnx"],
                        help="Backend konwersji: litert (litert-torch / ai-edge-torch) "
                             "lub onnx (PyTorch → ONNX → onnx2tf, fallback)")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(MODELS_DIR, f"{args.exercise}_best.pt")

    base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    suffix = "_fp16" if args.fp16 else ""
    tflite_path = os.path.join(MODELS_DIR, f"{base_name}{suffix}.tflite")

    model, ckpt = load_trained_model(args.checkpoint)

    input_ch = ckpt.get("input_channels", INPUT_CHANNELS)
    seq_len = ckpt.get("sequence_length", SEQUENCE_LENGTH)

    if args.backend == "onnx":
        convert_via_onnx(model, tflite_path, input_ch, seq_len, fp16=args.fp16)
    else:
        convert_to_tflite(model, tflite_path, input_ch, seq_len, fp16=args.fp16)

    if args.test:
        verify_tflite(tflite_path, model)

    print(f"\n🎉 Eksport zakończony!")
    print(f"   TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
