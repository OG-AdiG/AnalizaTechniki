"""
export_model.py — Konwersja modelu PyTorch → ONNX → TensorFlow Lite.

Użycie:
    python model/export_model.py
    python model/export_model.py --exercise squat --quantize
    python model/export_model.py --test

Pipeline: PyTorch (.pt) → ONNX (.onnx) → TFLite (.tflite)
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


def export_to_onnx(model: TemporalCNN, output_path: str,
                   input_channels: int = INPUT_CHANNELS,
                   sequence_length: int = SEQUENCE_LENGTH):
    """Eksportuje model PyTorch do formatu ONNX."""
    dummy_input = torch.randn(1, input_channels, sequence_length)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"✅ ONNX zapisany: {output_path}")


def convert_onnx_to_tflite(onnx_path: str, tflite_path: str, quantize: bool = False):
    """
    Konwertuje ONNX → TF SavedModel → TFLite.
    """
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    tf_rep = prepare(onnx_model)
    savedmodel_dir = onnx_path.replace(".onnx", "_savedmodel")
    tf_rep.export_graph(savedmodel_dir)
    print(f"✅ TF SavedModel: {savedmodel_dir}")

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)

    if quantize:
        print("🔧 Kwantyzacja INT8...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]

        def representative_dataset():
            for _ in range(100):
                data = np.random.randn(1, INPUT_CHANNELS, SEQUENCE_LENGTH).astype(np.float32)
                yield [data]

        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"✅ TFLite zapisany: {tflite_path} ({file_size_mb:.2f} MB)")


def verify_tflite(tflite_path: str, pytorch_model: TemporalCNN):
    """Weryfikuje, że TFLite daje podobne wyniki co PyTorch."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_input = np.random.randn(1, INPUT_CHANNELS, SEQUENCE_LENGTH).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]["index"])

    pytorch_model.eval()
    with torch.no_grad():
        pytorch_input = torch.FloatTensor(test_input)
        pytorch_output = pytorch_model(pytorch_input).numpy()

    max_diff = np.max(np.abs(tflite_output - pytorch_output))
    print(f"\n🔍 Weryfikacja TFLite vs PyTorch:")
    print(f"   Max różnica: {max_diff:.6f}")

    if max_diff < 1e-3:
        print("   ✅ Wyniki zgodne!")
    elif max_diff < 1e-1:
        print("   ⚠️  Niewielka różnica (akceptowalna)")
    else:
        print("   ❌ Duża różnica — sprawdź konwersję!")


def main():
    parser = argparse.ArgumentParser(description="Eksport modelu → ONNX → TFLite")
    parser.add_argument("--exercise", type=str, default=ACTIVE_EXERCISE,
                        choices=list(EXERCISE_CLASSES.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(MODELS_DIR, f"{args.exercise}_best.pt")

    base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    onnx_path = os.path.join(MODELS_DIR, f"{base_name}.onnx")
    tflite_path = os.path.join(MODELS_DIR, f"{base_name}.tflite")

    model, ckpt = load_trained_model(args.checkpoint)

    input_ch = ckpt.get("input_channels", INPUT_CHANNELS)
    seq_len = ckpt.get("sequence_length", SEQUENCE_LENGTH)

    export_to_onnx(model, onnx_path, input_ch, seq_len)
    convert_onnx_to_tflite(onnx_path, tflite_path, quantize=args.quantize)

    if args.test:
        verify_tflite(tflite_path, model)

    print(f"\n🎉 Eksport zakończony!")
    print(f"   ONNX:   {onnx_path}")
    print(f"   TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
