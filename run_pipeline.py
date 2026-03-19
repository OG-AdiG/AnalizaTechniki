"""
run_pipeline.py — Skrypt uruchamiający cały pipeline od A do Z.

Użycie:
    python run_pipeline.py --keypoints_dir ścieżka/do/keypointów
    python run_pipeline.py --keypoints_dir ścieżka --exercise squat
    python run_pipeline.py --step 2 --exercise pushup

Pipeline:
1. Normalizacja keypointów (z modelu pose estimation kolegi)
2. Trening modelu TCN
3. Eksport do TFLite
4. Weryfikacja wyników

Struktura oczekiwanych folderów:
    keypoints_dir/
    ├── correct/           ← pliki .npy z poprawnymi wykonaniami
    ├── knee_valgus/       ← pliki .npy z błędem knee_valgus (multi-class)
    ├── heel_lift/         ← itd.
    └── error/             ← alternatywnie: binary (correct/error)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.config import (
    KEYPOINTS_DIR, MODELS_DIR, ACTIVE_EXERCISE, EXERCISE_CLASSES, DATA_DIR,
)


def setup_directories(exercise: str):
    """Tworzy strukturę katalogów projektu."""
    labels = EXERCISE_CLASSES[exercise]["labels"]

    dirs_to_create = [
        DATA_DIR,
        MODELS_DIR,
    ]

    # Twórz podkatalogi per klasa błędu
    for label_name in labels.values():
        dirs_to_create.append(
            os.path.join(KEYPOINTS_DIR, exercise, label_name)
        )

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"  📁 {d}")


def step1_normalize_keypoints(keypoints_dir: str, exercise: str):
    """Krok 1: Normalizacja keypointów."""
    from data_pipeline.extract_keypoints import process_directory

    labels = EXERCISE_CLASSES[exercise]["labels"]

    print("\n" + "=" * 60)
    print("KROK 1: Normalizacja keypointów")
    print("=" * 60)

    for label_id, label_name in labels.items():
        input_dir = os.path.join(keypoints_dir, label_name)
        output_dir = os.path.join(KEYPOINTS_DIR, exercise, label_name)

        if os.path.exists(input_dir):
            print(f"\n📂 Przetwarzanie: {label_name}")
            process_directory(input_dir, output_dir)
        else:
            print(f"  ⚠ Brak katalogu: {input_dir}")

    # Fallback: binary format (correct/error)
    for fallback_name in ["correct", "error"]:
        input_dir = os.path.join(keypoints_dir, fallback_name)
        if os.path.exists(input_dir):
            output_dir = os.path.join(KEYPOINTS_DIR, exercise, fallback_name)
            print(f"\n📂 Przetwarzanie (fallback): {fallback_name}")
            process_directory(input_dir, output_dir)


def step2_train(exercise: str):
    """Krok 2: Trening modelu TCN."""
    from model.train import train

    print("\n" + "=" * 60)
    print(f"KROK 2: Trening modelu TCN ({exercise})")
    print("=" * 60)

    train(exercise=exercise)


def step3_export(exercise: str):
    """Krok 3: Eksport do TFLite."""
    from model.export_model import (
        load_trained_model, export_to_onnx,
        convert_onnx_to_tflite, verify_tflite,
    )

    print("\n" + "=" * 60)
    print(f"KROK 3: Eksport modelu → ONNX → TFLite ({exercise})")
    print("=" * 60)

    checkpoint_path = os.path.join(MODELS_DIR, f"{exercise}_best.pt")
    onnx_path = os.path.join(MODELS_DIR, f"{exercise}_best.onnx")
    tflite_path = os.path.join(MODELS_DIR, f"{exercise}_best.tflite")

    model, ckpt = load_trained_model(checkpoint_path)
    input_ch = ckpt.get("input_channels", 63)
    seq_len = ckpt.get("sequence_length", 30)

    export_to_onnx(model, onnx_path, input_ch, seq_len)
    convert_onnx_to_tflite(onnx_path, tflite_path)
    verify_tflite(tflite_path, model)


def main():
    parser = argparse.ArgumentParser(
        description="SEE Trainer — Pipeline trenowania modelu analizy techniki"
    )
    parser.add_argument(
        "--keypoints_dir", type=str, default=None,
        help="Ścieżka do katalogu z keypointami (correct/ + error_types/ w środku)"
    )
    parser.add_argument(
        "--exercise", type=str, default=ACTIVE_EXERCISE,
        choices=list(EXERCISE_CLASSES.keys()),
        help="Ćwiczenie do trenowania"
    )
    parser.add_argument(
        "--step", type=int, default=0, choices=[0, 1, 2, 3],
        help="Uruchom konkretny krok (0=wszystkie)"
    )
    parser.add_argument(
        "--setup-only", action="store_true",
        help="Tylko utwórz strukturę folderów"
    )
    args = parser.parse_args()

    print("🏋️ SEE Trainer — Pipeline trenowania modelu analizy techniki")
    print(f"   Ćwiczenie: {args.exercise}")
    print(f"   Klasy: {list(EXERCISE_CLASSES[args.exercise]['labels'].values())}")

    print("\n📁 Tworzenie struktury katalogów:")
    setup_directories(args.exercise)

    if args.setup_only:
        print("\n✅ Struktura utworzona. Umieść pliki keypointów w odpowiednich folderach.")
        return

    if args.step == 0 or args.step == 1:
        if args.keypoints_dir:
            step1_normalize_keypoints(args.keypoints_dir, args.exercise)
        else:
            print("\n⚠ Pomiń krok 1 (brak --keypoints_dir)")

    if args.step == 0 or args.step == 2:
        step2_train(args.exercise)

    if args.step == 0 or args.step == 3:
        step3_export(args.exercise)

    print("\n" + "=" * 60)
    print("🎉 Pipeline zakończony!")
    print("=" * 60)
    print(f"\nPliki wynikowe:")
    print(f"  📊 Model PyTorch: {os.path.join(MODELS_DIR, f'{args.exercise}_best.pt')}")
    print(f"  📊 Model ONNX:    {os.path.join(MODELS_DIR, f'{args.exercise}_best.onnx')}")
    print(f"  📱 Model TFLite:  {os.path.join(MODELS_DIR, f'{args.exercise}_best.tflite')}")


if __name__ == "__main__":
    main()
