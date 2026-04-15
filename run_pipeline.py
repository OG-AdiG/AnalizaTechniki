"""
run_pipeline.py — Skrypt uruchamiający cały pipeline od A do Z.

Użycie:
    # Pełny pipeline: wideo → keypoints → normalizacja → trening → eksport
    python run_pipeline.py --video_dir D:\\Pushups --exercise pushup

    # Tylko ekstrakcja z wideo
    python run_pipeline.py --video_dir D:\\Pushups --exercise pushup --step 0

    # Tylko trening (jeśli keypoints już wyekstrahowane)
    python run_pipeline.py --exercise pushup --step 2

    # Pełny pipeline z wideo
    python run_pipeline.py --video_dir D:\\Pushups --exercise pushup --mirror

Pipeline:
0. Ekstrakcja keypointów z wideo (SimCC)
1. Normalizacja keypointów
2. Trening modelu TCN
3. Eksport do TFLite

Struktura folderów wideo:
    D:\\Pushups\\
    ├── correct/           ← filmiki z poprawnym wykonaniem
    ├── flared_elbows/     ← filmiki z błędem flared_elbows
    ├── high_hips/         ← itd.
    ├── sagging_hips/
    ├── partial_rom_top/
    ├── partial_rom_bottom/
    └── setup/             ← pozycja startowa
"""

import os
import sys
import argparse

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.config import (
    KEYPOINTS_DIR, MODELS_DIR, ACTIVE_EXERCISE, EXERCISE_CLASSES, DATA_DIR,
    PUSHUP_VIDEOS_DIR,
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


def step0_extract_from_video(video_dir: str, exercise: str,
                              mirror: bool = False, checkpoint: str = None):
    """
    Krok 0: Ekstrakcja keypointów z plików wideo za pomocą SimCC.
    Przetwarza strukturę folderów: video_dir/{klasa}/*.mp4 → keypoints/{klasa}/*.npy

    Args:
        video_dir: Ścieżka do folderu z filmami (np. D:\\Pushups)
        exercise: Nazwa ćwiczenia
        mirror: Czy generować lustrzane odbicia
        checkpoint: Ścieżka do checkpointu SimCC
    """
    import torch
    import torchvision.transforms as T
    from video_to_keypoints import (
        load_simcc_model, process_directory as process_video_dir,
        DEFAULT_CHECKPOINT,
    )

    print("\n" + "=" * 60)
    print("KROK 0: Ekstrakcja keypointów z wideo (SimCC)")
    print("=" * 60)

    labels = EXERCISE_CLASSES[exercise]["labels"]

    # Inicjalizacja modelu SimCC
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simcc_model = load_simcc_model(checkpoint, device)
    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    total_videos = 0
    for label_id, label_name in labels.items():
        input_dir = os.path.join(video_dir, label_name)
        output_dir = os.path.join(KEYPOINTS_DIR, exercise, label_name)

        if os.path.exists(input_dir):
            print(f"\n📂 Klasa: {label_name}")
            process_video_dir(
                input_dir, output_dir, simcc_model, device,
                normalize_transform, preview=False, mirror=mirror
            )

            # Zlicz przetworzone pliki
            npy_count = len([f for f in os.listdir(output_dir)
                            if f.endswith(".npy")]) if os.path.exists(output_dir) else 0
            total_videos += npy_count
        else:
            print(f"  ⚠ Brak katalogu: {input_dir}")

    print(f"\n✅ Wyekstrahowano keypoints z {total_videos} filmów")


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
        "--video_dir", type=str, default=None,
        help="Ścieżka do folderu z filmami (np. D:\\Pushups)"
    )
    parser.add_argument(
        "--keypoints_dir", type=str, default=None,
        help="Ścieżka do katalogu z keypointami (jeśli już wyekstrahowane)"
    )
    parser.add_argument(
        "--exercise", type=str, default=ACTIVE_EXERCISE,
        choices=list(EXERCISE_CLASSES.keys()),
        help="Ćwiczenie do trenowania"
    )
    parser.add_argument(
        "--step", type=int, default=-1, choices=[-1, 0, 1, 2, 3],
        help="Uruchom konkretny krok (-1=wszystkie)"
    )
    parser.add_argument(
        "--setup-only", action="store_true",
        help="Tylko utwórz strukturę folderów"
    )
    parser.add_argument(
        "--mirror", action="store_true",
        help="Generuj lustrzane odbicia (podwaja dane)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Ścieżka do checkpointu SimCC (dla step0)"
    )
    args = parser.parse_args()

    print("🏋️ SEE Trainer — Pipeline trenowania modelu analizy techniki")
    print(f"   Ćwiczenie: {args.exercise}")
    print(f"   Klasy: {list(EXERCISE_CLASSES[args.exercise]['labels'].values())}")

    print("\n📁 Tworzenie struktury katalogów:")
    setup_directories(args.exercise)

    if args.setup_only:
        print("\n✅ Struktura utworzona. Umieść pliki w odpowiednich folderach.")
        return

    # Krok 0: Ekstrakcja z wideo
    if args.step in [-1, 0]:
        if args.video_dir:
            step0_extract_from_video(
                args.video_dir, args.exercise,
                mirror=args.mirror, checkpoint=args.checkpoint
            )
        else:
            print("\n⚠ Pomiń krok 0 (brak --video_dir)")

    # Krok 1: Normalizacja keypointów
    if args.step in [-1, 1]:
        if args.keypoints_dir:
            step1_normalize_keypoints(args.keypoints_dir, args.exercise)
        else:
            print("\n⚠ Pomiń krok 1 (brak --keypoints_dir)")

    # Krok 2: Trening
    if args.step in [-1, 2]:
        step2_train(args.exercise)

    # Krok 3: Eksport
    if args.step in [-1, 3]:
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
