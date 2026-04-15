"""
test_model.py — Testowanie modelu Analizy Techniki na wideo (per-rep flow).

Skrypt przetwarza wideo i klasyfikuje KAŻDE POWTÓRZENIE osobno:
- SimCC → 21 keypointów
- RepCounter → wykrywa granice repów
- TCN → klasyfikuje technikę PER REP (nie per klatkę!)
- Wyświetla wynik na ekranie + podsumowanie na końcu

Użycie:
    # Użytkownik wybiera ćwiczenie
    python test_model.py --video test.mp4 --exercise pushup

    # Auto-detekcja ćwiczenia
    python test_model.py --video test.mp4 --exercise auto

    # Z modelem PyTorch zamiast TFLite
    python test_model.py --video test.mp4 --exercise pushup --pytorch
"""

import os
import sys
import argparse
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    print("❌ Brak mediapipe. Zainstaluj: pip install mediapipe>=0.10.0")
    sys.exit(1)

try:
    import torch
    import torchvision.transforms as T
    from PIL import Image
except ImportError:
    print("❌ Brak torch/torchvision. Zainstaluj: pip install torch torchvision Pillow")
    sys.exit(1)

# Importy z projektu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.config import (
    EXERCISE_CLASSES, SEQUENCE_LENGTH, INPUT_CHANNELS, POSE_MODEL_DIR,
    MODELS_DIR,
)
from model.rep_classifier import RepClassifier
from model.exercise_detector import ExerciseDetector
# Import z pose_training — eksportowane przez video_to_keypoints
from video_to_keypoints import (
    SKELETON_DRAW, ensure_mediapipe_model, MP_MODEL_PATH,
    load_simcc_model, DEFAULT_CHECKPOINT,
    simcc_to_keypoints, crop_and_pad, INPUT_HEIGHT, INPUT_WIDTH,
)


# Kolory
COLOR_CORRECT = (0, 200, 0)
COLOR_ERROR = (0, 0, 255)
COLOR_SETUP = (255, 200, 0)
COLOR_PENDING = (200, 200, 200)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_SKELETON = (0, 255, 100)


def draw_rep_history(frame, rep_results, max_show=8):
    """Rysuje historię repów na ekranie (prawy panel)."""
    h, w = frame.shape[:2]
    x_start = w - 260
    y_start = 10

    # Tło panelu
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start - 10, y_start - 5),
                  (w - 5, y_start + 30 + min(len(rep_results), max_show) * 28),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "Historia repow:", (x_start, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # Pokaż ostatnie N repów
    visible = rep_results[-max_show:]
    for i, rep in enumerate(visible):
        y = y_start + 38 + i * 28
        name = rep.get("class_name", "?")
        conf = rep.get("confidence", 0)
        num = rep.get("rep_number", i + 1)

        if name == "correct":
            color = COLOR_CORRECT
            icon = "OK"
        elif name == "setup":
            color = COLOR_SETUP
            icon = "SU"
        else:
            color = COLOR_ERROR
            icon = "!!"

        text = f"Rep {num}: [{icon}] {name} ({conf:.0%})"
        cv2.putText(frame, text, (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def draw_summary_bar(frame, classifier):
    """Rysuje pasek podsumowania na dole ekranu."""
    h, w = frame.shape[:2]
    summary = classifier.get_summary()

    # Tło paska
    cv2.rectangle(frame, (0, h - 90), (w, h), COLOR_BLACK, -1)

    # Bieżący stan
    state = classifier.get_current_state()
    phase = state["phase"]
    angle = state["angle_smooth"]
    buf = state["buffered_frames"]
    reps = state["rep_count"]

    # Linia 1: Stan
    status = f"Faza: {phase.upper()} | Kat: {angle:.0f}deg | Bufor: {buf} kl."
    cv2.putText(frame, status, (10, h - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

    # Linia 2: Podsumowanie
    total = summary["total_reps"]
    correct = summary["correct_reps"]
    acc = summary["accuracy"]

    if total > 0:
        sum_text = f"Repy: {reps} | Poprawne: {correct}/{total} ({acc:.0%})"
        errors = summary["error_breakdown"]
        if errors:
            top_errors = sorted(errors.items(), key=lambda x: -x[1])[:3]
            err_str = ", ".join(f"{k}:{v}" for k, v in top_errors)
            sum_text += f" | Bledy: {err_str}"
    else:
        sum_text = f"Repy: {reps} | Oczekiwanie na powtorzenia..."

    color = COLOR_CORRECT if acc > 0.7 else COLOR_ERROR if total > 0 else COLOR_PENDING
    cv2.putText(frame, sum_text, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Ostatni rep (jeśli nowy)
    last_rep = state["last_rep"]
    if last_rep:
        last_name = last_rep.get("class_name", "?")
        last_conf = last_rep.get("confidence", 0)
        last_num = last_rep.get("rep_number", 0)
        last_color = COLOR_CORRECT if last_name == "correct" else COLOR_ERROR
        last_text = f"Ostatni: Rep {last_num} = {last_name} ({last_conf:.0%})"
        cv2.putText(frame, last_text, (w - 350, h - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, last_color, 1)


def main():
    parser = argparse.ArgumentParser(description="Testowanie modelu na wideo (per-rep)")
    parser.add_argument("--video", type=str, required=True, help="Ścieżka do filmu")
    parser.add_argument("--exercise", type=str, required=True,
                        choices=list(EXERCISE_CLASSES.keys()) + ["auto"],
                        help="Nazwa ćwiczenia (lub 'auto')")
    parser.add_argument("--model", type=str, default=None,
                        help="Ścieżka do modelu .tflite")
    parser.add_argument("--pytorch", action="store_true",
                        help="Użyj modelu PyTorch (.pt) zamiast TFLite")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Ścieżka do checkpointu SimCC")
    args = parser.parse_args()

    # Detekcja ćwiczenia (auto lub manual)
    exercise = args.exercise
    exercise_detector = None

    if exercise == "auto":
        exercise_detector = ExerciseDetector(mode="heuristic")
        exercise = "pushup"  # Domyślne — zmieni się po detekcji
        print("🔍 Tryb auto-detekcji ćwiczenia")

    # Lokalizacja modelu techniki
    if args.model is None:
        if args.pytorch:
            args.model = os.path.join(MODELS_DIR, f"{exercise}_best.pt")
        else:
            args.model = os.path.join(MODELS_DIR, f"{exercise}_best.tflite")

    model_type = "pytorch" if args.pytorch else "tflite"

    if not os.path.exists(args.model):
        print(f"⚠ Model nie znaleziony: {args.model}")
        print(f"  Klasyfikacja techniki wyłączona (tylko zliczanie repów)")
        args.model = None

    # Inicjalizacja RepClassifier
    classifier = RepClassifier(
        exercise=exercise,
        model_path=args.model,
        model_type=model_type,
    )
    print(f"🏋️ Ćwiczenie: {exercise}")
    print(f"📋 Klasy: {list(classifier.labels.values())}")

    # Ładowanie modelu SimCC (pose estimation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simcc_model = load_simcc_model(args.checkpoint, device)
    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # MediaPipe setup (TYLKO BB)
    ensure_mediapipe_model()
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć wideo: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = 30
    frame_step = max(1, round(fps / target_fps))
    print(f"📹 Wideo FPS: {fps:.1f} → co {frame_step}. klatkę ({total_frames} kl., {total_frames/fps:.1f}s)")

    # Okno
    cv2.namedWindow("SEE Trainer - Per-Rep Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SEE Trainer - Per-Rep Analysis", 800, 600)

    last_bbox = None
    raw_frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_rep_flash_frames = 0  # Flash efekt przy nowym repie

    with PoseLandmarker.create_from_options(options) as landmarker:
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if raw_frame_idx % frame_step != 0:
                    raw_frame_idx += 1
                    continue

                raw_frame_idx += 1
                h, w = frame.shape[:2]

                # === KROK 1: MediaPipe → Bounding Box ===
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(raw_frame_idx * 1000 / fps)
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    landmarks_list = results.pose_landmarks[0]
                    x_coords = [lm.x * w for lm in landmarks_list]
                    y_coords = [lm.y * h for lm in landmarks_list]
                    px1, px2 = min(x_coords), max(x_coords)
                    py1, py2 = min(y_coords), max(y_coords)

                    bw = px2 - px1
                    bh = py2 - py1
                    cx = px1 + bw / 2
                    cy = py1 + bh / 2

                    last_bbox = (cx, cy, bw, bh)

                # === KROK 2: SimCC → 21 keypointów ===
                keypoints_21 = np.zeros((21, 3), dtype=np.float32)

                if last_bbox is not None:
                    cx, cy, bw, bh = last_bbox
                    bw *= 1.2
                    bh *= 1.2

                    crop, rx1, ry1, rx2, ry2 = crop_and_pad(
                        frame, cx, cy, bw, bh, INPUT_WIDTH, INPUT_HEIGHT
                    )
                    crop_h, crop_w = crop.shape[:2]

                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(crop_rgb)
                        input_tensor = T.Resize((INPUT_HEIGHT, INPUT_WIDTH))(pil_img)
                        input_tensor = T.ToTensor()(input_tensor)
                        input_tensor = normalize_transform(input_tensor)
                        input_tensor = input_tensor.unsqueeze(0).to(device)

                        pred_x, pred_y, pred_vis = simcc_model(input_tensor)
                        kps = simcc_to_keypoints(pred_x, pred_y, pred_vis)
                        keypoints_21 = kps[0].cpu().numpy()

                        # Rysuj szkielet
                        for (i, j) in SKELETON_DRAW:
                            x1n, y1n, c1 = keypoints_21[i]
                            x2n, y2n, c2 = keypoints_21[j]
                            if c1 > 0.3 and c2 > 0.3:
                                pt1 = (int(rx1 + x1n * crop_w), int(ry1 + y1n * crop_h))
                                pt2 = (int(rx1 + x2n * crop_w), int(ry1 + y2n * crop_h))
                                cv2.line(frame, pt1, pt2, COLOR_SKELETON, 2)

                        # Auto-detect ćwiczenia
                        if exercise_detector is not None:
                            detected = exercise_detector.detect(keypoints_21)
                            if detected != "unknown" and detected != exercise:
                                exercise = detected
                                classifier = RepClassifier(exercise, args.model, model_type)
                                print(f"🔄 Wykryto ćwiczenie: {exercise}")
                else:
                    cv2.putText(frame, "BRAK POZY", (w // 2 - 100, 50),
                                font, 1, COLOR_ERROR, 2)

                # === KROK 3: Per-rep klasyfikacja ===
                rep_result = classifier.process_frame(keypoints_21)

                if rep_result is not None:
                    # Nowy rep — flash efekt
                    new_rep_flash_frames = 15
                    name = rep_result["class_name"]
                    conf = rep_result["confidence"]
                    num = rep_result["rep_number"]
                    print(f"  🏋️ Rep {num}: {name} ({conf:.0%})")

                # Flash efekt nowego repa
                if new_rep_flash_frames > 0:
                    alpha = new_rep_flash_frames / 15.0
                    last_result = classifier.rep_results[-1] if classifier.rep_results else None
                    if last_result:
                        flash_color = COLOR_CORRECT if last_result["class_name"] == "correct" else COLOR_ERROR
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, 5), flash_color, -1)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    new_rep_flash_frames -= 1

                # === RYSOWANIE UI ===
                draw_rep_history(frame, classifier.rep_results)
                draw_summary_bar(frame, classifier)

                # Tytuł
                title = f"SEE Trainer | {exercise.upper()} | Per-Rep Analysis"
                cv2.putText(frame, title, (10, 30), font, 0.6, COLOR_WHITE, 1)

                cv2.imshow("SEE Trainer - Per-Rep Analysis", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

    # Podsumowanie końcowe
    summary = classifier.get_summary()
    print("\n" + "=" * 60)
    print("📊 PODSUMOWANIE SETA")
    print("=" * 60)
    print(f"  Ćwiczenie:   {summary['exercise']}")
    print(f"  Łączne repy: {summary['total_reps']}")
    print(f"  Poprawne:    {summary['correct_reps']}")
    print(f"  Dokładność:  {summary['accuracy']:.0%}")
    if summary["error_breakdown"]:
        print(f"  Błędy:")
        for error_name, count in summary["error_breakdown"].items():
            print(f"    - {error_name}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
