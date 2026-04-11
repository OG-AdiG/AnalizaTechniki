"""
test_model.py — Testowanie modelu Analizy Techniki na nowym wideo.

Skrypt ładuje wyeksportowany model TFLite (lub PyTorch), przetwarza wideo:
- MediaPipe Pose → bounding box (detekcja osoby)
- Model SimCC → 21 keypointów (x, y, confidence) 0-1
- Normalizacja keypointów + sliding window → model TCN
- Wyświetla klasę predykcji na ekranie w czasie rzeczywistym

Użycie:
    python test_model.py --video test.mp4 --exercise pullup_overhand
    python test_model.py --video test.mp4 --model path/to/model.tflite --exercise pullup_overhand
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
    import tensorflow as tf
except ImportError:
    print("❌ Brak tensorflow. Zainstaluj: pip install tensorflow")
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
)
from video_to_keypoints import (
    SKELETON_DRAW, ensure_mediapipe_model, MP_MODEL_PATH,
    load_simcc_model, DEFAULT_CHECKPOINT,
)
from data_pipeline.extract_keypoints import normalize_keypoints, filter_low_confidence

# Import z pose_training
sys.path.insert(0, POSE_MODEL_DIR)
from model import simcc_to_keypoints
from dataset import crop_and_pad
from config import INPUT_HEIGHT, INPUT_WIDTH


def main():
    parser = argparse.ArgumentParser(description="Testowanie modelu na wideo")
    parser.add_argument("--video", type=str, required=True, help="Ścieżka do filmu")
    parser.add_argument("--exercise", type=str, required=True,
                        choices=list(EXERCISE_CLASSES.keys()), help="Nazwa ćwiczenia")
    parser.add_argument("--model", type=str, default=None,
                        help="Ścieżka do modelu .tflite (domyślnie z saved_models/)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Ścieżka do checkpointu SimCC")
    args = parser.parse_args()

    # Zlokalizuj model TFLite (technika)
    if args.model is None:
        args.model = os.path.join("saved_models", f"{args.exercise}_best.tflite")

    if not os.path.exists(args.model):
        print(f"❌ Nie znaleziono modelu: {args.model}")
        print(f"Użyj: python run_pipeline.py --step 3 --exercise {args.exercise}")
        sys.exit(1)

    print(f"🤖 Ładowanie modelu TFLite (technika): {args.model}")
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ładowanie modelu SimCC (pose estimation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simcc_model = load_simcc_model(args.checkpoint, device)
    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Pobieranie informacji o ćwiczeniu
    classes = EXERCISE_CLASSES[args.exercise]["labels"]
    print(f"📋 Wykrywane klasy: {classes}")

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
    target_fps = 30
    frame_step = max(1, round(fps / target_fps))
    print(f"📹 Wideo FPS: {fps:.1f} → Pobieranie co {frame_step}. klatkę")

    # Bufor na sekwencję (ostatnie 30 klatek)
    sequence_buffer = []
    last_bbox = None

    # Okna
    cv2.namedWindow("SEE Trainer - Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SEE Trainer - Test", 600, 800)

    raw_frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    with PoseLandmarker.create_from_options(options) as landmarker:
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Pomiń klatki żeby uzyskać 30 FPS
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
                pose_detected = False

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

                        pred_x, pred_y = simcc_model(input_tensor)
                        kps = simcc_to_keypoints(pred_x, pred_y)
                        keypoints_21 = kps[0].cpu().numpy()
                        pose_detected = True

                        # Rysuj szkielet
                        for (i, j) in SKELETON_DRAW:
                            x1n, y1n, c1 = keypoints_21[i]
                            x2n, y2n, c2 = keypoints_21[j]
                            if c1 > 0.3 and c2 > 0.3:
                                pt1 = (int(rx1 + x1n * crop_w), int(ry1 + y1n * crop_h))
                                pt2 = (int(rx1 + x2n * crop_w), int(ry1 + y2n * crop_h))
                                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "BRAK POZY", (w//2-100, 50), font, 1, (0, 0, 255), 2)

                # === KROK 3: Normalizacja + bufor ===
                frame_kps = np.expand_dims(keypoints_21, axis=0)
                frame_kps = filter_low_confidence(frame_kps)
                norm_kp = normalize_keypoints(frame_kps)
                sequence_buffer.append(norm_kp[0].flatten())

                if len(sequence_buffer) > SEQUENCE_LENGTH:
                    sequence_buffer.pop(0)

                # === KROK 4: Inferencja TCN ===
                prediction_text = "Zbieranie danych..."
                color = (255, 255, 0)

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    seq_array = np.array(sequence_buffer, dtype=np.float32)

                    expected_shape = input_details[0]['shape']

                    if expected_shape[1] == INPUT_CHANNELS and expected_shape[2] == SEQUENCE_LENGTH:
                        model_input = np.transpose(seq_array, (1, 0)).reshape(
                            (1, INPUT_CHANNELS, SEQUENCE_LENGTH)
                        )
                    else:
                        model_input = seq_array.reshape(
                            (1, SEQUENCE_LENGTH, INPUT_CHANNELS)
                        )

                    interpreter.set_tensor(input_details[0]['index'], model_input)
                    interpreter.invoke()
                    logits = interpreter.get_tensor(output_details[0]['index'])[0]

                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / exp_logits.sum()

                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]

                    if confidence > 0.4:
                        pred_class = classes[pred_idx]

                        if pred_class == "correct":
                            color = (0, 255, 0)
                            prediction_text = f"POPRAWNE ({confidence:.0%})"
                        elif pred_class == "setup":
                            color = (255, 200, 0)
                            prediction_text = f"SETUP ({confidence:.0%})"
                        else:
                            color = (0, 0, 255)
                            prediction_text = f"BLAD: {pred_class} ({confidence:.0%})"
                    else:
                        prediction_text = "Niepewny..."
                        color = (200, 200, 200)

                # Wyświetlanie
                cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
                cv2.putText(frame, prediction_text, (20, h - 30), font, 1.2, color, 3)

                buf_status = f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
                cv2.putText(frame, buf_status, (w - 200, h - 30), font, 0.6, (255, 255, 255), 1)

                cv2.imshow("SEE Trainer - Test", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
