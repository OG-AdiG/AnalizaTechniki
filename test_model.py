"""
test_model.py — Testowanie modelu Analizy Techniki na nowym wideo.

Skrypt ładuje wyeksportowany model TFLite (lub PyTorch), przetwarza wideo
przez MediaPipe Pose, zbiera sekwencje 30 klatek i na żywo wyświetla
przewidywaną klasę (np. "correct", "kipping", "partial_rom_top") na ekranie.

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

# Importy z projektu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.config import EXERCISE_CLASSES, SEQUENCE_LENGTH, INPUT_CHANNELS
from video_to_keypoints import MEDIAPIPE_TO_OURS, SKELETON_DRAW, ensure_model, MODEL_PATH
from data_pipeline.extract_keypoints import normalize_keypoints, filter_low_confidence


def main():
    parser = argparse.ArgumentParser(description="Testowanie modelu na wideo")
    parser.add_argument("--video", type=str, required=True, help="Ścieżka do filmu")
    parser.add_argument("--exercise", type=str, required=True, 
                        choices=list(EXERCISE_CLASSES.keys()), help="Nazwa ćwiczenia")
    parser.add_argument("--model", type=str, default=None, 
                        help="Ścieżka do modelu .tflite (domyślnie z saved_models/)")
    args = parser.parse_args()

    # Zlokalizuj model
    if args.model is None:
        args.model = os.path.join("saved_models", f"{args.exercise}_best.tflite")
        
    if not os.path.exists(args.model):
        print(f"❌ Nie znaleziono modelu: {args.model}")
        print(f"Użyj: python run_pipeline.py --step 3 --exercise {args.exercise}")
        sys.exit(1)

    print(f"🤖 Ładowanie modelu TFLite: {args.model}")
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Pobieranie informacji o ćwiczeniu
    classes = EXERCISE_CLASSES[args.exercise]["labels"]
    print(f"📋 Wykrywane klasy: {classes}")

    # MediaPipe setup
    ensure_model()
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć wideo: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 30
    frame_step = max(1, round(fps / target_fps))
    print(f"📹 Wideo FPS: {fps:.1f} → Pobieranie co {frame_step}. klatkę")

    # Bufor na sekwencję (ostanie 30 klatek)
    sequence_buffer = []

    # Okna
    cv2.namedWindow("SEE Trainer - Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SEE Trainer - Test", 600, 800)

    raw_frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    with PoseLandmarker.create_from_options(options) as landmarker:
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

            # MediaPipe Detekcja
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(raw_frame_idx * 1000 / fps)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            keypoints_21 = np.zeros((21, 3), dtype=np.float32)
            pose_detected = False

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                pose_detected = True
                landmarks = results.pose_landmarks[0]
                
                # Zbieranie punktów (tak jak w video_to_keypoints.py)
                for mp_idx, our_idx in MEDIAPIPE_TO_OURS.items():
                    lm = landmarks[mp_idx]
                    keypoints_21[our_idx] = [lm.x, lm.y, lm.visibility]

                # Mostek i biodra
                keypoints_21[19] = [
                    (landmarks[11].x + landmarks[12].x) / 2,
                    (landmarks[11].y + landmarks[12].y) / 2,
                    min(landmarks[11].visibility, landmarks[12].visibility)
                ]
                keypoints_21[20] = [
                    (landmarks[23].x + landmarks[24].x) / 2,
                    (landmarks[23].y + landmarks[24].y) / 2,
                    min(landmarks[23].visibility, landmarks[24].visibility)
                ]

                # Rysuj szkielet zielony
                for (i, j) in SKELETON_DRAW:
                    x1, y1, c1 = keypoints_21[i]
                    x2, y2, c2 = keypoints_21[j]
                    if c1 > 0.3 and c2 > 0.3:
                        cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)
            else:
                # Brak pozy = rysuj szkielet na czerwono info
                cv2.putText(frame, "BRAK POZY", (w//2-100, 50), font, 1, (0, 0, 255), 2)

            # Format dla funkcji treningowych: (num_frames, 21, 3)
            # Tworzymy tablicę (1, 21, 3) dla pojedynczej klatki
            frame_kps = np.expand_dims(keypoints_21, axis=0)
            
            # 1. Filtrowanie (niski confidence = (0,0))
            frame_kps = filter_low_confidence(frame_kps)
            
            # 2. Normalizacja (centrowanie mid_hip + skalowanie bark-biodro)
            norm_kp = normalize_keypoints(frame_kps)
            
            # Wyciągamy klatkę z [1, 21, 3] z powrotem do [21, 3] i spłaszczamy do [63]
            sequence_buffer.append(norm_kp[0].flatten())
            
            # Utrzymaj max określony ciąg
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)

            # Inferencja TFLite tylko jeśli mamy pełne okno (30 klatek = 1 sekunda ruchu)
            prediction_text = "Zbieranie danych..."
            color = (255, 255, 0) # Cyan

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Konwersja do tensor: shape = [1, 30, 63] lub [1, 63, 30]
                seq_array = np.array(sequence_buffer, dtype=np.float32) # [30, 63]
                
                expected_shape = input_details[0]['shape']
                
                # Zobacz jak był wyeksportowany model TFLite (transpozycja)
                if expected_shape[1] == INPUT_CHANNELS and expected_shape[2] == SEQUENCE_LENGTH:
                    # TFLite chce [1, 63, 30]
                    model_input = np.transpose(seq_array, (1, 0)).reshape((1, INPUT_CHANNELS, SEQUENCE_LENGTH))
                else:
                    # TFLite chce [1, 30, 63]
                    model_input = seq_array.reshape((1, SEQUENCE_LENGTH, INPUT_CHANNELS))

                # Uruchom model TFLite
                interpreter.set_tensor(input_details[0]['index'], model_input)
                interpreter.invoke()
                outputs = interpreter.get_tensor(output_details[0]['index'])[0]

                # Pobierz predykcję (Softmax bo TCN zwraca "czyste" logity/prawdopodobieństwa)
                pred_idx = np.argmax(outputs)
                confidence = outputs[pred_idx]
                
                # Pomiń jak mała pewność
                if confidence > 0.6:
                    pred_class = classes[pred_idx]
                    
                    if pred_class == "correct":
                        color = (0, 255, 0) # Green
                        prediction_text = f"POPRAWNE ({confidence:.0%})"
                    else:
                        color = (0, 0, 255) # Red
                        prediction_text = f"BLAD: {pred_class} ({confidence:.0%})"
                else:
                    prediction_text = "Niepewny..."
                    color = (200, 200, 200)

            # Wyświetlanie predykcji na wideo (czarne tło dla tekstu)
            cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, prediction_text, (20, h - 30), font, 1.2, color, 3)

            # Postęp buffora
            buf_status = f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
            cv2.putText(frame, buf_status, (w - 200, h - 30), font, 0.6, (255, 255, 255), 1)

            cv2.imshow("SEE Trainer - Test", frame)
            
            # Q przerywa
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
