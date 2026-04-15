"""
video_to_keypoints.py — Ekstrakcja keypointów z wideo za pomocą modelu SimCC.

MediaPipe służy TYLKO do detekcji bounding box osoby.
Właściwa ekstrakcja 21 keypointów wykonywana jest przez nasz model SimCC
(z repo pose_training).

Konwertuje nagrania wideo → pliki .npy w formacie zespołowym (21 keypointów × 3 wartości).

Użycie:
    # Pojedynczy plik
    python video_to_keypoints.py --video film.mp4 --output data/keypoints/pullup_overhand/correct/film1.npy

    # Cały folder
    python video_to_keypoints.py --video_dir filmy/correct/ --output_dir data/keypoints/pullup_overhand/correct/

    # Z podglądem
    python video_to_keypoints.py --video film.mp4 --output out.npy --preview

    # Z lustrzanym odbiciem
    python video_to_keypoints.py --video_dir filmy/ --output_dir data/keypoints/ --mirror

Pipeline: Klatka → MediaPipe (BB) → crop_and_pad → SimCC model → simcc_to_keypoints → [21, 3] → .npy
"""

import os
import sys
import argparse
import urllib.request
import numpy as np

try:
    import cv2
except ImportError:
    print("❌ Brak opencv-python. Zainstaluj: pip install opencv-python>=4.8.0")
    sys.exit(1)

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
    print("❌ Brak torch/torchvision/Pillow. Zainstaluj: pip install torch torchvision Pillow")
    sys.exit(1)

# Import konfiguracji z naszego projektu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.config import POSE_MODEL_DIR, SKELETON, NUM_KEYPOINTS

# Import modelu SimCC z repozytorium kolegi (pose_training)
# Musimy chwilowo wyczyścić cache 'model' z sys.modules, żeby Python nie załadował
# naszego pakietu scratch/model tylko model.py z pose_training
import sys
_cached_model = sys.modules.pop("model", None)
sys.path.insert(0, POSE_MODEL_DIR)

try:
    import model as pose_model
    import dataset as pose_dataset
    import config as pose_config

    PoseEstimationModel = pose_model.PoseEstimationModel
    simcc_to_keypoints = pose_model.simcc_to_keypoints
    crop_and_pad = pose_dataset.crop_and_pad
    INPUT_HEIGHT = pose_config.INPUT_HEIGHT
    INPUT_WIDTH = pose_config.INPUT_WIDTH
    SIMCC_SPLIT_RATIO = pose_config.SIMCC_SPLIT_RATIO
except ImportError as e:
    print(f"❌ Nie można zaimportować z pose_training ({POSE_MODEL_DIR}): {e}")
    sys.exit(1)
finally:
    # Przywróć oryginalne środowisko żeby niczego nie zepsuć reszcie kodu
    sys.path.pop(0)
    if _cached_model is not None:
        sys.modules["model"] = _cached_model

# Pary lewo-prawo do lustrzanego odbicia
MIRROR_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18),
]

# Połączenia kości do rysowania szkieletu (podgląd)
SKELETON_DRAW = [
    (0, 19), (19, 3), (19, 4),
    (3, 5), (5, 7), (4, 6), (6, 8),
    (19, 20), (20, 9), (20, 10),
    (9, 11), (11, 13), (10, 12), (12, 14),
    (13, 15), (13, 17), (14, 16), (14, 18),
]

# Docelowe FPS
TARGET_FPS = 30

# Ścieżka do modelu MediaPipe (tylko do bounding box)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MP_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")
MP_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

# Domyślna ścieżka do checkpointu SimCC
DEFAULT_CHECKPOINT = os.path.join(POSE_MODEL_DIR, "checkpoints", "finetune_best.pth")


def ensure_mediapipe_model():
    """Pobiera model MediaPipe Pose jeśli nie istnieje (używany tylko do BB)."""
    if os.path.exists(MP_MODEL_PATH):
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"📥 Pobieranie modelu MediaPipe Pose (heavy, ~30MB)...")
    print(f"   {MP_MODEL_URL}")

    try:
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)
        print(f"   ✅ Pobrano → {MP_MODEL_PATH}")
    except Exception as e:
        print(f"   ❌ Błąd pobierania: {e}")
        print(f"   Pobierz ręcznie z: {MP_MODEL_URL}")
        print(f"   I umieść w: {MP_MODEL_PATH}")
        sys.exit(1)


def load_simcc_model(checkpoint_path: str, device: torch.device):
    """Ładuje model SimCC pose estimation."""
    print(f"🧠 Ładowanie modelu SimCC z: {checkpoint_path}")
    model = PoseEstimationModel(num_keypoints=NUM_KEYPOINTS, pretrained_backbone=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"   ✅ Model załadowany ({device})")
    return model


def extract_keypoints_from_video(video_path: str,
                                  simcc_model,
                                  device: torch.device,
                                  normalize_transform,
                                  preview: bool = False) -> np.ndarray:
    """
    Ekstrakcja keypointów z pliku wideo.

    Flow: MediaPipe (bounding box) → crop → SimCC model → 21 keypointów (x,y,conf) 0-1

    Args:
        video_path: Ścieżka do pliku wideo
        simcc_model: Załadowany model SimCC
        device: torch device (cuda/cpu)
        normalize_transform: ImageNet normalizacja
        preview: Czy pokazać podgląd ze szkieletem

    Returns:
        np.ndarray o kształcie (num_frames, 21, 3) — x, y, confidence (0-1)
    """
    ensure_mediapipe_model()

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Nie można otworzyć wideo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Subsampling: jeśli wideo jest np. 60 FPS, bierzemy co 2. klatkę → 30 FPS
    frame_step = max(1, round(fps / TARGET_FPS))
    effective_fps = fps / frame_step

    print(f"  📹 {os.path.basename(video_path)}: {width}×{height}, "
          f"{fps:.0f} FPS → {effective_fps:.0f} FPS (co {frame_step}. klatka), "
          f"{total_frames} klatek ({total_frames/fps:.1f}s)")

    # MediaPipe — TYLKO do bounding box
    mp_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    CONF_THRESHOLD = 0.3
    all_keypoints = []
    frames_with_pose = 0
    raw_frame_idx = 0
    saved_frame_idx = 0
    last_bbox = None

    with PoseLandmarker.create_from_options(mp_options) as landmarker:
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Subsampling
                if raw_frame_idx % frame_step != 0:
                    raw_frame_idx += 1
                    continue

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

                # === KROK 2: SimCC model → 21 keypointów ===
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

                        # SimCC dekodowanie → [1, 21, 3] (x_norm, y_norm, conf) 0-1
                        kps = simcc_to_keypoints(pred_x, pred_y, pred_vis)
                        kps = kps[0].cpu().numpy()  # [21, 3]

                        keypoints_21 = kps
                        frames_with_pose += 1

                        if preview:
                            draw_skeleton(frame, keypoints_21, crop_w, crop_h,
                                          rx1, ry1)
                else:
                    if preview:
                        cv2.putText(frame, "BRAK DETEKCJI OSOBY", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                all_keypoints.append(keypoints_21)
                saved_frame_idx += 1

                if preview:
                    info = f"Klatka {saved_frame_idx} (src: {raw_frame_idx}/{total_frames})"
                    cv2.putText(frame, info, (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow("Ekstrakcja keypointow SimCC - [Q] aby zamknac", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("  ⏹ Przerwano przez użytkownika")
                        break

                raw_frame_idx += 1

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    result = np.array(all_keypoints, dtype=np.float32)

    detection_rate = frames_with_pose / max(saved_frame_idx, 1) * 100
    print(f"  ✅ {saved_frame_idx} klatek @ {effective_fps:.0f} FPS "
          f"(z {total_frames} @ {fps:.0f} FPS), "
          f"detekcja pozy: {detection_rate:.0f}%")

    if detection_rate < 50:
        print(f"  ⚠️ Niska detekcja ({detection_rate:.0f}%)! "
              f"Sprawdź czy osoba jest widoczna na filmie.")

    return result


def draw_skeleton(frame, keypoints_21, crop_w, crop_h, rx1, ry1):
    """Rysuje szkielet na klatce wideo (przeliczając z crop na oryginał)."""
    h, w = frame.shape[:2]

    # Konwersja z normalized 0-1 (w przestrzeni cropa) na piksele oryginału
    points = []
    for idx in range(21):
        x_norm, y_norm, conf = keypoints_21[idx]
        if conf > 0.3:
            orig_x = int(rx1 + x_norm * crop_w)
            orig_y = int(ry1 + y_norm * crop_h)
            points.append((orig_x, orig_y, conf))
        else:
            points.append(None)

    # Rysuj kości
    for (i, j) in SKELETON_DRAW:
        if points[i] is not None and points[j] is not None:
            pt1 = (points[i][0], points[i][1])
            pt2 = (points[j][0], points[j][1])
            cv2.line(frame, pt1, pt2, (0, 255, 100), 2)

    # Rysuj keypointy
    for p in points:
        if p is not None:
            pt = (p[0], p[1])
            conf = p[2]
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.5 else (0, 100, 255)
            cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pt, 5, (0, 0, 0), 1)


def mirror_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Lustrzane odbicie: odwraca X, zamienia lewe ↔ prawe."""
    mirrored = keypoints.copy()
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
    for left_idx, right_idx in MIRROR_PAIRS:
        mirrored[:, [left_idx, right_idx]] = mirrored[:, [right_idx, left_idx]]
    return mirrored


def process_single_video(video_path: str, output_path: str,
                          simcc_model, device, normalize_transform,
                          preview: bool = False, mirror: bool = False):
    """Przetwarza jeden plik wideo → .npy (+ opcjonalnie lustrzane)."""
    keypoints = extract_keypoints_from_video(
        video_path, simcc_model, device, normalize_transform, preview=preview
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, keypoints)
    print(f"  💾 Zapisano: {output_path} → kształt {keypoints.shape}")

    if mirror:
        mirror_path = output_path.replace(".npy", "_mirror.npy")
        mirrored = mirror_keypoints(keypoints)
        np.save(mirror_path, mirrored)
        print(f"  🪞 Lustrzane: {mirror_path} → kształt {mirrored.shape}")


def process_directory(video_dir: str, output_dir: str,
                       simcc_model, device, normalize_transform,
                       preview: bool = False, mirror: bool = False):
    """Przetwarza wszystkie pliki wideo w katalogu."""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm",
                        ".MP4", ".AVI", ".MOV", ".MKV")
    videos = sorted([
        f for f in os.listdir(video_dir)
        if f.endswith(video_extensions)
    ])

    if not videos:
        print(f"❌ Brak plików wideo w: {video_dir}")
        return

    print(f"\n📂 Znaleziono {len(videos)} filmów w: {video_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for i, video_name in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video_name}")
        video_path = os.path.join(video_dir, video_name)
        output_name = os.path.splitext(video_name)[0] + ".npy"
        output_path = os.path.join(output_dir, output_name)

        try:
            process_single_video(
                video_path, output_path, simcc_model, device,
                normalize_transform, preview=preview, mirror=mirror
            )
        except Exception as e:
            print(f"  ❌ Błąd: {e}")

    print(f"\n✅ Przetworzono {len(videos)} filmów → {output_dir}")
    if mirror:
        print(f"   + {len(videos)} lustrzanych odbić")


def main():
    parser = argparse.ArgumentParser(
        description="Ekstrakcja keypointów z wideo (MediaPipe BB + SimCC → 21 punktów → .npy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python video_to_keypoints.py --video film.mp4 --output wynik.npy
  python video_to_keypoints.py --video_dir filmy/correct/ --output_dir data/keypoints/pullup_overhand/correct/
  python video_to_keypoints.py --video film.mp4 --output wynik.npy --preview
  python video_to_keypoints.py --video_dir filmy/ --output_dir data/ --mirror
        """)

    parser.add_argument("--video", type=str, help="Ścieżka do pojedynczego pliku wideo")
    parser.add_argument("--output", type=str, help="Ścieżka wyjściowa .npy (dla --video)")
    parser.add_argument("--video_dir", type=str, help="Katalog z plikami wideo")
    parser.add_argument("--output_dir", type=str, help="Katalog wyjściowy na pliki .npy")
    parser.add_argument("--preview", action="store_true", help="Podgląd ze szkieletem (Q = zamknij)")
    parser.add_argument("--mirror", action="store_true", help="Generuj lustrzane odbicie")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help=f"Ścieżka do checkpointu SimCC (domyślnie: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--pose_model_dir", type=str, default=POSE_MODEL_DIR,
                        help=f"Ścieżka do repo pose_training (domyślnie: {POSE_MODEL_DIR})")

    args = parser.parse_args()

    print("🦴 Ekstrakcja keypointów z wideo (MediaPipe BB + SimCC)")
    print("=" * 60)

    # Inicjalizacja modelu SimCC
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simcc_model = load_simcc_model(args.checkpoint, device)
    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if args.video:
        if not args.output:
            args.output = os.path.splitext(args.video)[0] + ".npy"
        process_single_video(
            args.video, args.output, simcc_model, device,
            normalize_transform, preview=args.preview, mirror=args.mirror
        )
    elif args.video_dir:
        if not args.output_dir:
            args.output_dir = args.video_dir.rstrip("/\\") + "_keypoints"
        process_directory(
            args.video_dir, args.output_dir, simcc_model, device,
            normalize_transform, preview=args.preview, mirror=args.mirror
        )
    else:
        parser.print_help()
        print("\n❌ Podaj --video (pojedynczy plik) lub --video_dir (folder)")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 Gotowe! Pliki .npy są gotowe do użycia w pipeline trenowania.")
    print("   Następny krok: python run_pipeline.py --keypoints_dir <folder> --exercise pullup_overhand")


if __name__ == "__main__":
    main()
