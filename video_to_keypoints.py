"""
video_to_keypoints.py — Ekstrakcja keypointów z wideo za pomocą MediaPipe Pose.

Konwertuje nagrania wideo → pliki .npy w formacie zespołowym (21 keypointów × 3 wartości).

Użycie:
    # Pojedynczy plik
    python video_to_keypoints.py --video film.mp4 --output data/keypoints/pullup_overhand/correct/film1.npy

    # Cały folder (przetwarza wszystkie filmy w folderze)
    python video_to_keypoints.py --video_dir filmy/correct/ --output_dir data/keypoints/pullup_overhand/correct/

    # Z podglądem (okno z nałożonym szkieletem)
    python video_to_keypoints.py --video film.mp4 --output out.npy --preview

    # Z lustrzanym odbiciem (generuje też _mirror.npy)
    python video_to_keypoints.py --video_dir filmy/ --output_dir data/keypoints/ --mirror

Mapowanie: MediaPipe Pose (33 punkty) → Format zespołowy (21 punktów)
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

# ============================================================
# MAPOWANIE MediaPipe (33 kps) → Nasze (21 kps)
# ============================================================
MEDIAPIPE_TO_OURS = {
    0: 0,     # NOSE → nos
    7: 1,     # LEFT_EAR → lewe_ucho
    8: 2,     # RIGHT_EAR → prawe_ucho
    11: 3,    # LEFT_SHOULDER → lewy_bark
    12: 4,    # RIGHT_SHOULDER → prawy_bark
    13: 5,    # LEFT_ELBOW → lewy_lokiec
    14: 6,    # RIGHT_ELBOW → prawy_lokiec
    15: 7,    # LEFT_WRIST → lewy_nadgarstek
    16: 8,    # RIGHT_WRIST → prawy_nadgarstek
    23: 9,    # LEFT_HIP → lewe_biodro
    24: 10,   # RIGHT_HIP → prawe_biodro
    25: 11,   # LEFT_KNEE → lewe_kolano
    26: 12,   # RIGHT_KNEE → prawe_kolano
    27: 13,   # LEFT_ANKLE → lewa_kostka
    28: 14,   # RIGHT_ANKLE → prawa_kostka
    31: 15,   # LEFT_FOOT_INDEX → lewy_palec_stopy
    32: 16,   # RIGHT_FOOT_INDEX → prawy_palec_stopy
    29: 17,   # LEFT_HEEL → lewa_pieta
    30: 18,   # RIGHT_HEEL → prawa_pieta
    # 19: mostek (obliczany) = średnia(left_shoulder, right_shoulder)
    # 20: srodek_bioder (obliczany) = średnia(left_hip, right_hip)
}

# Pary lewo-prawo do lustrzanego odbicia
MIRROR_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18),
]

# Połączenia kości do rysowania szkieletu
SKELETON_DRAW = [
    (0, 19), (19, 3), (19, 4),
    (3, 5), (5, 7), (4, 6), (6, 8),
    (19, 20), (20, 9), (20, 10),
    (9, 11), (11, 13), (10, 12), (12, 14),
    (13, 15), (13, 17), (14, 16), (14, 18),
]

# Docelowe FPS (model oczekuje 30 FPS — 30 klatek = 1 sekunda)
TARGET_FPS = 30

# Ścieżka do modelu MediaPipe
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"


def ensure_model():
    """Pobiera model MediaPipe Pose jeśli nie istnieje."""
    if os.path.exists(MODEL_PATH):
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"📥 Pobieranie modelu MediaPipe Pose (heavy, ~30MB)...")
    print(f"   {MODEL_URL}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"   ✅ Pobrano → {MODEL_PATH}")
    except Exception as e:
        print(f"   ❌ Błąd pobierania: {e}")
        print(f"   Pobierz ręcznie z: {MODEL_URL}")
        print(f"   I umieść w: {MODEL_PATH}")
        sys.exit(1)


def extract_keypoints_from_video(video_path: str,
                                  preview: bool = False) -> np.ndarray:
    """
    Ekstrakcja keypointów z pliku wideo za pomocą MediaPipe Pose.

    Args:
        video_path: Ścieżka do pliku wideo (mp4/avi/mov)
        preview: Czy pokazać podgląd z nałożonym szkieletem

    Returns:
        np.ndarray o kształcie (num_frames, 21, 3) — x, y, confidence
    """
    ensure_model()

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

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_keypoints = []
    frames_with_pose = 0
    raw_frame_idx = 0
    saved_frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Subsampling — pomijaj klatki jeśli FPS > TARGET_FPS
            if raw_frame_idx % frame_step != 0:
                raw_frame_idx += 1
                continue

            # Konwersja BGR → RGB dla MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Timestamp w milisekundach (bazowany na oryginalnym FPS)
            timestamp_ms = int(raw_frame_idx * 1000 / fps)

            # Detekcja
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Konwersja landmarks → 21 keypointów
            keypoints_21 = np.zeros((21, 3), dtype=np.float32)

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]  # Pierwsza (jedyna) osoba
                frames_with_pose += 1

                # Mapowanie bezpośrednie (19 punktów)
                for mp_idx, our_idx in MEDIAPIPE_TO_OURS.items():
                    lm = landmarks[mp_idx]
                    keypoints_21[our_idx] = [lm.x, lm.y, lm.visibility]

                # Obliczane punkty
                # Mostek (19) = średnia(left_shoulder, right_shoulder)
                l_sh = landmarks[11]
                r_sh = landmarks[12]
                keypoints_21[19] = [
                    (l_sh.x + r_sh.x) / 2,
                    (l_sh.y + r_sh.y) / 2,
                    min(l_sh.visibility, r_sh.visibility),
                ]

                # Środek bioder (20) = średnia(left_hip, right_hip)
                l_hip = landmarks[23]
                r_hip = landmarks[24]
                keypoints_21[20] = [
                    (l_hip.x + r_hip.x) / 2,
                    (l_hip.y + r_hip.y) / 2,
                    min(l_hip.visibility, r_hip.visibility),
                ]

                if preview:
                    draw_skeleton(frame, keypoints_21, width, height)
            else:
                if preview:
                    cv2.putText(frame, "BRAK DETEKCJI POZY", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            all_keypoints.append(keypoints_21)
            saved_frame_idx += 1

            if preview:
                info = f"Klatka {saved_frame_idx} (src: {raw_frame_idx}/{total_frames})"
                cv2.putText(frame, info, (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Ekstrakcja keypointow - [Q] aby zamknac", frame)
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


def draw_skeleton(frame, keypoints_21, width, height):
    """Rysuje szkielet 21 keypointów na klatce wideo."""
    h, w = frame.shape[:2]

    for (i, j) in SKELETON_DRAW:
        x1, y1, c1 = keypoints_21[i]
        x2, y2, c2 = keypoints_21[j]
        if c1 > 0.3 and c2 > 0.3:
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(frame, pt1, pt2, (0, 255, 100), 2)

    for idx in range(21):
        x, y, conf = keypoints_21[idx]
        if conf > 0.3:
            pt = (int(x * w), int(y * h))
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
                          preview: bool = False, mirror: bool = False):
    """Przetwarza jeden plik wideo → .npy (+ opcjonalnie lustrzane)."""
    keypoints = extract_keypoints_from_video(video_path, preview=preview)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, keypoints)
    print(f"  💾 Zapisano: {output_path} → kształt {keypoints.shape}")

    if mirror:
        mirror_path = output_path.replace(".npy", "_mirror.npy")
        mirrored = mirror_keypoints(keypoints)
        np.save(mirror_path, mirrored)
        print(f"  🪞 Lustrzane: {mirror_path} → kształt {mirrored.shape}")


def process_directory(video_dir: str, output_dir: str,
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
            process_single_video(video_path, output_path,
                                  preview=preview, mirror=mirror)
        except Exception as e:
            print(f"  ❌ Błąd: {e}")

    print(f"\n✅ Przetworzono {len(videos)} filmów → {output_dir}")
    if mirror:
        print(f"   + {len(videos)} lustrzanych odbić")


def main():
    parser = argparse.ArgumentParser(
        description="Ekstrakcja keypointów z wideo (MediaPipe → 21 punktów → .npy)",
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

    args = parser.parse_args()

    print("🦴 Ekstrakcja keypointów z wideo (MediaPipe → 21 punktów)")
    print("=" * 60)

    if args.video:
        if not args.output:
            args.output = os.path.splitext(args.video)[0] + ".npy"
        process_single_video(args.video, args.output,
                              preview=args.preview, mirror=args.mirror)
    elif args.video_dir:
        if not args.output_dir:
            args.output_dir = args.video_dir.rstrip("/\\") + "_keypoints"
        process_directory(args.video_dir, args.output_dir,
                           preview=args.preview, mirror=args.mirror)
    else:
        parser.print_help()
        print("\n❌ Podaj --video (pojedynczy plik) lub --video_dir (folder)")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 Gotowe! Pliki .npy są gotowe do użycia w pipeline trenowania.")
    print("   Następny krok: python run_pipeline.py --keypoints_dir <folder> --exercise pullup_overhand")


if __name__ == "__main__":
    main()
