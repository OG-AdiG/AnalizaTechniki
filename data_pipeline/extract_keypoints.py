"""
extract_keypoints.py — Adapter danych keypointów z modelu pose estimation.

Twój kolega z zespołu robi model pose estimation, który zwraca:
    keypoints: [21, 3] — x, y, confidence (normalizacja 0-1)

Ten moduł:
- Przyjmuje dane w formacie zespołowym (21 kps × 3 dim)
- Normalizuje: centrowanie (mid_hip) + skalowanie (bark-biodro)
- Obsługuje ładowanie danych offline z plików .npy / .csv
- Przetwarza katalogi z nagraniami (wideo → keypoints)

UWAGA: Ekstrakcja z wideo wymaga modelu kolegi (osobny pipeline).
       Ten skrypt obsługuje normalizację i ładowanie gotowych keypoints.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.config import (
    NUM_KEYPOINTS, KEYPOINT_DIMS, LANDMARK_INDEX, MIN_CONFIDENCE,
)


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalizacja sekwencji keypointów:
    1. Centrowanie względem mid_hip (indeks 20)
    2. Skalowanie względem odległości bark-biodro

    Confidence (3. wymiar) NIE jest modyfikowany.

    Args:
        keypoints: (num_frames, 21, 3) — x, y, confidence

    Returns:
        (num_frames, 21, 3) — znormalizowane współrzędne (x,y zmienione, confidence bez zmian)
    """
    normalized = keypoints.copy()
    mid_hip_idx = LANDMARK_INDEX["mid_hip"]       # 20
    l_shoulder_idx = LANDMARK_INDEX["left_shoulder"]   # 3
    r_shoulder_idx = LANDMARK_INDEX["right_shoulder"]  # 4
    l_hip_idx = LANDMARK_INDEX["left_hip"]             # 9
    r_hip_idx = LANDMARK_INDEX["right_hip"]            # 10

    last_valid_hip = None
    last_valid_scale = None

    for i in range(len(normalized)):
        frame = normalized[i].copy()

        # 1. Wyznacz stabilny hip_center (środek bioder)
        if frame[mid_hip_idx, 2] >= MIN_CONFIDENCE:
            hip_center = frame[mid_hip_idx, :2]
        elif frame[l_hip_idx, 2] >= MIN_CONFIDENCE and frame[r_hip_idx, 2] >= MIN_CONFIDENCE:
            hip_center = (frame[l_hip_idx, :2] + frame[r_hip_idx, :2]) / 2.0
        elif last_valid_hip is not None:
            hip_center = last_valid_hip
        else:
            hip_center = np.array([0.5, 0.5]) # Fallback (środek obrazu)
            
        last_valid_hip = hip_center

        # 2. Wyznacz stabilny scale (odległość bark-biodro)
        left_dist = 0
        right_dist = 0
        valid_dists = 0
        
        if frame[l_shoulder_idx, 2] >= MIN_CONFIDENCE and frame[l_hip_idx, 2] >= MIN_CONFIDENCE:
            left_dist = np.linalg.norm(frame[l_shoulder_idx, :2] - frame[l_hip_idx, :2])
            valid_dists += 1
            
        if frame[r_shoulder_idx, 2] >= MIN_CONFIDENCE and frame[r_hip_idx, 2] >= MIN_CONFIDENCE:
            right_dist = np.linalg.norm(frame[r_shoulder_idx, :2] - frame[r_hip_idx, :2])
            valid_dists += 1

        if valid_dists == 2:
            scale = (left_dist + right_dist) / 2.0
        elif valid_dists == 1:
            scale = left_dist + right_dist # Jedno to 0
        elif last_valid_scale is not None:
            scale = last_valid_scale
        else:
            scale = 0.3 # Fallback domyślny

        scale = max(scale, 1e-4) # Zabezpieczenie przed dzieleniem przez 0
        last_valid_scale = scale

        # 3. Zastosuj transformację (tylko x, y — nie confidence)
        frame[:, :2] -= hip_center
        frame[:, :2] /= scale

        normalized[i] = frame

    return normalized


def filter_low_confidence(keypoints: np.ndarray,
                          min_conf: float = MIN_CONFIDENCE) -> np.ndarray:
    """
    Zeruje współrzędne keypointów z confidence poniżej progu.
    To pomaga modelowi ignorować niestabilne detekcje.

    Args:
        keypoints: (num_frames, 21, 3)
        min_conf: minimalny próg confidence

    Returns:
        (num_frames, 21, 3) — z wyzerowanymi niestabilnymi punktami
    """
    filtered = keypoints.copy()
    mask = filtered[:, :, 2] < min_conf  # (num_frames, 21)
    filtered[mask, :2] = 0.0
    # Confidence zostawiamy (model może się nauczyć ignorować niskie)
    return filtered


def load_keypoints_npy(filepath: str,
                       normalize: bool = True,
                       filter_conf: bool = True) -> np.ndarray:
    """
    Ładuje keypoints z pliku .npy i opcjonalnie normalizuje.

    Args:
        filepath: Ścieżka do pliku .npy z kształtem (T, 21, 3)
        normalize: Czy zastosować normalizację
        filter_conf: Czy filtrować niestabilne punkty

    Returns:
        np.ndarray (T, 21, 3)
    """
    keypoints = np.load(filepath).astype(np.float32)

    if keypoints.ndim != 3 or keypoints.shape[1:] != (NUM_KEYPOINTS, KEYPOINT_DIMS):
        raise ValueError(
            f"Oczekiwany kształt: (T, {NUM_KEYPOINTS}, {KEYPOINT_DIMS}), "
            f"got: {keypoints.shape}"
        )

    if normalize:
        keypoints = normalize_keypoints(keypoints)

    if filter_conf:
        keypoints = filter_low_confidence(keypoints)

    return keypoints


def load_keypoints_csv(filepath: str,
                       normalize: bool = True,
                       filter_conf: bool = True) -> np.ndarray:
    """
    Ładuje keypoints z pliku CSV.

    Oczekiwany format CSV: 63 kolumny na wiersz (21 punktów × 3 wartości),
    każdy wiersz = jedna klatka.

    Args:
        filepath: Ścieżka do pliku CSV
        normalize, filter_conf: jak wyżej

    Returns:
        np.ndarray (T, 21, 3)
    """
    data = np.loadtxt(filepath, delimiter=",", dtype=np.float32)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != NUM_KEYPOINTS * KEYPOINT_DIMS:
        raise ValueError(
            f"Oczekiwane {NUM_KEYPOINTS * KEYPOINT_DIMS} kolumn, got: {data.shape[1]}"
        )

    keypoints = data.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIMS)

    if normalize:
        keypoints = normalize_keypoints(keypoints)

    if filter_conf:
        keypoints = filter_low_confidence(keypoints)

    return keypoints


def process_directory(input_dir: str, output_dir: str):
    """
    Przetwarza katalog z plikami keypointów (.npy lub .csv).
    Normalizuje i zapisuje do output_dir.

    Args:
        input_dir: Katalog wejściowy z plikami keypointów
        output_dir: Katalog wyjściowy na znormalizowane pliki .npy
    """
    os.makedirs(output_dir, exist_ok=True)

    supported_ext = (".npy", ".csv")
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(supported_ext)]

    if not files:
        print(f"⚠ Brak plików keypointów w: {input_dir}")
        return

    print(f"📂 Znaleziono {len(files)} plików w: {input_dir}")

    for filename in sorted(files):
        filepath = os.path.join(input_dir, filename)
        output_name = os.path.splitext(filename)[0] + ".npy"
        output_path = os.path.join(output_dir, output_name)

        try:
            print(f"  🔄 Przetwarzanie: {filename}...", end=" ")

            if filename.endswith(".npy"):
                keypoints = load_keypoints_npy(filepath)
            else:
                keypoints = load_keypoints_csv(filepath)

            np.save(output_path, keypoints)
            print(f"✅ {keypoints.shape[0]} klatek → {output_name}")
        except Exception as e:
            print(f"❌ Błąd: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Przetwarzanie keypointów z modelu pose estimation (normalizacja)"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Katalog z plikami keypointów (.npy lub .csv)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Katalog wyjściowy na znormalizowane pliki .npy"
    )
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
    print("\n✅ Przetwarzanie zakończone!")


if __name__ == "__main__":
    main()
