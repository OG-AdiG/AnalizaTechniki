"""
dataset.py — PyTorch Dataset do ładowania sekwencji keypointów z plików .npy

Obsługuje dwa tryby:
A) single_rep_mode=True (NOWY):
   Każdy plik .npy = jedno powtórzenie → pad/resize do SEQUENCE_LENGTH
   Używane gdy dane treningowe to pliki wideo z 1 repem każdy.

B) single_rep_mode=False (legacy):
   Sliding window dzieli długie nagrania na okna po 30 klatek.

Format wejściowy: (T, 21, 3) — 21 keypointów × (x, y, confidence)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.config import (
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE,
    KEYPOINTS_DIR,
    ACTIVE_EXERCISE,
    EXERCISE_CLASSES,
    NUM_KEYPOINTS,
    KEYPOINT_DIMS,
    BATCH_SIZE,
)


class ExerciseDataset(Dataset):
    """
    Dataset ładujący sekwencje keypointów.

    Każda próbka to tensor (INPUT_CHANNELS, SEQUENCE_LENGTH) = (63, 30)
    i etykieta (int) — 0=setup, 1=correct, 2-6=typy błędów.
    """

    def __init__(self, sequences: list, labels: list, augment: bool = False):
        """
        Args:
            sequences: Lista tablic numpy (seq_len, 21, 3) — okna z keypoints
            labels: Lista etykiet (int)
            augment: Czy stosować augmentację danych
        """
        self.sequences = sequences
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()  # (seq_len, 21, 3)

        if self.augment:
            seq = self._augment(seq)

        # Spłaszcz (seq_len, 21, 3) → (seq_len, 63)
        seq_flat = seq.reshape(seq.shape[0], -1)  # (seq_len, 63)

        # Transponuj na (63, seq_len) — Conv1D oczekuje (channels, length)
        seq_tensor = torch.FloatTensor(seq_flat).permute(1, 0)  # (63, seq_len)
        label_tensor = torch.LongTensor([self.labels[idx]])

        return seq_tensor, label_tensor.squeeze()

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """
        Augmentacja sekwencji:
        - Losowy szum Gaussowski na x,y (nie na confidence)
        - Losowe skalowanie x,y
        - Losowe przesunięcie x,y
        - Losowe odbicie lustrzane (zamiana left↔right)
        - Losowe przyspieszenie/spowolnienie (temporal jitter)
        """
        # Szum na współrzędne x,y
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, seq[:, :, :2].shape)
            seq[:, :, :2] += noise.astype(np.float32)

        # Losowe skalowanie x,y
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            seq[:, :, :2] *= scale

        # Losowe przesunięcie x,y (drobne)
        if np.random.random() < 0.3:
            shift = np.random.uniform(-0.02, 0.02, (1, 1, 2))
            seq[:, :, :2] += shift.astype(np.float32)

        # Odbicie lustrzane (LEFT ↔ RIGHT)
        if np.random.random() < 0.3:
            seq = self._mirror_keypoints(seq)

        # Temporal jitter — losowe przyspieszenie/spowolnienie (0.8x-1.2x)
        if np.random.random() < 0.4:
            seq = self._temporal_jitter(seq)

        return seq

    def _mirror_keypoints(self, seq: np.ndarray) -> np.ndarray:
        """
        Odbicie lustrzane — zamiana lewych i prawych keypointów.
        Odwraca x-ową współrzędną i zamienia pary lewy↔prawy.
        """
        mirrored = seq.copy()

        # Odwróć współrzędną X (zakładamy znormalizowane dane)
        mirrored[:, :, 0] = -mirrored[:, :, 0]

        # Zamiana par lewy ↔ prawy
        # Pary indeksów: (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16), (17,18)
        swap_pairs = [
            (1, 2),    # lewe_ucho ↔ prawe_ucho
            (3, 4),    # lewy_bark ↔ prawy_bark
            (5, 6),    # lewy_lokiec ↔ prawy_lokiec
            (7, 8),    # lewy_nadgarstek ↔ prawy_nadgarstek
            (9, 10),   # lewe_biodro ↔ prawe_biodro
            (11, 12),  # lewe_kolano ↔ prawe_kolano
            (13, 14),  # lewa_kostka ↔ prawa_kostka
            (15, 16),  # lewy_palec ↔ prawy_palec
            (17, 18),  # lewa_pieta ↔ prawa_pieta
        ]

        for left_idx, right_idx in swap_pairs:
            mirrored[:, [left_idx, right_idx]] = mirrored[:, [right_idx, left_idx]]

        return mirrored

    def _temporal_jitter(self, seq: np.ndarray) -> np.ndarray:
        """
        Temporal jitter — losowe przyspieszenie/spowolnienie.
        Symuluje różne tempo wykonania ćwiczenia (0.8x-1.2x).
        Interpoluje do oryginalnej długości.
        """
        num_frames = seq.shape[0]
        speed_factor = np.random.uniform(0.8, 1.2)
        new_len = max(3, int(num_frames * speed_factor))

        # Indeksy do resamplingu
        old_indices = np.linspace(0, num_frames - 1, new_len)
        new_indices = np.linspace(0, new_len - 1, num_frames)

        # Interpolacja per keypoint × dim
        result = np.zeros_like(seq)
        for kp in range(seq.shape[1]):
            for dim in range(seq.shape[2]):
                values = seq[:, kp, dim]
                # Najpierw resample do new_len
                resampled = np.interp(
                    np.linspace(0, len(values) - 1, new_len),
                    np.arange(len(values)),
                    values,
                )
                # Potem resample z powrotem do num_frames
                result[:, kp, dim] = np.interp(
                    np.linspace(0, len(resampled) - 1, num_frames),
                    np.arange(len(resampled)),
                    resampled,
                )

        return result


def resize_sequence(keypoints: np.ndarray,
                    target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Resize sekwencji do target_len klatek (dla single_rep_mode).
    - Jeśli za krótka: pad ostatnią klatką
    - Jeśli za długa: równomierne subsamplowanie
    - Jeśli równa: bez zmian
    """
    num_frames = keypoints.shape[0]

    if num_frames == target_len:
        return keypoints
    elif num_frames < target_len:
        padded = np.empty(
            (target_len, keypoints.shape[1], keypoints.shape[2]),
            dtype=np.float32,
        )
        padded[:num_frames] = keypoints
        padded[num_frames:] = keypoints[-1]
        return padded
    else:
        indices = np.linspace(0, num_frames - 1, target_len, dtype=int)
        return keypoints[indices]


def create_sliding_windows(keypoints: np.ndarray,
                           window_size: int = SEQUENCE_LENGTH,
                           stride: int = SEQUENCE_STRIDE) -> list:
    """
    Dzieli sekwencję na okna za pomocą sliding window.

    Args:
        keypoints: (num_frames, 21, 3) — cała sekwencja
        window_size: Rozmiar okna (liczba klatek)
        stride: Przesunięcie między oknami

    Returns:
        Lista tablic numpy, każda (window_size, 21, 3)
    """
    windows = []
    num_frames = keypoints.shape[0]

    if num_frames < window_size:
        # Za krótkie — padowanie zerami
        padded = np.zeros(
            (window_size, keypoints.shape[1], keypoints.shape[2]),
            dtype=np.float32,
        )
        padded[:num_frames] = keypoints
        windows.append(padded)
    else:
        for start in range(0, num_frames - window_size + 1, stride):
            window = keypoints[start : start + window_size]
            windows.append(window)

    return windows


def load_dataset(exercise: str = ACTIVE_EXERCISE,
                 single_rep_mode: bool = True,
                 test_size: float = 0.2,
                 random_state: int = 42):
    """
    Ładuje pełny dataset dla danego ćwiczenia z plików .npy.

    Args:
        exercise: Nazwa ćwiczenia (np. "pushup")
        single_rep_mode: True = każdy plik to 1 rep (resize do SEQUENCE_LENGTH)
                         False = sliding window (legacy)
        test_size: Udział zbioru testowego
        random_state: Seed losowości

    Returns:
        train_loader, val_loader — PyTorch DataLoadery
    """
    exercise_dir = os.path.join(KEYPOINTS_DIR, exercise)
    labels_config = EXERCISE_CLASSES[exercise]["labels"]

    all_sequences = []
    all_labels = []

    # Sprawdź dostępne podkatalogi
    for label_id, label_name in labels_config.items():
        label_dir = os.path.join(exercise_dir, label_name)
        effective_label_id = label_id  # Nie nadpisuj pętliowej zmiennej (Python
                                       # binduje ją late — bugi w klauzurach)

        if not os.path.exists(label_dir):
            # Fallback: binarny format (correct/error) — używane gdy mamy
            # tylko dwie kategorie zamiast pełnej listy błędów.
            if label_id == 0 or label_name == "correct":
                label_dir = os.path.join(exercise_dir, "correct")
                effective_label_id = 0
            else:
                label_dir = os.path.join(exercise_dir, "error")
                # Wszystkie błędy mapujemy na jeden "zbiorczy" label — ale
                # załadujemy je tylko raz (pierwsza iteracja z label_id != 0).
                # Dla kolejnych iteracji z label_id > 1 pomijamy, żeby nie
                # duplikować tych samych plików N razy.
                if label_id > 1:
                    continue
                effective_label_id = 1
            if not os.path.exists(label_dir):
                continue

        npy_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])

        for npy_file in npy_files:
            keypoints = np.load(os.path.join(label_dir, npy_file))

            # Sprawdź czy plik nie jest pusty (Błąd MediaPipe w wycięciu sylwetki)
            if len(keypoints.shape) < 3 or keypoints.shape[0] == 0:
                print(f"  ⚠ Pomijanie uszkodzonego/pustego pliku: {npy_file}")
                continue

            if single_rep_mode:
                # Każdy plik = 1 rep → resize do SEQUENCE_LENGTH
                resized = resize_sequence(keypoints, SEQUENCE_LENGTH)
                all_sequences.append(resized)
                all_labels.append(effective_label_id)
            else:
                # Legacy: sliding window
                windows = create_sliding_windows(keypoints)
                all_sequences.extend(windows)
                all_labels.extend([effective_label_id] * len(windows))

        if npy_files:
            icon = "✅" if label_name == "correct" else "⚙️" if label_name == "setup" else "❌"
            print(f"  {icon} {label_name}: {len(npy_files)} plików")

    if len(all_sequences) == 0:
        raise FileNotFoundError(
            f"Brak plików .npy w {exercise_dir}. "
            f"Uruchom najpierw ekstrakcję keypointów!"
        )

    # Statystyki
    print(f"\n📊 Łącznie: {len(all_sequences)} próbek (single_rep={single_rep_mode})")
    for label_id, label_name in labels_config.items():
        count = all_labels.count(label_id)
        if count > 0:
            print(f"   {label_name}: {count}")

    # Podział train/val
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        all_sequences, all_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=all_labels,
    )

    train_dataset = ExerciseDataset(train_seqs, train_labels, augment=True)
    val_dataset = ExerciseDataset(val_seqs, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"📦 Train: {len(train_dataset)} próbek, Val: {len(val_dataset)} próbek\n")

    return train_loader, val_loader


if __name__ == "__main__":
    # Szybki test ładowania datasetu
    try:
        train_loader, val_loader = load_dataset()
        batch_x, batch_y = next(iter(train_loader))
        print(f"Batch X shape: {batch_x.shape}")  # (batch, 63, 30)
        print(f"Batch Y shape: {batch_y.shape}")   # (batch,)
        print(f"Labels in batch: {batch_y.tolist()}")
    except FileNotFoundError as e:
        print(f"⚠ {e}")
