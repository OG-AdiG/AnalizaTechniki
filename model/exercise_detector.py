"""
exercise_detector.py — Detekcja ćwiczenia z keypointów.

Dwa tryby:
1. HEURYSTYKA — szybka detekcja z kątów ciała (0 compute extra)
   Wystarczająca dla 2-3 ćwiczeń o wyraźnie różnej pozycji ciała.

2. MODEL — mały ExerciseClassifier (Conv1D → FC)
   Potrzebny dla 20+ ćwiczeń (bench press vs french press, deadlift vs rows).
   Rozmiar: ~50-100KB TFLite, ~10% kosztu modelu techniki.

Użycie:
    # Heurystyka
    detector = ExerciseDetector(mode="heuristic")
    exercise = detector.detect(keypoints_frame)  # "pushup" / "pullup" / "dips" / "unknown"

    # Model (po wytrenowaniu)
    detector = ExerciseDetector(mode="model", model_path="exercise_classifier.tflite")
    exercise = detector.detect_from_sequence(keypoints_30_frames)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import (
    EXERCISE_CLASSES, LANDMARK_INDEX, NUM_KEYPOINTS,
    INPUT_CHANNELS, SEQUENCE_LENGTH,
)
from model.angle_calculator import calculate_angle


# ============================================================
# TRYB 1: Heurystyka (0 compute extra)
# ============================================================

class HeuristicExerciseDetector:
    """
    Rozpoznaje ćwiczenie na podstawie kątów i pozycji ciała.
    Nie wymaga żadnego dodatkowego modelu — używa keypointów
    które i tak mamy z SimCC.

    Reguły:
    - Pompka: ciało horyzontalne (shoulder→hip→ankle ≈ 160-180°),
              nadgarstki niżej niż barki
    - Podciąganie: ciało wertykalne, ręce nad głową

    Koszt: ~5 operacji arytmetycznych per klatkę.
    """

    def __init__(self, vote_window: int = 30):
        """
        Args:
            vote_window: ile ostatnich klatek bierze do majority vote
        """
        self.vote_window = vote_window
        self.vote_history = []

    def detect(self, frame: np.ndarray) -> str:
        """
        Wykrywa ćwiczenie z jednej klatki keypointów.
        Używa majority vote z ostatnich N klatek.

        Args:
            frame: (21, 3) — keypointy jednej klatki

        Returns:
            nazwa ćwiczenia ("pushup", "pullup", "dips", "unknown")
        """
        # Wykrywanie z jednej klatki (surowe)
        raw_detection = self._detect_single_frame(frame)
        self.vote_history.append(raw_detection)

        # Przycinanie historii
        if len(self.vote_history) > self.vote_window:
            self.vote_history.pop(0)

        # Majority vote (ignoruj "unknown")
        votes = [v for v in self.vote_history if v != "unknown"]
        if not votes:
            return "unknown"

        # Najczęstszy wynik
        from collections import Counter
        counter = Counter(votes)
        return counter.most_common(1)[0][0]

    def _detect_single_frame(self, frame: np.ndarray) -> str:
        """Detekcja z jednej klatki (bez vote)."""
        if frame.shape[0] < NUM_KEYPOINTS:
            return "unknown"

        # Sprawdź confidence
        avg_conf = np.mean(frame[:, 2]) if frame.shape[1] >= 3 else 1.0
        if avg_conf < 0.3:
            return "unknown"

        # Keypoints
        l_shoulder = frame[LANDMARK_INDEX["left_shoulder"]]
        r_shoulder = frame[LANDMARK_INDEX["right_shoulder"]]
        l_hip = frame[LANDMARK_INDEX["left_hip"]]
        r_hip = frame[LANDMARK_INDEX["right_hip"]]
        l_ankle = frame[LANDMARK_INDEX["left_ankle"]]
        r_ankle = frame[LANDMARK_INDEX["right_ankle"]]
        l_wrist = frame[LANDMARK_INDEX["left_wrist"]]
        r_wrist = frame[LANDMARK_INDEX["right_wrist"]]

        # Kąt ciała: shoulder → hip → ankle (orientacja tułowia)
        body_angles = []
        for s, h, a in [(l_shoulder, l_hip, l_ankle),
                        (r_shoulder, r_hip, r_ankle)]:
            if s[2] > 0.3 and h[2] > 0.3 and a[2] > 0.3:
                angle = calculate_angle(s[:2], h[:2], a[:2])
                body_angles.append(angle)

        if not body_angles:
            return "unknown"

        body_angle = np.mean(body_angles)

        # Pozycja nadgarstków vs barków
        avg_wrist_y = np.mean([l_wrist[1], r_wrist[1]])
        avg_shoulder_y = np.mean([l_shoulder[1], r_shoulder[1]])
        avg_hip_y = np.mean([l_hip[1], r_hip[1]])

        # === REGUŁY DETEKCJI ===

        # POMPKA: ciało prawie proste (160-180°), nadgarstki na wysokości barków lub niżej
        # W układzie 0-1 (y=0 góra, y=1 dół): nadgarstki y ≈ barki y
        wrists_at_shoulder_level = abs(avg_wrist_y - avg_shoulder_y) < 0.15

        # PODCIĄGANIE: ciało wertykalne, nadgarstki POWYŻEJ barków (y mniejszy)
        wrists_above_shoulders = avg_wrist_y < avg_shoulder_y - 0.1

        # DIPY: ciało wertykalne, nadgarstki PONIŻEJ barków i w okolicach bioder
        wrists_below_shoulders = avg_wrist_y > avg_shoulder_y + 0.1
        wrists_near_hips = abs(avg_wrist_y - avg_hip_y) < 0.25

        if body_angle > 150 and wrists_at_shoulder_level:
            return "pushup"
        elif wrists_above_shoulders and avg_shoulder_y < avg_hip_y:
            # W układzie 0-1: y=0 to góra, więc shoulder_y < hip_y = barki wyżej
            return "pullup"
        elif wrists_below_shoulders and wrists_near_hips and avg_shoulder_y < avg_hip_y:
            return "dips"

        return "unknown"

    def reset(self):
        """Resetuje historię głosowania."""
        self.vote_history = []


# ============================================================
# TRYB 2: Model (dla 20+ ćwiczeń)
# ============================================================

class ExerciseClassifierModel:
    """
    Mały model klasyfikatora ćwiczenia.
    Architektura: 2× Conv1D → GAP → FC(num_exercises)
    Rozmiar: ~50-100KB TFLite

    Na razie: ARCHITEKTURA GOTOWA, model do wytrenowania
    gdy pojawią się dane z więcej niż 2 ćwiczeń.
    """

    def __init__(self, model_path: str = None, exercise_names: list = None):
        self.model_path = model_path
        self.exercise_names = exercise_names or list(EXERCISE_CLASSES.keys())
        self.interpreter = None

        if model_path is not None and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Ładuje model TFLite."""
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect_from_sequence(self, sequence: np.ndarray) -> str:
        """
        Rozpoznaje ćwiczenie z sekwencji 30 klatek.

        Args:
            sequence: (30, 21, 3) — sekwencja keypointów

        Returns:
            nazwa ćwiczenia
        """
        if self.interpreter is None:
            return "unknown"

        # (30, 21, 3) → (30, 63) → (63, 30) → (1, 63, 30)
        seq_flat = sequence.reshape(sequence.shape[0], -1)
        seq_t = seq_flat.T
        model_input = seq_t.reshape(1, INPUT_CHANNELS, SEQUENCE_LENGTH)

        # Sprawdź format wejścia
        expected_shape = self.input_details[0]["shape"]
        if (expected_shape[1] == SEQUENCE_LENGTH
                and expected_shape[2] == INPUT_CHANNELS):
            model_input = np.transpose(model_input, (0, 2, 1))

        expected_dtype = self.input_details[0]["dtype"]
        model_input = model_input.astype(expected_dtype)
        self.interpreter.set_tensor(self.input_details[0]["index"], model_input)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0].astype(np.float32)

        pred_idx = int(np.argmax(logits))
        if pred_idx < len(self.exercise_names):
            return self.exercise_names[pred_idx]
        return "unknown"


# ============================================================
# WRAPPER — automatycznie wybiera tryb
# ============================================================

class ExerciseDetector:
    """
    Wrapper: automatycznie używa heurystyki lub modelu.

    Jeśli model_path podany i plik istnieje → użyj modelu.
    W przeciwnym razie → heurystyka.
    """

    def __init__(self, mode: str = "auto", model_path: str = None,
                 vote_window: int = 30):
        """
        Args:
            mode: "heuristic", "model", lub "auto"
            model_path: ścieżka do modelu TFLite exercise classifier
            vote_window: okno majority vote (tylko heurystyka)
        """
        if mode == "auto":
            if model_path and os.path.exists(model_path):
                mode = "model"
            else:
                mode = "heuristic"

        self.mode = mode

        if mode == "heuristic":
            self._detector = HeuristicExerciseDetector(vote_window)
        elif mode == "model":
            self._detector = ExerciseClassifierModel(model_path)
        else:
            raise ValueError(f"Nieznany tryb: {mode}")

    def detect(self, frame: np.ndarray) -> str:
        """Wykrywa ćwiczenie z jednej klatki (heurystyka)."""
        if isinstance(self._detector, HeuristicExerciseDetector):
            return self._detector.detect(frame)
        return "unknown"

    def detect_from_sequence(self, sequence: np.ndarray) -> str:
        """Wykrywa ćwiczenie z sekwencji 30 klatek (model)."""
        if isinstance(self._detector, ExerciseClassifierModel):
            return self._detector.detect_from_sequence(sequence)
        # Fallback: heurystyka na ostatniej klatce
        return self.detect(sequence[-1])

    def reset(self):
        """Resetuje stan."""
        if hasattr(self._detector, "reset"):
            self._detector.reset()


# ============================================================
# Architektura PyTorch (do trenowania)
# ============================================================

def get_exercise_classifier_model():
    """
    Zwraca architekturę PyTorch modelu klasyfikatora ćwiczeń.
    Do wytrenowania gdy będzie więcej niż 2 ćwiczenia.

    Rozmiar: ~50KB (float32), ~25KB (float16)
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("⚠ PyTorch nie zainstalowany — architektura niedostępna")
        return None

    class ExerciseClassifier(nn.Module):
        """Mały klasyfikator ćwiczenia: (batch, 63, 30) → (batch, num_exercises)"""

        def __init__(self, num_exercises=len(EXERCISE_CLASSES)):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(INPUT_CHANNELS, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),

                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(32, num_exercises),
            )

        def forward(self, x):
            x = self.conv(x)         # (batch, 64, 7)
            x = self.gap(x)          # (batch, 64, 1)
            x = x.squeeze(-1)       # (batch, 64)
            x = self.fc(x)          # (batch, num_exercises)
            return x

    return ExerciseClassifier


if __name__ == "__main__":
    print("=" * 60)
    print("TEST ExerciseDetector")
    print("=" * 60)

    # Test heurystyki
    detector = ExerciseDetector(mode="heuristic")
    print(f"  Tryb: {detector.mode}")

    # Symuluj klatkę pompki — ciało horyzontalne
    pushup_frame = np.zeros((21, 3), dtype=np.float32)
    pushup_frame[:, 2] = 0.9  # confidence

    # Ustawiamy klatkę pompki (barki, biodra, kostki na poziomie)
    pushup_frame[3] = [0.3, 0.3, 0.9]   # left_shoulder
    pushup_frame[4] = [0.7, 0.3, 0.9]   # right_shoulder
    pushup_frame[7] = [0.2, 0.3, 0.9]   # left_wrist (na poziomie barków)
    pushup_frame[8] = [0.8, 0.3, 0.9]   # right_wrist
    pushup_frame[9] = [0.4, 0.5, 0.9]   # left_hip
    pushup_frame[10] = [0.6, 0.5, 0.9]  # right_hip
    pushup_frame[13] = [0.5, 0.7, 0.9]  # left_ankle
    pushup_frame[14] = [0.5, 0.7, 0.9]  # right_ankle

    # Podaj kilka klatek żeby voting miał dane
    for _ in range(35):
        result = detector.detect(pushup_frame)
    print(f"  Pompka (hor. ciało): {result}")

    detector.reset()

    # Symuluj podciąganie — ręce nad głową
    pullup_frame = np.zeros((21, 3), dtype=np.float32)
    pullup_frame[:, 2] = 0.9

    pullup_frame[3] = [0.4, 0.3, 0.9]   # left_shoulder
    pullup_frame[4] = [0.6, 0.3, 0.9]   # right_shoulder
    pullup_frame[7] = [0.35, 0.1, 0.9]  # left_wrist (nad głową!)
    pullup_frame[8] = [0.65, 0.1, 0.9]  # right_wrist
    pullup_frame[9] = [0.45, 0.6, 0.9]  # left_hip
    pullup_frame[10] = [0.55, 0.6, 0.9] # right_hip
    pullup_frame[13] = [0.45, 0.9, 0.9] # left_ankle
    pullup_frame[14] = [0.55, 0.9, 0.9] # right_ankle

    for _ in range(35):
        result = detector.detect(pullup_frame)
    print(f"  Podciąganie (vert.): {result}")

    # Test architektury modelu
    ExerciseClassifier = get_exercise_classifier_model()
    if ExerciseClassifier:
        import torch
        model = ExerciseClassifier()
        params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model ExerciseClassifier:")
        print(f"    Parametry: {params:,}")
        print(f"    Rozmiar: ~{params * 4 / 1024:.0f}KB (float32)")
        dummy = torch.randn(2, INPUT_CHANNELS, SEQUENCE_LENGTH)
        out = model(dummy)
        print(f"    Input:  {dummy.shape}")
        print(f"    Output: {out.shape}")
        print(f"  ✅ Architektura OK")
