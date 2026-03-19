"""
rep_counter.py — Moduł zliczania powtórzeń ćwiczeń.

Zlicza powtórzenia na podstawie analizy cykli kąta w stawach.
Obsługuje 3 ćwiczenia:
    - Squat:  kąt kolana (biodro-kolano-kostka)
    - Push-up: kąt łokcia (ramię-łokieć-nadgarstek)
    - Lunge:  kąt kolana przedniego (biodro-kolano-kostka)

Format danych: 21 keypointów × (x, y, confidence).
"""

import numpy as np
from model.angle_calculator import ExerciseAngleAnalyzer
from model.config import EXERCISE_CLASSES, ACTIVE_EXERCISE


class RepCounter:
    """
    Licznik powtórzeń oparty na cyklach kąta w stawach.

    Algorytm:
    1. Śledzi kąt kluczowego stawu w czasie
    2. Wykrywa przejścia: pozycja górna → dolna → górna = 1 powtórzenie
    3. Stosuje histerezę (progi up/down) dla odporności na szum
    """

    def __init__(self, exercise: str = ACTIVE_EXERCISE):
        self.exercise = exercise
        self.config = EXERCISE_CLASSES[exercise]
        self.rep_config = self.config["rep_phases"]
        self.landmarks = self.config["key_landmarks"]

        # Inicjalizuj analizator kątów
        self.angle_analyzer = ExerciseAngleAnalyzer(exercise)

        # Progi
        self.up_threshold = self.rep_config["up_threshold"]
        self.down_threshold = self.rep_config["down_threshold"]

        # Śledzony kąt
        self.tracking_joints = self.rep_config["angle_joint"]

        # Stan
        self.rep_count = 0
        self.phase = "up"
        self.angle_history = []

    def reset(self):
        """Resetuje licznik."""
        self.rep_count = 0
        self.phase = "up"
        self.angle_history = []

    def _get_tracking_angle(self, frame: np.ndarray) -> float:
        """
        Oblicza kąt stawu używanego do śledzenia powtórzeń.
        Uśrednia wartości z lewej i prawej strony.

        Args:
            frame: (21, 3) — współrzędne jednej klatki

        Returns:
            Średni kąt w stopniach
        """
        angles = []
        for side in ["left", "right"]:
            angle = self.angle_analyzer.compute_angle(
                frame, self.tracking_joints, side
            )
            if not np.isnan(angle):
                angles.append(angle)

        return np.mean(angles) if angles else 0.0

    def update(self, frame: np.ndarray) -> dict:
        """
        Aktualizuje licznik na podstawie nowej klatki.

        Args:
            frame: (21, 3) — współrzędne jednej klatki

        Returns:
            dict z aktualnym stanem:
            - rep_count: liczba powtórzeń
            - phase: aktualna faza ("up" / "down")
            - angle: aktualny kąt
            - new_rep: czy właśnie ukończono powtórzenie
        """
        angle = self._get_tracking_angle(frame)
        self.angle_history.append(angle)

        new_rep = False

        if self.phase == "up" and angle < self.down_threshold:
            self.phase = "down"
        elif self.phase == "down" and angle > self.up_threshold:
            self.phase = "up"
            self.rep_count += 1
            new_rep = True

        return {
            "rep_count": self.rep_count,
            "phase": self.phase,
            "angle": angle,
            "new_rep": new_rep,
        }

    def count_from_sequence(self, sequence: np.ndarray) -> dict:
        """
        Zlicza powtórzenia z pełnej sekwencji klatek.

        Args:
            sequence: (num_frames, 21, 3)

        Returns:
            dict z wynikami
        """
        self.reset()
        results = []

        for frame in sequence:
            result = self.update(frame)
            results.append(result)

        return {
            "exercise": self.exercise,
            "total_reps": self.rep_count,
            "angle_history": self.angle_history,
            "phase_changes": results,
        }


if __name__ == "__main__":
    # Test z syntetycznymi danymi — 3 ćwiczenia
    for exercise in ["squat", "pushup", "lunge"]:
        print(f"\n{'='*40}")
        print(f"TEST RepCounter: {exercise.upper()}")
        print(f"{'='*40}")

        counter = RepCounter(exercise)

        # Symuluj 5 powtórzeń
        fake_sequence = np.random.rand(150, 21, 3).astype(np.float32)
        fake_sequence[:, :, 2] = 0.9  # Wysokie confidence

        result = counter.count_from_sequence(fake_sequence)
        print(f"  Powtórzenia: {result['total_reps']}")
        print(f"  (Z losowymi danymi wynik będzie niedokładny)")
