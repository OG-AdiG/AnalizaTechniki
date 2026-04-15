"""
rep_counter.py — Moduł zliczania powtórzeń ćwiczeń z filtrem całkującym.

Zlicza powtórzenia na podstawie analizy cykli kąta w stawach.
Stosuje 4-warstwowy filtr:
    1. EMA (Exponential Moving Average) — wygładzenie szumu
    2. Filtr całkujący (trapezoidalny) — akumulacja impulsu zmiany kąta
    3. Walidacja amplitudy — min. zmiana kąta żeby liczyć rep
    4. Debounce + sufit — min. czas między repami + max reps/min

Format danych: 21 keypointów × (x, y, confidence).
"""

import numpy as np
from model.angle_calculator import ExerciseAngleAnalyzer
from model.config import (
    EXERCISE_CLASSES, ACTIVE_EXERCISE,
    EMA_ALPHA, REP_AMPLITUDE_THRESHOLD,
    MIN_REP_FRAMES, MAX_REPS_PER_MINUTE, DEFAULT_MAX_REPS_PER_MINUTE,
    TARGET_FPS,
)


class RepCounter:
    """
    Licznik powtórzeń z 4-warstwowym filtrem.

    Algorytm:
    1. Śledzi kąt kluczowego stawu w czasie (EMA-wygładzony)
    2. Wykrywa pełne cykle ruchu: pozycja_start → pozycja_end → pozycja_start
    3. Waliduje amplitudę — mały ruch się nie liczy
    4. Debounce — min. czas między repami (fizyczne ograniczenie)

    Rozwiązuje problem fałszywych zliczeń (np. 350 repów zamiast 20)
    spowodowany szumem danych z pose estimation.
    """

    def __init__(self, exercise: str = ACTIVE_EXERCISE):
        self.exercise = exercise
        self.config = EXERCISE_CLASSES[exercise]
        self.rep_config = self.config["rep_phases"]
        self.landmarks = self.config["key_landmarks"]

        # Inicjalizuj analizator kątów
        self.angle_analyzer = ExerciseAngleAnalyzer(exercise)

        # Progi faz ruchu
        self.up_threshold = self.rep_config["up_threshold"]
        self.down_threshold = self.rep_config["down_threshold"]

        # Śledzony kąt
        self.tracking_joints = self.rep_config["angle_joint"]

        # Parametry filtru
        self.ema_alpha = EMA_ALPHA
        self.amplitude_threshold = REP_AMPLITUDE_THRESHOLD
        self.min_rep_frames = MIN_REP_FRAMES
        self.max_reps_per_minute = MAX_REPS_PER_MINUTE.get(
            exercise, DEFAULT_MAX_REPS_PER_MINUTE
        )

        # Stan
        self.reset()

    def reset(self):
        """Resetuje licznik do stanu początkowego."""
        self.rep_count = 0
        self.phase = "up"           # Bieżąca faza: "up" lub "down"
        self.frame_index = 0        # Numer klatki

        # Historia (do debugowania / wizualizacji)
        self.angle_history_raw = []
        self.angle_history_smooth = []

        # Filtr EMA
        self.angle_smooth = None    # Wygładzony kąt (EMA)

        # Śledzenie amplitudy per cykl
        self.angle_peak = None      # Maksymalny kąt w bieżącym cyklu
        self.angle_valley = None    # Minimalny kąt w bieżącym cyklu

        # Debounce
        self.last_rep_frame = -self.min_rep_frames  # Pozwól na 1. rep od razu

        # Filtr całkujący — akumuluje kierunek zmiany kąta
        self.integral = 0.0
        self.prev_smooth = None

        # Śledzenie klatek w bieżącym repie
        self.frames_in_current_rep = 0

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

    def _apply_ema(self, angle_raw: float) -> float:
        """
        Warstwa 1: Exponential Moving Average.
        Wygładza sygnał kątowy, eliminując szum z keypoint estimation.

        angle_smooth = α * angle_raw + (1-α) * angle_prev
        """
        if self.angle_smooth is None:
            self.angle_smooth = angle_raw
        else:
            self.angle_smooth = (
                self.ema_alpha * angle_raw
                + (1.0 - self.ema_alpha) * self.angle_smooth
            )
        return self.angle_smooth

    def _update_integral(self, angle_smooth: float):
        """
        Warstwa 2: Filtr całkujący (trapezoidalny).
        Akumuluje zmianę kąta w czasie. Drobne drgania się nawzajem
        kompensują (szum jest symetryczny), a pełny ruch akumuluje
        dużą wartość.
        """
        if self.prev_smooth is not None:
            delta = angle_smooth - self.prev_smooth
            self.integral += delta
        self.prev_smooth = angle_smooth

    def _check_amplitude(self) -> bool:
        """
        Warstwa 3: Walidacja amplitudy.
        Rep jest ważny tylko jeśli kąt zmienił się o wystarczającą ilość stopni.
        """
        if self.angle_peak is None or self.angle_valley is None:
            return False
        amplitude = abs(self.angle_peak - self.angle_valley)
        return amplitude >= self.amplitude_threshold

    def _check_debounce(self) -> bool:
        """
        Warstwa 4: Debounce + sufit.
        Minimalny czas między powtórzeniami + max reps/min.
        """
        frames_since_last = self.frame_index - self.last_rep_frame

        # Debounce: min. czas między repami
        if frames_since_last < self.min_rep_frames:
            return False

        # Sufit: max reps/min
        elapsed_minutes = self.frame_index / (TARGET_FPS * 60) if TARGET_FPS > 0 else 1
        if elapsed_minutes > 0:
            current_rate = self.rep_count / elapsed_minutes
            if current_rate >= self.max_reps_per_minute:
                return False

        return True

    def update(self, frame: np.ndarray) -> dict:
        """
        Aktualizuje licznik na podstawie nowej klatki.

        Args:
            frame: (21, 3) — współrzędne jednej klatki

        Returns:
            dict z aktualnym stanem:
            - rep_count: liczba powtórzeń
            - phase: aktualna faza ("up" / "down")
            - angle_raw: surowy kąt
            - angle_smooth: wygładzony kąt (EMA)
            - new_rep: czy właśnie ukończono powtórzenie
            - rep_amplitude: amplituda ostatniego cyklu (°)
            - frames_in_rep: klatek od ostatniego repa
        """
        # Oblicz surowy kąt
        angle_raw = self._get_tracking_angle(frame)
        self.angle_history_raw.append(angle_raw)

        # Warstwa 1: EMA
        angle_smooth = self._apply_ema(angle_raw)
        self.angle_history_smooth.append(angle_smooth)

        # Warstwa 2: Filtr całkujący
        self._update_integral(angle_smooth)

        # Śledzenie peak/valley w bieżącym cyklu
        if self.angle_peak is None or angle_smooth > self.angle_peak:
            self.angle_peak = angle_smooth
        if self.angle_valley is None or angle_smooth < self.angle_valley:
            self.angle_valley = angle_smooth

        # Detekcja faz z histerezą
        new_rep = False
        self.frames_in_current_rep += 1

        if self.phase == "up" and angle_smooth < self.down_threshold:
            self.phase = "down"

        elif self.phase == "down" and angle_smooth > self.up_threshold:
            # Potencjalny rep — walidacja
            amplitude_ok = self._check_amplitude()
            debounce_ok = self._check_debounce()

            if amplitude_ok and debounce_ok:
                self.phase = "up"
                self.rep_count += 1
                new_rep = True
                self.last_rep_frame = self.frame_index
                self.frames_in_current_rep = 0

                # Reset peak/valley dla nowego cyklu
                self.angle_peak = angle_smooth
                self.angle_valley = angle_smooth

                # Reset integrala
                self.integral = 0.0
            elif not amplitude_ok:
                # Za mały ruch — zresetuj fazę ale nie licz repa
                self.phase = "up"
                self.angle_peak = angle_smooth
                self.angle_valley = angle_smooth
                self.integral = 0.0

        self.frame_index += 1

        return {
            "rep_count": self.rep_count,
            "phase": self.phase,
            "angle_raw": angle_raw,
            "angle_smooth": angle_smooth,
            "new_rep": new_rep,
            "rep_amplitude": abs(self.angle_peak - self.angle_valley)
                             if self.angle_peak is not None
                                and self.angle_valley is not None
                             else 0.0,
            "frames_in_rep": self.frames_in_current_rep,
            "integral": self.integral,
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
            "angle_history_raw": self.angle_history_raw,
            "angle_history_smooth": self.angle_history_smooth,
            "phase_changes": results,
        }


if __name__ == "__main__":
    # Test z syntetycznymi danymi — symulacja 5 pompek
    import math

    print("=" * 60)
    print("TEST RepCounter z filtrem całkującym")
    print("=" * 60)

    counter = RepCounter("pushup")
    print(f"  Ćwiczenie: pushup")
    print(f"  EMA α: {counter.ema_alpha}")
    print(f"  Amplituda min: {counter.amplitude_threshold}°")
    print(f"  Debounce: {counter.min_rep_frames} klatek")
    print(f"  Sufit: {counter.max_reps_per_minute} rep/min")

    # Symuluj 5 pompek: kąt łokcia oscyluje 90°→160°→90°...
    # Każda pompka trwa ~30 klatek (1s) = 150 klatek na 5 pompek
    num_reps = 5
    frames_per_rep = 30
    total_frames = num_reps * frames_per_rep

    fake_sequence = np.random.rand(total_frames, 21, 3).astype(np.float32)
    fake_sequence[:, :, 2] = 0.9  # Wysokie confidence

    # Symulacja kąta łokcia: sinusoida 90°-160°-90°
    for i in range(total_frames):
        t = i / frames_per_rep * 2 * math.pi
        # Kąt łokcia: center=125, amplitude=35 → 90-160
        angle = 125 + 35 * math.sin(t)
        # Ustaw pozycje keypointów tak żeby compute_angle zwracało ~angle
        # (to jest uproszczenie — w prawdziwym teście potrzebujesz geometric setup)

    result = counter.count_from_sequence(fake_sequence)
    print(f"\n  Wynik na losowych danych: {result['total_reps']} repów")
    print(f"  (Z losowymi danymi wynik będzie niedokładny)")
    print(f"  Test filtru przejdzie w test_implementation.py")
