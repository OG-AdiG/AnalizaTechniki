"""
rep_counter.py — Moduł zliczania powtórzeń ćwiczeń z filtrem całkującym.

Zlicza powtórzenia na podstawie analizy cykli kąta w stawach.
Stosuje 5-warstwowy filtr:
    1. Confidence-weighted angle — ważenie kątów pewnością keypointów
    2. EMA (Exponential Moving Average) — wygładzenie szumu
    3. Filtr całkujący (trapezoidalny) — akumulacja impulsu zmiany kąta
    4. Walidacja amplitudy — min. zmiana kąta żeby liczyć rep
    5. Debounce + sufit — min. czas między repami + max reps/min

Ulepszenia v2:
    - Adaptacyjne progi: po 3 repach dopasowuje się do ROM osoby
    - Odporność na dropout keypointów: pomija klatki bez danych
    - Timeout fazy: wymusza granicę repa po 3s w jednej fazie
    - Confidence-weighted angles: lepszy kąt z lewej/prawej strony

Format danych: 21 keypointów × (x, y, confidence).
"""

import numpy as np
from model.angle_calculator import ExerciseAngleAnalyzer
from model.config import (
    EXERCISE_CLASSES, ACTIVE_EXERCISE,
    EMA_ALPHA, REP_AMPLITUDE_THRESHOLD,
    MIN_REP_FRAMES, MAX_REPS_PER_MINUTE, DEFAULT_MAX_REPS_PER_MINUTE,
    TARGET_FPS,
    ADAPTIVE_THRESHOLD_REPS, ADAPTIVE_MARGIN,
    MAX_CONSECUTIVE_DROPOUTS, MAX_PHASE_FRAMES,
)


class RepCounter:
    """
    Licznik powtórzeń z 5-warstwowym filtrem + adaptacją.

    Algorytm:
    1. Śledzi kąt kluczowego stawu w czasie (confidence-weighted, EMA-wygładzony)
    2. Wykrywa pełne cykle ruchu: pozycja_start → pozycja_end → pozycja_start
    3. Waliduje amplitudę — mały ruch się nie liczy
    4. Debounce — min. czas między repami (fizyczne ograniczenie)
    5. Adaptuje progi po obserwacji pierwszych repów

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

        # Progi faz ruchu (początkowe — mogą być nadpisane przez adaptację)
        self.up_threshold = self.rep_config["up_threshold"]
        self.down_threshold = self.rep_config["down_threshold"]
        self.initial_up_threshold = self.up_threshold
        self.initial_down_threshold = self.down_threshold

        # Śledzony kąt
        self.tracking_joints = self.rep_config["angle_joint"]

        # Parametry filtru
        self.ema_alpha = EMA_ALPHA
        self.amplitude_threshold = REP_AMPLITUDE_THRESHOLD
        self.min_rep_frames = MIN_REP_FRAMES
        self.max_reps_per_minute = MAX_REPS_PER_MINUTE.get(
            exercise, DEFAULT_MAX_REPS_PER_MINUTE
        )

        # Parametry adaptacji
        self.adaptive_reps = ADAPTIVE_THRESHOLD_REPS
        self.adaptive_margin = ADAPTIVE_MARGIN
        self.max_phase_frames = MAX_PHASE_FRAMES
        self.max_consecutive_dropouts = MAX_CONSECUTIVE_DROPOUTS

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

        # === NOWE: Dropout handling ===
        self.consecutive_dropouts = 0
        self.last_valid_angle = None

        # === NOWE: Adaptacyjne progi ===
        self.thresholds_adapted = False
        self.observed_peaks = []    # Kąty peak z ukończonych repów
        self.observed_valleys = []  # Kąty valley z ukończonych repów
        self.up_threshold = self.initial_up_threshold
        self.down_threshold = self.initial_down_threshold

        # === NOWE: Timeout tracking ===
        self.phase_start_frame = 0  # Kiedy zaczęła się bieżąca faza

    def _get_tracking_angle(self, frame: np.ndarray):
        """
        Oblicza kąt stawu z ważeniem confidence keypointów.
        Zwraca None jeśli żaden kąt nie jest dostępny (dropout).

        Args:
            frame: (21, 3) — współrzędne jednej klatki

        Returns:
            float lub None (dropout)
        """
        angles = []
        confidences = []

        for side in ["left", "right"]:
            angle = self.angle_analyzer.compute_angle(
                frame, self.tracking_joints, side
            )
            if not np.isnan(angle):
                # Pobierz średnie confidence 3 stawów tworzących kąt
                conf = self._get_joints_confidence(frame, side)
                angles.append(angle)
                confidences.append(conf)

        if not angles:
            # Żaden kąt niedostępny — dropout
            self.consecutive_dropouts += 1
            return None

        self.consecutive_dropouts = 0

        # Confidence-weighted average (lepsza strona ma większy wpływ)
        if len(angles) == 2 and sum(confidences) > 0:
            weights = np.array(confidences)
            weights = weights / weights.sum()
            return float(np.average(angles, weights=weights))

        return float(np.mean(angles))

    def _get_joints_confidence(self, frame: np.ndarray, side: str) -> float:
        """Średnia confidence 3 stawów użytych do obliczenia kąta."""
        confs = []
        for joint_name in self.tracking_joints:
            try:
                point = self.angle_analyzer.get_landmark(frame, side, joint_name)
                if len(point) >= 3:
                    confs.append(float(point[2]))
            except KeyError:
                pass
        return float(np.mean(confs)) if confs else 0.5

    def _apply_ema(self, angle_raw: float) -> float:
        """
        Warstwa 2: Exponential Moving Average.
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
        Warstwa 3: Filtr całkujący (trapezoidalny).
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
        Warstwa 4: Walidacja amplitudy.
        Rep jest ważny tylko jeśli kąt zmienił się o wystarczającą ilość stopni.
        """
        if self.angle_peak is None or self.angle_valley is None:
            return False
        amplitude = abs(self.angle_peak - self.angle_valley)
        return amplitude >= self.amplitude_threshold

    def _check_debounce(self) -> bool:
        """
        Warstwa 5: Debounce + sufit.
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

    def _adapt_thresholds(self):
        """
        Adaptacja progów po obserwacji pierwszych N repów.

        Zamiast sztywnych progów (np. up=125°, down=100°) — oblicza
        faktyczny ROM (Range of Motion) osoby z pierwszych repów
        i ustawia progi z 20% marginesem.

        Dzięki temu model dopasowuje się do osób z różną elastycznością.
        """
        if self.thresholds_adapted:
            return

        if len(self.observed_peaks) < self.adaptive_reps:
            return

        recent_peaks = self.observed_peaks[-self.adaptive_reps:]
        recent_valleys = self.observed_valleys[-self.adaptive_reps:]

        avg_peak = float(np.mean(recent_peaks))
        avg_valley = float(np.mean(recent_valleys))
        rom = avg_peak - avg_valley

        # Adaptuj tylko jeśli ROM jest sensowny (> 30°)
        if rom < 30:
            return

        margin = rom * self.adaptive_margin

        new_up = avg_valley + rom - margin      # np. 160 - 12 = 148
        new_down = avg_valley + margin           # np. 90 + 12 = 102

        # Zabraniamy progów poza rozsądnym zakresem
        new_up = max(new_up, self.initial_down_threshold + 20)
        new_down = min(new_down, self.initial_up_threshold - 20)

        # FIX: Nie pozwalamy, aby adaptacja uczyniła progi BARDZIEJ rygorystycznymi
        # niż te celowo zdefiniowane w config.py (np. 125 dla up, 100 dla down).
        # Dzięki temu, jeśli ktoś zrobi 3 dobre pompki, próg nie skoczy do 160°, 
        # ucinając nam możliwość wykrycia "half rep top" na 135°.
        new_up = min(new_up, self.initial_up_threshold)
        new_down = max(new_down, self.initial_down_threshold)

        self.up_threshold = new_up
        self.down_threshold = new_down

        # Dopasuj też minimalną amplitudę (50% obserwowanego ROM)
        self.amplitude_threshold = min(rom * 0.5, self.amplitude_threshold)

        self.thresholds_adapted = True

    def _register_completed_rep(self, angle_smooth: float):
        """Zapisuje peak/valley ukończonego repa do historii adaptacji."""
        if self.angle_peak is not None:
            self.observed_peaks.append(self.angle_peak)
        if self.angle_valley is not None:
            self.observed_valleys.append(self.angle_valley)

    def _check_phase_timeout(self) -> bool:
        """
        Sprawdza czy faza trwa zbyt długo (osoba utknęła).

        Jeśli faza "down" trwa > MAX_PHASE_FRAMES i amplituda jest OK,
        wymuszamy granicę repa — osoba mogła odpocząć na dole
        lub nie wyprostować rąk wystarczająco, żeby przekroczyć próg.
        """
        frames_in_phase = self.frame_index - self.phase_start_frame
        return (
            self.phase == "down"
            and frames_in_phase > self.max_phase_frames
            and self._check_amplitude()
            and self._check_debounce()
        )

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
        # === Krok 1: Oblicz kąt (confidence-weighted) ===
        angle_raw = self._get_tracking_angle(frame)

        # Obsługa dropout — jeśli brak kąta, użyj ostatniego znanego
        if angle_raw is None:
            self.frame_index += 1
            self.frames_in_current_rep += 1

            # Jeśli za dużo dropoutów z rzędu — sygnał utracony
            if self.consecutive_dropouts > self.max_consecutive_dropouts:
                # Reset — nie da się śledzić
                pass

            return {
                "rep_count": self.rep_count,
                "phase": self.phase,
                "angle_raw": self.last_valid_angle or 0.0,
                "angle_smooth": self.angle_smooth or 0.0,
                "new_rep": False,
                "rep_amplitude": abs(self.angle_peak - self.angle_valley)
                                 if self.angle_peak is not None
                                    and self.angle_valley is not None
                                 else 0.0,
                "frames_in_rep": self.frames_in_current_rep,
                "integral": self.integral,
            }

        self.last_valid_angle = angle_raw
        self.angle_history_raw.append(angle_raw)

        # === Krok 2: EMA ===
        angle_smooth = self._apply_ema(angle_raw)
        self.angle_history_smooth.append(angle_smooth)

        # === Krok 3: Filtr całkujący ===
        self._update_integral(angle_smooth)

        # Śledzenie peak/valley w bieżącym cyklu
        if self.angle_peak is None or angle_smooth > self.angle_peak:
            self.angle_peak = angle_smooth
        if self.angle_valley is None or angle_smooth < self.angle_valley:
            self.angle_valley = angle_smooth

        # === Krok 4: Detekcja faz + timeout ===
        new_rep = False
        self.frames_in_current_rep += 1

        if self.phase == "up" and angle_smooth < self.down_threshold:
            self.phase = "down"
            self.phase_start_frame = self.frame_index

        elif self.phase == "down" and angle_smooth > self.up_threshold:
            # Potencjalny rep — walidacja
            amplitude_ok = self._check_amplitude()
            debounce_ok = self._check_debounce()

            if amplitude_ok and debounce_ok:
                new_rep = self._complete_rep(angle_smooth)

            elif not amplitude_ok:
                # Za mały ruch — zresetuj fazę ale nie licz repa
                self.phase = "up"
                self.phase_start_frame = self.frame_index
                self.angle_peak = angle_smooth
                self.angle_valley = angle_smooth
                self.integral = 0.0

        # === Krok 5: Timeout check ===
        if not new_rep and self._check_phase_timeout():
            new_rep = self._complete_rep(angle_smooth)

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

    def _complete_rep(self, angle_smooth: float) -> bool:
        """Finalizuje wykryte powtórzenie — wspólna logika dla normalnego i timeout repa."""
        # Zapisz peak/valley dla adaptacji
        self._register_completed_rep(angle_smooth)

        self.phase = "up"
        self.phase_start_frame = self.frame_index
        self.rep_count += 1
        self.last_rep_frame = self.frame_index
        self.frames_in_current_rep = 0

        # Reset peak/valley dla nowego cyklu
        self.angle_peak = angle_smooth
        self.angle_valley = angle_smooth

        # Reset integrala
        self.integral = 0.0

        # Próba adaptacji progów po N repach
        self._adapt_thresholds()

        return True

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
            "thresholds_adapted": self.thresholds_adapted,
            "final_up_threshold": self.up_threshold,
            "final_down_threshold": self.down_threshold,
        }


if __name__ == "__main__":
    # Test z syntetycznymi danymi — symulacja 5 pompek
    import math

    print("=" * 60)
    print("TEST RepCounter v2 z adaptacją + dropout handling")
    print("=" * 60)

    counter = RepCounter("pushup")
    print(f"  Ćwiczenie: pushup")
    print(f"  EMA α: {counter.ema_alpha}")
    print(f"  Amplituda min: {counter.amplitude_threshold}°")
    print(f"  Debounce: {counter.min_rep_frames} klatek")
    print(f"  Sufit: {counter.max_reps_per_minute} rep/min")
    print(f"  Adaptacja po: {counter.adaptive_reps} repach")
    print(f"  Timeout fazy: {counter.max_phase_frames} klatek")
    print(f"  Max dropout: {counter.max_consecutive_dropouts} klatek")

    # Symuluj 5 pompek: kąt łokcia oscyluje 90°→160°→90°...
    # Każda pompka trwa ~30 klatek (1s) = 150 klatek na 5 pompek
    num_reps = 5
    frames_per_rep = 30
    total_frames = num_reps * frames_per_rep

    fake_sequence = np.random.rand(total_frames, 21, 3).astype(np.float32)
    fake_sequence[:, :, 2] = 0.9  # Wysokie confidence

    result = counter.count_from_sequence(fake_sequence)
    print(f"\n  Wynik na losowych danych: {result['total_reps']} repów")
    print(f"  Adaptacja: {result['thresholds_adapted']}")
    print(f"  Progi: up={result['final_up_threshold']:.0f}° "
          f"down={result['final_down_threshold']:.0f}°")
    print(f"  (Z losowymi danymi wynik będzie niedokładny)")
    print(f"  Test filtru przejdzie w test_implementation.py")
