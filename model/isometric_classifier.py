"""
isometric_classifier.py — Klasyfikacja techniki dla ćwiczeń IZOMETRYCZNYCH.

Zamiast liczyć powtórzenia (jak RepClassifier), liczy CZAS trwania
w każdej klasie techniki. Używa sliding window do klasyfikacji
co 0.5 sekundy.

Flow:
    1. Bufor zbiera klatki (sliding window)
    2. Co ISOMETRIC_WINDOW_STRIDE klatek → klasyfikacja okna
    3. Wynik dodawany do timeline'u: [(klasa, czas_start, czas_end), ...]
    4. Summary: "Plank 45s: 30s correct, 10s hips_low, 5s hips_high"

Użycie:
    classifier = IsometricClassifier("plank", "saved_models/plank_best.tflite")
    for frame in video_frames:
        result = classifier.process_frame(keypoints_21)
        if result is not None:
            print(f"[{result['time']:.1f}s] {result['class_name']} ({result['confidence']:.0%})")
    print(classifier.get_summary())
"""

import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import (
    EXERCISE_CLASSES, ACTIVE_EXERCISE,
    SEQUENCE_LENGTH, INPUT_CHANNELS,
    TARGET_FPS,
    ISOMETRIC_WINDOW_SIZE, ISOMETRIC_WINDOW_STRIDE,
    ISOMETRIC_MIN_CONFIDENCE,
)
from data_pipeline.extract_keypoints import normalize_keypoints, filter_low_confidence


class IsometricClassifier:
    """
    Klasyfikator techniki dla ćwiczeń izometrycznych (plank, hollow body, l-sit).

    Zamiast wykrywać granice repów, używa sliding window co 0.5s
    i klasyfikuje każde okno osobno. Wynik to timeline z segmentami
    czasowymi per klasa.
    """

    def __init__(self, exercise: str = ACTIVE_EXERCISE,
                 model_path: str = None,
                 model_type: str = "tflite"):
        """
        Args:
            exercise: nazwa ćwiczenia (np. "plank")
            model_path: ścieżka do modelu TFLite (.tflite) lub PyTorch (.pt)
            model_type: "tflite" lub "pytorch"
        """
        self.exercise = exercise
        self.config = EXERCISE_CLASSES[exercise]
        self.labels = self.config["labels"]
        self.num_classes = self.config["num_classes"]

        # Parametry sliding window
        self.window_size = ISOMETRIC_WINDOW_SIZE
        self.window_stride = ISOMETRIC_WINDOW_STRIDE
        self.min_confidence = ISOMETRIC_MIN_CONFIDENCE
        self.fps = TARGET_FPS

        # Bufor klatek
        self.frame_buffer = []
        self.total_frames = 0
        self.frames_since_last_classify = 0

        # Timeline — lista wyników klasyfikacji
        # Każdy element: {"time": float, "class_name": str, "confidence": float, ...}
        self.timeline = []

        # Segmenty — ciągłe bloki tej samej klasy
        # [(class_name, start_time, end_time), ...]
        self.segments = []
        self._current_segment_class = None
        self._current_segment_start = 0.0

        # Czy ćwiczenie się rozpoczęło (pierwszy non-setup)
        self.exercise_started = False

        # Model
        self.model_type = model_type
        self.model = None
        self.interpreter = None

        if model_path is not None:
            self._load_model(model_path, model_type)

    def _load_model(self, model_path: str, model_type: str):
        """Ładuje model do inferencji."""
        if model_type == "tflite":
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        elif model_type == "pytorch":
            import torch
            from model.model import TemporalCNN
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self.model = TemporalCNN(
                input_channels=checkpoint.get("input_channels", INPUT_CHANNELS),
                num_classes=checkpoint.get("num_classes", self.num_classes),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        else:
            raise ValueError(f"Nieznany typ modelu: {model_type}")

    def process_frame(self, keypoints_21: np.ndarray) -> dict:
        """
        Przetwarza jedną klatkę keypointów.
        Zwraca wynik klasyfikacji co WINDOW_STRIDE klatek.

        Args:
            keypoints_21: (21, 3) — keypointy jednej klatki

        Returns:
            dict z wynikiem klasyfikacji lub None
            {
                "time": 3.5,              # czas w sekundach
                "class_name": "correct",
                "class_id": 1,
                "confidence": 0.92,
                "all_probs": {...},
                "total_time": 3.5,
                "exercise_time": 2.0,     # czas od startu ćwiczenia (bez setup)
            }
        """
        self.frame_buffer.append(keypoints_21.copy())
        self.total_frames += 1
        self.frames_since_last_classify += 1

        # Twardy limit bufora — trzymamy max 2× window_size
        max_buffer = self.window_size * 2
        if len(self.frame_buffer) > max_buffer:
            self.frame_buffer = self.frame_buffer[-max_buffer:]

        # Czy czas na klasyfikację?
        if (self.frames_since_last_classify >= self.window_stride
                and len(self.frame_buffer) >= self.window_size):
            self.frames_since_last_classify = 0
            return self._classify_window()

        return None

    def _classify_window(self) -> dict:
        """Klasyfikuje ostatnie window_size klatek."""
        # Pobierz okno
        window = self.frame_buffer[-self.window_size:]
        frames = np.array(window, dtype=np.float32)  # (window_size, 21, 3)

        # Normalizacja (taka sama jak w treningu)
        frames = normalize_keypoints(frames)
        frames = filter_low_confidence(frames)

        # Resize do SEQUENCE_LENGTH (gdyby window_size != SEQUENCE_LENGTH)
        frames = self._resize_sequence(frames, SEQUENCE_LENGTH)

        # Klasyfikuj
        if self.interpreter is not None:
            result = self._infer_tflite(frames)
        elif self.model is not None:
            result = self._infer_pytorch(frames)
        else:
            # Brak modelu — dummy
            result = {
                "class_id": -1,
                "class_name": "no_model",
                "confidence": 0.0,
                "all_probs": {},
            }

        # Dodaj metadane czasowe
        current_time = self.total_frames / self.fps
        result["time"] = current_time
        result["total_time"] = current_time

        # Detekcja startu ćwiczenia
        if not self.exercise_started and result["class_name"] != "setup":
            self.exercise_started = True
            self._exercise_start_time = current_time
            print(f"  [START] Cwiczenie izometryczne @ {current_time:.1f}s")

        if self.exercise_started:
            result["exercise_time"] = current_time - self._exercise_start_time
        else:
            result["exercise_time"] = 0.0

        # Aktualizuj timeline
        self.timeline.append(result)

        # Aktualizuj segmenty (ciągłe bloki tej samej klasy)
        self._update_segments(result["class_name"], current_time)

        return result

    def _update_segments(self, class_name: str, current_time: float):
        """Aktualizuje listę segmentów (ciągłych bloków tej samej klasy)."""
        window_duration = self.window_stride / self.fps

        if class_name != self._current_segment_class:
            # Zamknij poprzedni segment
            if self._current_segment_class is not None:
                self.segments.append({
                    "class": self._current_segment_class,
                    "start": self._current_segment_start,
                    "end": current_time,
                    "duration": current_time - self._current_segment_start,
                })
            # Otwórz nowy segment
            self._current_segment_class = class_name
            self._current_segment_start = current_time - window_duration

    def _resize_sequence(self, frames: np.ndarray, target_len: int) -> np.ndarray:
        """Resize sekwencji do target_len."""
        num_frames = frames.shape[0]
        if num_frames == target_len:
            return frames
        elif num_frames == 0:
            return np.zeros((target_len, 21, 3), dtype=np.float32)
        elif num_frames < target_len:
            padded = np.empty((target_len, frames.shape[1], frames.shape[2]),
                              dtype=np.float32)
            padded[:num_frames] = frames
            padded[num_frames:] = frames[-1]
            return padded
        else:
            indices = np.linspace(0, num_frames - 1, target_len, dtype=int)
            return frames[indices]

    def _infer_tflite(self, frames: np.ndarray) -> dict:
        """Inferencja przez TFLite."""
        seq_flat = frames.reshape(frames.shape[0], -1)
        seq_transposed = seq_flat.T
        model_input = seq_transposed.reshape(1, INPUT_CHANNELS, SEQUENCE_LENGTH)

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

        return self._logits_to_result(logits)

    def _infer_pytorch(self, frames: np.ndarray) -> dict:
        """Inferencja przez PyTorch."""
        import torch
        seq_flat = frames.reshape(frames.shape[0], -1)
        seq_transposed = seq_flat.T
        tensor = torch.FloatTensor(seq_transposed).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor).numpy()[0]

        return self._logits_to_result(logits)

    def _logits_to_result(self, logits: np.ndarray) -> dict:
        """Konwertuje logity na wynik klasyfikacji."""
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        class_name = self.labels.get(class_id, f"unknown_{class_id}")

        all_probs = {
            self.labels.get(i, f"class_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "all_probs": all_probs,
        }

    def get_summary(self) -> dict:
        """
        Zwraca podsumowanie ćwiczenia izometrycznego.

        Returns:
            {
                "exercise": "plank",
                "exercise_type": "isometric",
                "total_time": 45.0,          # cały czas nagrania
                "exercise_time": 40.0,       # czas od startu (bez setup)
                "time_per_class": {          # sekundy per klasa
                    "correct": 30.0,
                    "hips_low": 5.5,
                    "hips_high": 4.5,
                },
                "accuracy": 0.75,            # % czasu w "correct"
                "segments": [...],           # ciągłe bloki
                "timeline": [...],           # pełna historia
            }
        """
        # Zamknij ostatni otwarty segment
        if self._current_segment_class is not None:
            current_time = self.total_frames / self.fps
            self.segments.append({
                "class": self._current_segment_class,
                "start": self._current_segment_start,
                "end": current_time,
                "duration": current_time - self._current_segment_start,
            })
            self._current_segment_class = None

        total_time = self.total_frames / self.fps

        # Oblicz czas per klasa z timeline (dokładniejsze niż segmenty)
        time_per_class = defaultdict(float)
        window_duration = self.window_stride / self.fps
        for entry in self.timeline:
            if entry["class_name"] != "setup":
                time_per_class[entry["class_name"]] += window_duration

        # Czas ćwiczenia (bez setup)
        exercise_time = sum(time_per_class.values())

        # Procent w "correct"
        correct_time = time_per_class.get("correct", 0.0)
        accuracy = correct_time / exercise_time if exercise_time > 0 else 0.0

        # Czas per klasa błędów (bez correct i setup)
        error_time = {
            k: round(v, 1) for k, v in time_per_class.items()
            if k not in ("correct", "setup", "no_model")
        }

        return {
            "exercise": self.exercise,
            "exercise_type": "isometric",
            "total_time": round(total_time, 1),
            "exercise_time": round(exercise_time, 1),
            "correct_time": round(correct_time, 1),
            "accuracy": round(accuracy, 3),
            "time_per_class": {k: round(v, 1) for k, v in time_per_class.items()},
            "error_time": error_time,
            "segments": self.segments,
            "num_classifications": len(self.timeline),
        }

    def reset(self):
        """Resetuje klasyfikator do stanu początkowego."""
        self.frame_buffer = []
        self.total_frames = 0
        self.frames_since_last_classify = 0
        self.timeline = []
        self.segments = []
        self._current_segment_class = None
        self._current_segment_start = 0.0
        self.exercise_started = False

    def get_current_state(self) -> dict:
        """Zwraca bieżący stan (do wyświetlania na ekranie)."""
        last = self.timeline[-1] if self.timeline else None
        return {
            "exercise": self.exercise,
            "exercise_type": "isometric",
            "total_time": round(self.total_frames / self.fps, 1),
            "exercise_time": last["exercise_time"] if last else 0.0,
            "current_class": last["class_name"] if last else "setup",
            "current_confidence": last["confidence"] if last else 0.0,
            "buffered_frames": len(self.frame_buffer),
        }


# ============================================================
# WRAPPER — automatycznie wybiera Rep vs Isometric
# ============================================================

def create_classifier(exercise: str, model_path: str = None,
                      model_type: str = "tflite"):
    """
    Factory — tworzy odpowiedni klasyfikator na podstawie exercise_type.

    Args:
        exercise: nazwa ćwiczenia
        model_path: ścieżka do modelu
        model_type: "tflite" lub "pytorch"

    Returns:
        RepClassifier lub IsometricClassifier
    """
    from model.config import ISOMETRIC_EXERCISES

    if exercise in ISOMETRIC_EXERCISES:
        return IsometricClassifier(exercise, model_path, model_type)
    else:
        from model.rep_classifier import RepClassifier
        return RepClassifier(exercise, model_path, model_type)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST IsometricClassifier (plank, bez modelu)")
    print("=" * 60)

    classifier = IsometricClassifier("plank", model_path=None)
    print(f"  Cwiczenie: {classifier.exercise}")
    print(f"  Klasy: {list(classifier.labels.values())}")
    print(f"  Window: {classifier.window_size} kl., stride: {classifier.window_stride} kl.")
    print(f"  FPS: {classifier.fps}")

    # Symuluj 3 sekundy planku (90 klatek @ 30 FPS)
    classifications = 0
    for i in range(90):
        fake_kps = np.random.rand(21, 3).astype(np.float32)
        fake_kps[:, 2] = 0.9
        result = classifier.process_frame(fake_kps)
        if result is not None:
            classifications += 1
            print(f"  [{result['time']:.1f}s] {result['class_name']} "
                  f"(conf: {result['confidence']:.0%})")

    summary = classifier.get_summary()
    print(f"\n  Podsumowanie:")
    print(f"    Czas calkowity: {summary['total_time']}s")
    print(f"    Czas cwiczenia: {summary['exercise_time']}s")
    print(f"    Correct: {summary['correct_time']}s ({summary['accuracy']:.0%})")
    print(f"    Czas per klasa: {summary['time_per_class']}")
    print(f"    Bledy: {summary['error_time']}")
    print(f"    Segmenty: {len(summary['segments'])}")
    print(f"    Klasyfikacji: {summary['num_classifications']}")
    print(f"  [OK] IsometricClassifier dziala")

    # Test factory
    print(f"\n  --- Test factory ---")
    from model.config import ISOMETRIC_EXERCISES
    print(f"  Izometryczne: {ISOMETRIC_EXERCISES}")

    for ex in ["pushup", "plank", "hollow_body"]:
        clf = create_classifier(ex)
        print(f"  {ex} -> {type(clf).__name__}")
