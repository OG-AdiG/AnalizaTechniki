"""
rep_classifier.py — Klasyfikacja techniki PER POWTÓRZENIE (nie per klatkę).

Rozwiązuje problem flickeringu — zamiast klasyfikować każdą klatkę osobno,
buforuje klatki między granicami powtórzeń i klasyfikuje CAŁY rep naraz.

Flow:
    1. RepCounter wykrywa granice repów (start/koniec)
    2. Bufor zbiera klatki między granicami
    3. Model TCN klasyfikuje cały rep → 1 wynik na 1 powtórzenie
    4. Historia repów → raport: "10 repów: 7 correct, 2 flared, 1 sagging"

Użycie:
    classifier = RepClassifier("pushup", "saved_models/pushup_best.tflite")
    for frame in video_frames:
        result = classifier.process_frame(keypoints_21)
        if result is not None:
            print(f"Rep {result['rep_number']}: {result['class']} ({result['confidence']:.0%})")
    print(classifier.get_summary())
"""

import os
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.rep_counter import RepCounter
from model.config import (
    EXERCISE_CLASSES, ACTIVE_EXERCISE,
    SEQUENCE_LENGTH, INPUT_CHANNELS,
    NUM_KEYPOINTS, KEYPOINT_DIMS,
)
from data_pipeline.extract_keypoints import normalize_keypoints, filter_low_confidence


class RepClassifier:
    """
    Klasyfikator techniki per powtórzenie.

    Łączy RepCounter (wykrywanie granic repów) z modelem TCN
    (klasyfikacja techniki). Eliminuje flickering — 1 klasyfikacja
    na 1 powtórzenie zamiast ciągłego skakania klas.
    """

    def __init__(self, exercise: str = ACTIVE_EXERCISE,
                 model_path: str = None,
                 model_type: str = "tflite"):
        """
        Args:
            exercise: nazwa ćwiczenia (np. "pushup")
            model_path: ścieżka do modelu TFLite (.tflite) lub PyTorch (.pt)
            model_type: "tflite" lub "pytorch"
        """
        self.exercise = exercise
        self.config = EXERCISE_CLASSES[exercise]
        self.labels = self.config["labels"]
        self.num_classes = self.config["num_classes"]

        # RepCounter z filtrem całkującym
        self.rep_counter = RepCounter(exercise)

        # Bufor klatek bieżącego repa (lista np.ndarray (21,3))
        self.frame_buffer = []

        # Historia wyników repów
        self.rep_results = []

        # Model techniki
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
        Zwraca wynik klasyfikacji JEŚLI rep się właśnie skończył.

        Args:
            keypoints_21: (21, 3) — keypointy jednej klatki

        Returns:
            dict z wynikiem klasyfikacji jeśli rep ukończony, inaczej None
            {
                "rep_number": 1,
                "class_id": 1,
                "class_name": "correct",
                "confidence": 0.87,
                "all_probs": {0: 0.02, 1: 0.87, ...},
                "rep_frames": 34,
                "rep_amplitude": 82.5,
            }
        """
        # Dodaj klatkę do bufora
        self.frame_buffer.append(keypoints_21.copy())

        # Aktualizuj RepCounter
        rep_state = self.rep_counter.update(keypoints_21)

        # === Detekcja startu ćwiczenia ===
        # Gdy RepCounter wykryje pierwsze przejście progu down (just_started),
        # oznacza to, że osoba właśnie zaczęła ruch — wszystkie wcześniejsze
        # klatki w buforze to setup (ustawianie, podchodzenie) i trzeba je
        # odrzucić. Zostawiamy TYLKO bieżącą klatkę jako start 1. repa.
        if rep_state.get("just_started", False):
            discarded = len(self.frame_buffer) - 1
            self.frame_buffer = self.frame_buffer[-1:]  # Zostaw tylko bieżącą
            if discarded > 0:
                print(f"  🧹 Odrzucono {discarded} klatek setup z bufora")

        if rep_state["new_rep"] or rep_state.get("partial_rep", False):
            # Rep ukończony (pełny lub partial) → klasyfikuj buforowane klatki
            is_partial = rep_state.get("partial_rep", False)
            result = self._classify_buffered_rep(rep_state)
            result["is_partial"] = is_partial
            self.rep_results.append(result)

            if is_partial:
                print(f"  ⚡ Partial rep wykryty → klasyfikacja: {result['class_name']}")

            # Zachowaj ostatnich kilka klatek jako start nowego repa
            # (overlap pomaga z kontekstem)
            overlap = min(5, len(self.frame_buffer))
            self.frame_buffer = self.frame_buffer[-overlap:]

            return result

        return None

    def _classify_buffered_rep(self, rep_state: dict) -> dict:
        """
        Klasyfikuje buforowane klatki jako jeden rep.
        Resize'uje do SEQUENCE_LENGTH i przepuszcza przez model.

        UWAGA: Dane treningowe (.npy) są SUROWE (bez normalizacji),
        więc tutaj też NIE normalizujemy — model oczekuje raw keypointów.
        """
        frames = np.array(self.frame_buffer, dtype=np.float32)  # (T, 21, 3)
        raw_frame_count = frames.shape[0]

        # Resize do SEQUENCE_LENGTH (padding lub subsampling)
        frames = self._resize_sequence(frames, SEQUENCE_LENGTH)

        # Klasyfikuj
        if self.interpreter is not None:
            result = self._infer_tflite(frames)
        elif self.model is not None:
            result = self._infer_pytorch(frames)
        else:
            # Brak modelu — dummy result
            result = {
                "class_id": -1,
                "class_name": "no_model",
                "confidence": 0.0,
                "all_probs": {},
            }

        result["rep_number"] = len(self.rep_results) + 1
        result["rep_frames"] = raw_frame_count
        result["rep_amplitude"] = rep_state.get("rep_amplitude", 0.0)

        # === Diagnostyka ===
        probs = result.get("all_probs", {})
        if probs:
            sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
            top2 = sorted_probs[:2]
            print(f"    📊 Bufor: {raw_frame_count} kl. → "
                  f"Top: {top2[0][0]}={top2[0][1]:.0%}"
                  + (f", {top2[1][0]}={top2[1][1]:.0%}" if len(top2) > 1 else ""))

        return result

    def _resize_sequence(self, frames: np.ndarray,
                         target_len: int) -> np.ndarray:
        """
        Resize sekwencji do target_len klatek.
        - Jeśli za krótka: zero-padding na końcu
        - Jeśli za długa: równomierne subsamplowanie
        """
        num_frames = frames.shape[0]

        if num_frames == target_len:
            return frames
        elif num_frames < target_len:
            # Zero-padding
            padded = np.zeros(
                (target_len, frames.shape[1], frames.shape[2]),
                dtype=np.float32,
            )
            padded[:num_frames] = frames
            return padded
        else:
            # Równomierne subsamplowanie
            indices = np.linspace(0, num_frames - 1, target_len, dtype=int)
            return frames[indices]

    def _infer_tflite(self, frames: np.ndarray) -> dict:
        """Inferencja przez TFLite."""
        # (SEQUENCE_LENGTH, 21, 3) → (SEQUENCE_LENGTH, 63) → (63, SEQUENCE_LENGTH)
        seq_flat = frames.reshape(frames.shape[0], -1)  # (30, 63)
        seq_transposed = seq_flat.T  # (63, 30)
        model_input = seq_transposed.reshape(1, INPUT_CHANNELS, SEQUENCE_LENGTH)

        # Sprawdź format wejścia TFLite (onnx2tf może transponować)
        expected_shape = self.input_details[0]["shape"]
        if (expected_shape[1] == SEQUENCE_LENGTH
                and expected_shape[2] == INPUT_CHANNELS):
            model_input = np.transpose(model_input, (0, 2, 1))

        model_input = model_input.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], model_input)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]

        return self._logits_to_result(logits)

    def _infer_pytorch(self, frames: np.ndarray) -> dict:
        """Inferencja przez PyTorch."""
        import torch

        seq_flat = frames.reshape(frames.shape[0], -1)  # (30, 63)
        seq_transposed = seq_flat.T  # (63, 30)
        tensor = torch.FloatTensor(seq_transposed).unsqueeze(0)  # (1, 63, 30)

        with torch.no_grad():
            logits = self.model(tensor).numpy()[0]

        return self._logits_to_result(logits)

    def _logits_to_result(self, logits: np.ndarray) -> dict:
        """Konwertuje logity na wynik klasyfikacji."""
        # Softmax
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
        Zwraca podsumowanie seta (wszystkich powtórzeń).

        Returns:
            {
                "exercise": "pushup",
                "total_reps": 10,
                "correct_reps": 7,
                "accuracy": 0.7,
                "error_breakdown": {"flared_elbows": 2, "sagging_hips": 1},
                "rep_details": [...]
            }
        """
        if not self.rep_results:
            return {
                "exercise": self.exercise,
                "total_reps": 0,
                "correct_reps": 0,
                "accuracy": 0.0,
                "error_breakdown": {},
                "rep_details": [],
            }

        # Filtruj setup — nie jest powtórzeniem
        real_reps = [
            r for r in self.rep_results
            if r.get("class_name") != "setup"
        ]

        total = len(real_reps)
        correct = sum(1 for r in real_reps if r.get("class_name") == "correct")

        error_counts = Counter(
            r["class_name"] for r in real_reps
            if r.get("class_name") not in ("correct", "setup", "no_model")
        )

        return {
            "exercise": self.exercise,
            "total_reps": total,
            "correct_reps": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "error_breakdown": dict(error_counts),
            "rep_details": self.rep_results,
        }

    def reset(self):
        """Resetuje klasyfikator do stanu początkowego."""
        self.rep_counter.reset()
        self.frame_buffer = []
        self.rep_results = []

    def get_current_state(self) -> dict:
        """Zwraca bieżący stan (do wyświetlania na ekranie)."""
        last_rep = self.rep_results[-1] if self.rep_results else None
        return {
            "rep_count": self.rep_counter.rep_count,
            "phase": self.rep_counter.phase,
            "angle_smooth": self.rep_counter.angle_smooth or 0.0,
            "buffered_frames": len(self.frame_buffer),
            "last_rep": last_rep,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TEST RepClassifier (bez modelu)")
    print("=" * 60)

    classifier = RepClassifier("pushup", model_path=None)
    print(f"  Ćwiczenie: {classifier.exercise}")
    print(f"  Klasy: {list(classifier.labels.values())}")

    # Symuluj 3 repa (losowe dane)
    for i in range(90):
        fake_kps = np.random.rand(21, 3).astype(np.float32)
        fake_kps[:, 2] = 0.9
        result = classifier.process_frame(fake_kps)
        if result is not None:
            print(f"  Rep {result['rep_number']}: {result['class_name']}")

    summary = classifier.get_summary()
    print(f"\n  Podsumowanie: {summary['total_reps']} repów")
    print(f"  (Z losowymi danymi wyniki będą randomowe)")
    print("  ✅ RepClassifier działa")
