"""
angle_calculator.py — Obliczanie kątów stawów z keypointów 2D.

Moduł odpowiada za:
- Wyznaczanie kątów w kluczowych stawach
- Porównywanie kątów z wzorcami biomechanicznymi
- Generowanie informacji o błędach technicznych
- Detekcję asymetrii lewo/prawo

Obsługuje wszystkie ćwiczenia zdefiniowane w EXERCISE_CLASSES (config.py).
Format danych: 21 keypointów × (x, y, confidence).
"""

import numpy as np
from model.config import EXERCISE_CLASSES, LANDMARK_INDEX, MIN_CONFIDENCE


def calculate_angle(point_a: np.ndarray,
                    point_b: np.ndarray,
                    point_c: np.ndarray) -> float:
    """
    Oblicza kąt (w stopniach) między trzema punktami 2D, z wierzchołkiem w point_b.

    Np. dla kąta w łokciu: point_a=ramię, point_b=łokieć, point_c=nadgarstek

    Args:
        point_a: Współrzędne pierwszego punktu (x, y) lub (x, y, conf)
        point_b: Współrzędne wierzchołka kąta
        point_c: Współrzędne trzeciego punktu

    Returns:
        Kąt w stopniach [0, 180]
    """
    a = np.array(point_a[:2], dtype=np.float64)
    b = np.array(point_b[:2], dtype=np.float64)
    c = np.array(point_c[:2], dtype=np.float64)

    ba = a - b
    bc = c - b

    dot = np.dot(ba, bc)
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product < 1e-8:
        return 0.0

    cos_angle = np.clip(dot / norm_product, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return float(angle)


def get_keypoint_confidence(frame: np.ndarray, idx: int) -> float:
    """Pobiera confidence keypointa z klatki (21, 3)."""
    if frame.shape[1] >= 3:
        return float(frame[idx, 2])
    return 1.0  # Jeśli brak confidence, zakładamy 100%


class ExerciseAngleAnalyzer:
    """
    Uniwersalny analizator kątów — obsługuje dowolne ćwiczenie.

    Konfiguracja pobierana z EXERCISE_CLASSES w config.py.
    """

    def __init__(self, exercise: str = "pushup"):
        if exercise not in EXERCISE_CLASSES:
            raise ValueError(f"Nieznane ćwiczenie: {exercise}. "
                             f"Dostępne: {list(EXERCISE_CLASSES.keys())}")
        self.exercise = exercise
        self.config = EXERCISE_CLASSES[exercise]
        self.landmarks_map = self.config.get("key_landmarks", {})
        self.angle_rules = self.config["angle_rules"]

    def get_landmark(self, frame: np.ndarray, side: str, joint: str) -> np.ndarray:
        """
        Pobierz współrzędne keypointa z klatki.

        Args:
            frame: (21, 3) — współrzędne klatki
            side: "left" lub "right"
            joint: nazwa stawu (np. "shoulder", "knee")

        Returns:
            np.ndarray (x, y, confidence)
        """
        # Najpierw spróbuj z side prefix
        key = f"{side}_{joint}"
        if key in self.landmarks_map:
            idx = self.landmarks_map[key]
            return frame[idx]

        # Spróbuj bez side (np. "sternum", "mid_hip", "nose")
        if joint in self.landmarks_map:
            idx = self.landmarks_map[joint]
            return frame[idx]

        # Spróbuj przez LANDMARK_INDEX
        if key in LANDMARK_INDEX:
            idx = LANDMARK_INDEX[key]
            return frame[idx]
        if joint in LANDMARK_INDEX:
            idx = LANDMARK_INDEX[joint]
            return frame[idx]

        raise KeyError(f"Nieznany punkt: {key} / {joint}")

    def compute_angle(self, frame: np.ndarray,
                      joint_names: tuple, side: str = "left") -> float:
        """
        Oblicza kąt dla podanych trzech stawów.

        Args:
            frame: (21, 3)
            joint_names: (joint_a, joint_b, joint_c) — kąt w joint_b
            side: "left" lub "right"

        Returns:
            Kąt w stopniach
        """
        points = []
        for joint in joint_names:
            try:
                point = self.get_landmark(frame, side, joint)
                points.append(point)
            except KeyError:
                return float("nan")

        if len(points) != 3:
            return float("nan")

        # Sprawdź confidence
        for p in points:
            if len(p) >= 3 and p[2] < MIN_CONFIDENCE:
                return float("nan")

        return calculate_angle(points[0], points[1], points[2])

    def analyze_frame(self, frame: np.ndarray) -> dict:
        """
        Analizuje pojedynczą klatkę — oblicza kąty i wykrywa błędy.

        Args:
            frame: (21, 3) — współrzędne jednej klatki

        Returns:
            dict z kątami, błędami i oceną
        """
        results = {
            "angles": {},
            "errors": [],
            "is_correct": True,
        }

        for rule_name, rule_config in self.angle_rules.items():
            joints = rule_config["joints"]

            # Oblicz kąty dla obu stron
            angles_both_sides = {}
            for side in ["left", "right"]:
                angle = self.compute_angle(frame, joints, side)
                if not np.isnan(angle):
                    angles_both_sides[f"{side}_{rule_name}"] = angle
                    results["angles"][f"{side}_{rule_name}"] = angle

            # Oblicz średni kąt
            valid_angles = list(angles_both_sides.values())
            if not valid_angles:
                continue

            avg = np.mean(valid_angles)
            results["angles"][f"avg_{rule_name}"] = avg

            # --- Detekcja asymetrii L/R ---
            if len(valid_angles) == 2:
                diff = abs(valid_angles[0] - valid_angles[1])
                results["angles"][f"asymmetry_{rule_name}"] = diff
                max_asymmetry = rule_config.get("max_asymmetry", 15.0)
                if diff > max_asymmetry:
                    results["errors"].append({
                        "type": f"asymmetry_{rule_name}",
                        "message": f"Asymetria w {rule_name}: {diff:.1f}°",
                        "angle_diff": diff,
                    })
                    results["is_correct"] = False

            # --- Sprawdź zakres ogólny ---
            if "correct_range" in rule_config:
                min_a, max_a = rule_config["correct_range"]
                if avg < min_a or avg > max_a:
                    results["errors"].append({
                        "type": rule_name,
                        "message": rule_config.get("error_name", rule_name),
                        "angle": avg,
                        "expected_range": (min_a, max_a),
                    })
                    results["is_correct"] = False

            # --- Sprawdź zakresy fazowe (top/bottom) ---
            range_top = rule_config.get("correct_range_top")
            range_bottom = rule_config.get("correct_range_bottom")
            if range_top or range_bottom:
                in_any_range = False
                if range_top and range_top[0] <= avg <= range_top[1]:
                    in_any_range = True
                if range_bottom and range_bottom[0] <= avg <= range_bottom[1]:
                    in_any_range = True
                if not in_any_range:
                    results["errors"].append({
                        "type": rule_name,
                        "message": rule_config.get("error_name", rule_name),
                        "angle": avg,
                        "expected_range_top": range_top,
                        "expected_range_bottom": range_bottom,
                    })
                    results["is_correct"] = False

        return results

    def analyze_sequence(self, sequence: np.ndarray) -> dict:
        """
        Analizuje sekwencję klatek i agreguje wyniki.

        Args:
            sequence: (num_frames, 21, 3)

        Returns:
            dict z zagregowanymi wynikami
        """
        frame_results = []
        error_counts = {}

        for frame in sequence:
            result = self.analyze_frame(frame)
            frame_results.append(result)

            for error in result["errors"]:
                etype = error["type"]
                error_counts[etype] = error_counts.get(etype, 0) + 1

        correct_frames = sum(1 for r in frame_results if r["is_correct"])
        total_frames = len(frame_results)

        # Zbierz statystyki kątów
        all_angles = {}
        for rule_name in self.angle_rules:
            values = [
                r["angles"].get(f"avg_{rule_name}", float("nan"))
                for r in frame_results
            ]
            valid = [v for v in values if not np.isnan(v)]
            if valid:
                all_angles[rule_name] = {
                    "min": min(valid),
                    "max": max(valid),
                    "mean": np.mean(valid),
                }

        return {
            "exercise": self.exercise,
            "total_frames": total_frames,
            "correct_frames": correct_frames,
            "accuracy": correct_frames / total_frames if total_frames > 0 else 0,
            "angle_stats": all_angles,
            "error_counts": error_counts,
            "overall_correct": correct_frames / total_frames > 0.7 if total_frames > 0 else False,
        }


if __name__ == "__main__":
    # Test z losowymi danymi — 3 ćwiczenia
    for exercise in ["pullup", "pushup", "dips"]:
        print(f"\n{'='*50}")
        print(f"TEST: {exercise.upper()}")
        print(f"{'='*50}")

        analyzer = ExerciseAngleAnalyzer(exercise)

        # Symuluj klatkę (21 keypointów, 3 dim)
        fake_frame = np.random.rand(21, 3).astype(np.float32)
        fake_frame[:, 2] = 0.9  # Ustawiamy wysoką confidence

        result = analyzer.analyze_frame(fake_frame)
        print(f"Kąty: {result['angles']}")
        print(f"Błędy: {result['errors']}")
        print(f"Poprawna: {result['is_correct']}")
