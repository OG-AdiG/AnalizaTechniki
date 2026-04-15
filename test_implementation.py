"""Test script to verify the implementation."""
import sys
import os
import math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TEST 1: Config imports")
print("=" * 60)
from model.config import (
    NUM_KEYPOINTS, KEYPOINT_DIMS, INPUT_CHANNELS,
    SEQUENCE_LENGTH, NUM_CLASSES, EXERCISE_CLASSES, LANDMARK_INDEX,
    ACTIVE_EXERCISE, EMA_ALPHA, REP_AMPLITUDE_THRESHOLD,
    MIN_REP_FRAMES, MAX_REPS_PER_MINUTE,
)
print(f"  NUM_KEYPOINTS = {NUM_KEYPOINTS}")
print(f"  KEYPOINT_DIMS = {KEYPOINT_DIMS}")
print(f"  INPUT_CHANNELS = {INPUT_CHANNELS}")
print(f"  SEQUENCE_LENGTH = {SEQUENCE_LENGTH}")
print(f"  NUM_CLASSES = {NUM_CLASSES}")
print(f"  ACTIVE_EXERCISE = {ACTIVE_EXERCISE}")
print(f"  EMA_ALPHA = {EMA_ALPHA}")
print(f"  REP_AMPLITUDE_THRESHOLD = {REP_AMPLITUDE_THRESHOLD}")
print(f"  MIN_REP_FRAMES = {MIN_REP_FRAMES}")
print(f"  Exercises: {list(EXERCISE_CLASSES.keys())}")
for ex, cfg in EXERCISE_CLASSES.items():
    print(f"    {ex}: {cfg['num_classes']} classes = {list(cfg['labels'].values())}")

assert NUM_KEYPOINTS == 21, f"Expected 21, got {NUM_KEYPOINTS}"
assert KEYPOINT_DIMS == 3, f"Expected 3, got {KEYPOINT_DIMS}"
assert INPUT_CHANNELS == 63, f"Expected 63, got {INPUT_CHANNELS}"
assert SEQUENCE_LENGTH == 30, f"Expected 30, got {SEQUENCE_LENGTH}"
assert ACTIVE_EXERCISE == "pushup", f"Expected 'pushup', got {ACTIVE_EXERCISE}"
assert len(EXERCISE_CLASSES) == 2, f"Expected 2 exercises, got {len(EXERCISE_CLASSES)}"
assert "pullup_overhand" in EXERCISE_CLASSES, "Missing pullup_overhand"
assert "pushup" in EXERCISE_CLASSES, "Missing pushup"
assert EXERCISE_CLASSES["pullup_overhand"]["num_classes"] == 7, "pullup_overhand should have 7 classes"
assert EXERCISE_CLASSES["pushup"]["num_classes"] == 7, "pushup should have 7 classes"
# Sprawdz klasy pushup — musza pasowac do folderow na dysku
pushup_labels = list(EXERCISE_CLASSES["pushup"]["labels"].values())
assert "setup" in pushup_labels, "pushup should have 'setup'"
assert "correct" in pushup_labels, "pushup should have 'correct'"
assert "flared_elbows" in pushup_labels, "pushup should have 'flared_elbows'"
assert "high_hips" in pushup_labels, "pushup should have 'high_hips'"
assert "sagging_hips" in pushup_labels, "pushup should have 'sagging_hips'"
# Parametry filtru
assert EMA_ALPHA > 0 and EMA_ALPHA < 1, f"EMA_ALPHA should be 0-1, got {EMA_ALPHA}"
assert REP_AMPLITUDE_THRESHOLD > 0, f"Amplitude threshold should be positive"
assert MIN_REP_FRAMES > 0, f"Min rep frames should be positive"
print("  ✅ Config OK\n")

print("=" * 60)
print("TEST 2: Angle calculation (2D)")
print("=" * 60)
import numpy as np
from model.angle_calculator import calculate_angle

a = calculate_angle(np.array([0, 0]), np.array([1, 0]), np.array([1, 1]))
print(f"  90deg test: {a:.1f}deg")
assert abs(a - 90.0) < 0.1, f"Expected 90, got {a}"

b = calculate_angle(np.array([0, 0]), np.array([1, 0]), np.array([2, 0]))
print(f"  180deg test: {b:.1f}deg")
assert abs(b - 180.0) < 0.1, f"Expected 180, got {b}"

c = calculate_angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1]))
print(f"  90deg test2: {c:.1f}deg")
assert abs(c - 90.0) < 0.1, f"Expected 90, got {c}"
print("  ✅ Angles OK\n")

print("=" * 60)
print("TEST 3: Angle Analyzer (2 exercises)")
print("=" * 60)
from model.angle_calculator import ExerciseAngleAnalyzer

for exercise in ["pullup_overhand", "pushup"]:
    analyzer = ExerciseAngleAnalyzer(exercise)
    fake_frame = np.random.rand(21, 3).astype(np.float32)
    fake_frame[:, 2] = 0.9
    result = analyzer.analyze_frame(fake_frame)
    print(f"  {exercise}: {len(result['angles'])} angles, {len(result['errors'])} errors")
print("  ✅ Analyzers OK\n")

print("=" * 60)
print("TEST 4: Normalize keypoints")
print("=" * 60)
from data_pipeline.extract_keypoints import normalize_keypoints, filter_low_confidence

fake_seq = np.random.rand(10, 21, 3).astype(np.float32)
fake_seq[:, :, 2] = 0.9
normalized = normalize_keypoints(fake_seq)
print(f"  Input shape:  {fake_seq.shape}")
print(f"  Output shape: {normalized.shape}")
assert normalized.shape == fake_seq.shape
# Check mid_hip is now at ~(0,0)
print(f"  Mid-hip after norm (frame0): {normalized[0, 20, :2]}")
print("  ✅ Normalization OK\n")

print("=" * 60)
print("TEST 5: RepCounter z filtrem calalkujacym")
print("=" * 60)
from model.rep_counter import RepCounter

counter = RepCounter("pushup")
print(f"  EMA alpha: {counter.ema_alpha}")
print(f"  Amplitude threshold: {counter.amplitude_threshold} deg")
print(f"  Min rep frames: {counter.min_rep_frames}")
print(f"  Max reps/min: {counter.max_reps_per_minute}")

# Test z losowymi danymi
fake_seq = np.random.rand(60, 21, 3).astype(np.float32)
fake_seq[:, :, 2] = 0.9
result = counter.count_from_sequence(fake_seq)
print(f"  Pushup (random data): {result['total_reps']} reps")
print(f"  Angle history length: {len(result['angle_history_raw'])}")
print(f"  Smooth history length: {len(result['angle_history_smooth'])}")

# Test update interfejsu
counter.reset()
fake_frame = np.random.rand(21, 3).astype(np.float32)
fake_frame[:, 2] = 0.9
state = counter.update(fake_frame)
assert "rep_count" in state, "Missing rep_count"
assert "phase" in state, "Missing phase"
assert "angle_raw" in state, "Missing angle_raw"
assert "angle_smooth" in state, "Missing angle_smooth"
assert "new_rep" in state, "Missing new_rep"
assert "rep_amplitude" in state, "Missing rep_amplitude"
assert "frames_in_rep" in state, "Missing frames_in_rep"
assert "integral" in state, "Missing integral"
print(f"  Update result keys: {list(state.keys())}")
print("  ✅ RepCounter OK\n")

print("=" * 60)
print("TEST 6: RepClassifier (per-rep flow)")
print("=" * 60)
from model.rep_classifier import RepClassifier

classifier = RepClassifier("pushup", model_path=None)
print(f"  Exercise: {classifier.exercise}")
print(f"  Labels: {list(classifier.labels.values())}")
print(f"  Num classes: {classifier.num_classes}")

# Test process_frame
fake_kps = np.random.rand(21, 3).astype(np.float32)
fake_kps[:, 2] = 0.9
result = classifier.process_frame(fake_kps)
# result powinien byc None (pierwszy frame nie moze byc novym repem)
print(f"  process_frame result: {result}")

# Test get_summary (pusty)
summary = classifier.get_summary()
assert summary["total_reps"] == 0, f"Expected 0 reps, got {summary['total_reps']}"
assert summary["accuracy"] == 0.0, "Expected 0% accuracy"
print(f"  Summary (empty): {summary}")

# Test get_current_state
state = classifier.get_current_state()
assert "rep_count" in state, "Missing rep_count in state"
assert "phase" in state, "Missing phase in state"
assert "buffered_frames" in state, "Missing buffered_frames"
print(f"  Current state: {state}")

# Test resize_sequence
from model.rep_classifier import RepClassifier
test_frames = np.random.rand(15, 21, 3).astype(np.float32)
resized = classifier._resize_sequence(test_frames, 30)
assert resized.shape == (30, 21, 3), f"Expected (30,21,3), got {resized.shape}"
print(f"  Resize (15->30): {test_frames.shape} -> {resized.shape}")

test_frames2 = np.random.rand(50, 21, 3).astype(np.float32)
resized2 = classifier._resize_sequence(test_frames2, 30)
assert resized2.shape == (30, 21, 3), f"Expected (30,21,3), got {resized2.shape}"
print(f"  Resize (50->30): {test_frames2.shape} -> {resized2.shape}")
print("  ✅ RepClassifier OK\n")

print("=" * 60)
print("TEST 7: ExerciseDetector (heurystyka)")
print("=" * 60)
from model.exercise_detector import ExerciseDetector

detector = ExerciseDetector(mode="heuristic")
print(f"  Mode: {detector.mode}")

# Test pompka — cialo horyzontalne
pushup_frame = np.zeros((21, 3), dtype=np.float32)
pushup_frame[:, 2] = 0.9
pushup_frame[3] = [0.3, 0.3, 0.9]   # left_shoulder
pushup_frame[4] = [0.7, 0.3, 0.9]   # right_shoulder
pushup_frame[7] = [0.2, 0.3, 0.9]   # left_wrist
pushup_frame[8] = [0.8, 0.3, 0.9]   # right_wrist
pushup_frame[9] = [0.4, 0.5, 0.9]   # left_hip
pushup_frame[10] = [0.6, 0.5, 0.9]  # right_hip
pushup_frame[13] = [0.5, 0.7, 0.9]  # left_ankle
pushup_frame[14] = [0.5, 0.7, 0.9]  # right_ankle

for _ in range(35):
    result_pushup = detector.detect(pushup_frame)
print(f"  Pompka (horyzontalna): {result_pushup}")
assert result_pushup == "pushup", f"Expected 'pushup', got {result_pushup}"

detector.reset()

# Test podciaganie — rece nad glowa
pullup_frame = np.zeros((21, 3), dtype=np.float32)
pullup_frame[:, 2] = 0.9
pullup_frame[3] = [0.4, 0.3, 0.9]   # left_shoulder
pullup_frame[4] = [0.6, 0.3, 0.9]   # right_shoulder
pullup_frame[7] = [0.35, 0.1, 0.9]  # left_wrist
pullup_frame[8] = [0.65, 0.1, 0.9]  # right_wrist
pullup_frame[9] = [0.45, 0.6, 0.9]  # left_hip
pullup_frame[10] = [0.55, 0.6, 0.9] # right_hip
pullup_frame[13] = [0.45, 0.9, 0.9] # left_ankle
pullup_frame[14] = [0.55, 0.9, 0.9] # right_ankle

for _ in range(35):
    result_pullup = detector.detect(pullup_frame)
print(f"  Podciaganie (wertykalna): {result_pullup}")
assert result_pullup == "pullup_overhand", f"Expected 'pullup_overhand', got {result_pullup}"

print("  ✅ ExerciseDetector OK\n")

print("=" * 60)
print("TEST 8: Dataset (single_rep_mode)")
print("=" * 60)
from data_pipeline.dataset import resize_sequence, ExerciseDataset

# Resize sequence
short_seq = np.random.rand(15, 21, 3).astype(np.float32)
resized = resize_sequence(short_seq, 30)
assert resized.shape == (30, 21, 3), f"Expected (30,21,3), got {resized.shape}"
print(f"  resize_sequence (15->30): OK")

long_seq = np.random.rand(50, 21, 3).astype(np.float32)
resized = resize_sequence(long_seq, 30)
assert resized.shape == (30, 21, 3), f"Expected (30,21,3), got {resized.shape}"
print(f"  resize_sequence (50->30): OK")

exact_seq = np.random.rand(30, 21, 3).astype(np.float32)
resized = resize_sequence(exact_seq, 30)
assert resized.shape == (30, 21, 3), f"Expected (30,21,3), got {resized.shape}"
print(f"  resize_sequence (30->30): OK")

# Temporal jitter
dataset = ExerciseDataset([short_seq], [0], augment=True)
print(f"  ExerciseDataset length: {len(dataset)}")
x, y = dataset[0]
print(f"  Sample shape: x={x.shape}, y={y.shape}")
assert x.shape[0] == 63, f"Expected channels=63, got {x.shape[0]}"
print("  ✅ Dataset OK\n")

try:
    print("=" * 60)
    print("TEST 9: Model forward pass")
    print("=" * 60)
    import torch
    from model.model import TemporalCNN, count_parameters

    model = TemporalCNN()
    x = torch.randn(2, 63, 30)
    out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {count_parameters(model):,}")
    assert out.shape == (2, 7), f"Expected (2, 7), got {out.shape}"

    # Test per exercise
    for ex_name, ex_cfg in EXERCISE_CLASSES.items():
        m = TemporalCNN(num_classes=ex_cfg["num_classes"])
        o = m(x)
        print(f"  {ex_name}: output={o.shape}")
        assert o.shape == (2, ex_cfg["num_classes"])

    print("  ✅ Model OK\n")
except ImportError:
    print("  ⚠ torch not installed, skipping model test\n")

try:
    print("=" * 60)
    print("TEST 10: ExerciseClassifier architektura")
    print("=" * 60)
    from model.exercise_detector import get_exercise_classifier_model
    import torch

    ExerciseClassifier = get_exercise_classifier_model()
    if ExerciseClassifier:
        ec_model = ExerciseClassifier()
        params = sum(p.numel() for p in ec_model.parameters())
        dummy = torch.randn(2, 63, 30)
        out = ec_model(dummy)
        print(f"  Params: {params:,}")
        print(f"  Size: ~{params * 4 / 1024:.0f}KB")
        print(f"  Input:  {dummy.shape}")
        print(f"  Output: {out.shape}")
        assert out.shape == (2, len(EXERCISE_CLASSES)), f"Expected (2, {len(EXERCISE_CLASSES)}), got {out.shape}"
        print("  ✅ ExerciseClassifier OK\n")
except ImportError:
    print("  ⚠ torch not installed, skipping\n")

print("=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
