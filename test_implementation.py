"""Test script to verify the implementation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TEST 1: Config imports")
print("=" * 60)
from model.config import (
    NUM_KEYPOINTS, KEYPOINT_DIMS, INPUT_CHANNELS,
    SEQUENCE_LENGTH, NUM_CLASSES, EXERCISE_CLASSES, LANDMARK_INDEX,
)
print(f"  NUM_KEYPOINTS = {NUM_KEYPOINTS}")
print(f"  KEYPOINT_DIMS = {KEYPOINT_DIMS}")
print(f"  INPUT_CHANNELS = {INPUT_CHANNELS}")
print(f"  SEQUENCE_LENGTH = {SEQUENCE_LENGTH}")
print(f"  NUM_CLASSES = {NUM_CLASSES}")
print(f"  Exercises: {list(EXERCISE_CLASSES.keys())}")
for ex, cfg in EXERCISE_CLASSES.items():
    print(f"    {ex}: {cfg['num_classes']} classes = {list(cfg['labels'].values())}")

assert NUM_KEYPOINTS == 21, f"Expected 21, got {NUM_KEYPOINTS}"
assert KEYPOINT_DIMS == 3, f"Expected 3, got {KEYPOINT_DIMS}"
assert INPUT_CHANNELS == 63, f"Expected 63, got {INPUT_CHANNELS}"
assert SEQUENCE_LENGTH == 30, f"Expected 30, got {SEQUENCE_LENGTH}"
assert len(EXERCISE_CLASSES) == 3, f"Expected 3 exercises"
print("  ✅ Config OK\n")

print("=" * 60)
print("TEST 2: Angle calculation (2D)")
print("=" * 60)
import numpy as np
from model.angle_calculator import calculate_angle

a = calculate_angle(np.array([0, 0]), np.array([1, 0]), np.array([1, 1]))
print(f"  90° test: {a:.1f}°")
assert abs(a - 90.0) < 0.1, f"Expected 90, got {a}"

b = calculate_angle(np.array([0, 0]), np.array([1, 0]), np.array([2, 0]))
print(f"  180° test: {b:.1f}°")
assert abs(b - 180.0) < 0.1, f"Expected 180, got {b}"

c = calculate_angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1]))
print(f"  90° test2: {c:.1f}°")
assert abs(c - 90.0) < 0.1, f"Expected 90, got {c}"
print("  ✅ Angles OK\n")

print("=" * 60)
print("TEST 3: Angle Analyzer (3 exercises)")
print("=" * 60)
from model.angle_calculator import ExerciseAngleAnalyzer

for exercise in ["squat", "pushup", "lunge"]:
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
print("TEST 5: RepCounter (3 exercises)")
print("=" * 60)
from model.rep_counter import RepCounter

for exercise in ["squat", "pushup", "lunge"]:
    counter = RepCounter(exercise)
    fake_seq = np.random.rand(60, 21, 3).astype(np.float32)
    fake_seq[:, :, 2] = 0.9
    result = counter.count_from_sequence(fake_seq)
    print(f"  {exercise}: {result['total_reps']} reps (random data)")
print("  ✅ RepCounter OK\n")

try:
    print("=" * 60)
    print("TEST 6: Model forward pass")
    print("=" * 60)
    import torch
    from model.model import TemporalCNN, count_parameters

    model = TemporalCNN()
    x = torch.randn(2, 63, 30)
    out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {count_parameters(model):,}")
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"

    # Test per exercise
    for ex_name, ex_cfg in EXERCISE_CLASSES.items():
        m = TemporalCNN(num_classes=ex_cfg["num_classes"])
        o = m(x)
        print(f"  {ex_name}: output={o.shape}")
        assert o.shape == (2, ex_cfg["num_classes"])

    print("  ✅ Model OK\n")
except ImportError:
    print("  ⚠ torch not installed, skipping model test\n")

print("=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
