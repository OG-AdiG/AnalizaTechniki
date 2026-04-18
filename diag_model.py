"""
Diagnostyka modelu — testuje PyTorch i TFLite na danych treningowych.
Sprawdza czy model w ogóle rozróżnia klasy.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.config import EXERCISE_CLASSES, SEQUENCE_LENGTH, INPUT_CHANNELS, MODELS_DIR

exercise = "pushup"
config = EXERCISE_CLASSES[exercise]
labels = config["labels"]
keypoints_dir = os.path.join("data", "keypoints", exercise)

# === 1. Test PyTorch ===
print("=" * 60)
print("DIAGNOSTYKA: PyTorch model na danych treningowych")
print("=" * 60)

import torch
from model.model import TemporalCNN
from data_pipeline.dataset import resize_sequence

pt_path = os.path.join(MODELS_DIR, f"{exercise}_best.pt")
ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
model = TemporalCNN(
    input_channels=ckpt.get("input_channels", INPUT_CHANNELS),
    num_classes=ckpt.get("num_classes", config["num_classes"]),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model: {pt_path}")
print(f"Val acc: {ckpt.get('val_acc', '?')}")
print(f"Epoch: {ckpt.get('epoch', '?')}")

# Testuj po 3 pliki z każdej klasy
for label_id, label_name in labels.items():
    label_dir = os.path.join(keypoints_dir, label_name)
    if not os.path.exists(label_dir):
        print(f"\n⚠ Brak: {label_dir}")
        continue

    npy_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])[:3]
    if not npy_files:
        continue

    print(f"\n{'='*40}")
    print(f"Klasa: {label_name} (id={label_id}) — {len(npy_files)} plików testowanych")
    print(f"{'='*40}")

    for npy_file in npy_files:
        kps = np.load(os.path.join(label_dir, npy_file)).astype(np.float32)
        # Dane z keypoints_dir SĄ JUŻ znormalizowane (step1 w pipeline)
        resized = resize_sequence(kps, SEQUENCE_LENGTH)

        seq_flat = resized.reshape(resized.shape[0], -1)  # (30, 63)
        seq_t = seq_flat.T  # (63, 30)
        tensor = torch.FloatTensor(seq_t).unsqueeze(0)  # (1, 63, 30)

        with torch.no_grad():
            logits = model(tensor).numpy()[0]

        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / exp_l.sum()
        pred_id = int(np.argmax(probs))
        pred_name = labels.get(pred_id, "?")
        conf = float(probs[pred_id])

        status = "✅" if pred_id == label_id else "❌"
        print(f"  {status} {npy_file}: pred={pred_name} ({conf:.0%})")

        # Top 3 probabilities
        sorted_idx = np.argsort(-probs)[:3]
        for rank, idx in enumerate(sorted_idx):
            print(f"      #{rank+1} {labels.get(idx, '?')}: {probs[idx]:.1%}")

print("\n" + "=" * 60)
print("GOTOWE")
