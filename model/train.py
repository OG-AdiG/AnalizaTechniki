"""
train.py — Pętla treningowa dla modelu Temporal CNN.

Użycie:
    python model/train.py
    python model/train.py --exercise squat

Obsługuje:
- Trening z walidacją
- Weighted cross-entropy (kompensacja nierówności klas)
- Early stopping + model checkpointing
- Raport klasyfikacji z metrykami per klasa
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import TemporalCNN, count_parameters
from model.config import (
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    MODELS_DIR, ACTIVE_EXERCISE, EXERCISE_CLASSES,
    INPUT_CHANNELS, SEQUENCE_LENGTH,
)
from data_pipeline.dataset import load_dataset


def compute_class_weights(train_loader, num_classes: int, device) -> torch.Tensor:
    """
    Oblicza wagi klas na podstawie częstotliwości w zbiorze treningowym.
    Rzadsze klasy dostają wyższe wagi.
    """
    class_counts = torch.zeros(num_classes)

    for _, labels in train_loader:
        for label in labels:
            class_counts[label.item()] += 1

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalizacja

    return weights.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Trenuje model przez jedną epokę."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Walidacja modelu."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def train(exercise: str = ACTIVE_EXERCISE):
    """Główna funkcja treningowa."""
    exercise_config = EXERCISE_CLASSES[exercise]
    num_classes = exercise_config["num_classes"]

    # Urządzenie
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Urządzenie: {device}")
    print(f"🏋️  Ćwiczenie: {exercise} ({num_classes} klas)")

    # Dane
    print("\n📂 Ładowanie datasetu...")
    train_loader, val_loader = load_dataset(exercise=exercise)

    # Model
    model = TemporalCNN(
        input_channels=INPUT_CHANNELS,
        num_classes=num_classes,
    ).to(device)
    print(f"\n🧠 Model: TemporalCNN ({count_parameters(model):,} parametrów)")
    print(f"   Input: ({INPUT_CHANNELS}, {SEQUENCE_LENGTH}) = 21kps × 3dim × 30 klatek")
    print(f"   Output: {num_classes} klas")

    # Weighted cross-entropy (kompensacja nierównowagi klas)
    class_weights = compute_class_weights(train_loader, num_classes, device)
    print(f"   Wagi klas: {class_weights.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optymalizator
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    # Katalog na modele
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, f"{exercise}_best.pt")

    print(f"\n🏋️ Rozpoczynam trening ({NUM_EPOCHS} epok)...\n")
    print(f"{'Epoka':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>10}")
    print("-" * 72)

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, preds, labels = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | "
              f"{val_loss:>10.4f} | {val_acc:>8.1%} | {current_lr:>10.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "exercise": exercise,
                "num_classes": num_classes,
                "input_channels": INPUT_CHANNELS,
                "sequence_length": SEQUENCE_LENGTH,
            }, best_model_path)
            print(f"       💾 Zapisano najlepszy model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⏹️  Early stopping po epoce {epoch}")
                break

    elapsed = time.time() - start_time
    print(f"\n⏱️  Czas treningu: {elapsed:.1f}s")

    # Raport
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n📊 Najlepszy model — epoka {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.1%})")

    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)

    label_names = list(exercise_config["labels"].values())
    print("\n📋 Classification Report:")
    print(classification_report(final_labels, final_preds, target_names=label_names))
    print("📊 Confusion Matrix:")
    print(confusion_matrix(final_labels, final_preds))

    print(f"\n✅ Model zapisany: {best_model_path}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", type=str, default=ACTIVE_EXERCISE,
                        choices=list(EXERCISE_CLASSES.keys()))
    args = parser.parse_args()
    train(exercise=args.exercise)
