"""
model.py — Architektura Temporal CNN (TCNN) do klasyfikacji techniki ćwiczeń.

Temporal CNN przetwarza sekwencje czasowe współrzędnych keypointów
i klasyfikuje je jako poprawne wykonanie lub konkretne typy błędów.

Input:  (batch, 63, 30) — 21 keypointów × 3 wartości (x,y,conf), 30 klatek
Output: (batch, num_classes) — prawdopodobieństwa klas (5 dla każdego ćwiczenia)
"""

import torch
import torch.nn as nn

from model.config import INPUT_CHANNELS, SEQUENCE_LENGTH, NUM_CLASSES, DROPOUT_RATE


class TemporalCNN(nn.Module):
    """
    Temporal Convolutional Network do analizy sekwencji ruchu.

    Architektura:
        Conv1D bloki (×3) → Dilated Conv → Global Average Pooling → FC → Klasyfikacja

    Input:  (batch, 63, 30)   — 21 keypointów × 3 dim, sekwencja 30 klatek
    Output: (batch, num_classes) — prawdopodobieństwa klas
    """

    def __init__(self,
                 input_channels: int = INPUT_CHANNELS,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT_RATE):
        super().__init__()

        # Bloki konwolucyjne 1D
        self.conv_blocks = nn.Sequential(
            # Blok 1: 63 → 64
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Blok 2: 64 → 128
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Blok 3: 128 → 256
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Dodatkowy blok z dilation dla szerszego kontekstu czasowego
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Klasyfikator
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 63, 30) — sekwencja keypointów

        Returns:
            (batch, num_classes) — logity klas
        """
        x = self.conv_blocks(x)          # (batch, 256, seq_len//4)
        x = self.dilated_conv(x)         # (batch, 256, seq_len//4)
        x = self.gap(x)                  # (batch, 256, 1)
        x = x.squeeze(-1)               # (batch, 256)
        x = self.classifier(x)          # (batch, num_classes)
        return x

    def predict(self, x: torch.Tensor) -> tuple:
        """
        Predykcja z prawdopodobieństwami (do użycia w inferencji).

        Returns:
            (predicted_class, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)
        return predicted, probs


def count_parameters(model: nn.Module) -> int:
    """Zlicza parametry modelu."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test modelu
    model = TemporalCNN()
    print(f"Model: TemporalCNN")
    print(f"Parametry: {count_parameters(model):,}")
    print(f"Input:  (batch, {INPUT_CHANNELS}, {SEQUENCE_LENGTH})")
    print(f"Output: (batch, {NUM_CLASSES})")

    # Dummy forward pass
    dummy = torch.randn(2, INPUT_CHANNELS, SEQUENCE_LENGTH)
    output = model(dummy)
    print(f"\nDummy output shape: {output.shape}")
    print(f"Dummy output:\n{output}")
