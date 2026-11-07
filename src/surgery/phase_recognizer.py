"""CNN+LSTM surgical phase recognition for seven Cholec80-style phases.

Training input is a tensor shaped [N, T, 3, H, W] and integer labels [N].
Frames must be resized and normalized using ImageNet statistics.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models


DEFAULT_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


class PhaseCNNLSTM(nn.Module):
    """ResNet18 frame encoder followed by an LSTM temporal classifier."""

    def __init__(self, num_classes: int = 7, hidden_size: int = 256,
                 pretrained: bool = True):
        super().__init__()
        # ResNet extracts spatial features from each video frame.
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_size = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.cnn = backbone

        # LSTM learns how frame-level features evolve over time.
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # clips: [batch, time, channels, height, width]
        batch, time_steps, channels, height, width = clips.shape
        frames = clips.reshape(batch * time_steps, channels, height, width)
        features = self.cnn(frames)
        features = features.reshape(batch, time_steps, -1)
        sequence_output, _ = self.lstm(features)
        # Last output summarizes the observed frame sequence.
        return self.classifier(sequence_output[:, -1, :])


class PhaseRecognizer:
    """Load a trained CNN+LSTM checkpoint and classify frame sequences."""

    def __init__(self, weights: str = "models/best_phase.pth",
                 phases: Optional[List[str]] = None,
                 device: Optional[str] = None):
        self.phases = phases or DEFAULT_PHASES
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PhaseCNNLSTM(num_classes=len(self.phases), pretrained=False)
        checkpoint = torch.load(weights, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, clip: torch.Tensor) -> Dict[str, float]:
        """Predict one clip. clip shape: [T, 3, H, W]."""
        if clip.ndim != 4:
            raise ValueError("clip must have shape [T, 3, H, W]")
        logits = self.model(clip.unsqueeze(0).to(self.device))
        probabilities = torch.softmax(logits, dim=1)[0]
        class_index = int(torch.argmax(probabilities).item())
        return {
            "phase": self.phases[class_index],
            "confidence": float(probabilities[class_index].item()),
        }


def train_phase_model(model: PhaseCNNLSTM, train_loader, val_loader,
                      epochs: int = 15, learning_rate: float = 1e-4,
                      output_path: str = "models/best_phase.pth",
                      device: Optional[str] = None) -> Dict[str, float]:
    """Minimal supervised training loop; loaders yield (clips, labels)."""
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        for clips, labels in train_loader:
            clips, labels = clips.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            loss = criterion(model(clips), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for clips, labels in val_loader:
                clips, labels = clips.to(device_obj), labels.to(device_obj)
                predictions = model(clips).argmax(dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())

        accuracy = correct / total if total else 0.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_accuracy,
                "epoch": epoch + 1,
            }, output_path)
        print(f"Epoch {epoch + 1}/{epochs}: validation accuracy={accuracy:.4f}")

    return {"best_validation_accuracy": best_accuracy}
