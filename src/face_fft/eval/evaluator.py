import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from typing import Dict, Any


class Evaluator:
    """
    Evaluates cross-model generalization for the classifier, strictly focusing
    on targeted metrics like F1 score and Confusion Matrix, per project guidelines.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)

                # Model outputs basic logits
                logits = self.model(inputs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().squeeze(1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels)

        # Calculate target metrics
        # Labels: 0 for Real, 1 for Synthetic
        # F1 score is typically focused on the positive class (Synthetic)
        f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        return {"f1_score": float(f1), "confusion_matrix": cm.tolist()}
