import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any


class Trainer:
    """
    Handles the training and validation loops for the spectral classification model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Binary classification target (1 = Synthetic, 0 = Real)
        self.criterion = nn.BCEWithLogitsLoss()

        # AdamW chosen for lightweight CNN robustness
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader.dataset)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * inputs.size(0)

        return total_loss / len(self.val_loader.dataset)

    def train(
        self, num_epochs: int, save_path: str = "best_model.pt"
    ) -> Dict[str, Any]:
        """
        Executes the training loop across specified epochs.
        """
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved new best model with val_loss: {val_loss:.4f}\n")
            else:
                print("\n")

        return history
