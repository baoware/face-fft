import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.models.pipeline_mode import PipelineMode
from face_fft.data.deepaction import get_deepaction_splits


def evaluate_loader(model, dataloader, device, desc="Evaluating"):
    """
    Evaluates the model on a provided dataloader.
    """
    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=desc, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            # Forward pass
            logits = model(inputs).squeeze(1)

            # Probabilities for AUC
            probs = torch.sigmoid(logits)

            # Binary predictions
            preds = (logits > 0.0).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(cm, arch_name):
    filename = f"confusion_matrix_{arch_name}.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
    plt.colorbar(cax)

    # Add text annotations
    for (i, j), z in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{z}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3", alpha=0.9),
        )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted REAL", "Predicted FAKE"], fontsize=12)
    ax.set_yticklabels(
        ["Actual REAL", "Actual FAKE"], fontsize=12, rotation=90, va="center"
    )

    plt.title(
        f"3D-FFT Confusion Matrix: {arch_name.upper()}",
        pad=20,
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved confusion matrix to {filename}")
    plt.close(fig)


def main():
    DATA_ROOT = "src/face_fft/data/deepaction_dataset"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SYNTH_MODELS = [
        "Veo",
        "RunwayML",
        "StableDiffusion",
        "CogVideoX5B",
        "VideoPoet",
        "BDAnimateDiffLightning",
    ]

    ARCHITECTURES = ["compact", "r3d_18", "mc3_18", "r2plus1d_18"]
    PIPELINE_MODE = PipelineMode.PIXEL_BASELINE

    print(f"Using Device: {DEVICE}")
    print("Loading DeepAction Test Dataset...")
    _, _, test_dataset = get_deepaction_splits(
        root_dir=DATA_ROOT,
        synth_models=SYNTH_MODELS,
        train_ratio=0.8,
        val_ratio=0.1,
        target_frames=8,
        target_size=(256, 256),
        seed=42,
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    ablation_results = {}

    for arch in ARCHITECTURES:
        print(f"Evaluating Architecture: {arch.upper()}")
        print("----------")

        weights_path = f"checkpoints/ablation_{arch}.pt"

        if not os.path.exists(weights_path):
            print(f"Warning: Checkpoint not found at {weights_path}. Skipping...")
            continue

        # Initialize Pipeline
        model = FaceFFTPipeline(
            log_scale=True,
            in_channels=3,
            num_classes=1,
            model_type=arch,
            mode=PIPELINE_MODE,
            temporal_frames=8,
            spatial_size=(256, 256),
        )

        # Load Weights
        model.load_state_dict(
            torch.load(weights_path, map_location=DEVICE, weights_only=True)
        )
        model.to(DEVICE)

        # Calculate Parameter Count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Evaluate
        y_true, y_pred, y_prob = evaluate_loader(
            model, test_loader, DEVICE, desc=f"Eval {arch.upper()}"
        )

        # Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)

        # Save Confusion Matrix Image
        plot_confusion_matrix(cm, arch)

        # Store in dict
        ablation_results[arch] = {
            "Params": f"{params/1e6:.1f}M",
            "Acc": acc,
            "Prec": prec,
            "Rec": rec,
            "F1": f1,
            "AUC": auc,
        }

    print(
        f"| {'Architecture':<12} | {'Params':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} |"
    )
    print(
        "|--------------|----------|------------|-----------|------------|------------|------------|"
    )

    for arch, r in ablation_results.items():
        print(
            f"| {arch.upper():<12} | {r['Params']:<8} | {r['Acc']*100:>8.2f}% | {r['Prec']:>9.4f} | {r['Rec']:>8.4f} | {r['F1']:>8.4f} | {r['AUC']:>8.4f} |"
        )


if __name__ == "__main__":
    main()
