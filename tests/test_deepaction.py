import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score
)
from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.data.deepaction import get_deepaction_splits

def evaluate_subset(synth_models, data_root, model, device, batch_size=4):
    """
    Evaluates the model on a specific subset of synthetic models.
    Re-uses your exact split logic to guarantee we only evaluate on the Test Set.
    """
    _, _, test_dataset = get_deepaction_splits(
        root_dir=data_root,
        synth_models=synth_models,
        train_ratio=0.8,
        val_ratio=0.1,
        target_frames=8,
        target_size=(256, 256),
        seed=42
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels =[]
    all_preds = []
    all_probs =[]

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Eval {synth_models[0] if len(synth_models)==1 else 'ALL'}", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            # forward pass
            logits = model(inputs).squeeze(1)
            
            # convert logits to probabilities (0.0 to 1.0) for AUC
            probs = torch.sigmoid(logits)
            
            # convert logits to binary predictions (threshold at 0.0 logit = 0.5 prob)
            preds = (logits > 0.0).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(cm, filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues', alpha=0.8)
    plt.colorbar(cax)

    # add text annotations
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f'{z}', ha='center', va='center', 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3', alpha=0.9))

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted REAL', 'Predicted FAKE'], fontsize=12)
    ax.set_yticklabels(['Actual REAL', 'Actual FAKE'], fontsize=12, rotation=90, va='center')
    
    plt.title('3D-FFT Model Confusion Matrix', pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\nSaved confusion matrix plot to {filename}")

def main():
    DATA_ROOT = "/scratch/rjr6zk/face-fft/src/face_fft/data/deepaction_dataset" 
    WEIGHTS_PATH = "checkpoints/deepaction_model.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError()

    # auto-discover models
    all_folders =[d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    SYNTH_MODELS =[d for d in all_folders if d != "Pexels" and not d.startswith(".")]
    
    print(f"Loading ResNet3D model on {DEVICE}...")
    print("----------")
    model = FaceFFTPipeline(
        log_scale=True, 
        in_channels=3, 
        num_classes=1,
        model_type="resnet"
    )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    print("Overall metrics:")
    print("----------")
    
    y_true, y_pred, y_prob = evaluate_subset(SYNTH_MODELS, DATA_ROOT, model, DEVICE)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nOverall Accuracy : {acc*100:.2f}%")
    print(f"Precision        : {prec:.4f}")
    print(f"Recall           : {rec:.4f}")
    print(f"F1-Score         : {f1:.4f}")
    print(f"ROC-AUC          : {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Negatives (Real -> Real)   : {cm[0, 0]}")
    print(f"False Positives (Real -> Fake)  : {cm[0, 1]}")
    print(f"False Negatives (Fake -> Real)  : {cm[1, 0]}")
    print(f"True Positives (Fake -> Fake)   : {cm[1, 1]}")
    
    plot_confusion_matrix(cm, "confusion_matrix.png")

    # per ai generator breakdown
    print("AI Generator Breakdown (Accuracy | F1 | AUC)")
    print("-----------")
    
    print(f"{'Generator Model':<25} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("----------")
    
    for sm in SYNTH_MODELS:
        yt, yp, yprob = evaluate_subset([sm], DATA_ROOT, model, DEVICE)
        
        # calculate metrics for this specific subset
        m_acc = accuracy_score(yt, yp)
        m_f1 = f1_score(yt, yp)
        m_auc = roc_auc_score(yt, yprob)
        
        print(f"{sm:<25} | {m_acc*100:>8.2f}% | {m_f1:>8.4f} | {m_auc:>8.4f}")

    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()