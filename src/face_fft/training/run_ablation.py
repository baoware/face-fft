import os
import random
import torch
import time
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.models.pipeline_mode import PipelineMode, parse_pipeline_mode
from face_fft.training.trainer import Trainer
from face_fft.data.deepaction import get_deepaction_splits
from face_fft.eval.deepaction_evaluator import evaluate_subset

def main():
    DATA_ROOT = "/scratch/rjr6zk/face-fft/src/face_fft/data/deepaction_dataset" 
    all_folders =[d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    SYNTH_MODELS =[d for d in all_folders if d != "Pexels" and not d.startswith(".")]

    # architectures to evaluate
    ARCHITECTURES =["compact", "r3d_18", "mc3_18", "r2plus1d_18"]
    
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    TARGET_FRAMES = 8
    TARGET_SIZE = (256, 256)
    # Switch ablation: pixel_baseline | fft_no_mask | fft_learnable_mask
    # Set FACE_FFT_MODE to pick the arm without editing this file, so the pixel and
    # spectral sweeps are the same code path with the same seeds.
    PIPELINE_MODE = parse_pipeline_mode(os.environ.get("FACE_FFT_MODE", "pixel_baseline"))

    # Each arm gets its own checkpoint directory. eval_ablation.py reads the
    # pixel-baseline path and eval_genvidbench.py reads no_filter/, so these names
    # are load-bearing -- do not rename without updating both evaluators.
    CKPT_DIR = {
        PipelineMode.PIXEL_BASELINE: "checkpoints",
        PipelineMode.FFT_NO_MASK: "checkpoints/no_filter",
        PipelineMode.FFT_LEARNABLE_MASK: "checkpoints/learnable_filter",
    }[PIPELINE_MODE]
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Pipeline mode: {PIPELINE_MODE}  ->  {CKPT_DIR}/")
    NUM_WORKERS = 0 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # seed weight init and loader shuffling so a rerun is reproducible;
    # only the split was seeded before, which is why the Apr 16 and Apr 20
    # runs of this script disagreed by ~9 accuracy points.
    # Override with FACE_FFT_SEED to escape a bad draw (see the per-arch reseed below).
    SEED = int(os.environ.get("FACE_FFT_SEED", 42))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"Base seed: {SEED}")

    print(f"Using Device: {DEVICE}")

    # load datasets
    print("Loading DeepAction Dataset...")
    print("----------")
    train_dataset, val_dataset, _ = get_deepaction_splits(
        root_dir=DATA_ROOT,
        synth_models=SYNTH_MODELS,
        train_ratio=0.8,
        val_ratio=0.1,
        target_frames=TARGET_FRAMES,
        target_size=TARGET_SIZE,
        seed=42
    )

    results = {}

    for arch_idx, arch in enumerate(ARCHITECTURES):
        # Reseed per architecture, derived from the base seed. Seeding once before the
        # loop makes each architecture depend on how much RNG the previous ones consumed,
        # so an architecture cannot be rerun in isolation and reproduce. Deriving the seed
        # from the index keeps every architecture independent, reproducible on its own,
        # and escapable via FACE_FFT_SEED if one draws a degenerate init (as
        # r2plus1d_18 did on 2026-09-18: val loss 27.89 at epoch 1, never recovered).
        arch_seed = SEED + arch_idx
        random.seed(arch_seed)
        np.random.seed(arch_seed)
        torch.manual_seed(arch_seed)
        torch.cuda.manual_seed_all(arch_seed)

        print(f"Experiment: Training Architecture [{arch.upper()}] (seed={arch_seed})")

        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # Initialize Pipeline
        model = FaceFFTPipeline(
            log_scale=True,
            in_channels=3,
            num_classes=1,
            model_type=arch,
            mode=PIPELINE_MODE,
            temporal_frames=TARGET_FRAMES,
            spatial_size=TARGET_SIZE,
        )
        
        # Parameter Count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=DEVICE, lr=1e-3)
        
        # Train
        save_path = f"{CKPT_DIR}/ablation_{arch}.pt"
        start_time = time.time()
        trainer.train(num_epochs=EPOCHS, save_path=save_path)
        train_time = time.time() - start_time
        
        # Evaluate
        model.load_state_dict(torch.load(save_path, weights_only=True))
        model.to(DEVICE)
        model.eval()
        y_true, y_pred, y_prob = evaluate_subset(SYNTH_MODELS, DATA_ROOT, model, DEVICE)
        
        # Store Results
        results[arch] = {
            "Params": f"{params/1e6:.1f}M",
            "Time (s)": f"{train_time:.0f}",
            "Acc": accuracy_score(y_true, y_pred),
            "Prec": precision_score(y_true, y_pred, zero_division=0),
            "Rec": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob)
        }


    print("| Architecture | Params | Time (s) | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
    print("|--------------|--------|----------|----------|-----------|--------|----------|---------|")
    for arch, r in results.items():
        print(f"| {arch.upper():<12} | {r['Params']:<6} | {r['Time (s)']:<8} | {r['Acc']:.4f}   | {r['Prec']:.4f}    | {r['Rec']:.4f} | {r['F1']:.4f}   | {r['AUC']:.4f}  |")

if __name__ == "__main__":
    main()