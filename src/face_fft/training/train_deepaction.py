import os
import torch
from torch.utils.data import DataLoader

from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.training.trainer import Trainer

from face_fft.data.deepaction import get_deepaction_splits

def main():
    DATA_ROOT = "/scratch/rjr6zk/face-fft/src/face_fft/data/deepaction_dataset" 
    all_folders =[d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    SYNTH_MODELS =[d for d in all_folders if d != "Pexels" and not d.startswith(".")]
    
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    TARGET_FRAMES = 8
    TARGET_SIZE = (256, 256)
    NUM_WORKERS = 0 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using Device: {DEVICE}")

    # load datasets
    print("Loading DeepAction Dataset...")
    print("----------")
    train_dataset, val_dataset, test_dataset = get_deepaction_splits(
        root_dir=DATA_ROOT,
        synth_models=SYNTH_MODELS,
        train_ratio=0.8,
        val_ratio=0.1,
        target_frames=TARGET_FRAMES,
        target_size=TARGET_SIZE,
        seed=42
    )

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # initialize the model
    print("Initializing FaceFFTPipeline...")
    print("----------")
    model = FaceFFTPipeline(
        log_scale=True, 
        in_channels=3, 
        base_channels=16, 
        num_classes=1,
        model_type="resnet"
    )

    # initialize the trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # run training
    print("\nStarting Training Loop...")
    print("----------")
    os.makedirs("checkpoints", exist_ok=True)
    history = trainer.train(
        num_epochs=EPOCHS, 
        save_path="checkpoints/best_face_fft_model.pt"
    )

    # evaluation
    print("\nTraining Complete! Evaluating on Test Set...")
    print("----------")
    
    # load the best model weights
    model.load_state_dict(torch.load("checkpoints/best_face_fft_model.pt"))
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            logits = model(inputs)
            
            loss = criterion(logits, labels)
            test_loss += loss.item() * inputs.size(0)
            
            # convert logits to binary predictions
            preds = (logits > 0.0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_test_loss = test_loss / total
    test_accuracy = correct / total

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()