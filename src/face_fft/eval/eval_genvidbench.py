import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F_t
from tqdm import tqdm

from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.models.pipeline_mode import PipelineMode

class GVB_FakesOnlyDataset(Dataset):
    """Loads ONLY Fake videos. Label is always 1."""
    def __init__(self, video_paths, target_frames=8, target_size=(256, 256)):
        self.video_paths = video_paths
        self.target_frames = target_frames
        self.target_size = target_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total_frames - 1, self.target_frames).astype(int)
            
            frames =[]
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            cap.release()

            if len(frames) == 0:
                raise ValueError()

            v = torch.from_numpy(np.array(frames)).permute(3, 0, 1, 2).float() / 255.0
            return v, 1
        except Exception:
            return torch.zeros((3, self.target_frames, *self.target_size)), 1

def main():
    DATA_ROOT = "/scratch/rjr6zk/face-fft/src/face_fft/data/genvidbench_dataset" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {DEVICE}")
    
    ARCHITECTURES =["compact", "r3d_18", "mc3_18", "r2plus1d_18"]
    GENERATORS = ["Sora", "Kling", "OpenSora"]
    # Options: PIXEL_BASELINE, FFT_NO_MASK, FFT_LEARNABLE_MASK
    PIPELINE_MODE = PipelineMode.FFT_LEARNABLE_MASK
    
    MAX_VIDS_PER_GEN = 70 

    results = {arch: {} for arch in ARCHITECTURES}

    for arch in ARCHITECTURES:
        weights_path = f"checkpoints/learnable_filter/ablation_{arch}.pt"
        if not os.path.exists(weights_path):
            print(f"Skipping {arch.upper()}: Checkpoint not found at {weights_path}")
            continue

        print(f"\nTesting Architecture: {arch.upper()}")
        
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
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        arch_total_caught = 0
        arch_total_vids = 0
        
        # Evaluate each generator independently
        for gen in GENERATORS:
            gen_dir = os.path.join(DATA_ROOT, gen)
            if not os.path.exists(gen_dir):
                print(f"Folder not found for {gen}")
                continue
                
            fake_vids =[os.path.join(dp, f) for dp, dn, filenames in os.walk(gen_dir) for f in filenames if f.endswith('.mp4')]
            
            if len(fake_vids) == 0:
                continue
                
            if len(fake_vids) > MAX_VIDS_PER_GEN:
                np.random.seed(10) # encara Messi
                fake_vids = np.random.choice(fake_vids, MAX_VIDS_PER_GEN, replace=False).tolist()
                
            test_dataset = GVB_FakesOnlyDataset(fake_vids)
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
            
            caught_fakes = 0
            total_fakes = 0
            
            with torch.no_grad():
                for inputs, _ in tqdm(test_loader, desc=f"   Detecting {gen}", leave=False):
                    inputs = inputs.to(DEVICE)
                    logits = model(inputs).squeeze(1)
                    preds = (logits > 0.0).float() # Prediction > 0 means FAKE
                    
                    caught_fakes += preds.sum().item()
                    total_fakes += inputs.size(0)
            
            gen_rate = (caught_fakes / total_fakes) * 100 if total_fakes > 0 else 0
            results[arch][gen] = gen_rate
            
            arch_total_caught += caught_fakes
            arch_total_vids += total_fakes
            
        # Calculate overall recall for this architecture
        overall_rate = (arch_total_caught / arch_total_vids) * 100 if arch_total_vids > 0 else 0
        results[arch]["OVERALL"] = overall_rate
        print(f"   > {arch.upper()} Overall Detection Rate: {overall_rate:.2f}%")

    print(f"| {'Architecture':<12} | {'Sora':<10} | {'Kling':<10} | {'OpenSora':<10} | {'OVERALL':<10} |")
    print("|--------------|------------|------------|------------|------------|")
    for arch in ARCHITECTURES:
        if arch in results and results[arch]:
            r = results[arch]
            sora_rate = f"{r.get('Sora', 0):.2f}%"
            kling_rate = f"{r.get('Kling', 0):.2f}%"
            opensora_rate = f"{r.get('OpenSora', 0):.2f}%"
            overall_rate = f"{r.get('OVERALL', 0):.2f}%"
            
            print(f"| {arch.upper():<12} | {sora_rate:>10} | {kling_rate:>10} | {opensora_rate:>10} | {overall_rate:>10} |")

if __name__ == "__main__":
    main()