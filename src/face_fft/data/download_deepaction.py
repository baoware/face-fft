import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError()
login(token=HF_TOKEN)

save_directory = "/scratch/rjr6zk/face-fft/src/face_fft/data/deepaction_dataset"
os.makedirs(save_directory, exist_ok=True)

snapshot_download(
    repo_id="faridlab/deepaction_v1",
    repo_type="dataset",
    local_dir=save_directory,
    local_dir_use_symlinks=False,
    token=HF_TOKEN
)

print("Download complete.")