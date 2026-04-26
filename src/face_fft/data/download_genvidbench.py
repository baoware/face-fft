import os
import zipfile
import tarfile
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
import shutil

def extract_archive(archive_path, extract_to):
    print(f"Extracting {archive_path}...")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        print(f"Unknown archive format: {archive_path}")

def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError()
    login(token=HF_TOKEN)

    base_dir = "/scratch/rjr6zk/face-fft/src/face_fft/data"
    gvb_dir = os.path.join(base_dir, "genvidbench_dataset")
    os.makedirs(gvb_dir, exist_ok=True)

    repo_id = "jian-0/GenVidBench"
    
    target_files = {
        "OpenAI_Sora.zip": "Sora",
        "keling.zip": "Kling",
        "OpenSora_13800.tar.gz": "OpenSora"
    }

    for filename, folder_name in target_files.items():
        extract_path = os.path.join(gvb_dir, folder_name)
        if os.path.exists(extract_path):
            print(f"{folder_name} already extracted at {extract_path}")
            continue
            
        print(f"Downloading {filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"GenVidBench/6.7m/{filename}",
                local_dir=gvb_dir,
                token=HF_TOKEN
            )
            extract_archive(downloaded_path, extract_path)
            os.remove(downloaded_path) # Clean up zip to save space
        except Exception as e:
            print(f"Failed to download/extract {filename}: {e}")

if __name__ == "__main__":
    main()