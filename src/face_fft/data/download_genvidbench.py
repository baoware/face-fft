"""
Download GenVidBench synthetic (and optionally real) video archives.

By default downloads only the three synthetic generator archives used for
evaluation.  Pass --include-real to also fetch the Real video archive if it
exists in the HuggingFace repo.  Use --list-files to inspect every file in
the repo before deciding what to download.
"""

import argparse
import os
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files, login


REPO_ID = "jian-0/GenVidBench"
HF_PREFIX = "GenVidBench/6.7m"

# Synthetic archives present in the original download
SYNTH_ARCHIVES = {
    "OpenAI_Sora.zip": "Sora",
    "keling.zip": "Kling",
    "OpenSora_13800.tar.gz": "OpenSora",
}

# Candidate filenames for the real-video archive (checked in order)
REAL_ARCHIVE_CANDIDATES = [
    "Real.zip",
    "real.zip",
    "real_videos.zip",
    "Real_videos.zip",
    "youtube.zip",
]


def extract_archive(archive_path: str, extract_to: str) -> None:
    print(f"Extracting {archive_path} -> {extract_to} ...")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    elif archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_to)
    else:
        print(f"  Unknown archive format, skipping extraction: {archive_path}")


def download_archive(
    filename: str,
    folder_name: str,
    gvb_dir: str,
    hf_token: str,
    hf_prefix: str = HF_PREFIX,
) -> bool:
    """Download and extract one archive; returns True on success."""
    extract_path = os.path.join(gvb_dir, folder_name)
    if os.path.exists(extract_path):
        print(f"  [{folder_name}] already extracted at {extract_path}, skipping.")
        return True

    print(f"  Downloading {filename} ...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"{hf_prefix}/{filename}",
            local_dir=gvb_dir,
            token=hf_token,
        )
        extract_archive(downloaded_path, extract_path)
        os.remove(downloaded_path)
        print(f"  [{folder_name}] done.")
        return True
    except Exception as e:
        print(f"  Failed to download/extract {filename}: {e}")
        return False


def find_real_archive(hf_token: str) -> Optional[str]:
    """Scan the repo file listing to find a real-video archive."""
    print("Scanning repo for real-video archive ...")
    try:
        all_files = list(list_repo_files(REPO_ID, repo_type="dataset", token=hf_token))
    except Exception as e:
        print(f"  Could not list repo files: {e}")
        return None

    for candidate in REAL_ARCHIVE_CANDIDATES:
        for f in all_files:
            if f.endswith(candidate) or f.endswith(candidate.lower()):
                print(f"  Found real-video archive: {f}")
                return f  # full repo path, e.g. "GenVidBench/6.7m/Real.zip"

    # Fallback: print everything under the prefix so the user can decide
    prefix_files = [f for f in all_files if HF_PREFIX in f]
    print(f"  No known real-video archive found.  Files under {HF_PREFIX}:")
    for f in prefix_files:
        print(f"    {f}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GenVidBench dataset archives"
    )
    parser.add_argument(
        "--base_dir",
        default="/scratch/rjr6zk/face-fft/src/face_fft/data",
        help="Parent directory for the genvidbench_dataset/ folder",
    )
    parser.add_argument(
        "--include-real",
        action="store_true",
        default=False,
        help="Also download the real-video archive (needed for combined training)",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        default=False,
        help="Print all files in the HuggingFace repo and exit",
    )
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set.")
    login(token=hf_token)

    if args.list_files:
        print(f"Files in {REPO_ID}:")
        for f in list_repo_files(REPO_ID, repo_type="dataset", token=hf_token):
            print(f"  {f}")
        return

    gvb_dir = os.path.join(args.base_dir, "genvidbench_dataset")
    os.makedirs(gvb_dir, exist_ok=True)

    # --- Synthetic archives ---
    print("=== Downloading synthetic archives ===")
    for filename, folder_name in SYNTH_ARCHIVES.items():
        download_archive(filename, folder_name, gvb_dir, hf_token)

    # --- Real-video archive (optional) ---
    if args.include_real:
        print("\n=== Downloading real-video archive ===")
        real_repo_path = find_real_archive(hf_token)
        if real_repo_path is not None:
            # Derive just the basename and a "Real" output folder
            real_filename = Path(real_repo_path).name
            real_hf_prefix = str(Path(real_repo_path).parent)
            download_archive(
                real_filename, "Real", gvb_dir, hf_token, hf_prefix=real_hf_prefix
            )
        else:
            print(
                "\nNo real-video archive could be located automatically.\n"
                "If you know the filename, download it manually with:\n"
                '  python -c "from huggingface_hub import hf_hub_download; '
                "hf_hub_download('jian-0/GenVidBench', 'GenVidBench/6.7m/<filename>', "
                "repo_type='dataset', local_dir='<gvb_dir>')\"\n"
                "Then extract it into genvidbench_dataset/Real/.\n"
                "\nAlternatively, train_combined.py will automatically fall back to "
                "DeepAction Pexels real videos if genvidbench_dataset/Real/ is absent."
            )

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
