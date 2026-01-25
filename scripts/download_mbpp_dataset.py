#!/usr/bin/env python3
"""
Download MBPP dataset from HuggingFace and save to blue storage.
This script downloads all splits (train, test, validation, prompt) and saves them as parquet files.
"""

import os
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

# Blue storage configuration
BLUE_STORAGE = os.getenv("BLUE_STORAGE", "/blue/qi855292.ucf/ji757406.ucf")
DATASET_DIR = Path(BLUE_STORAGE) / "datasets" / "mbpp" / "full"

# MBPP dataset configuration
DATASET_REPO = "google-research-datasets/mbpp"
SPLITS = {
    'train': 'full/train-00000-of-00001.parquet',
    'test': 'full/test-00000-of-00001.parquet',
    'validation': 'full/validation-00000-of-00001.parquet',
    'prompt': 'full/prompt-00000-of-00001.parquet'
}

def download_mbpp_dataset():
    """Download all MBPP dataset splits to blue storage."""
    print(f"[MBPP Download] Target directory: {DATASET_DIR}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_path in SPLITS.items():
        output_file = DATASET_DIR / f"{split_name}-00000-of-00001.parquet"

        if output_file.exists():
            print(f"[MBPP Download] {split_name} already exists at {output_file}")
            # Verify it's readable
            try:
                df = pd.read_parquet(output_file)
                print(f"[MBPP Download] {split_name}: {len(df)} examples verified")
                continue
            except Exception as e:
                print(f"[MBPP Download] {split_name} corrupted, re-downloading: {e}")

        print(f"[MBPP Download] Downloading {split_name} from HuggingFace...")
        try:
            # Download from HuggingFace hub
            hf_file = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=split_path,
                repo_type="dataset",
                cache_dir=str(DATASET_DIR.parent / "hf_cache")
            )

            # Load and save to our target location
            df = pd.read_parquet(hf_file)
            df.to_parquet(output_file, index=False)

            print(f"[MBPP Download] {split_name}: {len(df)} examples saved to {output_file}")

        except Exception as e:
            print(f"[MBPP Download] Error downloading {split_name}: {e}")
            raise

    print(f"\n[MBPP Download] All splits downloaded successfully!")
    print(f"[MBPP Download] Location: {DATASET_DIR}")
    print(f"\nTo use offline dataset, set environment variable:")
    print(f'  export MBPP_DATASET_PATH="{DATASET_DIR}"')

if __name__ == "__main__":
    download_mbpp_dataset()
