# MBPP Offline Dataset Setup

This guide explains how to download and use the MBPP dataset from blue storage for offline training and testing.

## One-Time Setup: Download Dataset

Run the download script once to save the MBPP dataset to blue storage:

```bash
bash scripts/download_mbpp_dataset.sh
```

This will:
- Download all MBPP splits (train, test, validation, prompt) from HuggingFace
- Save them to `/blue/qi855292.ucf/ji757406.ucf/datasets/mbpp/full/`
- Create parquet files that can be loaded without internet access

**Expected output:**
```
[MBPP Download] Target directory: /blue/qi855292.ucf/ji757406.ucf/datasets/mbpp/full
[MBPP Download] Downloading train from HuggingFace...
[MBPP Download] train: 374 examples saved
[MBPP Download] Downloading test from HuggingFace...
[MBPP Download] test: 500 examples saved
[MBPP Download] Downloading validation from HuggingFace...
[MBPP Download] validation: 90 examples saved
[MBPP Download] Downloading prompt from HuggingFace...
[MBPP Download] prompt: 10 examples saved
[MBPP Download] ✓ All splits downloaded successfully!
```

## Using Offline Dataset

The training and testing scripts are already configured to use offline storage:

### Training
```bash
bash scripts/mas_train_mbpp.sh
```

### Testing
```bash
bash scripts/mas_test_mbpp.sh
```

Both scripts automatically set:
```bash
export MBPP_DATASET_PATH="/blue/qi855292.ucf/ji757406.ucf/datasets/mbpp/full"
```

## How It Works

The `MbppDataset` class in `Datasets/mbpp_dataset.py` checks for the `MBPP_DATASET_PATH` environment variable:

1. **If set and path exists**: Loads dataset from blue storage (offline mode)
   ```
   [MBPP Dataset] Loading train from offline storage: /blue/.../train-00000-of-00001.parquet
   ```

2. **If not set**: Falls back to HuggingFace online download
   ```
   [MBPP Dataset] Loading train from HuggingFace (online)
   ```

## Benefits

- **No internet required** during training/testing
- **Faster loading** from local storage
- **Consistent data** across runs
- **Saves home directory space** (datasets stored in blue storage)

## Storage Location

```
/blue/qi855292.ucf/ji757406.ucf/
├── datasets/
│   └── mbpp/
│       └── full/
│           ├── train-00000-of-00001.parquet      (~50KB)
│           ├── test-00000-of-00001.parquet       (~65KB)
│           ├── validation-00000-of-00001.parquet (~12KB)
│           └── prompt-00000-of-00001.parquet     (~2KB)
└── checkpoints/
    └── mas_router/
        └── mas_mbpp_train_6.pth
```

## Troubleshooting

### Dataset not found error
```
FileNotFoundError: Offline dataset file not found
```
**Solution**: Run the download script:
```bash
bash scripts/download_mbpp_dataset.sh
```

### Falls back to online mode
If you see `[MBPP Dataset] Loading from HuggingFace (online)`, the environment variable is not set.

**Solution**: Ensure your script sets:
```bash
export MBPP_DATASET_PATH="/blue/qi855292.ucf/ji757406.ucf/datasets/mbpp/full"
```
