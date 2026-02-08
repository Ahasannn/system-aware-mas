"""
Train latency and length predictors from baseline MAS inference CSV data.

Usage:
    python Experiments/train_latency_length_predictor.py \
        --csv logs/generate_data_for_latency_length_predictor/test_dry_run_v2.csv \
        --save-dir checkpoints/predictors
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import torch
from loguru import logger

# Latency estimator
from MAR.SystemRouter.latency_estimator import (
    prepare_latency_estimator_dataset,
    LatencyEstimator,
    LatencyEstimatorConfig,
    train_latency_estimator,
    save_latency_estimator,
)

# Length estimator
from MAR.SystemRouter.length_estimator import (
    prepare_length_estimator_dataset,
    LengthEstimator,
    LengthEstimatorConfig,
    train_length_estimator,
    save_length_estimator,
    encode_prompts,
    load_length_records_from_csv,
    build_length_estimator_metadata,
    LengthEstimatorDataset,
)
from MAR.SystemRouter.system_aware_router import SemanticEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train latency and length predictors")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the inference CSV with step-level telemetry.",
    )
    parser.add_argument(
        "--record-type",
        type=str,
        default="step",
        help="CSV record_type to filter on (default: step).",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints/predictors", help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency estimator training.",
    )
    parser.add_argument(
        "--skip-length",
        action="store_true",
        help="Skip length estimator training.",
    )
    return parser.parse_args()


def train_latency(args, device):
    logger.info("=" * 60)
    logger.info("LATENCY ESTIMATOR TRAINING")
    logger.info("=" * 60)

    logger.info(f"Loading data from {args.csv} (record_type={args.record_type})")
    dataset, metadata = prepare_latency_estimator_dataset(
        args.csv, record_type=args.record_type
    )
    logger.info(f"Dataset size: {len(dataset)} records")
    logger.info(f"Models: {metadata.model_vocab.tokens}")
    logger.info(f"Roles: {metadata.role_vocab.tokens}")
    logger.info(f"Strategies: {metadata.strategy_vocab.tokens}")

    if len(dataset) == 0:
        logger.error("No records found for latency estimator. Skipping.")
        return

    config = LatencyEstimatorConfig(
        num_numerical_features=metadata.num_numerical_features,
        embedding_dim=16,
        hidden_dims=[128, 64],
        dropout=0.1,
    )
    model = LatencyEstimator(
        num_numerical_features=config.num_numerical_features,
        num_strategies=len(metadata.strategy_vocab),
        num_roles=len(metadata.role_vocab),
        num_models=len(metadata.model_vocab),
        embedding_dim=config.embedding_dim,
        hidden_dims=list(config.hidden_dims),
        dropout=config.dropout,
    )

    logger.info(f"Training for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    start = time.time()
    epoch_losses = train_latency_estimator(
        model,
        dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    elapsed = time.time() - start
    logger.info(f"Training done in {elapsed:.1f}s")
    for i, loss in enumerate(epoch_losses):
        if i % max(1, len(epoch_losses) // 10) == 0 or i == len(epoch_losses) - 1:
            logger.info(f"  Epoch {i+1}/{args.epochs}: loss={loss:.6f}")

    save_path = os.path.join(args.save_dir, "latency_estimator.pth")
    save_latency_estimator(save_path, model, metadata, config)
    logger.info(f"Saved latency estimator checkpoint: {save_path}")
    logger.info(f"Checkpoint size: {os.path.getsize(save_path) / 1024:.1f} KB")


def train_length(args, device):
    logger.info("=" * 60)
    logger.info("LENGTH ESTIMATOR TRAINING")
    logger.info("=" * 60)

    logger.info("Loading SemanticEncoder for prompt embeddings...")
    encoder = SemanticEncoder(device=device)
    logger.info(f"Encoder embedding_dim: {encoder.embedding_dim}")

    logger.info(f"Loading data from {args.csv} (record_type={args.record_type})")
    records = load_length_records_from_csv(args.csv, record_type=args.record_type)
    logger.info(f"Records loaded: {len(records)}")

    if len(records) == 0:
        logger.error("No records found for length estimator. Skipping.")
        return

    metadata = build_length_estimator_metadata(
        records, semantic_dim=encoder.embedding_dim
    )
    logger.info(f"Models: {metadata.model_vocab.tokens}")
    logger.info(f"Roles: {metadata.role_vocab.tokens}")
    logger.info(f"Strategies: {metadata.strategy_vocab.tokens}")

    logger.info("Encoding prompts...")
    embeddings = encode_prompts(
        encoder, [r.prompt for r in records], batch_size=64
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    dataset = LengthEstimatorDataset(records, metadata, embeddings)

    config = LengthEstimatorConfig(
        semantic_dim=encoder.embedding_dim,
        semantic_compress_dim=64,
        embedding_dim=16,
        mlp_hidden=(128, 64, 32),
        dropout=0.1,
    )
    model = LengthEstimator(
        config,
        num_models=len(metadata.model_vocab),
        num_roles=len(metadata.role_vocab),
        num_strategies=len(metadata.strategy_vocab),
    )

    logger.info(f"Training for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    start = time.time()
    epoch_losses = train_length_estimator(
        model,
        dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    elapsed = time.time() - start
    logger.info(f"Training done in {elapsed:.1f}s")
    for i, loss in enumerate(epoch_losses):
        if i % max(1, len(epoch_losses) // 10) == 0 or i == len(epoch_losses) - 1:
            logger.info(f"  Epoch {i+1}/{args.epochs}: loss={loss:.6f}")

    save_path = os.path.join(args.save_dir, "length_estimator.pth")
    save_length_estimator(save_path, model, metadata, config)
    logger.info(f"Saved length estimator checkpoint: {save_path}")
    logger.info(f"Checkpoint size: {os.path.getsize(save_path) / 1024:.1f} KB")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"CSV: {args.csv}")
    logger.info(f"Save dir: {args.save_dir}")

    if not args.skip_latency:
        train_latency(args, device)

    if not args.skip_length:
        train_length(args, device)

    logger.info("All done.")
