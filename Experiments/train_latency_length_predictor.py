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
from MAR.InfraMind.latency_estimator import (
    LatencyEstimator,
    LatencyEstimatorConfig,
    LatencyEstimatorDataset,
    save_latency_estimator,
    load_latency_records_from_csv,
    build_latency_estimator_metadata,
)

# Length estimator
from MAR.InfraMind.length_estimator import (
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

# Quality estimator
from MAR.InfraMind.quality_estimator import (
    QualityEstimator,
    QualityEstimatorConfig,
    QualityEstimatorDataset,
    load_quality_records_from_csv,
    build_quality_estimator_metadata,
    train_quality_estimator,
    save_quality_estimator,
)
from MAR.InfraMind.inframind_router import SemanticEncoder


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
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip quality estimator training.",
    )
    parser.add_argument(
        "--length-epochs",
        type=int,
        default=None,
        help="Override epoch count for length estimator (default: use --epochs).",
    )
    parser.add_argument(
        "--latency-epochs",
        type=int,
        default=None,
        help="Override epoch count for latency estimator (default: use --epochs).",
    )
    return parser.parse_args()


def _evaluate_latency(model, dataset, device, split_name="test", batch_size=64, log_transform=False, metadata=None):
    """Evaluate latency estimator with per-head metrics in original space."""
    from torch.utils.data import DataLoader
    import math as _math

    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_ttft_pred, all_tpot_pred = [], []
    all_ttft_true, all_tpot_true = [], []
    with torch.no_grad():
        for x_num, strategy_id, role_id, model_id, ttft_target, tpot_target in loader:
            ttft_pred, tpot_pred = model(
                x_num.to(device), strategy_id.to(device),
                role_id.to(device), model_id.to(device),
            )
            all_ttft_pred.append(ttft_pred.cpu())
            all_tpot_pred.append(tpot_pred.cpu())
            all_ttft_true.append(ttft_target)
            all_tpot_true.append(tpot_target)

    ttft_pred = torch.cat(all_ttft_pred)
    tpot_pred = torch.cat(all_tpot_pred)
    ttft_true = torch.cat(all_ttft_true)
    tpot_true = torch.cat(all_tpot_true)

    # Inverse-transform back to original latency space
    if log_transform and metadata is not None:
        ttft_pred = torch.expm1(ttft_pred * max(metadata.ttft_log_std, 1e-8) + metadata.ttft_log_mean).clamp(min=0)
        tpot_pred = torch.expm1(tpot_pred * max(metadata.tpot_log_std, 1e-8) + metadata.tpot_log_mean).clamp(min=0)
        ttft_true = torch.expm1(ttft_true * max(metadata.ttft_log_std, 1e-8) + metadata.ttft_log_mean).clamp(min=0)
        tpot_true = torch.expm1(tpot_true * max(metadata.tpot_log_std, 1e-8) + metadata.tpot_log_mean).clamp(min=0)

    n = len(ttft_pred)

    def _head_metrics(pred, true, name):
        errors = pred - true
        abs_errors = errors.abs()
        mse = (errors ** 2).mean().item()
        rmse = _math.sqrt(mse)
        mae = abs_errors.mean().item()
        median_ae = abs_errors.median().item()
        ss_res = (errors ** 2).sum().item()
        ss_tot = ((true - true.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

        logger.info(f"  --- {name} ---")
        logger.info(f"    MSE      = {mse:.6f}")
        logger.info(f"    RMSE     = {rmse:.6f}")
        logger.info(f"    MAE      = {mae:.6f}")
        logger.info(f"    MedAE    = {median_ae:.6f}")
        logger.info(f"    R²       = {r2:.4f}")
        logger.info(f"    Target  : mean={true.mean():.4f}, std={true.std():.4f}, "
                     f"min={true.min():.4f}, max={true.max():.4f}")
        logger.info(f"    Predict : mean={pred.mean():.4f}, std={pred.std():.4f}, "
                     f"min={pred.min():.4f}, max={pred.max():.4f}")

        # Percentage-based accuracy
        within_10pct = ((abs_errors / true.clamp(min=1e-6)) <= 0.10).float().mean().item() * 100
        within_25pct = ((abs_errors / true.clamp(min=1e-6)) <= 0.25).float().mean().item() * 100
        within_50pct = ((abs_errors / true.clamp(min=1e-6)) <= 0.50).float().mean().item() * 100
        logger.info(f"    Within ±10%: {within_10pct:.1f}%")
        logger.info(f"    Within ±25%: {within_25pct:.1f}%")
        logger.info(f"    Within ±50%: {within_50pct:.1f}%")
        return {"mse": mse, "rmse": rmse, "mae": mae, "median_ae": median_ae, "r2": r2}

    logger.info(f"")
    logger.info(f"{'=' * 60}")
    logger.info(f"LATENCY PREDICTOR — {split_name.upper()} EVALUATION  (n={n})")
    logger.info(f"{'=' * 60}")
    ttft_m = _head_metrics(ttft_pred, ttft_true, "TTFT")
    tpot_m = _head_metrics(tpot_pred, tpot_true, "TPOT")

    # Sample predictions
    logger.info(f"")
    logger.info(f"  Sample predictions (first 10 from {split_name}):")
    logger.info(f"  {'TTFT_pred':>10} {'TTFT_true':>10} {'TPOT_pred':>10} {'TPOT_true':>10}")
    logger.info(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(min(10, n)):
        logger.info(f"  {ttft_pred[i].item():10.4f} {ttft_true[i].item():10.4f} "
                     f"{tpot_pred[i].item():10.4f} {tpot_true[i].item():10.4f}")
    logger.info(f"{'=' * 60}")

    return {"ttft": ttft_m, "tpot": tpot_m}


def train_latency(args, device):
    logger.info("=" * 60)
    logger.info("LATENCY ESTIMATOR TRAINING")
    logger.info("=" * 60)

    logger.info(f"Loading data from {args.csv} (record_type={args.record_type})")
    records = load_latency_records_from_csv(args.csv, record_type=args.record_type)
    logger.info(f"Records loaded: {len(records)}")

    if len(records) == 0:
        logger.error("No records found for latency estimator. Skipping.")
        return

    # --- Train / Val / Test split (80/10/10, by item_id for no leakage) ---
    import random
    from collections import defaultdict

    item_to_indices = defaultdict(list)
    for i, rec in enumerate(records):
        key = rec.item_id if rec.item_id else str(i)
        item_to_indices[key].append(i)

    unique_items = sorted(item_to_indices.keys())
    rng = random.Random(42)
    rng.shuffle(unique_items)

    n_items = len(unique_items)
    n_train = int(n_items * 0.80)
    n_val = int(n_items * 0.10)

    train_items = set(unique_items[:n_train])
    val_items = set(unique_items[n_train:n_train + n_val])
    test_items = set(unique_items[n_train + n_val:])

    train_idx = [i for item in train_items for i in item_to_indices[item]]
    val_idx = [i for item in val_items for i in item_to_indices[item]]
    test_idx = [i for item in test_items for i in item_to_indices[item]]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    logger.info(f"Split — unique items: {n_items} "
                f"(train={len(train_items)}, val={len(val_items)}, test={len(test_items)})")
    logger.info(f"Split — records: total={len(records)} "
                f"(train={len(train_records)}, val={len(val_records)}, test={len(test_records)})")

    # Build metadata from TRAIN only (includes target normalization stats)
    metadata = build_latency_estimator_metadata(
        train_records, compute_target_stats=True,
    )
    logger.info(f"Models: {metadata.model_vocab.tokens}")
    logger.info(f"Roles: {metadata.role_vocab.tokens}")
    logger.info(f"Strategies: {metadata.strategy_vocab.tokens}")

    # Log target distributions
    raw_ttft = [r.ttft for r in train_records]
    raw_tpot = [r.tpot for r in train_records]
    logger.info(f"TTFT stats (train): mean={sum(raw_ttft)/len(raw_ttft):.4f}, "
                f"min={min(raw_ttft):.4f}, max={max(raw_ttft):.4f}")
    logger.info(f"TPOT stats (train): mean={sum(raw_tpot)/len(raw_tpot):.4f}, "
                f"min={min(raw_tpot):.4f}, max={max(raw_tpot):.4f}")

    log_transform = True
    logger.info(f"Log-transform: {log_transform}")
    logger.info(f"Target normalization — TTFT: mean={metadata.ttft_log_mean:.4f}, std={metadata.ttft_log_std:.4f}")
    logger.info(f"Target normalization — TPOT: mean={metadata.tpot_log_mean:.4f}, std={metadata.tpot_log_std:.4f}")

    config = LatencyEstimatorConfig(
        num_numerical_features=metadata.num_numerical_features,
        embedding_dim=16,
        hidden_dims=[128, 64, 32],
        dropout=0.1,
        log_transform=log_transform,
    )

    train_dataset = LatencyEstimatorDataset(train_records, metadata, log_transform=log_transform)
    val_dataset = LatencyEstimatorDataset(val_records, metadata, log_transform=log_transform)
    test_dataset = LatencyEstimatorDataset(test_records, metadata, log_transform=log_transform)

    model = LatencyEstimator(
        num_numerical_features=config.num_numerical_features,
        num_strategies=len(metadata.strategy_vocab),
        num_roles=len(metadata.role_vocab),
        num_models=len(metadata.model_vocab),
        embedding_dim=config.embedding_dim,
        hidden_dims=list(config.hidden_dims),
        dropout=config.dropout,
    )

    # --- Training with val monitoring and early stopping ---
    latency_epochs = getattr(args, "latency_epochs", None) or args.epochs
    logger.info(f"Training for {latency_epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    best_val_loss = float("inf")
    best_state = None
    patience = 20
    patience_counter = 0

    from torch.utils.data import DataLoader
    import copy

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    start = time.time()
    for epoch in range(latency_epochs):
        # Train
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x_num, strategy_id, role_id, model_id, ttft_target, tpot_target in train_loader:
            x_num = x_num.to(device)
            strategy_id = strategy_id.to(device)
            role_id = role_id.to(device)
            model_id = model_id.to(device)
            ttft_target = ttft_target.to(device)
            tpot_target = tpot_target.to(device)

            optimizer.zero_grad()
            ttft_pred, tpot_pred = model(x_num, strategy_id, role_id, model_id)
            loss = loss_fn(ttft_pred, ttft_target) + loss_fn(tpot_pred, tpot_target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x_num.size(0)
            train_n += x_num.size(0)
        train_loss = train_loss_sum / max(train_n, 1)

        # Validate
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x_num, strategy_id, role_id, model_id, ttft_target, tpot_target in val_loader:
                ttft_pred, tpot_pred = model(
                    x_num.to(device), strategy_id.to(device),
                    role_id.to(device), model_id.to(device),
                )
                loss = loss_fn(ttft_pred, ttft_target.to(device)) + loss_fn(tpot_pred, tpot_target.to(device))
                val_loss_sum += loss.item() * x_num.size(0)
                val_n += x_num.size(0)
        val_loss = val_loss_sum / max(val_n, 1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if epoch % max(1, latency_epochs // 20) == 0 or epoch == latency_epochs - 1 or marker:
            logger.info(f"  Epoch {epoch+1:>4}/{latency_epochs}: "
                        f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{marker}")

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    elapsed = time.time() - start
    logger.info(f"Training done in {elapsed:.1f}s")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model (val_loss={best_val_loss:.6f})")

    # --- Evaluate on all splits (metrics in original latency space) ---
    _evaluate_latency(model, train_dataset, device, split_name="train",
                      batch_size=args.batch_size, log_transform=log_transform, metadata=metadata)
    _evaluate_latency(model, val_dataset, device, split_name="val",
                      batch_size=args.batch_size, log_transform=log_transform, metadata=metadata)
    _evaluate_latency(model, test_dataset, device, split_name="test",
                      batch_size=args.batch_size, log_transform=log_transform, metadata=metadata)

    save_path = os.path.join(args.save_dir, "latency_estimator.pth")
    save_latency_estimator(save_path, model, metadata, config)
    logger.info(f"Saved latency estimator checkpoint: {save_path}")
    logger.info(f"Checkpoint size: {os.path.getsize(save_path) / 1024:.1f} KB")


def _evaluate_length(model, dataset, device, split_name="test", batch_size=64, log_transform=False):
    """Evaluate length estimator and print detailed metrics (in original token space)."""
    from torch.utils.data import DataLoader
    import math as _math

    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for semantic, ptc, strategy_id, role_id, model_id, targets in loader:
            preds = model(
                semantic.to(device),
                ptc.to(device),
                strategy_id.to(device),
                role_id.to(device),
                model_id.to(device),
            )
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Inverse-transform back to token space for interpretable metrics
    if log_transform:
        preds = torch.expm1(preds)
        targets = torch.expm1(targets)

    n = len(preds)
    errors = preds - targets
    abs_errors = errors.abs()
    mse = (errors ** 2).mean().item()
    rmse = _math.sqrt(mse)
    mae = abs_errors.mean().item()
    # R²
    ss_res = (errors ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    # Median absolute error
    median_ae = abs_errors.median().item()

    logger.info(f"")
    logger.info(f"{'=' * 60}")
    logger.info(f"LENGTH PREDICTOR — {split_name.upper()} EVALUATION  (n={n})")
    logger.info(f"{'=' * 60}")
    logger.info(f"  MSE      = {mse:.2f}")
    logger.info(f"  RMSE     = {rmse:.2f} tokens")
    logger.info(f"  MAE      = {mae:.2f} tokens")
    logger.info(f"  MedAE    = {median_ae:.2f} tokens")
    logger.info(f"  R²       = {r2:.4f}")

    # Percentage-based accuracy buckets
    within_10pct = ((abs_errors / targets.clamp(min=1.0)) <= 0.10).float().mean().item() * 100
    within_25pct = ((abs_errors / targets.clamp(min=1.0)) <= 0.25).float().mean().item() * 100
    within_50pct = ((abs_errors / targets.clamp(min=1.0)) <= 0.50).float().mean().item() * 100
    logger.info(f"  Within ±10% of true length: {within_10pct:.1f}%")
    logger.info(f"  Within ±25% of true length: {within_25pct:.1f}%")
    logger.info(f"  Within ±50% of true length: {within_50pct:.1f}%")

    logger.info(f"")
    logger.info(f"  Target  distribution: mean={targets.mean():.1f}, std={targets.std():.1f}, "
                f"min={targets.min():.0f}, max={targets.max():.0f}")
    logger.info(f"  Predict distribution: mean={preds.mean():.1f}, std={preds.std():.1f}, "
                f"min={preds.min():.0f}, max={preds.max():.0f}")

    logger.info(f"")
    logger.info(f"  Sample predictions (first 10 from {split_name}):")
    logger.info(f"  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}")
    logger.info(f"  {'-'*10}  {'-'*8}  {'-'*8}")
    for i in range(min(10, n)):
        logger.info(f"  {preds[i].item():10.1f}  {targets[i].item():8.0f}  {errors[i].item():+8.1f}")
    logger.info(f"{'=' * 60}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "median_ae": median_ae, "r2": r2,
            "within_10_pct": within_10pct, "within_25_pct": within_25pct, "within_50_pct": within_50pct}


def train_length(args, device, encoder=None):
    logger.info("=" * 60)
    logger.info("LENGTH ESTIMATOR TRAINING")
    logger.info("=" * 60)

    # Reuse encoder if already loaded
    if encoder is None:
        logger.info("Loading SemanticEncoder for prompt embeddings...")
        encoder = SemanticEncoder(device=device)
    logger.info(f"Encoder embedding_dim: {encoder.embedding_dim}")

    logger.info(f"Loading data from {args.csv} (record_type={args.record_type})")
    records = load_length_records_from_csv(args.csv, record_type=args.record_type)
    logger.info(f"Records loaded: {len(records)}")

    if len(records) == 0:
        logger.error("No records found for length estimator. Skipping.")
        return

    # --- Train / Val / Test split (80/10/10, by item_id for no leakage) ---
    import random
    from collections import defaultdict

    item_to_indices = defaultdict(list)
    for i, rec in enumerate(records):
        item_to_indices[rec.item_id].append(i)

    unique_items = sorted(item_to_indices.keys())
    rng = random.Random(42)
    rng.shuffle(unique_items)

    n_items = len(unique_items)
    n_train = int(n_items * 0.80)
    n_val = int(n_items * 0.10)

    train_items = set(unique_items[:n_train])
    val_items = set(unique_items[n_train:n_train + n_val])
    test_items = set(unique_items[n_train + n_val:])

    train_idx = [i for item in train_items for i in item_to_indices[item]]
    val_idx = [i for item in val_items for i in item_to_indices[item]]
    test_idx = [i for item in test_items for i in item_to_indices[item]]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    logger.info(f"Split — unique items (queries): {n_items} "
                f"(train={len(train_items)}, val={len(val_items)}, test={len(test_items)})")
    logger.info(f"Split — records: total={len(records)} "
                f"(train={len(train_records)}, val={len(val_records)}, test={len(test_records)})")

    # Build metadata from TRAIN only
    metadata = build_length_estimator_metadata(
        train_records, semantic_dim=encoder.embedding_dim
    )
    logger.info(f"Models: {metadata.model_vocab.tokens}")
    logger.info(f"Roles: {metadata.role_vocab.tokens}")
    logger.info(f"Strategies: {metadata.strategy_vocab.tokens}")

    # Encode all queries at once (prompt field is now 'query', not 'prompt_base')
    logger.info("Encoding queries...")
    all_embeddings = encode_prompts(encoder, [r.prompt for r in records], batch_size=64)
    logger.info(f"Embeddings shape: {all_embeddings.shape}")

    train_emb = all_embeddings[train_idx]
    val_emb = all_embeddings[val_idx]
    test_emb = all_embeddings[test_idx]

    # Log target and feature distributions
    raw_targets = [float(r.output_length) for r in train_records]
    logger.info(f"Target stats (train, raw tokens): mean={sum(raw_targets)/len(raw_targets):.1f}, "
                f"min={min(raw_targets):.0f}, max={max(raw_targets):.0f}")
    raw_ptc = [float(r.prompt_tokens) for r in train_records]
    logger.info(f"Prompt tokens stats (train): mean={sum(raw_ptc)/len(raw_ptc):.1f}, "
                f"min={min(raw_ptc):.0f}, max={max(raw_ptc):.0f}")

    log_transform = True
    logger.info(f"Log-transform: {log_transform} (training on log1p(tokens), expm1 at inference)")

    config = LengthEstimatorConfig(
        semantic_dim=encoder.embedding_dim,
        semantic_compress_dim=64,
        embedding_dim=16,
        mlp_hidden=(128, 64, 32),
        dropout=0.1,
        log_transform=log_transform,
    )

    train_dataset = LengthEstimatorDataset(train_records, metadata, train_emb, log_transform=log_transform)
    val_dataset = LengthEstimatorDataset(val_records, metadata, val_emb, log_transform=log_transform)
    test_dataset = LengthEstimatorDataset(test_records, metadata, test_emb, log_transform=log_transform)

    model = LengthEstimator(
        config,
        num_models=len(metadata.model_vocab),
        num_roles=len(metadata.role_vocab),
        num_strategies=len(metadata.strategy_vocab),
    )

    # --- Training with val monitoring and early stopping ---
    length_epochs = getattr(args, "length_epochs", None) or args.epochs
    logger.info(f"Training for {length_epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    best_val_loss = float("inf")
    best_state = None
    patience = 20
    patience_counter = 0

    from torch.utils.data import DataLoader
    import copy

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    start = time.time()
    for epoch in range(length_epochs):
        # Train
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for semantic, ptc, strategy_id, role_id, model_id, targets in train_loader:
            semantic = semantic.to(device)
            ptc = ptc.to(device)
            strategy_id = strategy_id.to(device)
            role_id = role_id.to(device)
            model_id = model_id.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(semantic, ptc, strategy_id, role_id, model_id)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * semantic.size(0)
            train_n += semantic.size(0)
        train_loss = train_loss_sum / max(train_n, 1)

        # Validate
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for semantic, ptc, strategy_id, role_id, model_id, targets in val_loader:
                preds = model(
                    semantic.to(device), ptc.to(device),
                    strategy_id.to(device), role_id.to(device),
                    model_id.to(device),
                )
                loss = loss_fn(preds, targets.to(device))
                val_loss_sum += loss.item() * semantic.size(0)
                val_n += semantic.size(0)
        val_loss = val_loss_sum / max(val_n, 1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if epoch % max(1, length_epochs // 20) == 0 or epoch == length_epochs - 1 or marker:
            logger.info(f"  Epoch {epoch+1:>4}/{length_epochs}: "
                        f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{marker}")

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    elapsed = time.time() - start
    logger.info(f"Training done in {elapsed:.1f}s")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model (val_loss={best_val_loss:.6f})")

    # --- Evaluate on all splits (metrics in original token space) ---
    _evaluate_length(model, train_dataset, device, split_name="train",
                     batch_size=args.batch_size, log_transform=log_transform)
    _evaluate_length(model, val_dataset, device, split_name="val",
                     batch_size=args.batch_size, log_transform=log_transform)
    _evaluate_length(model, test_dataset, device, split_name="test",
                     batch_size=args.batch_size, log_transform=log_transform)

    save_path = os.path.join(args.save_dir, "length_estimator.pth")
    save_length_estimator(save_path, model, metadata, config)
    logger.info(f"Saved length estimator checkpoint: {save_path}")
    logger.info(f"Checkpoint size: {os.path.getsize(save_path) / 1024:.1f} KB")


def _evaluate_quality(model, dataset, device, split_name="test", batch_size=64):
    """Evaluate quality estimator and print detailed metrics."""
    from torch.utils.data import DataLoader
    import math as _math

    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for semantic, response_emb, strategy_id, role_id, model_id, targets in loader:
            preds = model(
                semantic.to(device),
                response_emb.to(device),
                strategy_id.to(device),
                role_id.to(device),
                model_id.to(device),
            )
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    n = len(preds)

    # --- Regression metrics ---
    errors = preds - targets
    abs_errors = errors.abs()
    mse = (errors ** 2).mean().item()
    rmse = _math.sqrt(mse)
    mae = abs_errors.mean().item()
    # R^2
    ss_res = (errors ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    logger.info(f"")
    logger.info(f"{'=' * 60}")
    logger.info(f"QUALITY PREDICTOR — {split_name.upper()} EVALUATION  (n={n})")
    logger.info(f"{'=' * 60}")
    logger.info(f"  MSE  = {mse:.4f}")
    logger.info(f"  RMSE = {rmse:.4f}")
    logger.info(f"  MAE  = {mae:.4f}")
    logger.info(f"  R²   = {r2:.4f}")

    # --- Per-score-bucket accuracy (within ±1 of true score) ---
    within_1 = (abs_errors <= 1.0).float().mean().item() * 100
    within_2 = (abs_errors <= 2.0).float().mean().item() * 100
    logger.info(f"  Within ±1 of true score: {within_1:.1f}%")
    logger.info(f"  Within ±2 of true score: {within_2:.1f}%")

    # --- Score distribution ---
    logger.info(f"")
    logger.info(f"  Target  distribution: mean={targets.mean():.2f}, std={targets.std():.2f}, "
                f"min={targets.min():.1f}, max={targets.max():.1f}")
    logger.info(f"  Predict distribution: mean={preds.mean():.2f}, std={preds.std():.2f}, "
                f"min={preds.min():.1f}, max={preds.max():.1f}")

    # --- Sample predictions (first 10) ---
    logger.info(f"")
    logger.info(f"  Sample predictions (first 10 from {split_name}):")
    logger.info(f"  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}")
    logger.info(f"  {'-'*10}  {'-'*8}  {'-'*8}")
    for i in range(min(10, n)):
        logger.info(f"  {preds[i].item():10.2f}  {targets[i].item():8.1f}  {errors[i].item():+8.2f}")
    logger.info(f"{'=' * 60}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
            "within_1_pct": within_1, "within_2_pct": within_2}


def train_quality(args, device, encoder=None):
    logger.info("=" * 60)
    logger.info("QUALITY ESTIMATOR TRAINING")
    logger.info("=" * 60)

    # Reuse encoder if already loaded (e.g. from length training)
    if encoder is None:
        logger.info("Loading SemanticEncoder for prompt embeddings...")
        encoder = SemanticEncoder(device=device)
    logger.info(f"Encoder embedding_dim: {encoder.embedding_dim}")

    logger.info(f"Loading data from {args.csv} (record_type={args.record_type})")
    records = load_quality_records_from_csv(args.csv, record_type=args.record_type)
    logger.info(f"Records loaded: {len(records)}")

    if len(records) == 0:
        logger.error("No records found for quality estimator (missing judge_score column?). Skipping.")
        return

    # --- Train / Val / Test split (80/10/10, by item_id for no leakage) ---
    import random
    # Group records by item_id (underlying query / problem) so all steps
    # for the same problem stay in the same split (no data leakage).
    from collections import defaultdict
    item_to_indices = defaultdict(list)
    for i, rec in enumerate(records):
        item_to_indices[rec.item_id].append(i)

    unique_items = sorted(item_to_indices.keys())
    rng = random.Random(42)
    rng.shuffle(unique_items)

    n_items = len(unique_items)
    n_train = int(n_items * 0.80)
    n_val = int(n_items * 0.10)

    train_items = set(unique_items[:n_train])
    val_items = set(unique_items[n_train:n_train + n_val])
    test_items = set(unique_items[n_train + n_val:])

    train_idx = [i for item in train_items for i in item_to_indices[item]]
    val_idx = [i for item in val_items for i in item_to_indices[item]]
    test_idx = [i for item in test_items for i in item_to_indices[item]]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    logger.info(f"Split — unique items (queries): {n_items} "
                f"(train={len(train_items)}, val={len(val_items)}, test={len(test_items)})")
    logger.info(f"Split — records: total={len(records)} "
                f"(train={len(train_records)}, val={len(val_records)}, test={len(test_records)})")

    # Build metadata from TRAIN only (prevents test leakage into vocab)
    metadata = build_quality_estimator_metadata(
        train_records, semantic_dim=encoder.embedding_dim
    )
    logger.info(f"Models: {metadata.model_vocab.tokens}")
    logger.info(f"Roles: {metadata.role_vocab.tokens}")
    logger.info(f"Strategies: {metadata.strategy_vocab.tokens}")

    # Encode all prompts and responses at once
    logger.info("Encoding prompts...")
    all_prompts = [rec.prompt for rec in records]
    all_prompt_embeddings = encode_prompts(encoder, all_prompts, batch_size=64)
    logger.info(f"Prompt embeddings shape: {all_prompt_embeddings.shape}")

    logger.info("Encoding responses...")
    all_responses = [rec.response or "" for rec in records]
    all_response_embeddings = encode_prompts(encoder, all_responses, batch_size=64)
    logger.info(f"Response embeddings shape: {all_response_embeddings.shape}")

    train_prompt_emb = all_prompt_embeddings[train_idx]
    val_prompt_emb = all_prompt_embeddings[val_idx]
    test_prompt_emb = all_prompt_embeddings[test_idx]

    train_response_emb = all_response_embeddings[train_idx]
    val_response_emb = all_response_embeddings[val_idx]
    test_response_emb = all_response_embeddings[test_idx]

    train_dataset = QualityEstimatorDataset(train_records, metadata, train_prompt_emb, train_response_emb)
    val_dataset = QualityEstimatorDataset(val_records, metadata, val_prompt_emb, val_response_emb)
    test_dataset = QualityEstimatorDataset(test_records, metadata, test_prompt_emb, test_response_emb)

    config = QualityEstimatorConfig(
        semantic_dim=encoder.embedding_dim,
        semantic_compress_dim=64,
        embedding_dim=16,
        mlp_hidden=(128, 64, 32),
        dropout=0.1,
    )
    model = QualityEstimator(
        config,
        num_models=len(metadata.model_vocab),
        num_roles=len(metadata.role_vocab),
        num_strategies=len(metadata.strategy_vocab),
    )

    # --- Training with val monitoring ---
    logger.info(f"Training for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    best_val_loss = float("inf")
    best_state = None
    patience = 15
    patience_counter = 0

    from torch.utils.data import DataLoader
    import copy

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    start = time.time()
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for semantic, response_emb, strategy_id, role_id, model_id, targets in train_loader:
            semantic = semantic.to(device)
            response_emb = response_emb.to(device)
            strategy_id = strategy_id.to(device)
            role_id = role_id.to(device)
            model_id = model_id.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(semantic, response_emb, strategy_id, role_id, model_id)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * semantic.size(0)
            train_n += semantic.size(0)
        train_loss = train_loss_sum / max(train_n, 1)

        # Validate
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for semantic, response_emb, strategy_id, role_id, model_id, targets in val_loader:
                preds = model(
                    semantic.to(device), response_emb.to(device),
                    strategy_id.to(device), role_id.to(device),
                    model_id.to(device),
                )
                loss = loss_fn(preds, targets.to(device))
                val_loss_sum += loss.item() * semantic.size(0)
                val_n += semantic.size(0)
        val_loss = val_loss_sum / max(val_n, 1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1 or marker:
            logger.info(f"  Epoch {epoch+1:>4}/{args.epochs}: "
                        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    elapsed = time.time() - start
    logger.info(f"Training done in {elapsed:.1f}s")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model (val_loss={best_val_loss:.4f})")

    # --- Evaluate on all splits ---
    _evaluate_quality(model, train_dataset, device, split_name="train", batch_size=args.batch_size)
    _evaluate_quality(model, val_dataset, device, split_name="val", batch_size=args.batch_size)
    test_metrics = _evaluate_quality(model, test_dataset, device, split_name="test", batch_size=args.batch_size)

    # --- Save ---
    save_path = os.path.join(args.save_dir, "quality_estimator.pth")
    save_quality_estimator(save_path, model, metadata, config)
    logger.info(f"Saved quality estimator checkpoint: {save_path}")
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

    # Share encoder between length and quality training
    shared_encoder = None
    if not args.skip_length or not args.skip_quality:
        logger.info("Loading SemanticEncoder (shared between length/quality)...")
        shared_encoder = SemanticEncoder(device=device)

    if not args.skip_length:
        train_length(args, device, encoder=shared_encoder)

    if not args.skip_quality:
        train_quality(args, device, encoder=shared_encoder)

    logger.info("All done.")
