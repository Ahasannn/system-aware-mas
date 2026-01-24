import argparse
import os
import sys

from loguru import logger
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MAR.SystemRouter.length_estimator import (  # noqa: E402
    LengthEstimator,
    LengthEstimatorBundle,
    LengthEstimatorConfig,
    prepare_length_estimator_dataset,
    save_length_estimator,
    train_length_estimator,
)
from MAR.SystemRouter.system_aware_router import SemanticEncoder  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train output length estimator from system router CSV.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to system router training CSV.")
    parser.add_argument("--record-type", type=str, default="role_step", help="CSV record_type filter.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Prompt embedding batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--min-length", type=int, default=1, help="Skip samples shorter than this length.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="checkpoints/length_estimator.pt",
        help="Checkpoint path to save the estimator.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SemanticEncoder(device=device)
    dataset, metadata = prepare_length_estimator_dataset(
        args.csv_path,
        encoder=encoder,
        record_type=args.record_type,
        batch_size=args.embed_batch_size,
        min_length=args.min_length,
    )

    if len(dataset) == 0:
        raise ValueError("No training samples found in the CSV.")

    config = LengthEstimatorConfig(semantic_dim=metadata.semantic_dim)
    model = LengthEstimator(
        config,
        num_models=len(metadata.model_vocab),
        num_roles=len(metadata.role_vocab),
        num_strategies=len(metadata.strategy_vocab),
    )

    losses = train_length_estimator(
        model,
        dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    logger.info("Training complete. Final loss: {:.4f}", losses[-1] if losses else 0.0)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_length_estimator(args.output_path, model, metadata, config)
    logger.info("Saved estimator to {}", args.output_path)

    bundle = LengthEstimatorBundle(model=model, metadata=metadata, encoder=encoder)
    logger.info("Bundle ready: {}", bundle)


if __name__ == "__main__":
    main()
