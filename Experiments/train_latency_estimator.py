import argparse
import os
import sys

from loguru import logger
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MAR.SystemRouter.latency_estimator import (  # noqa: E402
    LatencyEstimator,
    LatencyEstimatorBundle,
    LatencyEstimatorConfig,
    prepare_latency_estimator_dataset,
    save_latency_estimator,
    train_latency_estimator,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train latency estimator from system router CSV.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to system router training CSV.")
    parser.add_argument("--record-type", type=str, default="role_step", help="CSV record_type filter.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--min-ttft", type=float, default=1e-6, help="Skip samples below this ttft.")
    parser.add_argument("--min-tpot", type=float, default=1e-6, help="Skip samples below this tpot.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="checkpoints/latency_estimator.pt",
        help="Checkpoint path to save the estimator.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, metadata = prepare_latency_estimator_dataset(
        args.csv_path,
        record_type=args.record_type,
        min_ttft=args.min_ttft,
        min_tpot=args.min_tpot,
    )

    if len(dataset) == 0:
        raise ValueError("No training samples found in the CSV.")

    config = LatencyEstimatorConfig(num_numerical_features=metadata.num_numerical_features)
    model = LatencyEstimator(
        num_numerical_features=config.num_numerical_features,
        num_models=len(metadata.model_vocab),
        num_roles=len(metadata.role_vocab),
        num_strategies=len(metadata.strategy_vocab),
        embedding_dim=config.embedding_dim,
        hidden_dims=list(config.hidden_dims),
        dropout=config.dropout,
    )

    losses = train_latency_estimator(
        model,
        dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    logger.info("Training complete. Final loss: {:.4f}", losses[-1] if losses else 0.0)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_latency_estimator(args.output_path, model, metadata, config)
    logger.info("Saved estimator to {}", args.output_path)

    bundle = LatencyEstimatorBundle(model=model, metadata=metadata)
    logger.info("Bundle ready: {}", bundle)


if __name__ == "__main__":
    main()
