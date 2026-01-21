"""
Pilot training loop for the hierarchical System-Aware Router on a small MBPP slice.
Quality is measured by executing MBPP tests; latency is measured via wall-clock time.
"""
import argparse
import math
import os
import sys
import random
import time
from typing import Any, Dict, List, Sequence

from loguru import logger
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Datasets.mbpp_dataset import MbppDataset  # noqa: E402
from MAR.SystemRouter.system_aware_router import SystemAwareRouter  # noqa: E402
from MAR.SystemRouter.env import SystemRouterEnv  # noqa: E402
from MAR.SystemRouter.trainer import SystemRouterTrainer  # noqa: E402
from MAR.Utils.telemetry import CsvTelemetryWriter  # noqa: E402
from MAR.Utils.request_patterns import RequestPattern  # noqa: E402
from MAR.Utils.request_shooter import RequestShooter, RequestResult  # noqa: E402


def sample_mbpp(limit: int, split: str = "train") -> List[dict]:
    ds = MbppDataset(split)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    picked = indices[:limit]
    return [ds[i] for i in picked]

SYSTEM_ROUTER_CSV_FIELDS: Sequence[str] = (
    "run_id",
    "epoch",
    "episode_index",
    "record_type",
    "dataset",
    "split",
    "item_id",
    "query",
    "tests_json",
    "topology",
    "role_set",
    "budget_total",
    "quality",
    "quality_is_solved",
    "quality_feedback",
    "workflow_latency_seconds",
    "final_response",
    "code_response",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "role_name",
    "step_index",
    "model_name",
    "strategy_name",
    "latency_seconds",
    "budget_remaining",
    "prompt_base",
    "response_final",
)


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_item_id(row: Any, fallback: int) -> Any:
    keys = ("task_id", "id", "ID", "index", "idx")
    getter = row.get if hasattr(row, "get") else None
    for key in keys:
        value = None
        if getter is not None:
            value = getter(key)
        elif isinstance(row, dict):
            value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if value == "":
            continue
        return value
    return fallback


def _save_checkpoint(
    checkpoint_dir: str, router: SystemAwareRouter, trainer: SystemRouterTrainer, epoch: int, run_id: str, args: Any
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"system_router_{run_id}_epoch_{epoch}.pt")
    payload = {
        "epoch": epoch,
        "run_id": run_id,
        "router_state_dict": router.state_dict(),
        "planner_optimizer_state_dict": trainer.planner_optimizer.state_dict(),
        "executor_optimizer_state_dict": trainer.executor_optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot training for System-Aware Router on MBPP slice.")
    parser.add_argument("--limit", type=int, default=50, help="Number of MBPP items to train on.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs over the slice.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens when calling LLMs.")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-episodes", type=int, default=1, help="Episodes to log per epoch.")
    parser.add_argument("--arrival-rate", type=float, default=0.0, help="Arrival rate (req/sec). 0 disables shooting.")
    parser.add_argument("--arrival-pattern", type=str, default="poisson", help="Arrival pattern (poisson/microburst/sustained).")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent in-flight requests.")
    parser.add_argument("--burst-duration", type=float, default=3.0, help="Burst duration for microburst.")
    parser.add_argument("--spike-intensity", type=float, default=10.0, help="Spike intensity for microburst.")
    parser.add_argument("--spike-period", type=float, default=20.0, help="Spike period for microburst.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/system_router", help="Checkpoint output dir.")
    parser.add_argument("--telemetry-csv", type=str, default="", help="CSV path for per-episode telemetry.")
    parser.add_argument("--run-id", type=str, default="", help="Run id for telemetry/checkpoints.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    router = SystemAwareRouter()
    env = SystemRouterEnv(router=router, max_tokens=args.max_tokens)
    trainer = SystemRouterTrainer(router)

    run_id = args.run_id or _default_run_id()
    telemetry_path = args.telemetry_csv or os.path.join("logs", f"system_router_mbpp_{run_id}.csv")
    telemetry_writer = CsvTelemetryWriter(telemetry_path, fieldnames=SYSTEM_ROUTER_CSV_FIELDS)
    logger.info("Telemetry CSV: {}", telemetry_path)

    data = sample_mbpp(limit=args.limit, split="train")
    logger.info("Loaded {} MBPP items for training slice", len(data))

    for epoch in range(args.epochs):
        planner_transitions = []
        executor_transitions = []
        workflow_latencies = []
        use_shooter = args.arrival_rate > 0.0 or args.concurrency > 1

        def process_episode(row_idx: int, row: Any, episode: Dict[str, Any]) -> None:
            query = str(row.get("task") or row["text"])
            tests = list(row["test_list"])
            item_id = _resolve_item_id(row, row_idx)

            planner_transitions.append(episode["planner_transition"])
            executor_transitions.extend(episode["executor_transitions"])
            workflow_latencies.append(episode["workflow_latency_seconds"])

            episode_index = epoch * len(data) + row_idx
            csv_rows: List[Dict[str, Any]] = [
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "episode_index": episode_index,
                    "record_type": "episode",
                    "dataset": "mbpp",
                    "split": "train",
                    "item_id": item_id,
                    "query": query,
                    "tests_json": tests,
                    "topology": episode["topology"],
                    "role_set": episode["role_set"],
                    "budget_total": episode["budget_total"],
                    "quality": episode["quality"],
                    "quality_is_solved": episode["quality_is_solved"],
                    "quality_feedback": episode["quality_feedback"],
                    "workflow_latency_seconds": episode["workflow_latency_seconds"],
                    "final_response": episode["response"],
                    "code_response": episode.get("code_response", ""),
                    "prompt_tokens": episode["token_counts"].get("prompt_tokens", 0),
                    "completion_tokens": episode["token_counts"].get("completion_tokens", 0),
                    "total_tokens": episode["token_counts"].get("total_tokens", 0),
                }
            ]

            for step in episode["executor_transitions"]:
                token_counts = step.get("token_counts", {})
                csv_rows.append(
                    {
                        "run_id": run_id,
                        "epoch": epoch,
                        "episode_index": episode_index,
                        "record_type": "role_step",
                        "dataset": "mbpp",
                        "split": "train",
                        "item_id": item_id,
                        "query": query,
                        "tests_json": tests,
                        "topology": episode["topology"],
                        "role_set": episode["role_set"],
                        "quality": step.get("quality", 0.0),
                        "workflow_latency_seconds": step.get("workflow_latency_seconds", 0.0),
                        "role_name": step.get("role", ""),
                        "step_index": step.get("step_index", 0),
                        "model_name": step.get("model", ""),
                        "strategy_name": step.get("strategy", ""),
                        "latency_seconds": step.get("latency_seconds", 0.0),
                        "budget_remaining": step.get("budget_remaining", 0.0),
                        "prompt_base": step.get("prompt_base", ""),
                        "response_final": step.get("response", ""),
                        "prompt_tokens": token_counts.get("prompt_tokens", 0),
                        "completion_tokens": token_counts.get("completion_tokens", 0),
                        "total_tokens": token_counts.get("total_tokens", 0),
                    }
                )
            telemetry_writer.append_rows(csv_rows)

            if len(planner_transitions) <= args.log_episodes:
                logger.info(
                    "episode={} topology={} role_set={} quality={:.3f} latency={:.3f}s budget={:.2f}s",
                    len(planner_transitions) - 1,
                    episode["topology"],
                    episode["role_set"],
                    episode["quality"],
                    episode["workflow_latency_seconds"],
                    episode["budget_total"],
                )
                for step in episode["executor_transitions"]:
                    logger.info(
                        "  role={} model={} strategy={} call_latency={:.3f}s budget_rem={:.2f}s",
                        step["role"],
                        step["model"],
                        step["strategy"],
                        step["latency_seconds"],
                        step["budget_remaining"],
                    )

        if use_shooter:
            pattern = RequestPattern(
                pattern=args.arrival_pattern,
                rate=args.arrival_rate,
                spike_intensity=args.spike_intensity,
                spike_period=args.spike_period,
                burst_duration=args.burst_duration,
                seed=args.seed + epoch,
            )
            shooter = RequestShooter(
                pattern,
                max_concurrency=args.concurrency,
                capture_output=True,
                collect_results=True,
            )

            def handler(row: Any) -> Dict[str, Any]:
                query = str(row.get("task") or row["text"])
                tests = list(row["test_list"])
                with torch.no_grad():
                    return env.step(
                        query=query,
                        tests=tests,
                        deterministic=args.deterministic,
                        latency_seed=query,
                    )

            results: List[RequestResult] = shooter.run(
                data,
                handler=handler,
                item_id_fn=lambda row, idx: str(_resolve_item_id(row, idx)),
            )
            for res in sorted(results, key=lambda r: r.index):
                if not res.success:
                    logger.warning("request_failed idx={} item_id={} error={}", res.index, res.item_id, res.error)
                    continue
                row = data[res.index]
                if res.output is None:
                    logger.warning("request_empty_output idx={} item_id={}", res.index, res.item_id)
                    continue
                process_episode(res.index, row, res.output)
        else:
            for row_idx, row in enumerate(data):
                query = str(row.get("task") or row["text"])
                tests = list(row["test_list"])
                episode = env.step(
                    query=query,
                    tests=tests,
                    deterministic=args.deterministic,
                    latency_seed=query,
                )
                process_episode(row_idx, row, episode)

        metrics = trainer.train_batch(planner_transitions, executor_transitions)
        avg_quality = sum(t["quality"] for t in planner_transitions) / max(len(planner_transitions), 1)
        avg_latency = sum(workflow_latencies) / max(len(workflow_latencies), 1)
        logger.info(
            "epoch={} planner_loss={:.4f} executor_loss={:.4f} avg_quality={:.4f} avg_latency={:.3f}s lambda={:.4f}",
            epoch,
            metrics.get("planner_loss", 0.0),
            metrics.get("executor_loss", 0.0),
            avg_quality,
            avg_latency,
            float(torch.nn.functional.softplus(router.lagrange_multiplier).detach().cpu().item()),
        )
        checkpoint_path = _save_checkpoint(args.checkpoint_dir, router, trainer, epoch, run_id, args)
        logger.info("Saved checkpoint: {}", checkpoint_path)

    # Final sanity: route one sample deterministically
    if data:
        sample = data[0]
        result = env.step(
            query=str(sample["text"]),
            tests=list(sample["test_list"]),
            deterministic=True,
            latency_seed="final",
        )
        logger.info(
            "final_sample topology={} role_set={} quality={:.4f} latency={:.3f}s",
            result["topology"],
            result["role_set"],
            result["quality"],
            result["workflow_latency_seconds"],
        )


if __name__ == "__main__":
    main()
