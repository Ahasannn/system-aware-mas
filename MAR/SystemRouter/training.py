import argparse
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
import torch

from MAR.Utils.log import ProgressTracker

from MAR.SystemRouter.datasets import SystemRouterSample, available_datasets, get_dataset_adapter
from MAR.SystemRouter.env import SystemRouterEnv
from MAR.SystemRouter.system_aware_router import SystemAwareRouter
from MAR.SystemRouter.trainer import SystemRouterTrainer
from MAR.Utils.request_patterns import RequestPattern
from MAR.Utils.request_shooter import RequestResult, RequestShooter
from MAR.Utils.telemetry import CsvTelemetryWriter


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
    "answer",
    "topology",
    "role_set",
    "budget_total",
    "arrival_rate",
    "arrival_pattern",
    "quality",
    "quality_is_solved",
    "quality_feedback",
    "quality_pred",
    "quality_gold",
    "workflow_latency_seconds",
    "llm_elapsed_seconds",
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
    "round_index",
    "dep_level",
    "spatial_predecessors",
    "spatial_successors",
    "observed_ttft",
    "observed_tpot",
    "llm_running",
    "llm_waiting",
    "llm_kv_cache_usage",
    "llm_ttft_avg",
    "llm_itl_avg",
    "llm_e2e_avg",
    "llm_queue_avg",
    "llm_inference_avg",
    "prompt_base",
    "response_final",
)


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _save_checkpoint(
    checkpoint_path: Optional[str],
    checkpoint_dir: str,
    router: SystemAwareRouter,
    trainer: SystemRouterTrainer,
    epoch: int,
    run_id: str,
    args: Any,
) -> str:
    if checkpoint_path:
        path = checkpoint_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    else:
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


def _parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _parse_str_list(raw: str) -> List[str]:
    values: List[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(item)
    return values


def _episode_has_agent_errors(episode: Dict[str, Any]) -> bool:
    for step in episode.get("executor_transitions", []):
        if not step.get("success", True):
            return True
        error = step.get("error", "")
        if isinstance(error, str) and error.strip():
            return True
    return False


def _build_arg_parser(default_dataset: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pilot training for System-Aware Router on multiple datasets.")
    parser.add_argument("--dataset", type=str, default=default_dataset, choices=available_datasets())
    parser.add_argument("--split", type=str, default="train", help="Dataset split to sample from.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Dataset file or root path (meaning depends on dataset).",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="",
        help="Optional train split path (gsm8k/humaneval).",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="",
        help="Optional test split path (gsm8k/humaneval).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="Dataset root directory (math/mmlu).",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.2,
        help="Train split ratio when explicit splits are unavailable.",
    )
    parser.add_argument("--role-domain", type=str, default="", help="Override the role domain.")
    parser.add_argument("--prompt-file", type=str, default="", help="Override FinalNode prompt file.")
    parser.add_argument("--limit", type=int, default=50, help="Number of items to train on.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs over the slice.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens when calling LLMs.")
    parser.add_argument("--request-timeout", type=float, default=600.0, help="Per-request timeout in seconds.")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-episodes", type=int, default=1, help="Episodes to log per epoch.")
    parser.add_argument("--arrival-rate", type=float, default=0.0, help="Arrival rate (req/sec). 0 disables shooting.")
    parser.add_argument("--arrival-rates", type=str, default="", help="Comma-separated arrival rates to sweep.")
    parser.add_argument(
        "--arrival-pattern",
        type=str,
        default="poisson",
        help="Arrival pattern (poisson/microburst/sustained).",
    )
    parser.add_argument("--arrival-patterns", type=str, default="", help="Comma-separated arrival patterns to sweep.")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent in-flight requests.")
    parser.add_argument("--burst-duration", type=float, default=3.0, help="Burst duration for microburst.")
    parser.add_argument("--spike-intensity", type=float, default=10.0, help="Spike intensity for microburst.")
    parser.add_argument("--spike-period", type=float, default=20.0, help="Spike period for microburst.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/system_router", help="Checkpoint output dir.")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Fixed checkpoint path to overwrite.")
    parser.add_argument("--resume-checkpoint", action="store_true", help="Resume from checkpoint path if present.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save a checkpoint every N successful episodes (0 disables).",
    )
    parser.add_argument("--telemetry-csv", type=str, default="", help="CSV path for per-episode telemetry.")
    parser.add_argument("--run-id", type=str, default="", help="Run id for telemetry/checkpoints.")
    return parser


def main(default_dataset: str = "mbpp") -> None:
    parser = _build_arg_parser(default_dataset)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = args.dataset_path.strip() or None
    train_path = args.train_path.strip() or None
    test_path = args.test_path.strip() or None
    dataset_root = args.dataset_root.strip() or None

    adapter = get_dataset_adapter(
        args.dataset,
        split=args.split,
        seed=args.seed,
        dataset_path=dataset_path,
        train_path=train_path,
        test_path=test_path,
        dataset_root=dataset_root,
        split_ratio=args.split_ratio,
    )
    role_domain = args.role_domain.strip() or adapter.role_domain
    prompt_file = args.prompt_file.strip() or adapter.prompt_file

    router = SystemAwareRouter(role_domain=role_domain)
    env = SystemRouterEnv(
        router=router,
        max_tokens=args.max_tokens,
        prompt_file=prompt_file,
        request_timeout=args.request_timeout,
        quality_fn=adapter.score_response,
    )
    trainer = SystemRouterTrainer(router)

    checkpoint_path = args.checkpoint_path.strip() or ""
    resume_checkpoint = args.resume_checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        if resume_checkpoint or args.checkpoint_path:
            payload = torch.load(checkpoint_path, map_location=router.device)
            router.load_state_dict(payload.get("router_state_dict", payload))
            planner_state = payload.get("planner_optimizer_state_dict")
            executor_state = payload.get("executor_optimizer_state_dict")
            if planner_state:
                trainer.planner_optimizer.load_state_dict(planner_state)
            if executor_state:
                trainer.executor_optimizer.load_state_dict(executor_state)
            logger.info("Resumed checkpoint from {}", checkpoint_path)

    run_id = args.run_id or _default_run_id()
    telemetry_path = args.telemetry_csv or os.path.join("logs", f"system_router_{adapter.dataset_name}_{run_id}.csv")
    telemetry_writer = CsvTelemetryWriter(telemetry_path, fieldnames=SYSTEM_ROUTER_CSV_FIELDS)
    logger.info("Telemetry CSV: {}", telemetry_path)

    data = adapter.sample(limit=args.limit, shuffle=True, seed=args.seed)
    logger.info("Loaded {} {} items for training slice", len(data), adapter.dataset_name)

    sweep_rates = _parse_float_list(args.arrival_rates) if args.arrival_rates else [args.arrival_rate]
    sweep_patterns = _parse_str_list(args.arrival_patterns) if args.arrival_patterns else [args.arrival_pattern]
    sweep_configs = [(rate, pattern) for rate in sweep_rates for pattern in sweep_patterns]
    global_epoch = 0
    episode_counter = 0
    last_checkpoint_episode = 0
    process_lock = threading.Lock()

    for sweep_idx, (arrival_rate, arrival_pattern) in enumerate(sweep_configs):
        for epoch in range(args.epochs):
            planner_transitions: List[Dict[str, Any]] = []
            executor_transitions: List[Dict[str, Any]] = []
            workflow_latencies: List[float] = []
            use_shooter = arrival_rate > 0.0 or args.concurrency > 1

            # Initialize progress tracker for this epoch
            phase_name = f"Epoch {global_epoch}"
            progress = ProgressTracker(
                total=len(data),
                phase=phase_name,
                log_interval=max(1, len(data) // 10),  # Log ~10 times per epoch
            )

            def process_episode(sample_idx: int, sample: SystemRouterSample, episode: Dict[str, Any]) -> None:
                nonlocal episode_counter, last_checkpoint_episode
                query = sample.query
                tests = list(sample.tests) if sample.tests else []
                item_id = sample.item_id
                answer = sample.answer

                if _episode_has_agent_errors(episode):
                    # Track failed episode
                    for step in episode.get("executor_transitions", []):
                        if not step.get("success", True) or step.get("error", "").strip():
                            logger.warning("Episode {} agent error: role={} model={} error={}",
                                sample_idx, step.get("role"), step.get("model"), step.get("error"))
                    progress.update(success=False, models=None)
                    return

                with process_lock:
                    planner_transitions.append(episode["planner_transition"])
                    executor_transitions.extend(episode["executor_transitions"])
                    workflow_latencies.append(episode["workflow_latency_seconds"])

                    episode_index = episode_counter
                    episode_counter += 1
                    quality_pred = episode.get("quality_pred")
                    quality_gold = episode.get("quality_gold") or answer
                    csv_rows: List[Dict[str, Any]] = [
                        {
                            "run_id": run_id,
                            "epoch": global_epoch,
                            "episode_index": episode_index,
                            "record_type": "episode",
                            "dataset": adapter.dataset_name,
                            "split": args.split,
                            "item_id": item_id,
                            "query": query,
                            "tests_json": tests if tests else "",
                            "answer": answer,
                            "topology": episode["topology"],
                            "role_set": episode["role_set"],
                            "budget_total": episode["budget_total"],
                            "arrival_rate": arrival_rate,
                            "arrival_pattern": arrival_pattern,
                            "quality": episode["quality"],
                            "quality_is_solved": episode.get("quality_is_solved"),
                            "quality_feedback": episode.get("quality_feedback"),
                            "quality_pred": quality_pred,
                            "quality_gold": quality_gold,
                            "workflow_latency_seconds": episode["workflow_latency_seconds"],
                            "llm_elapsed_seconds": episode.get("llm_elapsed_seconds", 0.0),
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
                                "epoch": global_epoch,
                                "episode_index": episode_index,
                                "record_type": "role_step",
                                "dataset": adapter.dataset_name,
                                "split": args.split,
                                "item_id": item_id,
                                "query": query,
                                "tests_json": tests if tests else "",
                                "answer": answer,
                                "topology": episode["topology"],
                                "role_set": episode["role_set"],
                                "arrival_rate": arrival_rate,
                                "arrival_pattern": arrival_pattern,
                                "quality": step.get("quality", 0.0),
                                "workflow_latency_seconds": step.get("workflow_latency_seconds", 0.0),
                                "llm_elapsed_seconds": step.get("llm_elapsed_seconds", 0.0),
                                "role_name": step.get("role", ""),
                                "step_index": step.get("step_index", 0),
                                "model_name": step.get("model", ""),
                                "strategy_name": step.get("strategy", ""),
                                "latency_seconds": step.get("latency_seconds", 0.0),
                                "budget_remaining": step.get("budget_remaining", 0.0),
                                "round_index": step.get("round_index", 0),
                                "dep_level": step.get("dep_level", 0),
                                "spatial_predecessors": step.get("spatial_predecessors", ""),
                                "spatial_successors": step.get("spatial_successors", ""),
                                "observed_ttft": step.get("observed_ttft", 0.0),
                                "observed_tpot": step.get("observed_tpot", 0.0),
                                "llm_running": step.get("llm_running", 0),
                                "llm_waiting": step.get("llm_waiting", 0),
                                "llm_kv_cache_usage": step.get("llm_kv_cache_usage", 0.0),
                                "llm_ttft_avg": step.get("llm_ttft_avg", 0.0),
                                "llm_itl_avg": step.get("llm_itl_avg", 0.0),
                                "llm_e2e_avg": step.get("llm_e2e_avg", 0.0),
                                "llm_queue_avg": step.get("llm_queue_avg", 0.0),
                                "llm_inference_avg": step.get("llm_inference_avg", 0.0),
                                "prompt_base": step.get("prompt_base", ""),
                                "response_final": step.get("response", ""),
                                "prompt_tokens": token_counts.get("prompt_tokens", 0),
                                "completion_tokens": token_counts.get("completion_tokens", 0),
                                "total_tokens": token_counts.get("total_tokens", 0),
                            }
                        )
                    telemetry_writer.append_rows(csv_rows)

                    if args.checkpoint_every > 0 and (episode_counter - last_checkpoint_episode) >= args.checkpoint_every:
                        saved_path = _save_checkpoint(
                            checkpoint_path or None,
                            args.checkpoint_dir,
                            router,
                            trainer,
                            global_epoch,
                            run_id,
                            args,
                        )
                        logger.info("Saved checkpoint: {}", saved_path)
                        last_checkpoint_episode = episode_counter

                    # Extract models used in this episode for statistics
                    models_used = [step.get("model", "") for step in episode["executor_transitions"] if step.get("model")]
                    progress.update(success=True, models=models_used)

            if use_shooter:
                pattern = RequestPattern(
                    pattern=arrival_pattern,
                    rate=arrival_rate,
                    spike_intensity=args.spike_intensity,
                    spike_period=args.spike_period,
                    burst_duration=args.burst_duration,
                    seed=args.seed + sweep_idx * 1000 + epoch,
                )
                shooter = RequestShooter(
                    pattern,
                    max_concurrency=args.concurrency,
                    capture_output=True,
                    collect_results=True,
                    on_result=None,
                )

                def handler(sample: SystemRouterSample) -> Dict[str, Any]:
                    with torch.no_grad():
                        return env.step(
                            query=sample.query,
                            tests=sample.tests,
                            deterministic=args.deterministic,
                            latency_seed=sample.query,
                            query_id=sample.item_id,
                            dataset_name=adapter.dataset_name,
                            sample=sample,
                        )

                def _handle_result(res: RequestResult) -> None:
                    if not res.success:
                        progress.update(success=False, models=None)
                        return
                    sample = data[res.index]
                    if res.output is None:
                        progress.update(success=False, models=None)
                        return
                    process_episode(res.index, sample, res.output)

                shooter.on_result = _handle_result
                shooter.run(
                    data,
                    handler=handler,
                    item_id_fn=lambda sample, idx: str(sample.item_id),
                )
            else:
                for sample_idx, sample in enumerate(data):
                    try:
                        episode = env.step(
                            query=sample.query,
                            tests=sample.tests,
                            deterministic=args.deterministic,
                            latency_seed=sample.query,
                            query_id=sample.item_id,
                            dataset_name=adapter.dataset_name,
                            sample=sample,
                        )
                    except Exception as exc:
                        logger.warning("Episode {} failed: {}", sample_idx, exc)
                        progress.update(success=False, models=None)
                        continue
                    process_episode(sample_idx, sample, episode)

            # Log final progress summary for this epoch
            progress.log_final_summary()

            metrics = trainer.train_batch(planner_transitions, executor_transitions)
            avg_quality = sum(t["quality"] for t in planner_transitions) / max(len(planner_transitions), 1)
            avg_latency = sum(workflow_latencies) / max(len(workflow_latencies), 1)
            logger.info(
                "epoch={} sweep={}/{} rate={} pattern={} planner_loss={:.4f} executor_loss={:.4f} avg_quality={:.4f} avg_latency={:.3f}s lambda={:.4f}",
                global_epoch,
                sweep_idx + 1,
                len(sweep_configs),
                arrival_rate,
                arrival_pattern,
                metrics.get("planner_loss", 0.0),
                metrics.get("executor_loss", 0.0),
                avg_quality,
                avg_latency,
                float(torch.nn.functional.softplus(router.lagrange_multiplier).detach().cpu().item()),
            )
            saved_path = _save_checkpoint(
                checkpoint_path or None,
                args.checkpoint_dir,
                router,
                trainer,
                global_epoch,
                run_id,
                args,
            )
            logger.info("Saved checkpoint: {}", saved_path)
            global_epoch += 1

    if data:
        sample = data[0]
        try:
            result = env.step(
                query=sample.query,
                tests=sample.tests,
                deterministic=True,
                latency_seed="final",
                query_id=sample.item_id,
                dataset_name=adapter.dataset_name,
                sample=sample,
            )
        except Exception as exc:
            logger.warning("final_sample_failed error={}", exc)
        else:
            logger.info(
                "final_sample topology={} role_set={} quality={:.4f} latency={:.3f}s",
                result["topology"],
                result["role_set"],
                result["quality"],
                result["workflow_latency_seconds"],
            )


if __name__ == "__main__":
    main()
