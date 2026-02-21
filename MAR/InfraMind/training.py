import argparse
import math
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
import torch

from MAR.Utils.log import ProgressTracker

from MAR.InfraMind.budget_provider import BudgetProvider
from MAR.InfraMind.datasets import InfraMindSample, available_datasets, get_dataset_adapter
from MAR.InfraMind.env import InfraMindEnv
from MAR.InfraMind.inframind_router import InfraMindRouter
from MAR.InfraMind.metrics_watcher import model_metrics as global_model_metrics
from MAR.InfraMind.trainer import InfraMindTrainer
from MAR.Utils.request_patterns import RequestPattern
from MAR.Utils.request_shooter import RequestResult, RequestShooter
from MAR.Utils.telemetry import CsvTelemetryWriter


INFRAMIND_CSV_FIELDS: Sequence[str] = (
    "run_id",
    "epoch",
    "episode_index",
    "record_type",
    "dataset",
    "split",
    "item_id",
    "topology",
    "role_set",
    "budget_total",
    "arrival_rate",
    "arrival_pattern",
    "quality",
    "quality_is_solved",
    "quality_feedback",
    "workflow_latency_seconds",
    "llm_elapsed_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "query",
    "response",
    "prompt_base",
    "role_name",
    "step_index",
    "model_name",
    "strategy_name",
    "latency_seconds",
    "budget_remaining",
    "round_index",
    "dep_level",
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
)


def _default_run_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    return f"{ts}_job{job_id}" if job_id else ts


def _save_checkpoint(
    checkpoint_path: Optional[str],
    checkpoint_dir: str,
    router: InfraMindRouter,
    trainer: InfraMindTrainer,
    epoch: int,
    run_id: str,
    args: Any,
    dataset_name: str,
) -> str:
    if checkpoint_path:
        # User specified an explicit path — always use it
        path = checkpoint_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    else:
        # No explicit path — include run_id so each job gets its own checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"inframind_{dataset_name}_{run_id}.pt")
    payload = {
        "epoch": epoch,
        "run_id": run_id,
        "dataset": dataset_name,
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


# ---------------------------------------------------------------------------
# Sweep-level circuit breaker threshold
# ---------------------------------------------------------------------------
CIRCUIT_BREAKER_FAILURE_RATE = 0.30  # skip training if >30% episodes failed


def _wait_for_vllm_drain(timeout: float = 60.0, poll_interval: float = 2.0) -> bool:
    """Wait until all vLLM servers have zero running+waiting requests.

    Returns True if drained within timeout, False otherwise.
    This prevents leftover requests from a high-load sweep from polluting
    the next sweep's latency measurements and causing cascade failures.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        total_active = 0
        for model_name, snap in global_model_metrics.items():
            running = snap.get("num_requests_running", 0.0)
            waiting = snap.get("num_requests_waiting", 0.0)
            total_active += running + waiting
        if total_active == 0:
            return True
        logger.debug(
            "[Drain] Waiting for vLLM queues to clear: {} active requests remaining",
            int(total_active),
        )
        time.sleep(poll_interval)
    # Log which servers still have active requests
    for model_name, snap in global_model_metrics.items():
        running = snap.get("num_requests_running", 0.0)
        waiting = snap.get("num_requests_waiting", 0.0)
        if running + waiting > 0:
            logger.warning(
                "[Drain] {} still has {} running + {} waiting after {}s timeout",
                model_name, int(running), int(waiting), timeout,
            )
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
    parser.add_argument("--request-timeout", type=float, default=1800.0, help="Per-request timeout in seconds (30 min, matches baseline).")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-episodes", type=int, default=1, help="Episodes to log per epoch.")
    parser.add_argument("--arrival-rate", type=float, default=0.0, help="Arrival rate (req/min). 0 disables shooting.")
    parser.add_argument("--arrival-rates", type=str, default="", help="Comma-separated arrival rates to sweep.")
    parser.add_argument(
        "--arrival-pattern",
        type=str,
        default="poisson",
        help="Arrival pattern (poisson/microburst/sustained).",
    )
    parser.add_argument("--arrival-patterns", type=str, default="", help="Comma-separated arrival patterns to sweep.")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent in-flight requests.")
    parser.add_argument("--training-batch-size", type=int, default=64, help="Number of episodes to accumulate before each training update.")
    parser.add_argument("--burst-duration", type=float, default=3.0, help="Burst duration for microburst.")
    parser.add_argument("--spike-intensity", type=float, default=10.0, help="Spike intensity for microburst.")
    parser.add_argument("--spike-period", type=float, default=20.0, help="Spike period for microburst.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/inframind", help="Checkpoint output dir.")
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
    parser.add_argument(
        "--latency-predictor",
        type=str,
        default="",
        help="Path to latency predictor checkpoint (required for predicted-latency state).",
    )
    parser.add_argument(
        "--length-predictor",
        type=str,
        default="",
        help="Path to length predictor checkpoint (required for predicted-latency state).",
    )
    parser.add_argument(
        "--budget-csv",
        type=str,
        default="",
        help="Path to baseline CSV with per-query latency budgets.",
    )
    parser.add_argument(
        "--budget-sweep",
        type=str,
        default="",
        help="Comma-separated budget values (seconds) to sweep, e.g. '10,20,30,50,100,200'. "
        "When set, each (arrival_rate, pattern, budget) combo becomes a separate sweep config. "
        "Overrides --budget-csv.",
    )
    parser.add_argument(
        "--mas-checkpoint",
        type=str,
        default="",
        help="Path to pretrained MAS Router checkpoint. Loads shared planner "
        "modules (task_classifier, collab/num/role) as initialization, "
        "skips MAS-only modules (llm_router). Executor starts fresh.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Inference-only mode: skip weight updates and checkpoint saves. "
        "Use with --deterministic and a loaded checkpoint for paper results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pipeline test mode: skip LLM calls, use mock responses/latencies. "
        "Tests the full planner → executor → trainer pipeline without vLLM servers.",
    )
    parser.add_argument(
        "--random-exploration",
        action="store_true",
        help="Phase 1 random exploration: uniform-random executor actions, skip training. "
        "Generates diverse (model, strategy) data for judge scoring.",
    )
    parser.add_argument(
        "--quality-predictor",
        type=str,
        default="",
        help="Path to quality predictor checkpoint (Phase 2 training with predicted quality).",
    )
    parser.add_argument(
        "--budget-min",
        type=float,
        default=0.0,
        help="Minimum budget for LogUniform per-item randomization. "
        "When both --budget-min and --budget-max are set, each item "
        "samples budget ~ LogUniform(min, max). Overrides --budget-sweep.",
    )
    parser.add_argument(
        "--budget-max",
        type=float,
        default=0.0,
        help="Maximum budget for LogUniform per-item randomization.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="Number of validation items. 0 disables validation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--lr-scheduler",
        action="store_true",
        help="Enable ReduceLROnPlateau for both optimizers.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience.",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="LR reduction factor.",
    )
    return parser


def _get_val_budget(sweep_budgets: List[float], budget_provider: Optional[BudgetProvider]) -> float:
    """Return a sensible budget for validation runs."""
    if sweep_budgets:
        return sweep_budgets[-1]
    return 60.0


# Validation budget scenarios for budget-awareness checking.
# (label, budget_seconds)  — kept small so validation is fast.
_VAL_SCENARIOS = [
    ("tight", 20.0),
    ("loose", 120.0),
]

# Composite score weights: accuracy is king, but latency efficiency matters.
# composite = accuracy_weight * solve_rate + efficiency_weight * budget_efficiency
_VAL_ACCURACY_WEIGHT = 0.80
_VAL_EFFICIENCY_WEIGHT = 0.20


def _run_validation_pass(
    env: InfraMindEnv,
    router: InfraMindRouter,
    val_data: List[InfraMindSample],
    adapter: Any,
    budget: float,
    label: str,
) -> Dict[str, Any]:
    """Run a single deterministic validation pass at a given budget."""
    solved = 0
    total = 0
    latencies: List[float] = []
    budget_violations = 0
    with torch.no_grad():
        for sample in val_data:
            try:
                episode = env.step(
                    query=sample.query,
                    tests=sample.tests,
                    deterministic=True,
                    latency_seed=sample.query,
                    query_id=sample.item_id,
                    dataset_name=adapter.dataset_name,
                    sample=sample,
                    budget_total=budget,
                )
            except Exception as exc:
                logger.warning("Validation [{}] episode failed: {}", label, exc)
                total += 1
                continue
            total += 1
            if episode.get("quality_is_solved"):
                solved += 1
            lat = episode.get("workflow_latency_seconds", 0.0)
            latencies.append(lat)
            if lat > budget:
                budget_violations += 1
    solve_rate = solved / max(total, 1)
    avg_latency = sum(latencies) / max(len(latencies), 1) if latencies else 0.0
    slo_compliance = 1.0 - (budget_violations / max(total, 1))
    # Budget efficiency: fraction of budget used on average (lower = more efficient)
    budget_efficiency = 1.0 - min(1.0, avg_latency / budget) if budget > 0 else 0.0
    return {
        "label": label,
        "budget": budget,
        "solve_rate": solve_rate,
        "solved": solved,
        "total": total,
        "avg_latency": avg_latency,
        "slo_compliance": slo_compliance,
        "budget_violations": budget_violations,
        "budget_efficiency": budget_efficiency,
    }


def _run_validation(
    env: InfraMindEnv,
    router: InfraMindRouter,
    val_data: List[InfraMindSample],
    adapter: Any,
    val_budget: float,
) -> Dict[str, Any]:
    """Run multi-scenario validation: tight + loose budget passes.

    Returns composite metrics combining accuracy and budget awareness.
    The composite_score is used for best-model selection and early stopping.
    """
    router.eval()
    try:
        passes: List[Dict[str, Any]] = []
        for label, budget in _VAL_SCENARIOS:
            # Drain vLLM between passes so they don't interfere
            if passes and not env.dry_run:
                _wait_for_vllm_drain(timeout=30.0)
            result = _run_validation_pass(env, router, val_data, adapter, budget, label)
            passes.append(result)
            logger.info(
                "  VAL [{}] budget={:.0f}s solve_rate={:.4f} ({}/{}) "
                "avg_latency={:.3f}s slo_compliance={:.2f}%",
                label, budget, result["solve_rate"], result["solved"],
                result["total"], result["avg_latency"],
                result["slo_compliance"] * 100,
            )
    finally:
        router.train()

    # Aggregate: average solve_rate and budget_efficiency across scenarios
    avg_solve_rate = sum(p["solve_rate"] for p in passes) / len(passes)
    avg_efficiency = sum(p["budget_efficiency"] for p in passes) / len(passes)
    avg_slo = sum(p["slo_compliance"] for p in passes) / len(passes)
    avg_latency = sum(p["avg_latency"] for p in passes) / len(passes)

    # Composite score for best-model selection
    composite = _VAL_ACCURACY_WEIGHT * avg_solve_rate + _VAL_EFFICIENCY_WEIGHT * avg_efficiency

    return {
        "val_solve_rate": avg_solve_rate,
        "val_solved": sum(p["solved"] for p in passes),
        "val_total": sum(p["total"] for p in passes),
        "val_avg_latency": avg_latency,
        "val_slo_compliance": avg_slo,
        "val_budget_efficiency": avg_efficiency,
        "val_composite_score": composite,
        "val_passes": passes,
    }


def main(default_dataset: str = "mbpp") -> None:
    parser = _build_arg_parser(default_dataset)
    args = parser.parse_args()

    # Random exploration forces skip_training
    if getattr(args, "random_exploration", False):
        args.skip_training = True

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

    budget_csv_path = args.budget_csv.strip() or None
    budget_provider: Optional[BudgetProvider] = None
    if budget_csv_path:
        budget_provider = BudgetProvider(budget_csv_path)
        logger.info("Budget provider loaded with rates: {}", budget_provider.arrival_rates)

    latency_predictor_path = args.latency_predictor.strip() or None
    length_predictor_path = args.length_predictor.strip() or None
    quality_predictor_path = getattr(args, "quality_predictor", "") or ""
    quality_predictor_path = quality_predictor_path.strip() or None
    router = InfraMindRouter(
        role_domain=role_domain,
        latency_predictor_path=latency_predictor_path,
        length_predictor_path=length_predictor_path,
        quality_predictor_path=quality_predictor_path,
    )
    # Enable random exploration mode on the router
    if getattr(args, "random_exploration", False):
        router.random_exploration = True

    env = InfraMindEnv(
        router=router,
        max_tokens=args.max_tokens,
        prompt_file=prompt_file,
        request_timeout=args.request_timeout,
        quality_fn=adapter.score_response,
        dry_run=getattr(args, "dry_run", False),
    )

    # Map role_domain to task classifier label index
    _DOMAIN_TASK_INDEX = {"Math": 0, "Commonsense": 1, "Code": 2}
    task_label = _DOMAIN_TASK_INDEX.get(role_domain, 0)
    trainer = InfraMindTrainer(router)

    checkpoint_path = args.checkpoint_path.strip() or ""
    resume_checkpoint = args.resume_checkpoint
    if resume_checkpoint and checkpoint_path and os.path.isfile(checkpoint_path):
        # Resume from a previous InfraMind checkpoint (full state: router + optimizers)
        payload = torch.load(checkpoint_path, map_location=router.device)
        router.load_state_dict(payload.get("router_state_dict", payload), strict=False)
        planner_state = payload.get("planner_optimizer_state_dict")
        executor_state = payload.get("executor_optimizer_state_dict")
        if planner_state:
            trainer.planner_optimizer.load_state_dict(planner_state)
        if executor_state:
            trainer.executor_optimizer.load_state_dict(executor_state)
        logger.info("Resumed InfraMind checkpoint from {}", checkpoint_path)
        # Clear so new saves go to a fresh file (with new job ID)
        checkpoint_path = ""
    else:
        # Fresh training — load MAS planner weights as initialization
        mas_checkpoint = args.mas_checkpoint.strip() if hasattr(args, "mas_checkpoint") else ""
        if mas_checkpoint and os.path.isfile(mas_checkpoint):
            router.load_mas_checkpoint(mas_checkpoint)
        elif mas_checkpoint:
            logger.warning("MAS checkpoint not found: {}", mas_checkpoint)

    run_id = args.run_id or _default_run_id()
    telemetry_path = args.telemetry_csv or os.path.join("logs", f"inframind_{adapter.dataset_name}_{run_id}.csv")
    telemetry_writer = CsvTelemetryWriter(telemetry_path, fieldnames=INFRAMIND_CSV_FIELDS)
    logger.info("Telemetry CSV: {}", telemetry_path)

    val_limit = getattr(args, "val_limit", 0)
    total_needed = args.limit + val_limit
    all_data = adapter.sample(limit=total_needed, shuffle=True, seed=args.seed)
    data = all_data[: args.limit]
    val_data: List[InfraMindSample] = all_data[args.limit : args.limit + val_limit] if val_limit > 0 else []
    logger.info("Loaded {} {} items for training, {} for validation", len(data), adapter.dataset_name, len(val_data))

    # Attach LR schedulers if requested and validation is enabled
    if getattr(args, "lr_scheduler", False) and val_limit > 0:
        trainer.attach_lr_schedulers(
            patience=getattr(args, "lr_patience", 2),
            factor=getattr(args, "lr_factor", 0.5),
        )

    sweep_rates = _parse_float_list(args.arrival_rates) if args.arrival_rates else [args.arrival_rate]
    sweep_patterns = _parse_str_list(args.arrival_patterns) if args.arrival_patterns else [args.arrival_pattern]
    sweep_budgets = _parse_float_list(args.budget_sweep) if args.budget_sweep else []

    # Budget randomization mode: LogUniform(min, max) per item
    budget_min = getattr(args, "budget_min", 0.0)
    budget_max = getattr(args, "budget_max", 0.0)
    use_random_budget = budget_min > 0.0 and budget_max > 0.0 and budget_max > budget_min
    if use_random_budget:
        # Single config per (rate, pattern); budget sampled per item
        sweep_configs = [(rate, pattern, 0.0) for rate in sweep_rates for pattern in sweep_patterns]
        logger.info("Budget randomization: LogUniform({:.0f}, {:.0f}) per item, {} configs",
                     budget_min, budget_max, len(sweep_configs))
    elif sweep_budgets:
        sweep_configs = [
            (rate, pattern, budget)
            for rate in sweep_rates
            for pattern in sweep_patterns
            for budget in sweep_budgets
        ]
        logger.info("Budget sweep: {} budgets × {} rates × {} patterns = {} configs",
                     len(sweep_budgets), len(sweep_rates), len(sweep_patterns), len(sweep_configs))
    else:
        sweep_configs = [(rate, pattern, 0.0) for rate in sweep_rates for pattern in sweep_patterns]
    global_epoch = 0
    episode_counter = 0
    last_checkpoint_episode = 0
    process_lock = threading.Lock()
    training_lock = threading.Lock()

    # Early stopping / best model tracking
    best_val_solve_rate = -1.0
    epochs_without_improvement = 0
    patience = getattr(args, "patience", 5)
    val_budget = _get_val_budget(sweep_budgets, budget_provider)

    for epoch in range(args.epochs):
        random.shuffle(sweep_configs)
        for sweep_idx, (arrival_rate, arrival_pattern, sweep_budget) in enumerate(sweep_configs):
            # ---- Inter-sweep cooldown: wait for vLLM queues to drain ----
            if sweep_idx > 0 and not env.dry_run:
                logger.info("[Sweep {}/{}] Waiting for vLLM queues to drain before rate={}...",
                            sweep_idx + 1, len(sweep_configs), arrival_rate)
                drained = _wait_for_vllm_drain(timeout=60.0)
                if not drained:
                    logger.warning("[Sweep] vLLM queues did not fully drain; proceeding anyway")

            planner_transitions: List[Dict[str, Any]] = []
            executor_transitions: List[Dict[str, Any]] = []
            workflow_latencies: List[float] = []
            workflow_qualities: List[float] = []  # Track quality separately
            epoch_metrics: Dict[str, float] = {}
            epoch_train_count: int = 0
            use_shooter = arrival_rate > 0.0 or args.concurrency > 1

            # Initialize progress tracker for this epoch
            if use_random_budget:
                budget_tag = f" budget=LogU({budget_min:.0f},{budget_max:.0f})"
            elif sweep_budget > 0.0:
                budget_tag = f" budget={sweep_budget:.0f}s"
            else:
                budget_tag = ""
            phase_name = f"Epoch {global_epoch} rate={arrival_rate}{budget_tag}"
            progress = ProgressTracker(
                total=len(data),
                phase=phase_name,
                log_interval=max(1, len(data) // 10),  # Log ~10 times per epoch
            )

            def process_episode(sample_idx: int, sample: InfraMindSample, episode: Dict[str, Any]) -> None:
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

                # Collect transitions under process_lock (fast)
                planner_batch = None
                executor_batch = None
                with process_lock:
                    planner_transitions.append(episode["planner_transition"])
                    executor_transitions.extend(episode["executor_transitions"])
                    workflow_latencies.append(episode["workflow_latency_seconds"])
                    workflow_qualities.append(episode.get("quality", 0.0))

                    episode_index = episode_counter

                    # Snapshot batch if ready (copy + clear is fast)
                    training_batch_size = args.training_batch_size
                    if not args.skip_training and len(planner_transitions) >= training_batch_size:
                        planner_batch = planner_transitions.copy()
                        executor_batch = executor_transitions.copy()
                        planner_transitions.clear()
                        executor_transitions.clear()

                # Train outside process_lock (doesn't block other episode processing)
                if planner_batch is not None:
                    with training_lock:
                        nonlocal epoch_metrics, epoch_train_count
                        batch_metrics = trainer.train_batch(planner_batch, executor_batch, task_label=task_label)
                        epoch_metrics = batch_metrics
                        epoch_train_count += 1

                with process_lock:
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
                            "topology": episode["topology"],
                            "role_set": episode["role_set"],
                            "budget_total": episode["budget_total"],
                            "arrival_rate": arrival_rate,
                            "arrival_pattern": arrival_pattern,
                            "quality": episode["quality"],
                            "quality_is_solved": episode.get("quality_is_solved"),
                            "quality_feedback": episode.get("quality_feedback"),
                            "workflow_latency_seconds": episode["workflow_latency_seconds"],
                            "llm_elapsed_seconds": episode.get("llm_elapsed_seconds", 0.0),
                            "prompt_tokens": episode["token_counts"].get("prompt_tokens", 0),
                            "completion_tokens": episode["token_counts"].get("completion_tokens", 0),
                            "total_tokens": episode["token_counts"].get("total_tokens", 0),
                            "query": query,
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
                                "topology": episode["topology"],
                                "role_set": episode["role_set"],
                                "arrival_rate": arrival_rate,
                                "arrival_pattern": arrival_pattern,
                                "quality": step.get("quality", 0.0),
                                "workflow_latency_seconds": step.get("workflow_latency_seconds", 0.0),
                                "llm_elapsed_seconds": step.get("llm_elapsed_seconds", 0.0),
                                "query": query,
                                "response": step.get("response", ""),
                                "prompt_base": step.get("prompt_base", ""),
                                "role_name": step.get("role", ""),
                                "step_index": step.get("step_index", 0),
                                "model_name": step.get("model", ""),
                                "strategy_name": step.get("strategy", ""),
                                "latency_seconds": step.get("latency_seconds", 0.0),
                                "budget_remaining": step.get("budget_remaining", 0.0),
                                "budget_total": episode["budget_total"],
                                "round_index": step.get("round_index", 0),
                                "dep_level": step.get("dep_level", 0),
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
                                "prompt_tokens": token_counts.get("prompt_tokens", 0),
                                "completion_tokens": token_counts.get("completion_tokens", 0),
                                "total_tokens": token_counts.get("total_tokens", 0),
                            }
                        )
                    telemetry_writer.append_rows(csv_rows)

                    if not args.skip_training and args.checkpoint_every > 0 and (episode_counter - last_checkpoint_episode) >= args.checkpoint_every:
                        saved_path = _save_checkpoint(
                            checkpoint_path or None,
                            args.checkpoint_dir,
                            router,
                            trainer,
                            global_epoch,
                            run_id,
                            args,
                            adapter.dataset_name,
                        )
                        logger.info("Saved checkpoint: {}", saved_path)
                        last_checkpoint_episode = episode_counter

                    # Extract usage stats from this episode
                    models_used = [step.get("model", "") for step in episode["executor_transitions"] if step.get("model")]
                    strategies_used = [step.get("strategy", "") for step in episode["executor_transitions"] if step.get("strategy")]
                    progress.update(
                        success=True,
                        models=models_used,
                        topology=episode.get("topology", ""),
                        strategies=strategies_used,
                        latency=episode.get("workflow_latency_seconds", 0.0),
                        quality=episode.get("quality", 0.0),
                    )

            # Pre-assign budgets: same budget for each training batch block
            # so advantage normalization compares items at the same budget level.
            batch_size = args.training_batch_size
            if use_random_budget:
                item_budgets: Dict[int, float] = {}
                for block_start in range(0, len(data), batch_size):
                    log_b = random.uniform(math.log(budget_min), math.log(budget_max))
                    block_budget = math.exp(log_b)
                    block_end = min(block_start + batch_size, len(data))
                    for i in range(block_start, block_end):
                        item_budgets[i] = block_budget

            def _get_budget(sample: InfraMindSample, sample_idx: int = 0) -> float:
                if use_random_budget:
                    return item_budgets.get(sample_idx, 60.0)
                if sweep_budget > 0.0:
                    return sweep_budget
                if budget_provider is not None:
                    return budget_provider.get_budget(str(sample.item_id), arrival_rate)
                return 60.0

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

                # Map sample identity to index for budget lookup
                _sample_to_idx: Dict[int, int] = {id(s): i for i, s in enumerate(data)}

                def handler(sample: InfraMindSample) -> Dict[str, Any]:
                    s_idx = _sample_to_idx.get(id(sample), 0)
                    budget = _get_budget(sample, s_idx)
                    if args.skip_training:
                        with torch.no_grad():
                            return env.step(
                                query=sample.query,
                                tests=sample.tests,
                                deterministic=args.deterministic,
                                latency_seed=sample.query,
                                query_id=sample.item_id,
                                dataset_name=adapter.dataset_name,
                                sample=sample,
                                budget_total=budget,
                            )
                    else:
                        return env.step(
                            query=sample.query,
                            tests=sample.tests,
                            deterministic=args.deterministic,
                            latency_seed=sample.query,
                            query_id=sample.item_id,
                            dataset_name=adapter.dataset_name,
                            sample=sample,
                            budget_total=budget,
                        )

                def _handle_result(res: RequestResult) -> None:
                    if not res.success:
                        logger.warning("Request {} failed: {}", res.index, res.error)
                        progress.update(success=False)
                        return
                    sample = data[res.index]
                    if res.output is None:
                        progress.update(success=False)
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
                            budget_total=_get_budget(sample, sample_idx),
                        )
                    except Exception as exc:
                        logger.warning("Episode {} failed: {}", sample_idx, exc)
                        progress.update(success=False)
                        continue
                    process_episode(sample_idx, sample, episode)

            # Log final progress summary for this epoch
            progress.log_final_summary()

            # ---- Circuit breaker: skip training if too many episodes failed ----
            failure_rate = progress.failed / max(progress.processed, 1)
            if failure_rate > CIRCUIT_BREAKER_FAILURE_RATE and not args.skip_training:
                logger.warning(
                    "CIRCUIT BREAKER: {:.1f}% episodes failed (threshold={:.0f}%). "
                    "Discarding {} planner + {} executor transitions to prevent "
                    "survivor-bias poisoning.",
                    failure_rate * 100,
                    CIRCUIT_BREAKER_FAILURE_RATE * 100,
                    len(planner_transitions),
                    len(executor_transitions),
                )
                planner_transitions.clear()
                executor_transitions.clear()
                workflow_latencies.clear()
                workflow_qualities.clear()
                # Still increment global epoch and drain vLLM before next sweep
                global_epoch += 1
                if not env.dry_run:
                    logger.info("[Cooldown] Waiting for vLLM queues to drain after circuit breaker...")
                    _wait_for_vllm_drain(timeout=90.0)
                continue

            # Train on any remaining transitions (last partial batch)
            if not args.skip_training and (planner_transitions or executor_transitions):
                remaining_metrics = trainer.train_batch(planner_transitions, executor_transitions, task_label=task_label)
                epoch_metrics = remaining_metrics
                epoch_train_count += 1

            metrics = epoch_metrics

            avg_quality = sum(workflow_qualities) / max(len(workflow_qualities), 1) if workflow_qualities else 0.0
            avg_latency = sum(workflow_latencies) / max(len(workflow_latencies), 1) if workflow_latencies else 0.0
            if use_random_budget:
                budget_label = f"LogU({budget_min:.0f},{budget_max:.0f})"
            elif sweep_budget > 0.0:
                budget_label = f"{sweep_budget:.0f}s"
            else:
                budget_label = "csv/default"

            if args.skip_training:
                logger.info(
                    "epoch={} sweep={}/{} rate={} pattern={} budget={} [inference-only] avg_latency={:.3f}s",
                    global_epoch,
                    sweep_idx + 1,
                    len(sweep_configs),
                    arrival_rate,
                    arrival_pattern,
                    budget_label,
                    avg_latency,
                )
            else:
                logger.info(
                    "epoch={} sweep={}/{} rate={} pattern={} budget={}"
                    " planner_loss={:.4f} utility={:.4f}"
                    " executor_loss={:.4f}"
                    " avg_quality={:.4f} avg_latency={:.3f}s"
                    " cost_ratio={:.3f} constraint_gap={:.3f}"
                    " e_entropy={:.3f}",
                    global_epoch,
                    sweep_idx + 1,
                    len(sweep_configs),
                    arrival_rate,
                    arrival_pattern,
                    budget_label,
                    metrics.get("planner_loss", 0.0),
                    metrics.get("planner_avg_utility", 0.0),
                    metrics.get("executor_loss", 0.0),
                    avg_quality,
                    avg_latency,
                    metrics.get("cost_ratio", 0.0),
                    metrics.get("constraint_gap", 0.0),
                    metrics.get("executor_entropy", 0.0),
                )
                saved_path = _save_checkpoint(
                    checkpoint_path or None,
                    args.checkpoint_dir,
                    router,
                    trainer,
                    global_epoch,
                    run_id,
                    args,
                    adapter.dataset_name,
                )
                logger.info("Saved checkpoint: {}", saved_path)
            global_epoch += 1

        # -- End of sweep configs for this epoch --
        # Validation, LR scheduling, early stopping
        if val_data and not args.skip_training:
            # Drain vLLM before validation so it starts clean
            if not env.dry_run:
                _wait_for_vllm_drain(timeout=60.0)

            val_metrics = _run_validation(
                env, router, val_data, adapter, val_budget,
            )
            val_sr = val_metrics["val_solve_rate"]
            val_composite = val_metrics["val_composite_score"]
            logger.info(
                "VALIDATION epoch={} solve_rate={:.4f} ({}/{}) avg_latency={:.3f}s "
                "slo_compliance={:.2f}% efficiency={:.2f}% composite={:.4f}",
                epoch, val_sr, val_metrics["val_solved"], val_metrics["val_total"],
                val_metrics["val_avg_latency"],
                val_metrics["val_slo_compliance"] * 100,
                val_metrics["val_budget_efficiency"] * 100,
                val_composite,
            )

            # Step LR schedulers on composite score (not just accuracy)
            trainer.step_schedulers(val_composite)

            # Best model checkpointing — use composite score
            if val_composite > best_val_solve_rate:
                best_val_solve_rate = val_composite
                epochs_without_improvement = 0
                best_path = os.path.join(
                    args.checkpoint_dir,
                    f"inframind_{adapter.dataset_name}_{run_id}_best.pt",
                )
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                payload = {
                    "epoch": epoch,
                    "run_id": run_id,
                    "dataset": adapter.dataset_name,
                    "router_state_dict": router.state_dict(),
                    "planner_optimizer_state_dict": trainer.planner_optimizer.state_dict(),
                    "executor_optimizer_state_dict": trainer.executor_optimizer.state_dict(),
                    "val_solve_rate": val_sr,
                    "val_composite_score": val_composite,
                    "val_slo_compliance": val_metrics["val_slo_compliance"],
                    "val_budget_efficiency": val_metrics["val_budget_efficiency"],
                    "val_passes": val_metrics["val_passes"],
                    "args": vars(args),
                }
                torch.save(payload, best_path)
                logger.info(
                    "New best model (composite={:.4f}, solve_rate={:.4f}) saved: {}",
                    val_composite, val_sr, best_path,
                )
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No improvement for {} epoch(s) (best_composite={:.4f}, current={:.4f})",
                    epochs_without_improvement, best_val_solve_rate, val_composite,
                )

            # Early stopping
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered after {} epochs without improvement",
                    epochs_without_improvement,
                )
                break

    if data and not args.skip_training:
        sample = data[0]
        if sweep_budgets:
            final_budget = sweep_budgets[-1]
        elif budget_provider is not None:
            last_rate = sweep_rates[-1] if sweep_rates else 0.0
            try:
                final_budget = budget_provider.get_budget(str(sample.item_id), last_rate)
            except KeyError:
                final_budget = 60.0
        else:
            final_budget = 60.0
        try:
            result = env.step(
                query=sample.query,
                tests=sample.tests,
                deterministic=True,
                latency_seed="final",
                query_id=sample.item_id,
                dataset_name=adapter.dataset_name,
                sample=sample,
                budget_total=final_budget,
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
