import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import argparse
import yaml
import json
import re
import torch
from loguru import logger
import torch.nn.functional as F
import glob

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile_full import llm_profile, model_base_urls
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging, ProgressTracker
from MAR.Utils.telemetry import CsvTelemetryWriter
from MAR.Utils.request_patterns import RequestPattern
from MAR.Utils.request_shooter import RequestShooter
from MAR.SystemRouter.metrics_watcher import start_metrics_watcher, model_metrics

from Datasets.humaneval_dataset import HumanEvalDataset, HumanEvalDataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _build_metrics_url_map_from_profile(base_urls: dict):
    """Build metrics URL map from model_base_urls for vLLM metrics collection."""
    urls = {}
    for name, base_url in base_urls.items():
        if not name or not base_url:
            continue
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        metrics_url = f"{base}/metrics"
        urls[name] = metrics_url
    return urls


def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)

def reset_vllm_logs():
    """Truncate vLLM model logs so each run starts fresh instead of appending."""
    for path in glob.glob("logs/vllm/*.log"):
        try:
            open(path, "w").close()
        except OSError:
            # Best-effort; ignore if file is locked or missing.
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on humaneval")
    parser.add_argument("--dataset_json", type=str, default="Datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="(Deprecated) If >0, only run the first N examples (applies to both train and test).",
    )
    parser.add_argument("--train_limit", type=int, default=0, help="If >0, only run the first N training examples.")
    parser.add_argument("--test_limit", type=int, default=0, help="If >0, only run the first N test examples.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional path to a router .pth checkpoint to load before training/testing.",
    )
    parser.add_argument('--lr', type=float, default=0.01,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=16,help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Default 10.")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the agentprune')
    parser.add_argument('--prompt_file', type=str, default='MAR/Roles/FinalNode/humaneval.json')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cost_rate', type=float, default=400.0)
    parser.add_argument('--max_agent', type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=600.0, help="Per-request timeout in seconds.")
    parser.add_argument("--arrival-rate", type=float, nargs='+', default=[0.0], help="Arrival rate(s) (req/sec) for test shooting. Provide multiple values to sweep.")
    parser.add_argument("--arrival-pattern", type=str, default="poisson", help="Arrival pattern for test shooting.")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent in-flight requests in test shooting.")
    parser.add_argument("--burst-duration", type=float, default=3.0, help="Burst duration for microburst.")
    parser.add_argument("--spike-intensity", type=float, default=10.0, help="Spike intensity for microburst.")
    parser.add_argument("--spike-period", type=float, default=20.0, help="Spike period for microburst.")
    parser.add_argument("--train-telemetry-csv", type=str, default="", help="CSV path for training telemetry.")
    parser.add_argument("--test-telemetry-csv", type=str, default="", help="CSV path for test telemetry.")
    parser.add_argument("--save-checkpoint", type=str, default="", help="Path to save training checkpoint (single file, updated periodically).")
    parser.add_argument("--split-ratio", type=float, default=0.2, help="Train/test split ratio (default 0.2 means 20% train, 80% test)")
    args = parser.parse_args()
    return args

BASELINE_TRAIN_FIELDS = (
    "run_id",
    "epoch",
    "batch_id",
    "episode_index",
    "record_type",
    "dataset",
    "split",
    "item_id",
    "topology",
    "role_set",
    "workflow_latency_seconds",
    "llm_elapsed_seconds",
    "quality_is_solved",
    "step_index",
    "round_index",
    "dep_level",
    "role_name",
    "llm_name",
    "latency_seconds",
    "spatial_predecessors",
    "spatial_successors",
)

# CSV fields for test telemetry with LLM system metrics
BASELINE_TEST_FIELDS = (
    "run_id",
    "dataset",
    "split",
    "batch_id",
    "item_id",
    "record_type",
    # Topology / workflow metadata
    "task_name",
    "reasoning_name",
    "graph_id",
    "num_agents",
    "agent_roles_json",
    "agent_llms_json",
    "role_llm_map_json",
    # Quality evaluation
    "quality_is_correct",
    "quality_feedback",
    "quality_state_json",
    "eval_duration_sec",
    "utility",
    # Timing
    "workflow_latency_seconds",
    "llm_elapsed_seconds",
    "arrival_rate",
    "arrival_pattern",
    # Step-level fields (for step records)
    "step_index",
    "round_index",
    "dep_level",
    "role_name",
    "llm_name",
    "latency_seconds",
    "node_id",
    "spatial_predecessors",
    "spatial_successors",
    # vLLM system metrics (pre-request snapshot)
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


if __name__ == '__main__':
    reset_vllm_logs()
    args = parse_args()
    fix_random_seed(1234)
    train_dataset = HumanEvalDataset('train', split_ratio=args.split_ratio, seed=1234)
    test_dataset = HumanEvalDataset('test', split_ratio=args.split_ratio, seed=1234)

    # Backwards compatible: `--limit` applies to both unless overridden.
    if args.limit and args.limit > 0:
        if not args.train_limit:
            args.train_limit = args.limit
        if not args.test_limit:
            args.test_limit = args.limit

    if args.train_limit and args.train_limit > 0:
        train_dataset.df = train_dataset.df.iloc[: args.train_limit].reset_index(drop=True)
    if args.test_limit and args.test_limit > 0:
        test_dataset.df = test_dataset.df.iloc[: args.test_limit].reset_index(drop=True)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"humaneval_{current_time}.txt"
    configure_logging(log_name=log_file)
    run_id = current_time
    train_telemetry_csv = args.train_telemetry_csv or f"logs/{args.domain}_{current_time}_train.csv"
    train_writer = CsvTelemetryWriter(train_telemetry_csv, fieldnames=BASELINE_TRAIN_FIELDS)
    logger.info(f"Train telemetry CSV: {train_telemetry_csv}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter(max_agent=args.max_agent,device=device).to(device)
    if args.checkpoint:
        router.load_state_dict(torch.load(args.checkpoint, map_location=device))
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    tasks = tasks_profile
    llms = llm_profile
    reasonings = reasoning_profile
    logger.info("Start training...")

    episode_counter = 0
    checkpoint_dir = os.path.join("checkpoints", "mas_router")
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(args.epochs):
        total_solved, total_executed = (0, 0)
        train_loader = HumanEvalDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Initialize progress tracker for training
        num_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
        train_progress = ProgressTracker(
            total=len(train_dataset),
            phase=f"Train Epoch {epoch}",
            log_interval=max(1, len(train_dataset) // 10),
        )

        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(f"humaneval_router_epoch{epoch}_new.pth", map_location=torch.device('cuda')))
            continue
        for i_batch, current_batch in enumerate(train_loader):
            start_ts = time.time()
            queries = [item['prompt'] for item in current_batch]
            tests = [item['test'] for item in current_batch]
            item_ids = [item.get("task_id", "") for item in current_batch]
            task_labels = [2 for _ in current_batch]
            tasks_y = torch.tensor(task_labels).to(device)
            optimizer.zero_grad()
            results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(
                queries,
                tasks,
                llms,
                reasonings,
                task_labels,
                prompt_file=args.prompt_file,
                item_ids=item_ids,
                dataset=args.domain,
                request_timeout=args.request_timeout,
            )

            skipped_indices = getattr(router, "last_skipped_indices", set())
            valid_indices = [idx for idx in range(len(queries)) if idx not in skipped_indices]
            if not valid_indices:
                logger.warning("All requests in batch {} timed out; skipping batch.", i_batch)
                continue

            valid_mask = torch.zeros(len(queries), dtype=torch.bool, device=tasks_y.device)
            valid_mask[valid_indices] = True
            tasks_probs_valid = tasks_probs[valid_mask] if isinstance(tasks_probs, torch.Tensor) else tasks_probs
            tasks_y_valid = tasks_y[valid_mask] if isinstance(tasks_y, torch.Tensor) else tasks_y
            task_loss = F.cross_entropy(tasks_probs_valid, tasks_y_valid)
            utilities = []
            answers_loss = []
            is_solved_list = []
            pattern = r'```python.*```'
            for idx in valid_indices:
                query = queries[idx]
                result = results[idx]
                test = tests[idx]
                log_prob = log_probs[idx]
                cost = costs[idx]
                match = re.search(pattern, result, re.DOTALL|re.MULTILINE)
                if match:
                    answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                    is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
                else:
                    is_solved = 0
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                utility = is_solved - cost * args.cost_rate
                utilities.append(utility)
                is_solved_list.append(is_solved)
                answer_loss = -log_prob * utility
                answers_loss.append(answer_loss)

            compact_workflows = getattr(router, "last_compact_workflows", [])
            if compact_workflows:
                csv_rows = []
                for idx, workflow in enumerate(compact_workflows):
                    if not workflow:
                        continue
                    episode_index = episode_counter
                    episode_counter += 1
                    item_id = workflow.get("item_id", item_ids[idx] if idx < len(item_ids) else "")
                    topology = workflow.get("topology", "")
                    role_set = "-".join(workflow.get("role_set", []))
                    workflow_latency = float(workflow.get("workflow_latency_seconds", 0.0))
                    llm_elapsed = float(workflow.get("llm_elapsed_seconds", workflow_latency))
                    quality_is_solved = int(is_solved_list[idx]) if idx < len(is_solved_list) else 0
                    base = {
                        "run_id": run_id,
                        "epoch": epoch,
                        "batch_id": i_batch,
                        "episode_index": episode_index,
                        "dataset": args.domain,
                        "split": "train",
                        "item_id": item_id,
                        "topology": topology,
                        "role_set": role_set,
                        "workflow_latency_seconds": workflow_latency,
                        "llm_elapsed_seconds": llm_elapsed,
                        "quality_is_solved": quality_is_solved,
                    }
                    csv_rows.append({**base, "record_type": "episode"})
                    for step in workflow.get("transitions", []):
                        csv_rows.append(
                            {
                                **base,
                                "record_type": "step",
                                "step_index": step.get("step_index", ""),
                                "round_index": step.get("round_index", ""),
                                "dep_level": step.get("dep_level", ""),
                                "role_name": step.get("role_name", ""),
                                "llm_name": step.get("llm_name", ""),
                                "latency_seconds": step.get("latency_seconds", 0.0),
                                "spatial_predecessors": step.get("spatial_predecessors", ""),
                                "spatial_successors": step.get("spatial_successors", ""),
                                "llm_elapsed_seconds": step.get("llm_elapsed_seconds", llm_elapsed),
                            }
                        )
                train_writer.append_rows(csv_rows)
            answer_loss = torch.stack(answers_loss).sum() / len(answers_loss)
            if isinstance(vae_loss, torch.Tensor):
                if vae_loss.ndim > 0 and vae_loss.shape[0] == len(queries):
                    vae_loss = vae_loss[valid_mask].mean()
                else:
                    vae_loss = vae_loss.mean()
            else:
                vae_loss = torch.tensor(vae_loss, device=device).mean()
            is_solved_tensor = torch.tensor(is_solved_list, dtype=torch.float32, device=device).unsqueeze(1)  # shape: [N, 1]
            if isinstance(agents_num, torch.Tensor) and agents_num.ndim > 0 and agents_num.shape[0] == len(queries):
                agents_num = agents_num[valid_mask]
            adjust_loss = ((1 - is_solved_tensor) * (router.num_determiner.max_agent - agents_num) + 0.25 * is_solved_tensor * agents_num).mean()

            loss = task_loss + answer_loss + vae_loss*0.001 # + adjust_loss
            loss.backward()
            optimizer.step()
            accuracy = total_solved / total_executed if total_executed else 0.0

            # Update progress tracker with model usage from workflows
            models_used = []
            for workflow in compact_workflows:
                if workflow:
                    for step in workflow.get("transitions", []):
                        llm = step.get("llm_name", "")
                        if llm:
                            models_used.append(llm)
            for _ in range(len(valid_indices)):
                train_progress.update(success=True, models=models_used[:len(models_used)//max(1, len(valid_indices))] if models_used else None)
            for _ in range(len(queries) - len(valid_indices)):
                train_progress.update(success=False, models=None)
            if (i_batch + 1) % 5 == 0 and args.save_checkpoint:
                # Periodic checkpoint update
                ckpt_path = args.save_checkpoint
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                tmp_path = ckpt_path + ".tmp"
                torch.save(router.state_dict(), tmp_path)
                os.replace(tmp_path, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")
        # Log training progress summary for this epoch
        train_progress.log_final_summary()

        # Save at end of each epoch
        if args.save_checkpoint:
            ckpt_path = args.save_checkpoint
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            tmp_path = ckpt_path + ".tmp"
            torch.save(router.state_dict(), tmp_path)
            os.replace(tmp_path, ckpt_path)
            logger.info(f"Saved checkpoint after epoch {epoch}: {ckpt_path}")
    logger.info("End training...")
    logger.info("Start testing...")

    # Start metrics watcher for vLLM system metrics collection
    metrics_url_map = _build_metrics_url_map_from_profile(model_base_urls)
    if metrics_url_map:
        logger.info(f"Starting metrics watcher for {len(metrics_url_map)} models: {list(metrics_url_map.keys())}")
        start_metrics_watcher(metrics_url_map, interval=1.0)
    else:
        logger.warning("No metrics URL map built - metrics collection disabled")

    # Sweep through arrival rates
    arrival_rates = args.arrival_rate if isinstance(args.arrival_rate, list) else [args.arrival_rate]
    pattern = r'```python.*```'

    # Create single CSV for all arrival rates with metrics fields
    telemetry_csv = args.test_telemetry_csv or f"logs/{args.domain}_{current_time}_telemetry.csv"
    logger.info(f"Telemetry CSV: {telemetry_csv}")
    quality_writer = CsvTelemetryWriter(telemetry_csv, fieldnames=BASELINE_TEST_FIELDS)

    for arrival_rate in arrival_rates:
        logger.info(f"Testing with arrival_rate={arrival_rate}, arrival_pattern={args.arrival_pattern}")
        total_solved, total_executed = (0, 0)
        test_loader = HumanEvalDataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        use_shooter = arrival_rate > 0.0 or args.concurrency > 1

        # Initialize progress tracker for testing
        test_progress = ProgressTracker(
            total=len(test_dataset),
            phase=f"Test (rate={arrival_rate})",
            log_interval=10,  # Log every 10 workflow completions
        )

        if use_shooter:
            pattern_gen = RequestPattern(
                pattern=args.arrival_pattern,
                rate=arrival_rate,
                spike_intensity=args.spike_intensity,
                spike_period=args.spike_period,
                burst_duration=args.burst_duration,
                seed=1234,
            )
            def on_workflow_complete(result):
                """Callback to log progress as each workflow completes."""
                test_progress.update(success=result.success, models=None)

            shooter = RequestShooter(
                pattern_gen,
                max_concurrency=args.concurrency,
                capture_output=True,
                collect_results=True,
                on_result=on_workflow_complete,
            )

            def handler(row):
                query = row["prompt"]
                test = row["test"]
                item_id = row.get("task_id", "")
                task_labels = [2]
                with torch.no_grad():
                    results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(
                        [query],
                        tasks,
                        llms,
                        reasonings,
                        task_labels,
                        prompt_file=args.prompt_file,
                        item_ids=[item_id],
                        dataset=args.domain,
                        split="test",
                        batch_id=0,
                        run_id=current_time,
                        request_timeout=args.request_timeout,
                    )

                workflows = getattr(router, "last_compact_workflows", [])
                wf_data = workflows[0] if workflows else {}
                w_latency = wf_data.get("workflow_latency_seconds", 0.0)
                l_elapsed = wf_data.get("llm_elapsed_seconds", 0.0)
                transitions = wf_data.get("transitions", [])

                skipped_indices = getattr(router, "last_skipped_indices", set())
                if 0 in skipped_indices:
                    return None
                return {
                    "query": query,
                    "test": test,
                    "item_id": item_id,
                    "result": results[0],
                    "cost": costs[0],
                    "log_prob": log_probs[0],
                    "workflow_latency_seconds": w_latency,
                    "llm_elapsed_seconds": l_elapsed,
                    "transitions": transitions,
                    "task_name": wf_data.get("task_name", ""),
                    "reasoning_name": wf_data.get("reasoning_name", ""),
                    "graph_id": wf_data.get("graph_id", ""),
                    "num_agents": wf_data.get("num_agents", 0),
                    "agent_roles_json": wf_data.get("agent_roles_json", ""),
                    "agent_llms_json": wf_data.get("agent_llms_json", ""),
                    "role_llm_map_json": wf_data.get("role_llm_map_json", ""),
                }

            items = test_dataset.df.to_dict("records")
            results = shooter.run(
                items,
                handler=handler,
                item_id_fn=lambda row, idx: str(row.get("task_id", idx)),
            )
            for res in sorted(results, key=lambda r: r.index):
                if not res.success:
                    continue
                payload = res.output
                if payload is None:
                    continue
                query = payload["query"]
                test = payload["test"]
                item_id = payload["item_id"]
                result = payload["result"]
                cost = payload["cost"]
                w_latency = payload.get("workflow_latency_seconds", 0.0)
                l_elapsed = payload.get("llm_elapsed_seconds", 0.0)
                transitions = payload.get("transitions", [])

                topo_meta = {
                    "task_name": payload.get("task_name", ""),
                    "reasoning_name": payload.get("reasoning_name", ""),
                    "graph_id": payload.get("graph_id", ""),
                    "num_agents": payload.get("num_agents", 0),
                    "agent_roles_json": payload.get("agent_roles_json", ""),
                    "agent_llms_json": payload.get("agent_llms_json", ""),
                    "role_llm_map_json": payload.get("role_llm_map_json", ""),
                }

                eval_start_ts = time.time()
                match = re.search(pattern, result, re.DOTALL | re.MULTILINE)
                if match:
                    answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                    is_solved, feedback, state = PyExecutor().execute(answer, [test], timeout=100)
                else:
                    feedback = "No python code block found in the model output."
                    state = ()
                    is_solved = 0
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                utility = is_solved - cost * args.cost_rate

                csv_rows = []
                csv_rows.append({
                    "run_id": current_time,
                    "dataset": args.domain,
                    "split": "test",
                    "batch_id": "",
                    "item_id": str(item_id),
                    "record_type": "episode",
                    **topo_meta,
                    "quality_is_correct": bool(is_solved),
                    "quality_feedback": feedback,
                    "quality_state_json": list(state) if state else "",
                    "eval_duration_sec": time.time() - eval_start_ts,
                    "utility": utility,
                    "workflow_latency_seconds": w_latency,
                    "llm_elapsed_seconds": l_elapsed,
                    "arrival_rate": arrival_rate,
                    "arrival_pattern": args.arrival_pattern,
                })
                for step in transitions:
                    csv_rows.append({
                        "run_id": current_time,
                        "dataset": args.domain,
                        "split": "test",
                        "batch_id": "",
                        "item_id": str(item_id),
                        "record_type": "step",
                        **topo_meta,
                        "quality_is_correct": bool(is_solved),
                        "workflow_latency_seconds": w_latency,
                        "llm_elapsed_seconds": l_elapsed,
                        "arrival_rate": arrival_rate,
                        "arrival_pattern": args.arrival_pattern,
                        "step_index": step.get("step_index", ""),
                        "round_index": step.get("round_index", ""),
                        "dep_level": step.get("dep_level", ""),
                        "role_name": step.get("role_name", ""),
                        "llm_name": step.get("llm_name", ""),
                        "latency_seconds": step.get("latency_seconds", 0.0),
                        "node_id": step.get("node_id", ""),
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
                    })
                quality_writer.append_rows(csv_rows)
            test_progress.log_final_summary()
            accuracy = total_solved / total_executed if total_executed else 0.0
            logger.info("Shot {} requests. Accuracy: {}", total_executed, accuracy)
        else:
            for i_batch, current_batch in enumerate(test_loader):
                start_ts = time.time()
                queries = [item['prompt'] for item in current_batch]
                tests = [item['test'] for item in current_batch]
                item_ids = [item["task_id"] for item in current_batch]
                task_labels = [2 for _ in current_batch]
                tasks_y = torch.tensor(task_labels).to(device)
                results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(
                    queries,
                    tasks,
                    llms,
                    reasonings,
                    task_labels,
                    prompt_file=args.prompt_file,
                    item_ids=item_ids,
                    dataset=args.domain,
                    split="test",
                    batch_id=i_batch,
                    run_id=current_time,
                    request_timeout=args.request_timeout,
                )

                compact_workflows = getattr(router, "last_compact_workflows", [])

                utilities = []
                skipped_indices = getattr(router, "last_skipped_indices", set())
                for idx, (item_id, query, result, test, log_prob, cost) in enumerate(
                    zip(item_ids, queries, results, tests, log_probs, costs)
                ):
                    if idx in skipped_indices:
                        test_progress.update(success=False, models=None)
                        continue

                    wf_data = compact_workflows[idx] if idx < len(compact_workflows) else {}
                    w_latency = wf_data.get("workflow_latency_seconds", 0.0)
                    l_elapsed = wf_data.get("llm_elapsed_seconds", 0.0)
                    transitions = wf_data.get("transitions", [])

                    topo_meta = {
                        "task_name": wf_data.get("task_name", ""),
                        "reasoning_name": wf_data.get("reasoning_name", ""),
                        "graph_id": wf_data.get("graph_id", ""),
                        "num_agents": wf_data.get("num_agents", 0),
                        "agent_roles_json": wf_data.get("agent_roles_json", ""),
                        "agent_llms_json": wf_data.get("agent_llms_json", ""),
                        "role_llm_map_json": wf_data.get("role_llm_map_json", ""),
                    }

                    eval_start_ts = time.time()
                    match = re.search(pattern, result, re.DOTALL|re.MULTILINE)
                    if match:
                        answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                        is_solved, feedback, state = PyExecutor().execute(answer, [test], timeout=100)
                    else:
                        feedback = "No python code block found in the model output."
                        state = ()
                        is_solved = 0
                    total_solved = total_solved + is_solved
                    total_executed = total_executed + 1
                    utility = is_solved - cost * args.cost_rate
                    utilities.append(utility)

                    csv_rows = []
                    csv_rows.append({
                        "run_id": current_time,
                        "dataset": args.domain,
                        "split": "test",
                        "batch_id": i_batch,
                        "item_id": str(item_id),
                        "record_type": "episode",
                        **topo_meta,
                        "quality_is_correct": bool(is_solved),
                        "quality_feedback": feedback,
                        "quality_state_json": list(state) if state else "",
                        "eval_duration_sec": time.time() - eval_start_ts,
                        "utility": utility,
                        "workflow_latency_seconds": w_latency,
                        "llm_elapsed_seconds": l_elapsed,
                        "arrival_rate": arrival_rate,
                        "arrival_pattern": args.arrival_pattern,
                    })
                    for step in transitions:
                        csv_rows.append({
                            "run_id": current_time,
                            "dataset": args.domain,
                            "split": "test",
                            "batch_id": i_batch,
                            "item_id": str(item_id),
                            "record_type": "step",
                            **topo_meta,
                            "quality_is_correct": bool(is_solved),
                            "workflow_latency_seconds": w_latency,
                            "llm_elapsed_seconds": l_elapsed,
                            "arrival_rate": arrival_rate,
                            "arrival_pattern": args.arrival_pattern,
                            "step_index": step.get("step_index", ""),
                            "round_index": step.get("round_index", ""),
                            "dep_level": step.get("dep_level", ""),
                            "role_name": step.get("role_name", ""),
                            "llm_name": step.get("llm_name", ""),
                            "latency_seconds": step.get("latency_seconds", 0.0),
                            "node_id": step.get("node_id", ""),
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
                        })
                    quality_writer.append_rows(csv_rows)
                    test_progress.update(success=True, models=None)

            # Log test progress summary
            test_progress.log_final_summary()
            accuracy = total_solved / total_executed if total_executed else 0.0
            logger.info(f"Final accuracy for arrival_rate={arrival_rate}: {accuracy}")
