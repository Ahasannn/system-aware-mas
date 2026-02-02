import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import argparse
import yaml
import json
import torch
from loguru import logger
import torch.nn.functional as F
import glob

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile_full import llm_profile, model_base_urls
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging, ProgressTracker
from MAR.Utils.telemetry import CsvTelemetryWriter
from MAR.Utils.request_patterns import RequestPattern
from MAR.Utils.request_shooter import RequestShooter
from MAR.SystemRouter.metrics_watcher import start_metrics_watcher, model_metrics
from Datasets.gsm8k_dataset import gsm_get_predict

from datasets import load_dataset

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
            pass

def gsm_data_process(dataset):
    """Process GSM8K dataset from HuggingFace format."""
    list_data_dict = []
    for i, data in enumerate(dataset):
        item = {"task": data["question"], "id": i}
        raw_answer = data["answer"]
        raw_answer_list = raw_answer.split("\n####")
        item["step"] = raw_answer_list[0].strip()
        item["answer"] = raw_answer_list[-1].replace(",", "").strip()
        list_data_dict.append(item)
    return list_data_dict

def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on gsm8k")
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
    parser.add_argument('--domain', type=str, default="gsm8k",help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the agentprune')
    parser.add_argument('--prompt_file', type=str, default='MAR/Roles/FinalNode/gsm8k.json')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cost_rate', type=float, default=400.0)
    parser.add_argument('--max_agent', type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=600.0, help="Per-request timeout in seconds.")
    parser.add_argument("--arrival-rate", type=float, nargs='+', default=[0.0], help="Arrival rate(s) (req/sec) for test shooting.")
    parser.add_argument("--arrival-pattern", type=str, default="poisson", help="Arrival pattern for test shooting.")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent in-flight requests in test shooting.")
    parser.add_argument("--train-telemetry-csv", type=str, default="", help="CSV path for training telemetry.")
    parser.add_argument("--test-telemetry-csv", type=str, default="", help="CSV path for test telemetry.")
    parser.add_argument("--save-checkpoint", type=str, default="", help="Path to save training checkpoint.")
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
    "quality_pred",
    "quality_gold",
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

    # Load GSM8K from HuggingFace
    hf_train = load_dataset("openai/gsm8k", "main", split="train")
    hf_test = load_dataset("openai/gsm8k", "main", split="test")

    train_dataset = gsm_data_process(hf_train)
    test_dataset = gsm_data_process(hf_test)

    # Backwards compatible: `--limit` applies to both unless overridden.
    if args.limit and args.limit > 0:
        if not args.train_limit:
            args.train_limit = args.limit
        if not args.test_limit:
            args.test_limit = args.limit

    if args.train_limit and args.train_limit > 0:
        train_dataset = train_dataset[:args.train_limit]
    if args.test_limit and args.test_limit > 0:
        test_dataset = test_dataset[:args.test_limit]

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"gsm8k_{current_time}.txt"
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
        train_batches = len(train_dataset) // args.batch_size

        # Initialize progress tracker for training
        train_progress = ProgressTracker(
            total=len(train_dataset),
            phase=f"Train Epoch {epoch}",
            log_interval=max(1, len(train_dataset) // 10),
        )

        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(f"gsm8k_router_epoch{epoch}.pth", map_location=torch.device('cuda')))
            continue

        for i_batch in range(train_batches):
            start_ts = time.time()
            current_batch = dataloader(train_dataset, args.batch_size, i_batch)
            queries = [item['task'] for item in current_batch]
            answers = [item['answer'] for item in current_batch]
            item_ids = [item['id'] for item in current_batch]
            task_labels = [0 for _ in current_batch]  # 0 = math task
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

            for idx in valid_indices:
                query = queries[idx]
                result = results[idx]
                true_answer = answers[idx]
                log_prob = log_probs[idx]
                cost = costs[idx]
                predict_answer = gsm_get_predict(result)
                try:
                    is_solved = float(predict_answer) == float(true_answer)
                except:
                    is_solved = False
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
            is_solved_tensor = torch.tensor(is_solved_list, dtype=torch.float32, device=device).unsqueeze(1)
            if isinstance(agents_num, torch.Tensor) and agents_num.ndim > 0 and agents_num.shape[0] == len(queries):
                agents_num = agents_num[valid_mask]

            loss = task_loss + answer_loss + vae_loss*0.001
            loss.backward()
            optimizer.step()
            accuracy = total_solved / total_executed if total_executed else 0.0

            # Update progress tracker
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
                ckpt_path = args.save_checkpoint
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                tmp_path = ckpt_path + ".tmp"
                torch.save(router.state_dict(), tmp_path)
                os.replace(tmp_path, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

        train_progress.log_final_summary()

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

    # Create single CSV for all arrival rates with metrics fields
    telemetry_csv = args.test_telemetry_csv or f"logs/{args.domain}_{current_time}_telemetry.csv"
    logger.info(f"Telemetry CSV: {telemetry_csv}")
    quality_writer = CsvTelemetryWriter(telemetry_csv, fieldnames=BASELINE_TEST_FIELDS)

    for arrival_rate in arrival_rates:
        logger.info(f"Testing with arrival_rate={arrival_rate}, arrival_pattern={args.arrival_pattern}")
        total_solved, total_executed = (0, 0)
        use_shooter = arrival_rate > 0.0 or args.concurrency > 1

        test_progress = ProgressTracker(
            total=len(test_dataset),
            phase=f"Test (rate={arrival_rate})",
            log_interval=10,
        )

        if use_shooter:
            pattern_gen = RequestPattern(
                pattern=args.arrival_pattern,
                rate=arrival_rate,
                seed=1234,
            )
            def on_workflow_complete(result):
                test_progress.update(success=result.success, models=None)

            shooter = RequestShooter(
                pattern_gen,
                max_concurrency=args.concurrency,
                capture_output=True,
                collect_results=True,
                on_result=on_workflow_complete,
            )

            def handler(row):
                query = row["task"]
                true_answer = row["answer"]
                item_id = row.get("id", "")
                task_labels = [0]
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
                    "true_answer": true_answer,
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

            results = shooter.run(
                test_dataset,
                handler=handler,
                item_id_fn=lambda row, idx: str(row.get("id", idx)),
            )
            for res in sorted(results, key=lambda r: r.index):
                if not res.success:
                    continue
                payload = res.output
                if payload is None:
                    continue
                query = payload["query"]
                true_answer = payload["true_answer"]
                item_id = payload["item_id"]
                result = payload["result"]
                cost = payload["cost"]
                w_latency = payload.get("workflow_latency_seconds", 0.0)
                l_elapsed = payload.get("llm_elapsed_seconds", 0.0)
                transitions = payload.get("transitions", [])

                # Topology metadata from compact workflow
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
                predict_answer = gsm_get_predict(result)
                try:
                    is_solved = float(predict_answer) == float(true_answer)
                except:
                    is_solved = False
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                utility = is_solved - cost * args.cost_rate

                # Build CSV rows: episode record + step records with metrics
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
                    "quality_pred": str(predict_answer),
                    "quality_gold": str(true_answer),
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
            test_batches = len(test_dataset) // args.batch_size
            for i_batch in range(test_batches):
                start_ts = time.time()
                current_batch = dataloader(test_dataset, args.batch_size, i_batch)
                queries = [item['task'] for item in current_batch]
                answers = [item['answer'] for item in current_batch]
                item_ids = [item["id"] for item in current_batch]
                task_labels = [0 for _ in current_batch]
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
                skipped_indices = getattr(router, "last_skipped_indices", set())

                for idx, (item_id, query, result, true_answer, log_prob, cost) in enumerate(
                    zip(item_ids, queries, results, answers, log_probs, costs)
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
                    predict_answer = gsm_get_predict(result)
                    try:
                        is_solved = float(predict_answer) == float(true_answer)
                    except:
                        is_solved = False
                    total_solved = total_solved + is_solved
                    total_executed = total_executed + 1
                    utility = is_solved - cost * args.cost_rate

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
                        "quality_pred": str(predict_answer),
                        "quality_gold": str(true_answer),
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

            test_progress.log_final_summary()
            accuracy = total_solved / total_executed if total_executed else 0.0
            logger.info(f"Final accuracy for arrival_rate={arrival_rate}: {accuracy}")
