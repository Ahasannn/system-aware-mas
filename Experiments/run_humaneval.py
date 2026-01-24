import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import argparse
import yaml
import json
import time
import re
import torch
from loguru import logger
import torch.nn.functional as F

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile_test import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging
from MAR.Utils.telemetry import CsvTelemetryWriter
from MAR.Utils.request_patterns import RequestPattern
from MAR.Utils.request_shooter import RequestShooter

from Datasets.humaneval_dataset import HumanEvalDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    parser.add_argument('--cost_rate', type=float, default=200.0)
    parser.add_argument('--max_agent', type=int, default=6)
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--arrival-rate", type=float, default=0.0, help="Arrival rate (req/sec) for test shooting. 0 = disabled.")
    parser.add_argument("--arrival-pattern", type=str, default="poisson", help="Arrival pattern (poisson/microburst/sustained).")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent in-flight requests in test shooting.")
    parser.add_argument("--burst-duration", type=float, default=3.0, help="Burst duration for microburst.")
    parser.add_argument("--spike-intensity", type=float, default=10.0, help="Spike intensity for microburst.")
    parser.add_argument("--spike-period", type=float, default=20.0, help="Spike period for microburst.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    fix_random_seed(1234)

    # Load HumanEval dataset (auto-downloads from Hugging Face)
    train_dataset_obj = HumanEvalDataset(split='train', split_ratio=0.2, seed=1234)
    test_dataset_obj = HumanEvalDataset(split='test', split_ratio=0.2, seed=1234)

    # Convert to list format for batch processing
    train_dataset = [train_dataset_obj[i] for i in range(len(train_dataset_obj))]
    test_dataset = [test_dataset_obj[i] for i in range(len(test_dataset_obj))]

    # Apply limits if specified
    if args.limit > 0:
        train_dataset = train_dataset[:args.limit]
        test_dataset = test_dataset[:args.limit]
        logger.info(f"Applied --limit={args.limit} to both train and test datasets")
    if args.train_limit > 0:
        train_dataset = train_dataset[:args.train_limit]
        logger.info(f"Applied --train_limit={args.train_limit} to training dataset")
    if args.test_limit > 0:
        test_dataset = test_dataset[:args.test_limit]
        logger.info(f"Applied --test_limit={args.test_limit} to test dataset")

    logger.info(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"humaneval_{current_time}.txt"
    configure_logging(log_name=log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter(max_agent=args.max_agent,device=device).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)

    # Load checkpoint if provided
    if args.checkpoint and os.path.isfile(args.checkpoint):
        router.load_state_dict(torch.load(args.checkpoint, map_location=device))
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    tasks = tasks_profile
    llms = llm_profile
    reasonings = reasoning_profile

    # Skip training if train_limit is 0 (test-only mode)
    if len(train_dataset) > 0:
        logger.info("Start training...")
        for epoch in range(args.epochs):
            if epoch < args.start_epoch:
                router.load_state_dict(torch.load(f"humaneval_router_epoch{epoch}.pth", map_location=torch.device('cuda')))
                continue
            logger.info(f"Epoch {epoch}",80*'-')
            train_batches = int(len(train_dataset)/args.batch_size)
            total_solved, total_executed = (0, 0)
            for i_batch in range(train_batches):
                logger.info(f"Batch {i_batch}",80*'-')
                start_ts = time.time()
                current_batch = dataloader(train_dataset,args.batch_size,i_batch)
                queries = [item['prompt'] for item in current_batch]
                tests = [item['test'] for item in current_batch]
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
                )

                task_loss = F.cross_entropy(tasks_probs, tasks_y)
                utilities = []
                answers_loss = []
                is_solved_list = []
                pattern = r'```python.*```'
                for query, result, test, log_prob, cost in zip(queries, results, tests, log_probs, costs):
                    match = re.search(pattern, result, re.DOTALL|re.MULTILINE)
                    if match:
                        answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                        is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
                    else:
                        answer = ""
                        is_solved = 0
                    total_solved = total_solved + is_solved
                    total_executed = total_executed + 1
                    utility = is_solved - cost * args.cost_rate
                    utilities.append(utility)
                    is_solved_list.append(is_solved)
                    answer_loss = -log_prob * utility
                    answers_loss.append(answer_loss)

                answer_loss = torch.stack(answers_loss).sum() / len(answers_loss)
                vae_loss = vae_loss.mean()
                is_solved_tensor = torch.tensor(is_solved_list, dtype=torch.float32, device=device).unsqueeze(1)  # shape: [N, 1]
                adjust_loss = ((1 - is_solved_tensor) * (router.num_determiner.max_agent - agents_num) + 0.25 * is_solved_tensor *  agents_num).mean()

                loss = task_loss + answer_loss + vae_loss*0.001 # + adjust_loss
                loss.backward()
                optimizer.step()

                accuracy = total_solved / total_executed
                logger.info(f"Batch time {time.time() - start_ts:.3f}")
                logger.info(f"Accuracy: {accuracy}")
                logger.info(f"utilities:{utilities}")
            logger.info(f"Epoch {epoch} Finishes",80*'-')
            torch.save(router.state_dict(), f"humaneval_router_epoch{epoch}.pth")
        logger.info("Finish training...")
    else:
        logger.info("Skipping training (train_limit=0 or no training data)")

    logger.info("Start testing...")

    total_solved, total_executed = (0, 0)
    telemetry_csv = f"logs/{args.domain}_{current_time}_telemetry.csv"
    logger.info(f"Telemetry CSV: {telemetry_csv}")
    quality_writer = CsvTelemetryWriter(telemetry_csv)

    use_shooter = args.arrival_rate > 0.0 or args.concurrency > 1

    if use_shooter:
        logger.info(f"Testing with request shooter - arrival_rate={args.arrival_rate}, pattern={args.arrival_pattern}, concurrency={args.concurrency}")

        pattern_obj = RequestPattern(
            pattern=args.arrival_pattern,
            rate=args.arrival_rate,
            spike_intensity=args.spike_intensity,
            spike_period=args.spike_period,
            burst_duration=args.burst_duration,
            seed=1234,
        )

        shooter = RequestShooter(
            pattern_obj,
            max_concurrency=args.concurrency,
            capture_output=True,
            collect_results=True,
        )

        def process_item(item, idx):
            """Handler function for each test item."""
            query = item['prompt']
            test = item['test']
            item_id = item.get("task_id", idx)
            task_label = 2

            with torch.no_grad():
                results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(
                    [query],
                    tasks,
                    llms,
                    reasonings,
                    [task_label],
                    prompt_file=args.prompt_file,
                    telemetry_path=telemetry_csv,
                    item_ids=[item_id],
                    dataset=args.domain,
                    split="test",
                    batch_id=0,
                    run_id=current_time,
                    request_timeout=args.request_timeout,
                )

            result = results[0]
            cost = costs[0]

            # Evaluate the result
            eval_start_ts = time.time()
            pattern = r'```python.*```'
            match = re.search(pattern, result, re.DOTALL|re.MULTILINE)
            if match:
                answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(answer, [test], timeout=100)
            else:
                feedback = "No python code block found in the model output."
                state = ()
                is_solved = 0

            utility = is_solved - cost * args.cost_rate

            return {
                'item_id': item_id,
                'is_solved': is_solved,
                'utility': utility,
                'feedback': feedback,
                'state': state,
                'eval_duration_sec': time.time() - eval_start_ts,
            }

        def handle_result(res):
            """Callback for completed requests."""
            nonlocal total_solved, total_executed
            if not res.success:
                logger.warning(f"Request failed idx={res.index} item_id={res.item_id} error={res.error}")
                return

            output = res.output
            total_solved += output['is_solved']
            total_executed += 1

            quality_writer.append_rows([
                {
                    "run_id": current_time,
                    "dataset": args.domain,
                    "split": "test",
                    "batch_id": 0,
                    "item_id": str(output['item_id']),
                    "record_type": "quality",
                    "quality_is_correct": bool(output['is_solved']),
                    "quality_feedback": output['feedback'],
                    "quality_state_json": list(output['state']) if output['state'] else "",
                    "eval_duration_sec": output['eval_duration_sec'],
                    "utility": output['utility'],
                }
            ])

            if total_executed % 10 == 0:
                accuracy = total_solved / total_executed if total_executed > 0 else 0
                logger.info(f"Progress: {total_executed}/{len(test_dataset)} - Accuracy: {accuracy:.3f}")

        shooter.on_result = handle_result
        shooter.run(
            test_dataset,
            handler=process_item,
            item_id_fn=lambda item, idx: str(item.get("task_id", idx)),
        )

    else:
        # Standard sequential testing (no shooting)
        logger.info("Testing sequentially (no request shooting)")
        test_batches = int(len(test_dataset)/args.batch_size)

        for i_batch in range(test_batches):
            logger.info(f"Batch {i_batch}",80*'-')
            start_ts = time.time()
            current_batch = dataloader(test_dataset,args.batch_size,i_batch)
            queries = [item['prompt'] for item in current_batch]
            tests = [item['test'] for item in current_batch]
            item_ids = [item.get("task_id", i_batch * args.batch_size + j) for j, item in enumerate(current_batch)]
            task_labels = [2 for _ in current_batch]

            with torch.no_grad():
                results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(
                    queries,
                    tasks,
                    llms,
                    reasonings,
                    task_labels,
                    prompt_file=args.prompt_file,
                    telemetry_path=telemetry_csv,
                    item_ids=item_ids,
                    dataset=args.domain,
                    split="test",
                    batch_id=i_batch,
                    run_id=current_time,
                    request_timeout=args.request_timeout,
                )

            utilities = []
            pattern = r'```python.*```'
            for item_id, query, result, test, log_prob, cost in zip(item_ids, queries, results, tests, log_probs, costs):
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
                quality_writer.append_rows(
                    [
                        {
                            "run_id": current_time,
                            "dataset": args.domain,
                            "split": "test",
                            "batch_id": i_batch,
                            "item_id": str(item_id),
                            "record_type": "quality",
                            "quality_is_correct": bool(is_solved),
                            "quality_feedback": feedback,
                            "quality_state_json": list(state) if state else "",
                            "eval_duration_sec": time.time() - eval_start_ts,
                            "utility": utility,
                        }
                    ]
                )

            accuracy = total_solved / total_executed if total_executed > 0 else 0
            logger.info(f"Batch time {time.time() - start_ts:.3f}")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"utilities:{utilities}")

    # Final test results
    final_accuracy = total_solved / total_executed if total_executed > 0 else 0
    logger.info(f"Finish testing... Final Accuracy: {final_accuracy:.3f} ({total_solved}/{total_executed})")
