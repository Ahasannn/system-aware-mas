"""LLM-as-a-Judge scoring for InfraMind Phase 1 CSV data.

Reads a Phase 1 exploration CSV, sends each agent response to a judge LLM
with a rubric (1-10 scale), and writes an output CSV with ``judge_score``
column added.

Usage:
    python Experiments/judge_responses.py \
        --input-csv logs/inframind_explore.csv \
        --output-csv logs/inframind_explore_scored.csv \
        --judge-model meta-llama/Llama-3.3-70B-Instruct \
        --judge-url http://localhost:8010/v1 \
        --max-concurrent 20
"""

import argparse
import asyncio
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Judge rubric prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a multi-agent math-solving system. Your task is \
to score an agent's response given the full prompt it received and the reasoning \
strategy it was instructed to follow.

You will be given:
1. The FULL PROMPT the agent received — this includes its role description, \
format requirements, the math problem, and any context or hints from co-agents \
in the pipeline.
2. The REASONING STRATEGY the agent was told to use (Flash, Concise, or DeepThink).
3. The agent's RESPONSE.

## Scoring Rubric (1-10)

### Correctness & Reasoning (primary criteria)
- **10**: Correct final answer with clear, complete, and logically sound reasoning.
- **9**: Correct final answer; reasoning is sound but has minor formatting or \
notation imperfections.
- **8**: Correct final answer; reasoning has small gaps or unnecessary steps but \
the core logic holds.
- **7**: Nearly correct — sound method but a minor arithmetic or algebraic slip \
leads to a slightly wrong answer.
- **6**: Correct general approach identified, but execution errors (calculation \
mistakes, sign errors, mis-substitution) produce a wrong answer.
- **5**: Partially correct reasoning that captures some key ideas, but the \
solution path is incomplete or diverges before reaching a valid answer.
- **4**: Shows awareness of the relevant math topic but the approach is \
fundamentally flawed or based on incorrect assumptions.
- **3**: Mostly wrong; only superficial or tangential relevance to the problem.
- **2**: Almost entirely wrong; response contains minimal mathematical content \
related to the problem.
- **1**: Completely wrong, empty, refuses to answer, or is irrelevant to the \
problem.

### Strategy Alignment (adjust up or down by 1 point)
- **Flash**: Reward brevity and directness. A Flash response that solves the \
problem in few steps without unnecessary elaboration is ideal. Do NOT penalize \
lack of detail.
- **Concise**: Expect a balanced response — key reasoning steps shown without \
excessive verbosity. Moderate detail is appropriate.
- **DeepThink**: Expect thorough step-by-step reasoning with justifications. \
Penalize shallow or overly brief responses that skip important steps.

### Co-Agent Context Usage (adjust up or down by 1 point)
- If hints or solutions from co-agents are provided in the prompt, evaluate \
whether the agent appropriately incorporated, verified, or built upon them.
- Reward agents that correctly identify and fix errors from co-agent hints.
- Penalize blind copying of co-agent output without verification or added value.

### Format Compliance
- Penalize by 1 point if the response ignores the explicit format requirements \
stated in its prompt (e.g., missing "The answer is X" on the last line).

### Automatic Low Scores
- Response that merely restates or paraphrases the question → cap at 2.
- Response that is truncated mid-reasoning with no answer → cap at 3.
- Response that outputs only a final answer with zero reasoning when strategy \
is DeepThink → cap at 4.

Output ONLY a single integer from 1 to 10. No explanation, no extra text."""

JUDGE_USER_TEMPLATE = """\
## Full Prompt Given to Agent
{prompt_base}

## Reasoning Strategy
{strategy_name}

## Agent Response
{response}

Score (1-10):"""


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def parse_score(text: str) -> Optional[int]:
    """Extract integer score from judge response."""
    text = text.strip()
    # Try first line
    first_line = text.split("\n")[0].strip()
    # Direct integer
    if first_line.isdigit():
        score = int(first_line)
        if 1 <= score <= 10:
            return score
    # Regex fallback: find first number in range 1-10
    match = re.search(r"\b(\d{1,2})\b", first_line)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score
    # Search entire response
    match = re.search(r"\b(\d{1,2})\s*/\s*10\b", text)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score
    match = re.search(r"(?:score|rating)\s*[:=]\s*(\d{1,2})", text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score
    return None


async def judge_single(
    client: "AsyncOpenAI",
    model: str,
    prompt_base: str,
    strategy_name: str,
    response: str,
    max_retries: int = 3,
    timeout: float = 300.0,
) -> Optional[int]:
    """Call the judge LLM and return a 1-10 score."""
    user_msg = JUDGE_USER_TEMPLATE.format(
        prompt_base=prompt_base,
        strategy_name=strategy_name,
        response=response,
    )
    for attempt in range(max_retries):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=64,
                    temperature=0.0,
                ),
                timeout=timeout,
            )
            text = resp.choices[0].message.content or ""
            score = parse_score(text)
            if score is not None:
                return score
            logger.warning("Could not parse score from judge response: {}", text[:100])
        except asyncio.TimeoutError:
            logger.warning("Judge timeout (attempt {}/{})", attempt + 1, max_retries)
        except Exception as e:
            logger.warning("Judge API error (attempt {}/{}): {}", attempt + 1, max_retries, e)
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_input_csv(path: str) -> List[Dict[str, str]]:
    """Load CSV and filter to role_step rows that have response data."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_scored_map(output_path: str) -> Dict[str, int]:
    """Load already-scored row identifiers with their scores for resume.

    Only rows with a non-empty ``judge_score`` are considered scored.
    Returns a mapping from row key to score.
    """
    if not os.path.isfile(output_path):
        return {}
    scored = {}
    with open(output_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_str = row.get("judge_score", "").strip()
            if score_str:
                try:
                    scored[_row_key(row)] = int(score_str)
                except ValueError:
                    pass
    return scored


def _row_key(row: Dict[str, str]) -> str:
    """Unique identifier for a row (episode + step + arrival rate)."""
    return f"{row.get('run_id', '')}|{row.get('episode_index', '')}|{row.get('step_index', '')}|{row.get('arrival_rate', '')}|{row.get('record_type', '')}"


async def run_judge_pipeline(args: argparse.Namespace) -> None:
    if AsyncOpenAI is None:
        logger.error("openai package not installed. Run: pip install openai")
        return

    client = AsyncOpenAI(
        base_url=args.judge_url,
        api_key=args.api_key or "EMPTY",
    )

    # Load input
    all_rows = load_input_csv(args.input_csv)
    logger.info("Loaded {} total rows from {}", len(all_rows), args.input_csv)

    # Determine output fieldnames (add judge_score)
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
    if "judge_score" not in fieldnames:
        fieldnames.append("judge_score")

    # Resume support: load existing scores (only rows with non-empty judge_score)
    existing_scores: Dict[str, int] = {}
    if args.resume and os.path.isfile(args.output_csv):
        existing_scores = load_scored_map(args.output_csv)
        logger.info("Resuming: {} rows already scored, will skip those", len(existing_scores))

    # Separate rows: role_step rows with response get judged, others pass through
    to_judge = []
    passthrough = []
    for row in all_rows:
        key = _row_key(row)
        if key in existing_scores:
            # Already scored — carry the score forward, treat as passthrough
            row["judge_score"] = str(existing_scores[key])
            passthrough.append(row)
            continue
        record_type = row.get("record_type", "")
        response = row.get("response", "").strip()
        prompt_base = row.get("prompt_base", "").strip()
        if record_type == "role_step" and response and prompt_base:
            to_judge.append(row)
        else:
            passthrough.append(row)

    logger.info("{} rows to judge, {} passthrough rows (incl. {} already scored)",
                len(to_judge), len(passthrough), len(existing_scores))

    # Always rewrite the full output file (fresh write with all rows)
    out_f = open(args.output_csv, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    # Write passthrough rows immediately (episode rows + already-scored rows)
    for row in passthrough:
        row.setdefault("judge_score", "")
        writer.writerow(row)
    out_f.flush()

    # Concurrent judging with semaphore
    semaphore = asyncio.Semaphore(args.max_concurrent)
    write_lock = asyncio.Lock()
    scored_count = 0
    failed_count = 0
    start_time = time.time()

    async def judge_and_write(row: Dict[str, str]) -> None:
        nonlocal scored_count, failed_count
        async with semaphore:
            score = await judge_single(
                client,
                model=args.judge_model,
                prompt_base=row.get("prompt_base", ""),
                strategy_name=row.get("strategy_name", ""),
                response=row.get("response", ""),
            )
        async with write_lock:
            if score is not None:
                row["judge_score"] = str(score)
                scored_count += 1
            else:
                row["judge_score"] = ""
                failed_count += 1
            writer.writerow(row)
            total = scored_count + failed_count
            if total % 100 == 0:
                out_f.flush()
                elapsed = time.time() - start_time
                rate = total / max(elapsed, 0.01)
                remaining = (len(to_judge) - total) / max(rate, 0.01)
                logger.info(
                    "Progress: {}/{} scored ({} failed), {:.1f} rows/s, ETA {:.0f}s",
                    scored_count, len(to_judge), failed_count, rate, remaining,
                )

    # Run all judge tasks
    tasks = [judge_and_write(row) for row in to_judge]
    await asyncio.gather(*tasks)

    out_f.flush()
    out_f.close()
    elapsed = time.time() - start_time
    logger.info(
        "Done: {}/{} scored, {} failed, {:.1f}s elapsed. Output: {}",
        scored_count, len(to_judge), failed_count, elapsed, args.output_csv,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge scoring for InfraMind Phase 1 exploration data.",
    )
    parser.add_argument(
        "--input-csv", type=str, required=True,
        help="Path to Phase 1 exploration CSV with response column.",
    )
    parser.add_argument(
        "--output-csv", type=str, required=True,
        help="Output CSV path (input CSV + judge_score column).",
    )
    parser.add_argument(
        "--judge-model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name for the judge LLM.",
    )
    parser.add_argument(
        "--judge-url", type=str, default="http://localhost:8010/v1",
        help="OpenAI-compatible API base URL for the judge LLM.",
    )
    parser.add_argument(
        "--api-key", type=str, default="",
        help="API key for the judge endpoint (default: EMPTY).",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=20,
        help="Maximum concurrent judge requests.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from partial output (append to existing output CSV).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_judge_pipeline(args))
