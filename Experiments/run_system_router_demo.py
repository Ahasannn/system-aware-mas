import argparse
import os
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger

from MAR.SystemRouter.system_aware_router import SystemAwareRouter
from MAR.SystemRouter.env import SystemRouterEnv


def _default_queries() -> List[str]:
    return [
        "Summarize why reinforcement learning can help multi-agent routing.",
        "How do I parallelize three LLM calls safely under a 5 second budget?",
        "List two risks when routing to a slower but smarter model.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo runner for the System-Aware Router.")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling actions.")
    parser.add_argument("--query", type=str, help="Single query to route. Defaults to a small built-in list.")
    parser.add_argument("--tests", nargs="*", help="Python tests (MBPP-style). Required for quality scoring.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens.")
    args = parser.parse_args()

    router = SystemAwareRouter()
    env = SystemRouterEnv(router=router, max_tokens=args.max_tokens)

    queries = [args.query] if args.query else _default_queries()
    for query in queries:
        result = env.step(
            query=query,
            tests=args.tests,
            deterministic=args.deterministic,
        )
        logger.info(
            "topology={} role_set={} latency={:.3f}s quality={:.3f} response={}",
            result["topology"],
            result["role_set"],
            result["workflow_latency_seconds"],
            result["quality"],
            result["response"][:200],
        )


if __name__ == "__main__":
    main()
