#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from MAR.Utils.const import MAR_ROOT


class ProgressTracker:
    """Track and log training/testing progress with comprehensive statistics."""

    def __init__(self, total: int, phase: str = "Processing", log_interval: int = 10):
        self.total = total
        self.phase = phase
        self.log_interval = log_interval
        self.processed = 0
        self.succeeded = 0
        self.failed = 0
        self.model_counts: Dict[str, int] = defaultdict(int)
        self.topology_counts: Dict[str, int] = defaultdict(int)
        self.strategy_counts: Dict[str, int] = defaultdict(int)
        self.total_latency = 0.0
        self.total_quality = 0.0
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._last_log_count = 0

    def update(
        self,
        success: bool = True,
        models: Optional[List[str]] = None,
        topology: Optional[str] = None,
        strategies: Optional[List[str]] = None,
        latency: Optional[float] = None,
        quality: Optional[float] = None,
    ) -> None:
        """Update progress after processing a request."""
        with self._lock:
            self.processed += 1
            if success:
                self.succeeded += 1
                if models:
                    for model in models:
                        self.model_counts[model] += 1
                if topology:
                    self.topology_counts[topology] += 1
                if strategies:
                    for strategy in strategies:
                        if strategy:  # Skip empty strategies
                            self.strategy_counts[strategy] += 1
                if latency is not None:
                    self.total_latency += latency
                if quality is not None:
                    self.total_quality += quality
            else:
                self.failed += 1
            if self.processed - self._last_log_count >= self.log_interval:
                self._log_progress()
                self._last_log_count = self.processed

    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed = time.time() - self.start_time
        pending = self.total - self.processed
        pct = (self.processed / self.total * 100) if self.total > 0 else 0
        rate = self.processed / elapsed if elapsed > 0 else 0
        eta = pending / rate if rate > 0 else 0

        logger.info(
            "[{}] Progress: {}/{} ({:.1f}%) | Pending: {} | Success: {} | Failed: {} | Rate: {:.2f} req/s | ETA: {:.1f}s",
            self.phase,
            self.processed,
            self.total,
            pct,
            pending,
            self.succeeded,
            self.failed,
            rate,
            eta,
        )

    def log_stats(self) -> None:
        """Log comprehensive usage statistics."""
        with self._lock:
            if self.succeeded == 0:
                logger.info("[{}] No successful requests to report", self.phase)
                return

            # Average metrics
            avg_latency = self.total_latency / self.succeeded if self.succeeded > 0 else 0.0
            avg_quality = self.total_quality / self.succeeded if self.succeeded > 0 else 0.0

            logger.info("[{}] ═══════════════════════════════════════", self.phase)
            logger.info("[{}] Performance Metrics:", self.phase)
            logger.info("[{}]   Average Quality: {:.4f}", self.phase, avg_quality)
            logger.info("[{}]   Average Latency: {:.3f}s", self.phase, avg_latency)

            # Topology distribution
            if self.topology_counts:
                logger.info("[{}] Topology Distribution:", self.phase)
                sorted_topologies = sorted(self.topology_counts.items(), key=lambda x: x[1], reverse=True)
                for topology, count in sorted_topologies:
                    pct = (count / self.succeeded * 100) if self.succeeded > 0 else 0
                    logger.info("[{}]   {:20s} : {:4d} ({:5.1f}%)", self.phase, topology, count, pct)

            # Model distribution
            if self.model_counts:
                logger.info("[{}] Model Distribution:", self.phase)
                sorted_models = sorted(self.model_counts.items(), key=lambda x: x[1], reverse=True)
                total_model_requests = sum(self.model_counts.values())
                for model, count in sorted_models:
                    # Extract short model name (last part after /)
                    short_name = model.split("/")[-1] if "/" in model else model
                    pct = (count / total_model_requests * 100) if total_model_requests > 0 else 0
                    logger.info("[{}]   {:30s} : {:4d} ({:5.1f}%)", self.phase, short_name, count, pct)

            # Strategy distribution
            if self.strategy_counts:
                logger.info("[{}] Strategy Distribution:", self.phase)
                sorted_strategies = sorted(self.strategy_counts.items(), key=lambda x: x[1], reverse=True)
                total_strategy_requests = sum(self.strategy_counts.values())
                for strategy, count in sorted_strategies:
                    pct = (count / total_strategy_requests * 100) if total_strategy_requests > 0 else 0
                    logger.info("[{}]   {:20s} : {:4d} ({:5.1f}%)", self.phase, strategy, count, pct)

            logger.info("[{}] ═══════════════════════════════════════", self.phase)

    def log_final_summary(self) -> None:
        """Log final summary when phase completes."""
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        logger.info(
            "[{}] Completed: {}/{} | Success: {} | Failed: {} | Total time: {:.1f}s | Avg rate: {:.2f} req/s",
            self.phase,
            self.processed,
            self.total,
            self.succeeded,
            self.failed,
            elapsed,
            rate,
        )
        self.log_stats()

def configure_logging(print_level: str = "INFO", logfile_level: str = "DEBUG", log_name:str = "log.txt") -> None:
    """
    Configure the logging settings for the application.

    Args:
        print_level (str): The logging level for console output.
        logfile_level (str): The logging level for file output.
    """
    logger.remove()
    logger.add(sys.stderr, level=print_level)
    logger.add(MAR_ROOT /f'logs/{log_name}', level=logfile_level)

def initialize_log_file(experiment_name: str, time_stamp: str) -> Path:
    """
    Initialize the log file with a start message and return its path.

    Args:
        mode (str): The mode of operation, used in the file path.
        time_stamp (str): The current timestamp, used in the file path.

    Returns:
        Path: The path to the initialized log file.
    """
    try:
        log_file_path = MAR_ROOT / f'result/{experiment_name}/logs/log_{time_stamp}.txt'
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'w') as file:
            file.write("============ Start ============\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise
    return log_file_path

def swarmlog(sender: str, text: str, cost: float,  prompt_tokens: int, complete_tokens: int, log_file_path: str) -> None:
    """
    Custom log function for swarm operations. Includes dynamic global variables.

    Args:
        sender (str): The name of the sender.
        text (str): The text message to log.
        cost (float): The cost associated with the operation.
        result_file (Path, optional): Path to the result file. Default is None.
        solution (list, optional): Solution data to be logged. Default is an empty list.
    """
    # Directly reference global variables for dynamic values
    formatted_message = (
        f"{sender} | 💵Total Cost: ${cost:.5f} | "
        f"Prompt Tokens: {prompt_tokens} | "
        f"Completion Tokens: {complete_tokens} | \n {text}"
    )
    print(formatted_message)

    try:
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'a') as file:
            file.write(f"{formatted_message}\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise


def main():
    configure_logging()
    # Example usage of swarmlog with dynamic values
    swarmlog("SenderName", "This is a test message.", 0.123)

if __name__ == "__main__":
    main()

