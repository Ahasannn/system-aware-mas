import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

from MAR.Utils.request_patterns import RequestPattern


@dataclass
class RequestResult:
    index: int
    item_id: str
    start_ts: float
    end_ts: float
    latency_seconds: float
    success: bool
    error: str
    output: Optional[Any]


class RequestShooter:
    def __init__(
        self,
        pattern: RequestPattern,
        max_concurrency: int = 8,
        poll_interval: float = 0.01,
        capture_output: bool = False,
        collect_results: bool = True,
        on_result: Optional[Callable[[RequestResult], None]] = None,
    ) -> None:
        self.pattern = pattern
        self.max_concurrency = max(1, int(max_concurrency))
        self.poll_interval = float(poll_interval)
        self.capture_output = bool(capture_output)
        self.collect_results = bool(collect_results)
        self.on_result = on_result

        self._semaphore = threading.Semaphore(self.max_concurrency)
        self._results: List[RequestResult] = []
        self._results_lock = threading.Lock()

    def run(
        self,
        items: Iterable[Any],
        handler: Callable[[Any], Any],
        item_id_fn: Optional[Callable[[Any, int], str]] = None,
    ) -> List[RequestResult]:
        threads: List[threading.Thread] = []
        if self.collect_results:
            with self._results_lock:
                self._results = []

        def _worker(item: Any, idx: int, item_id: str) -> None:
            try:
                start_wall = time.time()
                start_perf = time.perf_counter()
                output: Optional[Any] = None
                success = False
                error = ""
                try:
                    result = handler(item)
                    success = True
                    if self.capture_output:
                        output = result
                except Exception as exc:
                    error = str(exc)
                end_perf = time.perf_counter()
                end_wall = time.time()
                request_result = RequestResult(
                    index=idx,
                    item_id=item_id,
                    start_ts=start_wall,
                    end_ts=end_wall,
                    latency_seconds=end_perf - start_perf,
                    success=success,
                    error=error,
                    output=output,
                )
                if self.collect_results:
                    with self._results_lock:
                        self._results.append(request_result)
                if self.on_result:
                    self.on_result(request_result)
            finally:
                self._semaphore.release()

        for idx, item in enumerate(items):
            self._semaphore.acquire()
            try:
                item_id = item_id_fn(item, idx) if item_id_fn else str(idx)
            except Exception:
                self._semaphore.release()
                raise
            thread = threading.Thread(target=_worker, args=(item, idx, item_id), daemon=True)
            thread.start()
            threads.append(thread)
            time.sleep(self.pattern.next_delay())

        for thread in threads:
            thread.join()

        if not self.collect_results:
            return []
        with self._results_lock:
            return list(self._results)
