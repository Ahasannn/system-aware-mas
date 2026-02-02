#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import multiprocessing as mp
from threading import Thread

def timeout_handler(_, __):
    raise TimeoutError()


def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)


class PropagatingThread(Thread):
    """Kept for backward compatibility."""
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


def _timeout_worker(func, args, result_queue):
    """Worker for multiprocessing-based timeout."""
    try:
        result = func(*args)
        result_queue.put(("ok", result))
    except BaseException as e:
        result_queue.put(("error", e))


def function_with_timeout(func, args, timeout):
    """Execute *func(*args)* with a hard timeout.

    Uses multiprocessing so the child process can be forcibly killed
    when the timeout expires.  The previous threading implementation
    could not actually stop runaway computations (Python threads are
    not killable), which caused training to hang indefinitely.
    """
    result_queue = mp.Queue()
    proc = mp.Process(target=_timeout_worker, args=(func, args, result_queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(1)
        if proc.is_alive():
            proc.kill()
        raise TimeoutError()

    status, value = result_queue.get(timeout=1)
    if status == "error":
        raise value
    return value
    

