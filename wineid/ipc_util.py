"""Cross-platform helpers for killable child processes and bounded pipe IO.

Windows has no ``select`` support for pipes and no POSIX process groups, so the
worker modules fall back to ``taskkill /T`` and deadline-bounded helper threads
there. POSIX keeps the cheaper select/session-group path.
"""
import concurrent.futures
import os
import signal
import subprocess
import time


def kill_tree(p):
    """Kill *p* and every descendant. Never raise if the process already exited."""
    if os.name == 'nt':
        try:
            subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            if p.poll() is None:
                p.kill()
    else:
        os.killpg(p.pid, signal.SIGKILL)


def with_deadline(fn, deadline):
    """Run blocking *fn* in a thread; raise TimeoutError once *deadline* passes.

    The helper thread may outlive the timeout. Callers kill the child process
    afterwards, which unblocks the pending read/write and lets the thread exit.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=max(0, deadline - time.monotonic()))
    except concurrent.futures.TimeoutError:
        raise TimeoutError('deadline') from None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
