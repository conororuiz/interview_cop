"""Lock-free-ish single-writer / single-reader ring buffer for float32 PCM.

We use a numpy-backed circular buffer with a threading.Lock around
producer/consumer pointer updates. Throughput is dominated by the audio
callback writing tiny ~30 ms blocks; the lock is held for a few microseconds
and is irrelevant in practice. We deliberately keep this implementation
simple rather than reaching for shared-memory or asyncio queues — it makes
the surrounding code easier to reason about and dataflow is linear.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class RingBuffer:
    def __init__(self, capacity_samples: int):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be > 0")
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._w = 0
        self._r = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def write(self, x: np.ndarray) -> int:
        """Write samples. Overwrites the oldest data if buffer is full."""
        x = np.asarray(x, dtype=np.float32)
        n = x.size
        if n == 0:
            return 0
        if n >= self._capacity:
            x = x[-self._capacity:]
            n = x.size

        with self._lock:
            first = min(n, self._capacity - self._w)
            self._buf[self._w:self._w + first] = x[:first]
            remaining = n - first
            if remaining > 0:
                self._buf[:remaining] = x[first:]
                self._w = remaining
            else:
                self._w = (self._w + first) % self._capacity

            new_size = self._size + n
            if new_size > self._capacity:
                # Drop oldest by moving the read pointer forward.
                self._r = (self._r + (new_size - self._capacity)) % self._capacity
                self._size = self._capacity
            else:
                self._size = new_size
        return n

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    def read(self, n: int) -> Optional[np.ndarray]:
        """Read up to n samples and advance the read pointer. None if empty."""
        with self._lock:
            if self._size == 0:
                return None
            n = min(n, self._size)
            out = np.empty(n, dtype=np.float32)
            first = min(n, self._capacity - self._r)
            out[:first] = self._buf[self._r:self._r + first]
            if n > first:
                out[first:] = self._buf[:n - first]
                self._r = n - first
            else:
                self._r = (self._r + first) % self._capacity
            self._size -= n
            return out

    def peek(self, n: int) -> Optional[np.ndarray]:
        """Return up to the last n samples without consuming."""
        with self._lock:
            if self._size == 0:
                return None
            n = min(n, self._size)
            start = (self._r + self._size - n) % self._capacity
            first = min(n, self._capacity - start)
            out = np.empty(n, dtype=np.float32)
            out[:first] = self._buf[start:start + first]
            if n > first:
                out[first:] = self._buf[:n - first]
            return out

    def drop_oldest(self, n: int) -> None:
        with self._lock:
            n = min(n, self._size)
            self._r = (self._r + n) % self._capacity
            self._size -= n
