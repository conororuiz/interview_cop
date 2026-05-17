import numpy as np
import pytest

from transcriber.pipeline.ring_buffer import RingBuffer


def test_basic_write_read():
    rb = RingBuffer(10)
    rb.write(np.arange(5, dtype=np.float32))
    assert rb.size == 5
    out = rb.read(3)
    assert out is not None
    assert np.allclose(out, [0, 1, 2])
    assert rb.size == 2


def test_overwrite_oldest_when_full():
    rb = RingBuffer(5)
    rb.write(np.arange(7, dtype=np.float32))  # 0,1,2,3,4,5,6 -> keeps last 5
    out = rb.read(5)
    assert out is not None
    assert np.allclose(out, [2, 3, 4, 5, 6])


def test_wrap_around():
    rb = RingBuffer(5)
    rb.write(np.arange(3, dtype=np.float32))
    rb.read(2)  # advance read pointer
    rb.write(np.arange(3, 7, dtype=np.float32))  # 3,4,5,6
    out = rb.read(5)
    assert out is not None
    assert np.allclose(out, [2, 3, 4, 5, 6])


def test_peek_does_not_consume():
    rb = RingBuffer(10)
    rb.write(np.arange(5, dtype=np.float32))
    assert np.allclose(rb.peek(3), [2, 3, 4])  # last 3
    assert rb.size == 5
