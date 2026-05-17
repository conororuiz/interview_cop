"""Segmenter unit tests.

Uses a fake VAD so tests are deterministic and don't depend on Silero's
ONNX runtime or model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from transcriber.pipeline.segmenter import Segmenter
from transcriber.pipeline.vad import SILERO_CHUNK, SILERO_SR


class FakeVAD:
    """VAD stub: returns a deterministic probability per chunk number."""

    threshold = 0.5

    def __init__(self, pattern: list[float]):
        self._pattern = pattern
        self._i = 0

    def process_chunk(self, chunk: np.ndarray) -> float:
        if chunk.size != SILERO_CHUNK:
            raise ValueError("bad chunk size")
        p = self._pattern[self._i % len(self._pattern)]
        self._i += 1
        return p

    def is_speech(self, prob: float) -> bool:
        return prob >= self.threshold


def _audio_block(n_chunks: int) -> np.ndarray:
    return np.full(n_chunks * SILERO_CHUNK, 0.01, dtype=np.float32)


def test_no_speech_yields_no_segments():
    vad = FakeVAD(pattern=[0.1])
    seg = Segmenter(vad)
    finals = seg.push(_audio_block(200), timestamp=0.0)
    assert finals == []
    assert seg.flush() == []


def test_continuous_speech_then_silence_closes_segment():
    # 100 chunks of speech (~3.2s) then 30 chunks of silence (~960ms > silence_tail).
    vad = FakeVAD(pattern=[0.9] * 100 + [0.1] * 30)
    seg = Segmenter(vad)
    audio = _audio_block(130)
    finals = seg.push(audio, timestamp=0.0)
    assert len(finals) == 1
    s = finals[0]
    assert s.segment_id == 0
    assert s.duration_s >= 1.0  # at least min_segment_ms
    assert s.speech_ratio > 0.5
    assert s.pcm.dtype == np.float32


def test_segment_ids_increment():
    # Speech → silence → speech → silence → speech (last unclosed).
    pattern = (
        [0.9] * 80 + [0.1] * 40   # close at id 0
        + [0.9] * 80 + [0.1] * 40 # close at id 1
        + [0.9] * 60              # open id 2, no close yet
    )
    vad = FakeVAD(pattern=pattern)
    seg = Segmenter(vad)
    finals = seg.push(_audio_block(len(pattern)), timestamp=0.0)
    assert [f.segment_id for f in finals] == [0, 1]


def test_peek_open_returns_snapshot():
    vad = FakeVAD(pattern=[0.9] * 200)
    seg = Segmenter(vad)
    seg.push(_audio_block(120), timestamp=0.0)
    snap = seg.peek_open(min_samples=SILERO_SR)  # require >= 1s
    assert snap is not None
    assert snap.segment_id == 0
    assert snap.pcm.size >= SILERO_SR
    assert snap.end_time > snap.start_time


def test_force_cut_on_max_segment(monkeypatch):
    # Force max to small value to trigger force-cut.
    from transcriber import config as cfg
    cfg.get_settings.cache_clear()  # rebuild from env
    monkeypatch.setenv("TRANSCRIBER_MAX_SEGMENT_MS", "2000")
    cfg.get_settings.cache_clear()

    vad = FakeVAD(pattern=[0.9] * 500)  # endless speech
    seg = Segmenter(vad)
    finals = seg.push(_audio_block(200), timestamp=0.0)
    assert finals, "expected a force-cut to fire"
    # Clean up env so other tests aren't affected.
    monkeypatch.delenv("TRANSCRIBER_MAX_SEGMENT_MS", raising=False)
    cfg.get_settings.cache_clear()
