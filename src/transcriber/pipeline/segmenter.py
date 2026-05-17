"""Streaming segmenter.

Consumes capture chunks (any size) and feeds the VAD in fixed 512-sample
chunks (the only size Silero VAD v5 accepts). Emits `AudioSegment` objects
ready for Whisper.

Strategy:
  * Internal small buffer to coalesce arbitrary input into 512-sample chunks.
  * Open a segment when we see N consecutive speech chunks (hysteresis).
  * Close a segment when we see `silence_tail_ms` of non-speech *after* the
    segment has reached at least `min_segment_ms`, OR force-cut if the
    segment exceeds `max_segment_ms`.
  * Force-cut tries to land on the lowest-probability frame within the last
    ~1.5 s, so we don't slice mid-word.
  * The first `segment_overlap_ms` of every segment include audio from
    *before* the speech onset, so words spanning the boundary survive.

Pure data-flow class — no threads, no I/O. Easy to unit-test.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import get_settings
from .vad import SILERO_CHUNK, SILERO_SR, SileroVAD

log = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    pcm: np.ndarray             # float32 mono, 16 kHz
    start_time: float           # seconds (monotonic from capture timeline)
    end_time: float
    speech_ratio: float         # fraction of chunks classified as speech
    mean_prob: float            # average speech probability inside the segment
    segment_id: int = -1        # monotonically increasing per opened segment

    @property
    def duration_s(self) -> float:
        return self.pcm.size / SILERO_SR


@dataclass
class OpenSegmentSnapshot:
    """Snapshot of the audio currently accumulated in an in-progress segment."""
    segment_id: int
    pcm: np.ndarray
    start_time: float
    end_time: float


@dataclass
class _OpenSegment:
    segment_id: int = -1
    chunks: list[np.ndarray] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    start_time: float = 0.0
    n_speech: int = 0
    n_total: int = 0

    @property
    def total_samples(self) -> int:
        return sum(c.size for c in self.chunks)


class Segmenter:
    def __init__(self, vad: SileroVAD, debug_probs: bool = False):
        s = get_settings()
        self.vad = vad
        self.debug_probs = debug_probs

        self.min_samples = int(s.min_segment_ms / 1000 * SILERO_SR)
        self.max_samples = int(s.max_segment_ms / 1000 * SILERO_SR)
        self.silence_tail_chunks = max(1, int((s.silence_tail_ms / 1000 * SILERO_SR) / SILERO_CHUNK))
        self.overlap_samples = int(s.segment_overlap_ms / 1000 * SILERO_SR)

        # Pre-segment ring of recent (chunk, prob) to use as overlap prefix.
        self._pre_ring_size_chunks = max(1, self.overlap_samples // SILERO_CHUNK)
        self._pre_ring: deque[tuple[np.ndarray, float]] = deque(maxlen=self._pre_ring_size_chunks)

        self._open: _OpenSegment | None = None
        self._consecutive_silence = 0
        self._consecutive_speech = 0
        self._open_required = 2  # require 2 consecutive speech chunks to open

        # Caller-relative time origin and per-chunk advance.
        self._t_cursor: float | None = None

        # Small input buffer to assemble 512-sample chunks from arbitrary blocks.
        self._inbuf = np.zeros(0, dtype=np.float32)

        # ID counter for segments. Each opened segment gets a unique id so
        # the orchestrator can match previews to their final transcription.
        self._next_segment_id = 0

        # Lock protects _open against concurrent peek_open() from preview worker.
        self._lock = threading.Lock()

        # Diagnostic counters
        self.total_chunks = 0
        self.speech_chunks = 0
        self.max_prob = 0.0

    def peek_open(self, min_samples: int = SILERO_SR) -> Optional[OpenSegmentSnapshot]:
        """Return a snapshot of the in-progress segment, if it has at least
        `min_samples` accumulated. Thread-safe."""
        with self._lock:
            if self._open is None:
                return None
            total = self._open.total_samples
            if total < min_samples:
                return None
            pcm = np.concatenate(self._open.chunks)
            return OpenSegmentSnapshot(
                segment_id=self._open.segment_id,
                pcm=pcm,
                start_time=self._open.start_time,
                end_time=self._open.start_time + pcm.size / SILERO_SR,
            )

    def push(self, pcm: np.ndarray, timestamp: float) -> list[AudioSegment]:
        """Feed arbitrary-length audio. `timestamp` is the monotonic time of the
        first sample of `pcm`. Returns any segments finalized as a result."""
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32, copy=False)

        # Time of the first unprocessed sample in our internal buffer.
        if self._t_cursor is None:
            self._t_cursor = timestamp

        # Append to internal buffer and dispatch in 512-sample chunks.
        if self._inbuf.size:
            self._inbuf = np.concatenate([self._inbuf, pcm])
        else:
            self._inbuf = pcm.copy() if pcm.flags.owndata is False else pcm

        finalized: list[AudioSegment] = []
        n_chunks = self._inbuf.size // SILERO_CHUNK
        with self._lock:
            for i in range(n_chunks):
                chunk = self._inbuf[i * SILERO_CHUNK : (i + 1) * SILERO_CHUNK]
                chunk_t = self._t_cursor + (i * SILERO_CHUNK) / SILERO_SR
                prob = self.vad.process_chunk(chunk)
                self.total_chunks += 1
                self.max_prob = max(self.max_prob, prob)
                if self.vad.is_speech(prob):
                    self.speech_chunks += 1
                if self.debug_probs:
                    log.debug("vad t=%.2f prob=%.2f speech=%s",
                              chunk_t, prob, self.vad.is_speech(prob))
                self._process_chunk(chunk.copy(), prob, chunk_t, finalized)

        consumed = n_chunks * SILERO_CHUNK
        self._t_cursor += consumed / SILERO_SR
        self._inbuf = self._inbuf[consumed:].copy() if consumed else self._inbuf
        return finalized

    def flush(self) -> list[AudioSegment]:
        """Finalize an open segment, if any. Call once on shutdown."""
        if self._open is not None and self._open.total_samples >= self.min_samples:
            seg = self._materialize(self._t_cursor or 0.0)
            self._open = None
            return [seg]
        self._open = None
        return []

    # --- internals ---

    def _process_chunk(
        self,
        chunk: np.ndarray,
        prob: float,
        chunk_t: float,
        out: list[AudioSegment],
    ) -> None:
        is_speech = self.vad.is_speech(prob)

        if self._open is None:
            # Maintain pre-ring as potential overlap prefix.
            self._pre_ring.append((chunk, prob))
            if is_speech:
                self._consecutive_speech += 1
                if self._consecutive_speech >= self._open_required:
                    # Open segment, including the pre-ring as prefix.
                    prefix_chunks = list(self._pre_ring)
                    self._pre_ring.clear()
                    start_t = chunk_t - (len(prefix_chunks) - 1) * SILERO_CHUNK / SILERO_SR
                    seg_id = self._next_segment_id
                    self._next_segment_id += 1
                    self._open = _OpenSegment(segment_id=seg_id, start_time=start_t)
                    for c, p in prefix_chunks:
                        self._open.chunks.append(c)
                        self._open.probs.append(p)
                        self._open.n_total += 1
                        if self.vad.is_speech(p):
                            self._open.n_speech += 1
                    self._consecutive_silence = 0
            else:
                self._consecutive_speech = 0
            return

        # Inside an open segment.
        self._open.chunks.append(chunk)
        self._open.probs.append(prob)
        self._open.n_total += 1
        if is_speech:
            self._open.n_speech += 1
            self._consecutive_silence = 0
        else:
            self._consecutive_silence += 1

        seg_samples = self._open.total_samples

        # Close on silence after min_samples.
        if (
            self._consecutive_silence >= self.silence_tail_chunks
            and seg_samples >= self.min_samples
        ):
            out.append(self._materialize(chunk_t + chunk.size / SILERO_SR))
            self._reset_after_close()
            return

        # Force-cut if we hit max_samples.
        if seg_samples >= self.max_samples:
            out.append(self._force_cut(chunk_t + chunk.size / SILERO_SR))
            self._reset_after_close()

    def _materialize(self, end_t: float) -> AudioSegment:
        assert self._open is not None
        pcm = np.concatenate(self._open.chunks) if self._open.chunks else np.zeros(0, dtype=np.float32)
        ratio = self._open.n_speech / max(1, self._open.n_total)
        mean_prob = float(np.mean(self._open.probs)) if self._open.probs else 0.0
        seg = AudioSegment(
            pcm=pcm,
            start_time=self._open.start_time,
            end_time=end_t,
            speech_ratio=ratio,
            mean_prob=mean_prob,
            segment_id=self._open.segment_id,
        )
        log.debug(
            "Segment %d closed: dur=%.2fs speech=%.0f%% mean_prob=%.2f",
            seg.segment_id, seg.duration_s, ratio * 100, mean_prob,
        )
        return seg

    def _force_cut(self, end_t: float) -> AudioSegment:
        assert self._open is not None
        lookback = min(len(self._open.probs), int(1.5 * SILERO_SR / SILERO_CHUNK))
        tail_probs = self._open.probs[-lookback:]
        rel = int(np.argmin(tail_probs))
        cut_chunk_idx = len(self._open.probs) - lookback + rel + 1
        if cut_chunk_idx < 1 or cut_chunk_idx >= len(self._open.probs):
            return self._materialize(end_t)
        kept_chunks = self._open.chunks[:cut_chunk_idx]
        kept_probs = self._open.probs[:cut_chunk_idx]
        kept = np.concatenate(kept_chunks)
        ratio = sum(1 for p in kept_probs if self.vad.is_speech(p)) / max(1, len(kept_probs))
        mean_prob = float(np.mean(kept_probs)) if kept_probs else 0.0
        log.debug("Segment %d force-cut at min-prob frame, kept=%.2fs",
                  self._open.segment_id, kept.size / SILERO_SR)
        return AudioSegment(
            pcm=kept,
            start_time=self._open.start_time,
            end_time=self._open.start_time + kept.size / SILERO_SR,
            speech_ratio=ratio,
            mean_prob=mean_prob,
            segment_id=self._open.segment_id,
        )

    def _reset_after_close(self) -> None:
        self._open = None
        self._consecutive_silence = 0
        self._consecutive_speech = 0
        # Keep VAD state — speech context can carry over (no .reset() here).
