"""End-to-end live pipeline.

Wires capture → segmenter → ASR → (translation) → UI events.

Threads:
  * PortAudio audio thread (callback) pushes capture chunks onto a queue.
  * `_capture_pump_loop` consumes them, runs the VU meter, and forwards to
    the segmenter queue.
  * `_segmenter_loop` runs VAD/segmentation and pushes finalized segments
    to the ASR queue.
  * `_asr_loop` runs Whisper (and translation) serialized on a single thread
    to avoid oversubscribing the GPU.
  * `_preview_loop` periodically peeks the segmenter's open segment and, if
    big enough, runs a *fast* Whisper pass on it for live caption output.
    Preview transcripts are emitted with a `segment_id` tying them to the
    eventual final, so the UI can replace them when the segment closes.

Auto-recovery: if the capture stream raises (e.g. user unplugs a USB DAC),
the supervisor catches it, re-runs the device picker, and restarts capture
without dropping the segmenter / ASR threads.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Iterator, Optional

import numpy as np

from ..ai.gemini_engine import AiPromptContext, GeminiResponder, detect_question
from ..asr.whisper_engine import Transcript, WhisperEngine
from ..audio.capture import AudioCapture, CaptureChunk, CaptureConfig, DeviceInfo
from ..audio.device_picker import select_capture_backend
from ..config import get_settings
from ..hardware.accel import AccelProfile, detect_accel
from .segmenter import AudioSegment, Segmenter
from .vad import SileroVAD

log = logging.getLogger(__name__)


# --- Event types pushed to UI ---

@dataclass
class EvtLoading:
    """Progress messages while heavy models load. The bridge thread emits these
    BEFORE EvtSystemReady so the GUI can show a meaningful loading screen."""
    stage: str        # "vad" | "whisper" | "translator" | "capture"
    message: str
    progress: float   # 0..1, best effort


@dataclass
class EvtSystemReady:
    device: DeviceInfo
    accel: AccelProfile
    whisper_model: str


@dataclass
class EvtAudioLevel:
    rms: float
    peak: float


@dataclass
class EvtSegmentDetected:
    segment_id: int
    start: float
    duration: float


@dataclass
class EvtPreviewTranscript:
    """In-progress transcription of an open segment. UI should replace any
    previous preview with the same segment_id, and clear when the matching
    EvtTranscript arrives."""
    segment_id: int
    text: str
    translation: Optional[str]
    language: str
    language_prob: float
    duration_s: float                  # how much audio is in the preview so far
    asr_seconds: float


@dataclass
class EvtTranscript:
    transcript: Transcript
    segment: AudioSegment
    translation: Optional[str] = None          # set only when needed (non-target language)
    queued_at: float = 0.0
    finished_at: float = 0.0
    translation_seconds: float = 0.0           # 0 if no translation happened

    @property
    def segment_id(self) -> int:
        return self.segment.segment_id

    @property
    def end_to_end_latency_s(self) -> float:
        return self.finished_at - self.segment.start_time


@dataclass
class EvtAiResponseStart:
    request_id: int
    detected_question: Optional[str]
    language: str


@dataclass
class EvtAiResponseDelta:
    request_id: int
    delta: str             # incremental text chunk


@dataclass
class EvtAiResponseDone:
    request_id: int
    full_text: str
    seconds: float


@dataclass
class EvtAiResponseError:
    request_id: int
    message: str


@dataclass
class EvtError:
    message: str
    fatal: bool = False


Event = (
    EvtLoading
    | EvtSystemReady
    | EvtAudioLevel
    | EvtSegmentDetected
    | EvtPreviewTranscript
    | EvtTranscript
    | EvtAiResponseStart
    | EvtAiResponseDelta
    | EvtAiResponseDone
    | EvtAiResponseError
    | EvtError
)


@dataclass
class _TranscriptRecord:
    """One finalised transcript kept in the AI context history."""
    text: str
    translation: Optional[str]
    language: str
    monotonic_time: float


# --- Orchestrator ---

class Orchestrator:
    def __init__(
        self,
        translator: Optional[Callable[[str, str], str]] = None,
        translator_factory: Optional[Callable[[], Callable[[str, str], str]]] = None,
        target_lang: str = "es",
    ):
        """
        Args:
            translator: a ready-to-use translation callable. Avoid for GUIs —
                use `translator_factory` so loading happens in the background.
            translator_factory: a zero-arg callable returning a translation
                callable. Called from the bridge thread during `start()`, so
                the main thread is never blocked by NLLB load.
            target_lang: ISO 639-1 of the target language. Defaults to "es".
        """
        s = get_settings()
        self._settings = s
        self._accel = detect_accel()
        self._capture: AudioCapture | None = None
        self._segmenter: Segmenter | None = None
        self._whisper: WhisperEngine | None = None
        self._translator = translator
        self._translator_factory = translator_factory
        self._target_lang = target_lang

        # Queues
        self._chunk_q: queue.Queue[CaptureChunk] = queue.Queue(maxsize=400)
        self._seg_q: queue.Queue[AudioSegment] = queue.Queue(maxsize=32)
        # Translation runs on its own worker so ASR can start the NEXT segment
        # while NLLB is still busy. Item: (segment, transcript, queued_at).
        self._trans_q: queue.Queue[tuple[AudioSegment, "Transcript", float]] = queue.Queue(maxsize=32)
        self._ai_q: queue.Queue[int] = queue.Queue(maxsize=8)   # carries request_ids
        self._event_q: asyncio.Queue[Event] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Threads
        self._stop_evt = threading.Event()
        self._t_segmenter: threading.Thread | None = None
        self._t_asr: threading.Thread | None = None
        self._t_capture_pump: threading.Thread | None = None
        self._t_preview: threading.Thread | None = None
        self._t_translation: threading.Thread | None = None
        self._t_ai: threading.Thread | None = None

        # AI responder (lazy: built on first request if not provided).
        self._ai_responder: GeminiResponder | None = None
        # History of finalised transcripts for AI context.
        self._history: deque[_TranscriptRecord] = deque(maxlen=64)
        self._history_lock = threading.Lock()
        self._ai_request_counter = itertools.count(1)

        # Single mutex around the Whisper model so finals and previews don't
        # run concurrently. faster-whisper is thread-safe but we still want
        # serialization to avoid VRAM contention on long segments.
        self._asr_lock = threading.Lock()

        # State for preview loop.
        self._last_finalized_id = -1
        self._last_preview_id = -1
        self._last_preview_samples = 0

        # Audio meter EMA
        self._meter_rms = 0.0
        self._meter_peak = 0.0
        self._meter_last_emit = 0.0

    # --- lifecycle ---

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._emit(EvtLoading(stage="vad",
                                     message="Cargando VAD (Silero)...", progress=0.05))
        await asyncio.to_thread(self._load_vad)

        await self._emit(EvtLoading(stage="whisper",
                                     message=f"Cargando Whisper {self._settings.whisper_model} (CUDA float16)...",
                                     progress=0.20))
        await asyncio.to_thread(self._load_whisper)

        if self._translator is None and self._translator_factory is not None:
            await self._emit(EvtLoading(stage="translator",
                                         message="Cargando traductor NLLB-200 1.3B (~5 GB)...",
                                         progress=0.55))
            await asyncio.to_thread(self._load_translator)

        await self._emit(EvtLoading(stage="capture",
                                     message="Abriendo dispositivo de audio...",
                                     progress=0.92))
        self._init_capture()
        self._start_workers()
        await self._emit(EvtSystemReady(
            device=self._capture.device,  # type: ignore[union-attr]
            accel=self._accel,
            whisper_model=self._whisper.model_name,  # type: ignore[union-attr]
        ))

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception as e:  # pragma: no cover
                log.warning("Error stopping capture: %s", e)
        for t in (self._t_capture_pump, self._t_segmenter, self._t_asr,
                  self._t_preview, self._t_translation, self._t_ai):
            if t is not None:
                t.join(timeout=2.0)

    # --- AI responder public API ---

    @property
    def ai_available(self) -> bool:
        return self._ai_responder is not None and self._ai_responder.is_available

    def clear_history(self) -> None:
        """Drop every stored transcript so the next AI request only sees what
        arrives FROM NOW ON. Called by the GUI when the user clicks 'Limpiar'.

        Also drains pending AI requests so a click queued just before clear
        doesn't run against the about-to-be-discarded context.
        """
        with self._history_lock:
            self._history.clear()
        try:
            while True:
                self._ai_q.get_nowait()
        except queue.Empty:
            pass

    def request_ai_response(self) -> int:
        """Trigger an AI response over the current transcript history.

        Returns the new request_id. The result will arrive as
        `EvtAiResponseStart` → many `EvtAiResponseDelta` → `EvtAiResponseDone`
        (or `EvtAiResponseError`) on the event queue, so any UI can subscribe.
        Returns -1 if no AI backend is configured.
        """
        if not self.ai_available:
            self._emit_threadsafe(EvtAiResponseError(
                request_id=-1,
                message="Gemini no está configurado. Define GEMINI_API_KEY en .env.",
            ))
            return -1
        request_id = next(self._ai_request_counter)
        try:
            self._ai_q.put_nowait(request_id)
        except queue.Full:
            self._emit_threadsafe(EvtAiResponseError(
                request_id=request_id,
                message="Cola de IA llena; espera a que termine la respuesta anterior.",
            ))
            return -1
        return request_id

    def events(self) -> AsyncIterator[Event]:
        async def gen():
            while not self._stop_evt.is_set() or not self._event_q.empty():
                try:
                    ev = await asyncio.wait_for(self._event_q.get(), timeout=0.5)
                    yield ev
                except asyncio.TimeoutError:
                    continue
        return gen()

    # --- helpers ---

    def _load_vad(self) -> None:
        vad = SileroVAD()
        self._segmenter = Segmenter(vad)

    def _load_whisper(self) -> None:
        self._whisper = WhisperEngine(accel=self._accel)

    def _load_translator(self) -> None:
        if self._translator_factory is None:
            return
        try:
            self._translator = self._translator_factory()
        except Exception as e:
            log.exception("Translator factory failed: %s", e)
            self._translator = None

    def _ensure_ai_responder(self) -> None:
        """Build the Gemini responder lazily. Safe to call repeatedly."""
        if self._ai_responder is not None:
            return
        try:
            self._ai_responder = GeminiResponder()
        except Exception as e:
            log.warning("Could not initialise Gemini responder: %s", e)
            self._ai_responder = None

    def _init_capture(self) -> None:
        cfg = CaptureConfig(
            sample_rate=self._settings.sample_rate,
            channels=self._settings.channels,
            block_ms=self._settings.capture_block_ms,
            device_hint=self._settings.audio_device_hint,
        )
        self._capture = select_capture_backend(cfg)
        self._capture.start()

    def _start_workers(self) -> None:
        self._t_capture_pump = threading.Thread(
            target=self._capture_pump_loop, name="capture-pump", daemon=True,
        )
        self._t_segmenter = threading.Thread(
            target=self._segmenter_loop, name="segmenter", daemon=True,
        )
        self._t_asr = threading.Thread(
            target=self._asr_loop, name="asr", daemon=True,
        )
        self._t_preview = threading.Thread(
            target=self._preview_loop, name="asr-preview", daemon=True,
        )
        self._t_capture_pump.start()
        self._t_segmenter.start()
        self._t_asr.start()
        if self._translator is not None:
            self._t_translation = threading.Thread(
                target=self._translation_loop, name="translation", daemon=True,
            )
            self._t_translation.start()
        if self._settings.preview_enabled:
            self._t_preview.start()

        # AI responder is built lazily, but the worker is always there so a
        # request can be honoured (or rejected cleanly) without a race.
        self._ensure_ai_responder()
        self._t_ai = threading.Thread(
            target=self._ai_worker_loop, name="ai-responder", daemon=True,
        )
        self._t_ai.start()

    # --- worker loops ---

    def _capture_pump_loop(self) -> None:
        assert self._capture is not None
        try:
            for chunk in self._capture.chunks(timeout=0.5):
                pcm = chunk.pcm
                if pcm.size:
                    peak = float(np.max(np.abs(pcm)))
                    rms = float(np.sqrt(np.mean(pcm * pcm)))
                    self._meter_rms = 0.9 * self._meter_rms + 0.1 * rms
                    self._meter_peak = max(0.85 * self._meter_peak, peak)
                    now = time.monotonic()
                    if now - self._meter_last_emit > 0.2:
                        self._meter_last_emit = now
                        self._emit_threadsafe(EvtAudioLevel(
                            rms=self._meter_rms, peak=self._meter_peak,
                        ))

                try:
                    self._chunk_q.put(chunk, timeout=0.2)
                except queue.Full:
                    log.warning("Chunk queue full — dropping. ASR may be falling behind.")
                if self._stop_evt.is_set():
                    break
        except Exception as e:
            log.exception("Capture pump failed: %s", e)
            self._emit_threadsafe(EvtError(f"Capture failed: {e}", fatal=False))
            if not self._stop_evt.is_set():
                self._attempt_capture_recovery()

    def _segmenter_loop(self) -> None:
        assert self._segmenter is not None
        while not self._stop_evt.is_set():
            try:
                chunk = self._chunk_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                segs = self._segmenter.push(chunk.pcm, chunk.timestamp)
                for s in segs:
                    self._emit_threadsafe(EvtSegmentDetected(
                        segment_id=s.segment_id,
                        start=s.start_time, duration=s.duration_s,
                    ))
                    try:
                        self._seg_q.put(s, timeout=0.5)
                    except queue.Full:
                        log.warning("ASR queue full; dropping oldest.")
                        try:
                            self._seg_q.get_nowait()
                        except queue.Empty:
                            pass
                        self._seg_q.put_nowait(s)
            except Exception as e:
                log.exception("Segmenter error: %s", e)

        try:
            for s in self._segmenter.flush():
                self._seg_q.put_nowait(s)
        except Exception:
            pass

    def _asr_loop(self) -> None:
        """Transcribe finalized segments. Hand off translation to its own worker.

        For target-language audio: emit `EvtTranscript` immediately (no translation).
        For other languages: hand off to `_translation_loop` which emits the
        final event with translation included.

        This keeps the ASR thread free to start the NEXT segment while NLLB
        is still busy with the previous one — which is what eliminates the
        burst-lag the user observed.
        """
        assert self._whisper is not None
        last_seg_end = 0.0
        while not self._stop_evt.is_set():
            try:
                seg = self._seg_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                queued_at = time.monotonic()
                if last_seg_end and seg.start_time - last_seg_end > 8.0:
                    self._whisper.reset_context()

                with self._asr_lock:
                    tr = self._whisper.transcribe(seg.pcm)
                last_seg_end = seg.end_time
                self._last_finalized_id = seg.segment_id

                needs_translation = (
                    self._translator is not None
                    and bool(tr.text)
                    and not tr.language.lower().startswith(self._target_lang)
                )

                if not needs_translation:
                    self._record_history(tr, translation=None)
                    self._emit_threadsafe(EvtTranscript(
                        transcript=tr, segment=seg, translation=None,
                        queued_at=queued_at, finished_at=time.monotonic(),
                    ))
                    continue

                # Hand off to translation worker; it will emit the EvtTranscript.
                try:
                    self._trans_q.put((seg, tr, queued_at), timeout=0.5)
                except queue.Full:
                    log.warning("Translation queue full; dropping oldest.")
                    try:
                        self._trans_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._trans_q.put_nowait((seg, tr, queued_at))
            except Exception as e:
                log.exception("ASR error: %s", e)
                self._emit_threadsafe(EvtError(f"ASR error: {e}"))

    def _translation_loop(self) -> None:
        """Dedicated translation worker. Emits the final EvtTranscript with translation."""
        assert self._translator is not None
        while not self._stop_evt.is_set():
            try:
                seg, tr, queued_at = self._trans_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                t0 = time.monotonic()
                translation = self._translator(tr.text, tr.language)
                dt = time.monotonic() - t0
            except Exception as e:
                log.exception("Translation failed: %s", e)
                translation = f"[translation error: {e}]"
                dt = 0.0
            self._record_history(tr, translation=translation)
            self._emit_threadsafe(EvtTranscript(
                transcript=tr, segment=seg, translation=translation,
                queued_at=queued_at, finished_at=time.monotonic(),
                translation_seconds=dt,
            ))

    def _preview_loop(self) -> None:
        """Periodically transcribe the in-progress segment for live captioning."""
        assert self._segmenter is not None and self._whisper is not None
        s = self._settings
        interval = s.preview_interval_ms / 1000.0
        min_samples = int(s.preview_min_audio_ms / 1000 * 16000)
        # Re-preview only if at least +1s of new audio has accumulated.
        growth_threshold_samples = 16000

        while not self._stop_evt.is_set():
            time.sleep(interval)
            if self._stop_evt.is_set():
                break

            snap = self._segmenter.peek_open(min_samples=min_samples)
            if snap is None:
                continue
            if snap.segment_id <= self._last_finalized_id:
                # The segment has already been finalized by the time we got here.
                continue
            if (
                snap.segment_id == self._last_preview_id
                and snap.pcm.size < self._last_preview_samples + growth_threshold_samples
            ):
                # Not enough new audio to bother re-running.
                continue

            try:
                # If the ASR worker is currently busy with a final, skip this
                # preview iteration — finals always win.
                acquired = self._asr_lock.acquire(timeout=0.05)
                if not acquired:
                    continue
                try:
                    tr = self._whisper.transcribe(snap.pcm, preview=True)
                finally:
                    self._asr_lock.release()

                if not tr.text:
                    continue

                translation: Optional[str] = None
                if (
                    self._translator is not None
                    and not tr.language.lower().startswith(self._target_lang)
                ):
                    try:
                        translation = self._translator(tr.text, tr.language)
                    except Exception as e:
                        log.debug("Preview translation failed: %s", e)

                self._last_preview_id = snap.segment_id
                self._last_preview_samples = snap.pcm.size
                self._emit_threadsafe(EvtPreviewTranscript(
                    segment_id=snap.segment_id,
                    text=tr.text,
                    translation=translation,
                    language=tr.language,
                    language_prob=tr.language_prob,
                    duration_s=tr.duration_s,
                    asr_seconds=tr.asr_seconds,
                ))
            except Exception as e:
                log.debug("Preview loop error (non-fatal): %s", e)

    # --- history + AI worker

    def _record_history(self, tr: Transcript, translation: Optional[str]) -> None:
        if not tr.text:
            return
        with self._history_lock:
            self._history.append(_TranscriptRecord(
                text=tr.text,
                translation=translation,
                language=tr.language,
                monotonic_time=time.monotonic(),
            ))

    def _build_ai_context(self) -> AiPromptContext:
        s = self._settings
        cutoff = time.monotonic() - s.ai_history_seconds
        with self._history_lock:
            recent = [r for r in self._history if r.monotonic_time >= cutoff]
            if not recent and self._history:
                # Fallback to the very last entry so we always have something.
                recent = [self._history[-1]]
        # Cap context length, keeping the MOST recent material.
        joined_parts: list[str] = []
        total = 0
        for r in reversed(recent):
            line = f"[{r.language}] {r.text}"
            if total + len(line) + 1 > s.ai_max_context_chars:
                break
            joined_parts.append(line)
            total += len(line) + 1
        joined = "\n".join(reversed(joined_parts))

        # Language for the response = language of the most recent transcript
        # (the listener probably wants the answer in the speaker's tongue).
        language = recent[-1].language if recent else self._settings.target_language

        # Question detection scans the most recent transcripts first.
        detected = None
        for r in reversed(recent):
            q = detect_question(r.text, r.language)
            if q:
                detected = q
                break

        return AiPromptContext(
            transcript_recent=joined or "(no transcript yet)",
            detected_question=detected,
            language=language,
        )

    def _ai_worker_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                request_id = self._ai_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._ai_responder is None or not self._ai_responder.is_available:
                self._emit_threadsafe(EvtAiResponseError(
                    request_id=request_id,
                    message="Gemini no está configurado. Define GEMINI_API_KEY en .env.",
                ))
                continue
            try:
                ctx = self._build_ai_context()
                self._emit_threadsafe(EvtAiResponseStart(
                    request_id=request_id,
                    detected_question=ctx.detected_question,
                    language=ctx.language,
                ))
                t0 = time.monotonic()
                parts: list[str] = []
                for delta in self._ai_responder.respond_streaming(ctx):
                    if self._stop_evt.is_set():
                        break
                    parts.append(delta)
                    self._emit_threadsafe(EvtAiResponseDelta(
                        request_id=request_id, delta=delta,
                    ))
                full = "".join(parts)
                self._emit_threadsafe(EvtAiResponseDone(
                    request_id=request_id, full_text=full,
                    seconds=time.monotonic() - t0,
                ))
            except Exception as e:
                log.exception("AI worker failed: %s", e)
                self._emit_threadsafe(EvtAiResponseError(
                    request_id=request_id, message=str(e),
                ))

    def _attempt_capture_recovery(self) -> None:
        """Reopen the capture stream with exponential backoff.

        Common triggers we recover from:
          * default output device changed (user plugged in headphones).
          * audio service restart on Windows.
          * brief PulseAudio sink disappearance on Linux.
        """
        delays = [1.0, 2.0, 4.0, 8.0, 16.0]
        for i, delay in enumerate(delays, start=1):
            if self._stop_evt.is_set():
                return
            log.info("Capture recovery attempt %d/%d in %.1fs...", i, len(delays), delay)
            time.sleep(delay)
            try:
                self._init_capture()
                # Restart the pump thread that died with the previous stream.
                self._t_capture_pump = threading.Thread(
                    target=self._capture_pump_loop, name="capture-pump", daemon=True,
                )
                self._t_capture_pump.start()
                dev = self._capture.device if self._capture else None
                log.info("Capture recovered on device: %s", dev.name if dev else "?")
                self._emit_threadsafe(EvtSystemReady(
                    device=dev,  # type: ignore[arg-type]
                    accel=self._accel,
                    whisper_model=self._whisper.model_name if self._whisper else "?",
                ))
                return
            except Exception as e:
                log.warning("Recovery attempt %d failed: %s", i, e)
        # Exhausted retries.
        log.error("Capture recovery exhausted all retries.")
        self._emit_threadsafe(EvtError(
            "Capture device could not be re-opened. Reproduce audio and restart the app.",
            fatal=True,
        ))

    # --- event plumbing ---

    async def _emit(self, ev: Event) -> None:
        await self._event_q.put(ev)

    def _emit_threadsafe(self, ev: Event) -> None:
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._event_q.put_nowait, ev)
        except RuntimeError:
            pass
