"""Bridge between the asyncio-driven Orchestrator and Qt's signal/slot system.

Runs the orchestrator's event loop in a dedicated QThread, then converts
each pipeline event into a Qt signal so widgets can update from the main
GUI thread without manual marshalling.
"""

from __future__ import annotations

import asyncio
import logging

from PySide6.QtCore import QObject, QThread, Signal

from ..pipeline.orchestrator import (
    EvtAudioLevel,
    EvtError,
    EvtLoading,
    EvtPreviewTranscript,
    EvtSegmentDetected,
    EvtSystemReady,
    EvtTranscript,
    Orchestrator,
)

log = logging.getLogger(__name__)


class OrchestratorBridge(QObject):
    """Lives in the Qt main thread, owns a QThread running asyncio."""

    # All signals carry the dataclass instance directly. Receivers connect
    # with type=QueuedConnection (the default for cross-thread) so they run
    # in the GUI thread.
    loading = Signal(object)             # EvtLoading
    system_ready = Signal(object)        # EvtSystemReady
    audio_level = Signal(object)         # EvtAudioLevel
    segment_detected = Signal(object)    # EvtSegmentDetected
    preview = Signal(object)             # EvtPreviewTranscript
    transcript = Signal(object)          # EvtTranscript
    error = Signal(object)               # EvtError
    stopped = Signal()

    def __init__(self, orchestrator: Orchestrator, parent=None):
        super().__init__(parent)
        self._orch = orchestrator
        self._thread = QThread(parent)
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop_async)
        self._thread.quit()
        self._thread.wait(3000)

    def _stop_async(self) -> None:
        asyncio.ensure_future(self._orch.stop(), loop=self._loop)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception as e:
            log.exception("Orchestrator thread crashed: %s", e)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            self.stopped.emit()

    async def _main(self) -> None:
        # Run start() as a background task so EvtLoading events emitted during
        # model loading are consumed by the event iterator below in real time
        # (otherwise we'd buffer all progress until start() returned).
        start_task = asyncio.create_task(self._safe_start())
        try:
            async for ev in self._orch.events():
                self._dispatch(ev)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("Bridge main failed: %s", e)
            self.error.emit(EvtError(f"Bridge: {e}", fatal=True))
        finally:
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _safe_start(self) -> None:
        try:
            await self._orch.start()
        except Exception as e:
            log.exception("Orchestrator start failed: %s", e)
            self.error.emit(EvtError(f"Inicialización falló: {e}", fatal=True))

    def _dispatch(self, ev) -> None:
        if isinstance(ev, EvtLoading):
            self.loading.emit(ev)
        elif isinstance(ev, EvtSystemReady):
            self.system_ready.emit(ev)
        elif isinstance(ev, EvtAudioLevel):
            self.audio_level.emit(ev)
        elif isinstance(ev, EvtSegmentDetected):
            self.segment_detected.emit(ev)
        elif isinstance(ev, EvtPreviewTranscript):
            self.preview.emit(ev)
        elif isinstance(ev, EvtTranscript):
            self.transcript.emit(ev)
        elif isinstance(ev, EvtError):
            self.error.emit(ev)
