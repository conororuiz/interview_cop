"""Textual TUI for the live transcriber.

Layout:

    ┌───────────────────────── Header / status ─────────────────────────┐
    │ Device · API · GPU · Model · Language detected · Live audio level │
    ├───────────────────────────── Transcript ──────────────────────────┤
    │ scrollable log of FINALIZED segments                              │
    │ timestamp · lang · text · [translation]                           │
    ├───────────────────────── Live caption ────────────────────────────┤
    │ italic / dim preview of the IN-PROGRESS segment, plus its         │
    │ translation. Replaced when the segment finalises.                 │
    ├───────────────────────────── Metrics ─────────────────────────────┤
    │ segments · avg RTF · last latency · throughput                    │
    └───────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, RichLog, Static

from ..pipeline.orchestrator import (
    EvtAudioLevel,
    EvtError,
    EvtPreviewTranscript,
    EvtSegmentDetected,
    EvtSystemReady,
    EvtTranscript,
    Orchestrator,
)


def _fmt_bar(level: float, width: int = 24) -> str:
    n = max(0, min(width, int(level * width)))
    return "█" * n + "░" * (width - n)


class StatusBar(Static):
    def __init__(self):
        super().__init__("Initializing...", id="status_bar")
        self.device_name = "—"
        self.api_name = "—"
        self.gpu_name = "—"
        self.model_name = "—"
        self.compute_type = "—"
        self.last_lang = "—"
        self.last_lang_prob = 0.0
        self.rms = 0.0
        self.peak = 0.0

    def refresh_text(self) -> None:
        bar = _fmt_bar(min(1.0, self.rms * 6))
        lang = (
            f"[bold cyan]{self.last_lang}[/] ({self.last_lang_prob:.2f})"
            if self.last_lang != "—" else "[dim]detecting…[/]"
        )
        self.update(
            f"[bold]Device[/]: {self.device_name} [dim]({self.api_name})[/]   "
            f"[bold]GPU[/]: {self.gpu_name}   "
            f"[bold]Model[/]: {self.model_name} [dim]({self.compute_type})[/]   "
            f"[bold]Lang[/]: {lang}\n"
            f"[bold]Audio[/] {bar} peak={self.peak:.2f}"
        )


class LiveCaption(Static):
    """Single-segment in-progress caption updated by EvtPreviewTranscript.

    Cleared whenever the matching EvtTranscript arrives, or when we see a
    newer segment_id (i.e. the previous one finalised but we missed the event)."""

    def __init__(self):
        super().__init__("", id="live_caption")
        self._current_id: int | None = None

    def show_preview(self, seg_id: int, lang: str, text: str, translation: str | None) -> None:
        self._current_id = seg_id
        tag = (lang or "??").upper()
        body = f"[dim italic]🎙  [{tag}…] {text}[/]"
        if translation:
            body += f"\n[dim italic]    → ES… {translation}[/]"
        self.update(body)

    def clear_if_matches(self, seg_id: int) -> None:
        if self._current_id == seg_id or self._current_id is None:
            self._current_id = None
            self.update("")

    def clear(self) -> None:
        self._current_id = None
        self.update("")


class MetricsBar(Static):
    def __init__(self):
        super().__init__("", id="metrics_bar")
        self.n_segments = 0
        self.total_asr_time = 0.0
        self.total_audio_time = 0.0
        self.last_latency = 0.0
        self.last_rtf = 0.0
        self.n_previews = 0

    def refresh_text(self) -> None:
        avg_rtf = (self.total_asr_time / self.total_audio_time) if self.total_audio_time else 0.0
        self.update(
            f"Segments: [bold]{self.n_segments}[/]   "
            f"Previews: [bold]{self.n_previews}[/]   "
            f"RTF avg: [bold]{avg_rtf:.2f}[/]   "
            f"RTF last: [bold]{self.last_rtf:.2f}[/]   "
            f"E2E latency last: [bold]{self.last_latency:.1f}s[/]"
        )


class TranscriberApp(App):
    CSS = """
    Screen { layout: vertical; }
    #status_bar {
        height: 3;
        padding: 0 1;
        background: $boost;
        color: $text;
    }
    #transcript {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    #live_caption {
        height: auto;
        min-height: 2;
        max-height: 6;
        padding: 0 1;
        border: round $accent;
        color: $text-muted;
    }
    #metrics_bar {
        height: 1;
        padding: 0 1;
        background: $boost;
    }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "clear", "Clear"),
    ]

    def __init__(self, orchestrator: Orchestrator):
        super().__init__()
        self._orch = orchestrator
        self._status = StatusBar()
        self._metrics = MetricsBar()
        self._caption = LiveCaption()
        self._transcript = RichLog(id="transcript", highlight=True, markup=True, wrap=True)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield self._status
        yield self._transcript
        yield self._caption
        yield self._metrics
        yield Footer()

    async def on_mount(self) -> None:
        self._transcript.write("[dim]Starting capture and loading models…[/]")
        self.run_worker(self._consume_events(), exclusive=True)
        self.run_worker(self._start_orchestrator(), exclusive=False)
        self.set_interval(0.1, self._status.refresh_text)
        self.set_interval(0.2, self._metrics.refresh_text)

    async def _start_orchestrator(self) -> None:
        try:
            await self._orch.start()
        except Exception as e:
            self._transcript.write(f"[bold red]Failed to start orchestrator:[/] {e}")

    async def _consume_events(self) -> None:
        async for ev in self._orch.events():
            if isinstance(ev, EvtSystemReady):
                self._status.device_name = ev.device.name
                self._status.api_name = ev.device.api_name
                self._status.gpu_name = ev.accel.gpu_name or "CPU only"
                self._status.model_name = ev.whisper_model
                self._status.compute_type = f"{ev.accel.ct2_device}/{ev.accel.ct2_compute_type}"
                self._transcript.write(
                    f"[green]Ready.[/] Capture device: [bold]{ev.device.name}[/] "
                    f"({ev.device.api_name}). Models loaded. Play audio now.\n"
                )
            elif isinstance(ev, EvtAudioLevel):
                self._status.rms = ev.rms
                self._status.peak = ev.peak
            elif isinstance(ev, EvtSegmentDetected):
                pass
            elif isinstance(ev, EvtPreviewTranscript):
                self._on_preview(ev)
            elif isinstance(ev, EvtTranscript):
                self._on_transcript(ev)
            elif isinstance(ev, EvtError):
                self._transcript.write(f"[bold red]Error:[/] {ev.message}")

    def _on_preview(self, ev: EvtPreviewTranscript) -> None:
        if not ev.text:
            return
        self._metrics.n_previews += 1
        self._caption.show_preview(ev.segment_id, ev.language, ev.text, ev.translation)
        # Update language hint in status from preview too so the user sees
        # the detected language as soon as we have a first guess.
        self._status.last_lang = ev.language
        self._status.last_lang_prob = ev.language_prob

    def _on_transcript(self, ev: EvtTranscript) -> None:
        tr = ev.transcript
        # Always clear the live caption for this segment when its final arrives.
        self._caption.clear_if_matches(ev.segment_id)
        if not tr.text:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        lang_tag = tr.language.upper() if tr.language else "??"
        line = f"[dim]{ts}[/] [bold yellow]\\[{lang_tag}][/] {tr.text}"
        if ev.translation:
            line += f"\n     [bold magenta]→ ES[/] {ev.translation}"
        self._transcript.write(line)

        self._status.last_lang = tr.language
        self._status.last_lang_prob = tr.language_prob

        self._metrics.n_segments += 1
        self._metrics.total_asr_time += tr.asr_seconds
        self._metrics.total_audio_time += tr.duration_s
        self._metrics.last_rtf = tr.rtf
        self._metrics.last_latency = ev.end_to_end_latency_s

    def action_clear(self) -> None:
        self._transcript.clear()
        self._caption.clear()

    async def action_quit(self) -> None:
        await self._orch.stop()
        self.exit()
