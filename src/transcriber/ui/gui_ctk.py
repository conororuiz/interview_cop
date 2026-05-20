"""Lightweight floating GUI built on CustomTkinter.

Why CustomTkinter and not PySide6:
  * No Qt → ~200 MB lighter install.
  * Built on tkinter (already bundled with Python on Windows/macOS) so the
    only added dep is CustomTkinter itself (~5 MB).
  * Pure GDI rendering on Windows: avoids the compositor / GPU rendering
    issues that plague Qt's translucent frameless windows on some setups.
  * Starts in ~300 ms.

Design:
  * Native window frame (Windows handles drag + resize natively) — far more
    reliable than rolling our own.
  * `-alpha 0.94` for semi-transparency.
  * `-topmost` so it stays floating above other windows; togglable.
  * Two `CTkTextbox` panels stacked vertically with a splitter feel.
  * Loading overlay that swaps to the live view when EvtSystemReady arrives.
  * Per-stage progress (VAD / Whisper / NLLB / capture) wired into the
    same EvtLoading events the PySide6 build already produced.

Threading: the orchestrator runs in a daemon thread with its own asyncio
loop. UI updates from that thread are scheduled with `self.root.after(0, ...)`
which is the only safe way to touch tkinter widgets from another thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Callable

import customtkinter as ctk
import tkinter as tk

from ..hardware.monitor import HardwareMonitor
from ..pipeline.orchestrator import (
    EvtAiResponseDelta,
    EvtAiResponseDone,
    EvtAiResponseError,
    EvtAiResponseStart,
    EvtAudioLevel,
    EvtError,
    EvtLoading,
    EvtPreviewTranscript,
    EvtSystemReady,
    EvtTranscript,
    Orchestrator,
)

log = logging.getLogger(__name__)


# ---------- Palette ----------

C_BG = "#0f1220"          # window background
C_PANEL = "#181d2c"       # text panel background
C_PANEL_LIGHT = "#1f2538"
C_BORDER = "#2a3148"
C_TEXT = "#e8edf5"
C_TEXT_DIM = "#aab2c5"
C_TEXT_MUTED = "#6e7689"
C_ACCENT = "#5cdbff"
C_ACCENT_DARK = "#2a8db8"
C_OK = "#5ee8a8"
C_WARN = "#f7c862"
C_DANGER = "#ff5c7c"
C_PREVIEW = "#7a8198"     # italic preview line colour

FONT_BODY = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_HEAD = ("Segoe UI", 8, "bold")
FONT_TS = ("Consolas", 9)


# ---------- Loading screen ----------

class LoadingFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=2)

        title = ctk.CTkLabel(
            self, text="Preparando el transcriptor",
            font=("Segoe UI", 22, "bold"), text_color=C_TEXT,
        )
        title.grid(row=1, column=0, padx=40, pady=(0, 6))

        self._step = ctk.CTkLabel(
            self, text="Iniciando…",
            font=FONT_BOLD, text_color=C_ACCENT,
        )
        self._step.grid(row=2, column=0, padx=40, pady=(4, 12))

        self._bar = ctk.CTkProgressBar(
            self, width=380, height=10,
            progress_color=C_ACCENT, fg_color=C_BORDER,
        )
        self._bar.set(0.02)
        self._bar.grid(row=3, column=0, padx=40, pady=(0, 18))

        hint = ctk.CTkLabel(
            self,
            text=(
                "La primera ejecución puede tardar más mientras se descargan los modelos.\n"
                "Whisper large-v3 (~3 GB) y NLLB-200 1.3B (~5 GB) quedan en caché para la próxima vez."
            ),
            font=FONT_SMALL, text_color=C_TEXT_MUTED, justify="center",
        )
        hint.grid(row=4, column=0, padx=40, pady=(0, 12))

    def set_loading(self, stage: str, message: str, progress: float) -> None:
        self._step.configure(text=message)
        self._bar.set(max(0.02, min(1.0, progress)))


# ---------- Reusable widgets ----------

class StatusStrip(ctk.CTkFrame):
    """Top status row: device · GPU · model · detected language · VU meter."""

    def __init__(self, master):
        super().__init__(master, fg_color="transparent", height=44)
        self.grid_columnconfigure(7, weight=1)

        col = 0
        self._device_lbl = self._chip("DISPOSITIVO", "—", col); col += 2
        self._gpu_lbl = self._chip("GPU", "—", col); col += 2
        self._model_lbl = self._chip("MODELO", "—", col); col += 2
        self._lang_lbl = self._chip("IDIOMA", "—", col, lang=True); col += 2

        self._level = LevelMeter(self, width=160, height=12)
        self._level.grid(row=0, column=col, padx=12, sticky="e", rowspan=2)

    def _chip(self, key: str, value: str, col: int, lang: bool = False) -> ctk.CTkLabel:
        k = ctk.CTkLabel(self, text=key, font=FONT_HEAD, text_color=C_TEXT_MUTED)
        k.grid(row=0, column=col, padx=(12, 4), pady=(6, 0), sticky="w")
        v = ctk.CTkLabel(
            self, text=value,
            font=FONT_BOLD,
            text_color=(C_ACCENT if lang else C_TEXT),
            anchor="w",
        )
        v.grid(row=1, column=col, padx=(12, 4), pady=(0, 6), sticky="w", columnspan=2)
        return v

    def set_device(self, name: str, api: str) -> None:
        self._device_lbl.configure(text=f"{name} · {api}")

    def set_gpu(self, name: str) -> None:
        self._gpu_lbl.configure(text=name)

    def set_model(self, name: str, compute: str) -> None:
        self._model_lbl.configure(text=f"{name} · {compute}")

    def set_language(self, lang: str, prob: float) -> None:
        if not lang:
            self._lang_lbl.configure(text="detectando…")
        else:
            self._lang_lbl.configure(text=f"{lang.upper()} · {prob:.2f}")

    def set_level(self, rms: float, peak: float) -> None:
        self._level.set_level(rms, peak)


class LevelMeter(ctk.CTkCanvas):
    """Compact VU meter using tk Canvas — drawn each set_level call."""

    def __init__(self, master, **kw):
        super().__init__(master, highlightthickness=0, bg=C_BG, **kw)
        self._rms = 0.0
        self._peak = 0.0
        self.bind("<Configure>", lambda e: self._render())

    def set_level(self, rms: float, peak: float) -> None:
        self._rms = rms
        self._peak = peak
        self._render()

    def _render(self) -> None:
        self.delete("all")
        w = max(1, int(self.winfo_width()))
        h = max(1, int(self.winfo_height()))
        # Background
        self.create_rectangle(0, 0, w, h, fill=C_BORDER, outline="")
        # RMS
        frac = max(0.0, min(1.0, self._rms * 6))
        bar_w = int(w * frac)
        if bar_w > 0:
            color = C_ACCENT
            if frac > 0.85:
                color = C_DANGER
            elif frac > 0.65:
                color = C_WARN
            self.create_rectangle(0, 0, bar_w, h, fill=color, outline="")
        # Peak marker
        peak = max(0.0, min(1.0, self._peak))
        x = int(w * peak)
        if x > 0:
            self.create_rectangle(x - 1, 0, x + 1, h, fill="#ffffff", outline="")


class TranscriptPanel(ctk.CTkFrame):
    """A panel with a header + a CTkTextbox + tracking for the live preview block.

    The CTkTextbox is a tk.Text underneath, so we use tk.Text APIs directly
    for fine-grained control of the preview block (mark + tag).
    """

    def __init__(self, master, title: str, subtitle: str, accent: str, placeholder: str):
        super().__init__(master, fg_color=C_PANEL, corner_radius=10,
                          border_width=1, border_color=C_BORDER)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent", height=24)
        head.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        tag = ctk.CTkLabel(head, text=title.upper(),
                            font=("Segoe UI", 9, "bold"),
                            text_color=accent)
        tag.pack(side="left")
        sub = ctk.CTkLabel(head, text=" · " + subtitle,
                            font=("Segoe UI", 9), text_color=C_TEXT_MUTED)
        sub.pack(side="left")

        self._text = ctk.CTkTextbox(
            self, fg_color=C_PANEL_LIGHT, text_color=C_TEXT,
            border_width=0, corner_radius=8, wrap="word",
            font=FONT_BODY,
        )
        self._text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        # Underlying tk.Text — access for tag config / marks.
        t: tk.Text = self._text._textbox
        t.tag_configure("ts", foreground=C_TEXT_MUTED, font=FONT_TS)
        t.tag_configure("lang", foreground=accent, font=("Segoe UI", 9, "bold"))
        t.tag_configure("body", foreground=C_TEXT, font=FONT_BODY)
        t.tag_configure("preview_label", foreground=C_TEXT_MUTED,
                         font=("Segoe UI", 9, "italic"))
        t.tag_configure("preview_body", foreground=C_PREVIEW,
                         font=("Segoe UI", 11, "italic"))
        t.tag_configure("placeholder", foreground=C_TEXT_MUTED,
                         font=("Segoe UI", 11, "italic"))
        t.tag_configure("err", foreground=C_DANGER, font=FONT_BOLD)

        self._t = t
        self._accent = accent
        self._placeholder = placeholder
        self._current_preview_seg: int | None = None
        self._show_placeholder()
        self._text.configure(state="disabled")

    # --- API

    def append_final(self, ts: str, lang: str, text: str) -> None:
        self._with_edit(lambda: self._do_append_final(ts, lang, text))

    def append_error(self, ts: str, message: str) -> None:
        def go():
            self._clear_placeholder_if_present()
            self._remove_preview_block()
            self._t.insert("end", f"\n{ts}  ", ("ts",))
            self._t.insert("end", "ERR ", ("err",))
            self._t.insert("end", message + "\n", ("err",))
            self._t.see("end")
        self._with_edit(go)

    def update_preview(self, seg_id: int, lang: str, text: str) -> None:
        self._with_edit(lambda: self._do_update_preview(seg_id, lang, text))

    def clear_preview(self, seg_id: int) -> None:
        def go():
            if self._current_preview_seg == seg_id:
                self._remove_preview_block()
        self._with_edit(go)

    def clear_all(self) -> None:
        def go():
            self._t.delete("1.0", "end")
            self._current_preview_seg = None
            self._show_placeholder()
        self._with_edit(go)

    # --- helpers

    def _with_edit(self, fn: Callable[[], None]) -> None:
        self._text.configure(state="normal")
        try:
            fn()
        finally:
            self._text.configure(state="disabled")

    def _show_placeholder(self) -> None:
        self._t.delete("1.0", "end")
        self._t.insert("1.0", self._placeholder, ("placeholder",))

    def _clear_placeholder_if_present(self) -> None:
        cur = self._t.get("1.0", "end").strip()
        if cur == self._placeholder:
            self._t.delete("1.0", "end")

    def _do_append_final(self, ts: str, lang: str, text: str) -> None:
        self._clear_placeholder_if_present()
        self._remove_preview_block()
        if self._t.index("end-1c") != "1.0":
            self._t.insert("end", "\n")
        self._t.insert("end", f"{ts}  ", ("ts",))
        self._t.insert("end", f"[{lang.upper()}] ", ("lang",))
        self._t.insert("end", text + "\n", ("body",))
        self._t.see("end")

    def _do_update_preview(self, seg_id: int, lang: str, text: str) -> None:
        self._clear_placeholder_if_present()
        # If preview is from a different segment, drop the old one.
        if self._current_preview_seg is not None and self._current_preview_seg != seg_id:
            self._remove_preview_block()

        if self._current_preview_seg == seg_id and self._t.tag_ranges("preview_range"):
            # Replace existing preview block.
            self._t.delete("preview_start", "preview_end")
            self._t.mark_set("preview_insert", "preview_start")
        else:
            # Create marks for the new preview block at the end.
            if self._t.index("end-1c") != "1.0":
                self._t.insert("end", "\n")
            self._t.mark_set("preview_start", "end-1c")
            self._t.mark_gravity("preview_start", "left")
            self._t.mark_set("preview_insert", "end-1c")
            self._current_preview_seg = seg_id

        self._t.insert("preview_insert", "⏵ EN VIVO · ", ("preview_label",))
        self._t.insert("preview_insert", lang.upper(), ("preview_label",))
        self._t.insert("preview_insert", "\n", ("preview_label",))
        self._t.insert("preview_insert", text, ("preview_body",))
        # Mark the end of the preview block.
        self._t.mark_set("preview_end", "preview_insert")
        self._t.mark_gravity("preview_end", "right")
        # Tag the whole range so we can find it next time.
        self._t.tag_remove("preview_range", "1.0", "end")
        self._t.tag_add("preview_range", "preview_start", "preview_end")
        self._t.see("end")

    def _remove_preview_block(self) -> None:
        if self._t.tag_ranges("preview_range"):
            self._t.delete("preview_start", "preview_end")
            self._t.tag_remove("preview_range", "1.0", "end")
        self._current_preview_seg = None


class AiResponsePanel(ctk.CTkFrame):
    """Panel that displays the streamed Gemini answer. Includes the trigger
    button and a 'detected question' line shown above the response body."""

    def __init__(self, master, on_respond: Callable[[], None]):
        super().__init__(master, fg_color=C_PANEL, corner_radius=10,
                          border_width=1, border_color=C_BORDER)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent", height=30)
        head.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        head.grid_columnconfigure(2, weight=1)

        tag = ctk.CTkLabel(head, text="TU RESPUESTA",
                            font=("Segoe UI", 9, "bold"),
                            text_color="#c084fc")
        tag.grid(row=0, column=0, sticky="w")
        sub = ctk.CTkLabel(head, text=" · Lo que dirías como entrevistado",
                            font=("Segoe UI", 9), text_color=C_TEXT_MUTED)
        sub.grid(row=0, column=1, sticky="w")

        self._respond_btn = ctk.CTkButton(
            head, text="✨  Responder", width=130, height=26,
            fg_color="#a855f7", hover_color="#c084fc", text_color="#fff",
            command=on_respond,
        )
        self._respond_btn.grid(row=0, column=3, sticky="e")

        self._text = ctk.CTkTextbox(
            self, fg_color=C_PANEL_LIGHT, text_color=C_TEXT,
            border_width=0, corner_radius=8, wrap="word",
            font=FONT_BODY,
        )
        self._text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        t: tk.Text = self._text._textbox
        t.tag_configure("placeholder", foreground=C_TEXT_MUTED,
                         font=("Segoe UI", 11, "italic"))
        t.tag_configure("ts", foreground=C_TEXT_MUTED, font=FONT_TS)
        t.tag_configure("q", foreground="#c084fc", font=("Segoe UI", 10, "italic"))
        t.tag_configure("body", foreground=C_TEXT, font=FONT_BODY)
        t.tag_configure("err", foreground=C_DANGER, font=FONT_BOLD)
        t.tag_configure("meta", foreground=C_TEXT_MUTED, font=FONT_SMALL)
        self._t = t

        self._show_placeholder()
        self._text.configure(state="disabled")
        self._current_request_id: int | None = None

    # --- API

    def set_button_enabled(self, enabled: bool, label: str | None = None) -> None:
        if enabled:
            self._respond_btn.configure(state="normal",
                                         text=label or "✨  Responder")
        else:
            self._respond_btn.configure(state="disabled",
                                         text=label or "Sin API key")

    def begin_response(self, request_id: int, question: str | None,
                        language: str) -> None:
        self._current_request_id = request_id
        self._with_edit(lambda: self._do_begin(question, language))
        self._respond_btn.configure(state="disabled", text="Pensando…")

    def append_delta(self, request_id: int, delta: str) -> None:
        if request_id != self._current_request_id:
            return
        self._with_edit(lambda: self._do_append_body(delta))

    def finish_response(self, request_id: int, seconds: float) -> None:
        if request_id != self._current_request_id:
            return
        self._with_edit(lambda: self._do_append_meta(f"  ({seconds:.1f}s)"))
        self._respond_btn.configure(state="normal", text="✨  Responder")

    def fail_response(self, request_id: int, message: str) -> None:
        self._with_edit(lambda: self._do_append_error(message))
        self._respond_btn.configure(state="normal", text="✨  Responder")

    def clear_all(self) -> None:
        self._with_edit(self._show_placeholder)
        self._current_request_id = None

    # --- internals

    def _with_edit(self, fn: Callable[[], None]) -> None:
        self._text.configure(state="normal")
        try:
            fn()
        finally:
            self._text.configure(state="disabled")

    def _show_placeholder(self) -> None:
        self._t.delete("1.0", "end")
        self._t.insert("1.0",
                        "Modo entrevista: pulsa “Responder” y Gemini te sugerirá, "
                        "en primera persona y en el idioma del audio, lo que "
                        "dirías como entrevistado.",
                        ("placeholder",))

    def _do_begin(self, question: str | None, language: str) -> None:
        if self._t.get("1.0", "end").strip().startswith("Pulsa"):
            self._t.delete("1.0", "end")
        ts = datetime.now().strftime("%H:%M:%S")
        if self._t.index("end-1c") != "1.0":
            self._t.insert("end", "\n")
        self._t.insert("end", f"\n{ts}  ", ("ts",))
        self._t.insert("end", f"[{language.upper()}] ", ("meta",))
        if question:
            self._t.insert("end", f"\n  ❓ {question}\n", ("q",))
        else:
            self._t.insert("end",
                            "\n  (sin pregunta explícita — respondo de forma "
                            "natural al contexto reciente)\n",
                            ("meta",))
        self._t.see("end")

    def _do_append_body(self, delta: str) -> None:
        self._t.insert("end", delta, ("body",))
        self._t.see("end")

    def _do_append_meta(self, meta: str) -> None:
        self._t.insert("end", meta, ("meta",))
        self._t.see("end")

    def _do_append_error(self, msg: str) -> None:
        self._t.insert("end", f"\n⚠ {msg}\n", ("err",))
        self._t.see("end")


class MetricsStrip(ctk.CTkFrame):
    """Bottom row: counters + CPU/GPU/VRAM + clear button."""

    def __init__(self, master, on_clear: Callable[[], None]):
        super().__init__(master, fg_color="transparent", height=32)
        self.grid_columnconfigure(99, weight=1)

        self._on_clear = on_clear
        self._n_segs = 0
        self._n_prev = 0
        self._total_asr = 0.0
        self._total_audio = 0.0
        self._last_rtf = 0.0
        self._last_latency = 0.0

        cols = []

        def chip(key: str) -> ctk.CTkLabel:
            col = len(cols) * 2
            cols.append(key)
            k = ctk.CTkLabel(self, text=key, font=FONT_HEAD, text_color=C_TEXT_MUTED)
            k.grid(row=0, column=col, padx=(12, 4), pady=4, sticky="w")
            v = ctk.CTkLabel(self, text="0", font=FONT_BOLD, text_color=C_TEXT)
            v.grid(row=0, column=col + 1, padx=(0, 6), pady=4, sticky="w")
            return v

        self._seg_v = chip("SEGS")
        self._prev_v = chip("PREVS")
        self._rtf_v = chip("RTF")
        self._lat_v = chip("E2E")
        self._cpu_v = chip("CPU")
        self._gpu_v = chip("GPU")
        self._vram_v = chip("VRAM")

        clear = ctk.CTkButton(
            self, text="🗑  Limpiar", width=90, height=24,
            fg_color="transparent", text_color=C_TEXT_DIM,
            hover_color=C_PANEL_LIGHT, border_width=1, border_color=C_BORDER,
            command=on_clear,
        )
        clear.grid(row=0, column=99, padx=8, pady=4, sticky="e")

    def on_transcript(self, tr_seconds: float, audio_seconds: float,
                       rtf: float, latency: float) -> None:
        self._n_segs += 1
        self._total_asr += tr_seconds
        self._total_audio += audio_seconds
        self._last_rtf = rtf
        self._last_latency = latency
        self._refresh()

    def on_preview(self) -> None:
        self._n_prev += 1
        self._prev_v.configure(text=str(self._n_prev))

    def set_hardware(self, cpu: float, gpu: float | None,
                      vram_used: float | None, vram_total: float | None) -> None:
        self._cpu_v.configure(text=f"{cpu:.0f}%")
        if gpu is not None:
            self._gpu_v.configure(text=f"{gpu:.0f}%")
        else:
            self._gpu_v.configure(text="—")
        if vram_used is not None and vram_total:
            self._vram_v.configure(text=f"{vram_used/1024:.1f}/{vram_total/1024:.1f} GB")
        else:
            self._vram_v.configure(text="—")

    def reset_counters(self) -> None:
        self._n_segs = 0
        self._n_prev = 0
        self._total_asr = 0.0
        self._total_audio = 0.0
        self._last_rtf = 0.0
        self._last_latency = 0.0
        self._refresh()

    def _refresh(self) -> None:
        avg = (self._total_asr / self._total_audio) if self._total_audio else 0.0
        self._seg_v.configure(text=str(self._n_segs))
        self._prev_v.configure(text=str(self._n_prev))
        self._rtf_v.configure(
            text=f"{avg:.2f}",
            text_color=(C_OK if avg < 0.4 else C_WARN if avg < 1.0 else C_DANGER),
        )
        self._lat_v.configure(
            text=f"{self._last_latency:.1f}s",
            text_color=(C_OK if self._last_latency < 6
                         else C_WARN if self._last_latency < 12
                         else C_DANGER),
        )


# ---------- Bridge thread (orchestrator events -> root.after) ----------

class _OrchThread(threading.Thread):
    def __init__(self, orchestrator: Orchestrator, schedule: Callable[[Callable[[], None]], None]):
        super().__init__(name="orch-asyncio", daemon=True)
        self._orch = orchestrator
        self._schedule = schedule
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_event: Callable[[object], None] | None = None

    def set_on_event(self, cb: Callable[[object], None]) -> None:
        self._on_event = cb

    def request_stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop_async)

    def _stop_async(self) -> None:
        try:
            asyncio.ensure_future(self._orch.stop(), loop=self._loop)
        except Exception:
            pass

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception:
            log.exception("Orchestrator thread crashed")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    async def _main(self) -> None:
        start_task = asyncio.create_task(self._safe_start())
        try:
            async for ev in self._orch.events():
                if self._on_event is not None:
                    # Schedule on tkinter main thread.
                    self._schedule(lambda e=ev: self._on_event(e))  # type: ignore[misc]
        finally:
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except Exception:
                    pass

    async def _safe_start(self) -> None:
        try:
            await self._orch.start()
        except Exception as e:
            log.exception("Orchestrator start failed: %s", e)
            if self._on_event is not None:
                self._schedule(lambda: self._on_event(EvtError(f"Inicialización falló: {e}", fatal=True)))  # type: ignore[misc]


# ---------- Main app ----------

class TranscriberApp:
    def __init__(self, orchestrator: Orchestrator):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Realtime Transcriber")
        self.root.geometry("860x720")
        self.root.minsize(620, 460)
        self.root.configure(fg_color=C_BG)
        # Semi-transparency (whole window — texts are bright enough to read clearly).
        try:
            self.root.attributes("-alpha", 0.94)
        except Exception:
            pass
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # --- Top bar with title + opacity + pin ----------------------------
        topbar = ctk.CTkFrame(self.root, fg_color="transparent")
        topbar.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        topbar.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(topbar, text="🎙  Realtime Transcriber",
                              font=FONT_TITLE, text_color=C_TEXT)
        title.grid(row=0, column=0, sticky="w")

        self._subtitle = ctk.CTkLabel(
            topbar, text="inicializando…",
            font=FONT_SMALL, text_color=C_TEXT_MUTED,
        )
        self._subtitle.grid(row=0, column=1, sticky="w", padx=10)

        opa_lbl = ctk.CTkLabel(topbar, text="Opacidad",
                                font=FONT_SMALL, text_color=C_TEXT_MUTED)
        opa_lbl.grid(row=0, column=2, padx=(0, 6))

        self._opacity = ctk.CTkSlider(
            topbar, from_=60, to=100, number_of_steps=40, width=110,
            progress_color=C_ACCENT, button_color=C_ACCENT, button_hover_color="#9be8ff",
            command=self._on_opacity_change,
        )
        self._opacity.set(94)
        self._opacity.grid(row=0, column=3, padx=(0, 12))

        self._pinned = True
        self._pin_btn = ctk.CTkButton(
            topbar, text="📌  Encima", width=92, height=26,
            fg_color=C_ACCENT_DARK, hover_color=C_ACCENT, text_color=C_TEXT,
            command=self._toggle_pin,
        )
        self._pin_btn.grid(row=0, column=4)

        # --- Content area (loading / live) ---------------------------------
        self._content = ctk.CTkFrame(self.root, fg_color="transparent")
        self._content.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(0, weight=1)

        self._loading = LoadingFrame(self._content)
        self._loading.grid(row=0, column=0, sticky="nsew")

        self._live = ctk.CTkFrame(self._content, fg_color="transparent")
        self._live.grid_columnconfigure(0, weight=1)
        # Three panels share remaining vertical space evenly (weights 2/2/2).
        self._live.grid_rowconfigure(1, weight=2)
        self._live.grid_rowconfigure(2, weight=2)
        self._live.grid_rowconfigure(3, weight=2)
        # NOT gridded yet — we swap into place when EvtSystemReady arrives.

        self._status = StatusStrip(self._live)
        self._status.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self._src_panel = TranscriptPanel(
            self._live, title="Original", subtitle="Transcripción del audio",
            accent=C_ACCENT,
            placeholder="Esperando audio… reproduce algo para empezar.",
        )
        self._src_panel.grid(row=1, column=0, sticky="nsew", pady=(4, 3))

        self._tgt_panel = TranscriptPanel(
            self._live, title="Español", subtitle="Traducción automática",
            accent=C_OK,
            placeholder="La traducción aparecerá aquí cuando el idioma no sea español.",
        )
        self._tgt_panel.grid(row=2, column=0, sticky="nsew", pady=(3, 3))

        self._ai_panel = AiResponsePanel(
            self._live, on_respond=self._on_respond_clicked,
        )
        self._ai_panel.grid(row=3, column=0, sticky="nsew", pady=(3, 0))

        self._metrics = MetricsStrip(self._live, on_clear=self._on_clear)
        self._metrics.grid(row=4, column=0, sticky="ew", pady=(6, 0))

        # --- Wire orchestrator thread --------------------------------------
        self._orch = orchestrator
        self._thread = _OrchThread(orchestrator, schedule=self._schedule)
        self._thread.set_on_event(self._on_event)
        self._thread.start()

        # --- Hardware sampling --------------------------------------------
        self._hw = HardwareMonitor()
        self._hw_after_id: str | None = None
        self._poll_hardware()

        # --- Close handler -------------------------------------------------
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- helpers
    def _schedule(self, fn: Callable[[], None]) -> None:
        """Thread-safe scheduling onto tkinter's main loop."""
        try:
            self.root.after(0, fn)
        except RuntimeError:
            # Root being destroyed.
            pass

    def _on_opacity_change(self, value: float) -> None:
        try:
            self.root.attributes("-alpha", value / 100.0)
        except Exception:
            pass

    def _toggle_pin(self) -> None:
        self._pinned = not self._pinned
        try:
            self.root.attributes("-topmost", self._pinned)
        except Exception:
            pass
        if self._pinned:
            self._pin_btn.configure(text="📌  Encima", fg_color=C_ACCENT_DARK)
        else:
            self._pin_btn.configure(text="📌  Normal", fg_color=C_BORDER)

    def _on_clear(self) -> None:
        self._src_panel.clear_all()
        self._tgt_panel.clear_all()
        self._ai_panel.clear_all()
        self._metrics.reset_counters()
        # Critical: also wipe the orchestrator's transcript history so the
        # next AI request only sees what arrives FROM NOW ON. Without this
        # Gemini would still be answering based on segments the user can no
        # longer see.
        try:
            self._orch.clear_history()
        except Exception as e:
            log.debug("clear_history failed: %s", e)

    def _on_respond_clicked(self) -> None:
        # Re-check availability at click time — the responder is built lazily
        # in the bridge thread, so it might not yet exist at first launch.
        if not self._orch.ai_available:
            self._ai_panel.fail_response(
                -1, "Gemini no está disponible. Define GEMINI_API_KEY en .env y reinicia.",
            )
            self._ai_panel.set_button_enabled(False, "Sin API key")
            return
        self._orch.request_ai_response()

    def _poll_hardware(self) -> None:
        try:
            s = self._hw.sample()
            self._metrics.set_hardware(s.cpu_percent, s.gpu_percent,
                                         s.vram_used_mb, s.vram_total_mb)
        except Exception as e:
            log.debug("hardware sample failed: %s", e)
        self._hw_after_id = self.root.after(1000, self._poll_hardware)

    def _swap_to_live(self) -> None:
        if self._live.winfo_manager() != "grid":
            self._loading.grid_remove()
            self._live.grid(row=0, column=0, sticky="nsew")

    # --- event handler from orchestrator (runs on tkinter main thread)
    def _on_event(self, ev) -> None:
        if isinstance(ev, EvtLoading):
            self._loading.set_loading(ev.stage, ev.message, ev.progress)
            self._subtitle.configure(text=ev.message)
            return
        if isinstance(ev, EvtSystemReady):
            self._status.set_device(ev.device.name, ev.device.api_name)
            self._status.set_gpu(ev.accel.gpu_name or "CPU")
            self._status.set_model(ev.whisper_model,
                                    f"{ev.accel.ct2_device}/{ev.accel.ct2_compute_type}")
            self._subtitle.configure(text=f"escuchando {ev.device.name}")
            self._swap_to_live()
            # Reflect AI availability in the button state.
            if self._orch.ai_available:
                self._ai_panel.set_button_enabled(True, "✨  Responder")
            else:
                self._ai_panel.set_button_enabled(
                    False, "Sin API key (Gemini)",
                )
            return
        if isinstance(ev, EvtAudioLevel):
            try:
                self._status.set_level(ev.rms, ev.peak)
            except Exception:
                pass
            return
        if isinstance(ev, EvtPreviewTranscript):
            if not ev.text:
                return
            self._metrics.on_preview()
            self._status.set_language(ev.language, ev.language_prob)
            self._src_panel.update_preview(ev.segment_id, ev.language, ev.text)
            if ev.translation:
                self._tgt_panel.update_preview(ev.segment_id, "es", ev.translation)
            return
        if isinstance(ev, EvtTranscript):
            tr = ev.transcript
            self._src_panel.clear_preview(ev.segment_id)
            self._tgt_panel.clear_preview(ev.segment_id)
            if not tr.text:
                return
            ts = datetime.now().strftime("%H:%M:%S")
            self._src_panel.append_final(ts, tr.language, tr.text)
            if ev.translation:
                self._tgt_panel.append_final(ts, "es", ev.translation)
            self._status.set_language(tr.language, tr.language_prob)
            self._metrics.on_transcript(
                tr_seconds=tr.asr_seconds, audio_seconds=tr.duration_s,
                rtf=tr.rtf, latency=ev.end_to_end_latency_s,
            )
            return
        if isinstance(ev, EvtAiResponseStart):
            self._ai_panel.begin_response(ev.request_id, ev.detected_question,
                                            ev.language)
            return
        if isinstance(ev, EvtAiResponseDelta):
            self._ai_panel.append_delta(ev.request_id, ev.delta)
            return
        if isinstance(ev, EvtAiResponseDone):
            self._ai_panel.finish_response(ev.request_id, ev.seconds)
            return
        if isinstance(ev, EvtAiResponseError):
            self._ai_panel.fail_response(ev.request_id, ev.message)
            return
        if isinstance(ev, EvtError):
            ts = datetime.now().strftime("%H:%M:%S")
            self._src_panel.append_error(ts, ev.message)
            return

    def _on_close(self) -> None:
        try:
            if self._hw_after_id is not None:
                self.root.after_cancel(self._hw_after_id)
        except Exception:
            pass
        try:
            self._hw.shutdown()
        except Exception:
            pass
        self._thread.request_stop()
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> int:
        self.root.mainloop()
        return 0


def run_gui_ctk(orchestrator: Orchestrator) -> int:
    app = TranscriberApp(orchestrator)
    return app.run()
