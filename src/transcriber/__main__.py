"""Main entry points.

Two console scripts are registered (see pyproject.toml):
  * `transcriber`     — Textual TUI in terminal
  * `transcriber-gui` — CustomTkinter floating GUI (lightweight)

`transcriber --gui` is also accepted as a shorthand. A `--qt-gui` flag
selects the PySide6 build instead, which is kept around as an alternative
but defaults to off because Qt's translucent frameless windows have known
rendering issues on some Windows / GPU-driver combinations.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Callable  # noqa: F401  (used in type hint comment)

from .logging_setup import setup_logging
from .pipeline.orchestrator import Orchestrator

log = logging.getLogger(__name__)


def _apply_cpu_mode(model_override: str | None = None,
                     translator_override: str | None = None) -> None:
    """Set env vars BEFORE settings are first read so the whole stack
    reconfigures for CPU-only execution.

    We also down-shift the default model sizes — large-v3 + NLLB-1.3B are
    unusable on CPU. Users can still override via --whisper-model /
    --translator or via the existing TRANSCRIBER_* env vars.
    """
    os.environ["TRANSCRIBER_COMPUTE_DEVICE"] = "cpu"
    os.environ.setdefault("TRANSCRIBER_WHISPER_MODEL", model_override or "medium")
    os.environ.setdefault("TRANSCRIBER_TRANSLATION_BACKEND",
                            translator_override or "nllb-600m")
    # Smaller models also benefit from a wider VAD window — speech detection
    # is the same but ASR call cost is higher, so larger segments amortise
    # the per-call overhead.
    os.environ.setdefault("TRANSCRIBER_MAX_SEGMENT_MS", "12000")
    os.environ.setdefault("TRANSCRIBER_PREVIEW_INTERVAL_MS", "3000")
    os.environ.setdefault("TRANSCRIBER_PREVIEW_MIN_AUDIO_MS", "3000")


def _build_orchestrator(no_translate: bool) -> Orchestrator:
    """Build the orchestrator with a translator FACTORY (not an instance).

    Heavy model loads (NLLB ~5 GB, Whisper ~3 GB) must NOT happen on the main
    thread or the GUI freezes before the window even appears. The bridge
    thread invokes the factory inside `Orchestrator.start()`.
    """
    if no_translate:
        return Orchestrator()

    def factory() -> "Callable[[str, str], str]":
        from .translation.nllb_engine import NLLBTranslator
        return NLLBTranslator().translate

    return Orchestrator(translator_factory=factory)


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only execution (simulates a machine "
                             "without a GPU). Auto-downsizes models: Whisper "
                             "medium + NLLB-600M.")
    parser.add_argument("--whisper-model", type=str, default=None,
                        help="Override Whisper model size "
                             "(tiny / base / small / medium / large-v2 / large-v3).")
    parser.add_argument("--translator", type=str, default=None,
                        help="Override translation backend "
                             "(nllb-1.3b / nllb-600m / deepl).")
    parser.add_argument("--no-translate", action="store_true",
                        help="Disable translation; transcribe only.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging.")


def _apply_common_flags(args: argparse.Namespace) -> None:
    """Apply env-var-style overrides from CLI flags. Must run BEFORE the
    settings cache is populated (i.e. before setup_logging / Orchestrator)."""
    if args.debug:
        os.environ.setdefault("TRANSCRIBER_LOG_LEVEL", "DEBUG")
    if args.cpu:
        _apply_cpu_mode(model_override=args.whisper_model,
                          translator_override=args.translator)
        log.warning(
            "CPU mode enabled — using Whisper=%s and translator=%s "
            "(set via --whisper-model / --translator to override).",
            os.environ["TRANSCRIBER_WHISPER_MODEL"],
            os.environ["TRANSCRIBER_TRANSLATION_BACKEND"],
        )
    else:
        if args.whisper_model:
            os.environ["TRANSCRIBER_WHISPER_MODEL"] = args.whisper_model
        if args.translator:
            os.environ["TRANSCRIBER_TRANSLATION_BACKEND"] = args.translator


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Real-time system-audio transcriber + translator.",
    )
    parser.add_argument("--gui", action="store_true",
                        help="Launch the lightweight CustomTkinter GUI.")
    parser.add_argument("--qt-gui", action="store_true",
                        help="Launch the heavier PySide6 GUI (experimental).")
    _add_common_flags(parser)
    args = parser.parse_args()
    _apply_common_flags(args)

    if args.qt_gui:
        return _run_qt_gui(args.no_translate)
    if args.gui:
        return _run_ctk_gui(args.no_translate)
    return _run_tui(args.no_translate)


def main_gui() -> int:
    """Console-script entry point for `transcriber-gui` — uses CustomTkinter."""
    parser = argparse.ArgumentParser(prog="transcriber-gui")
    parser.add_argument("--qt", action="store_true",
                        help="Use the PySide6 build instead of CustomTkinter.")
    _add_common_flags(parser)
    args = parser.parse_args()
    _apply_common_flags(args)

    if args.qt:
        return _run_qt_gui(args.no_translate)
    return _run_ctk_gui(args.no_translate)


def _run_tui(no_translate: bool) -> int:
    setup_logging(quiet_console=True)
    from .ui.tui import TranscriberApp
    orch = _build_orchestrator(no_translate)
    app = TranscriberApp(orch)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            asyncio.run(orch.stop())
        except Exception:
            pass
    return 0


def _run_ctk_gui(no_translate: bool) -> int:
    # CustomTkinter handles its own console output cleanly, so we can leave
    # console logging enabled to aid debugging.
    setup_logging(quiet_console=False)
    try:
        from .ui.gui_ctk import run_gui_ctk
    except ImportError as e:
        print(
            "CustomTkinter not installed. Install with:\n"
            "    pip install customtkinter\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        return 1
    orch = _build_orchestrator(no_translate)
    return run_gui_ctk(orch)


def _run_qt_gui(no_translate: bool) -> int:
    setup_logging(quiet_console=False)
    try:
        from .ui.gui import run_gui
    except ImportError as e:
        print(
            "PySide6 not installed (optional). Install with:\n"
            "    pip install PySide6\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        return 1
    orch = _build_orchestrator(no_translate)
    return run_gui(orch)


if __name__ == "__main__":
    sys.exit(main())
