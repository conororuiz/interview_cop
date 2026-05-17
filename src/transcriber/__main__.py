"""Main entry point: launches the live TUI transcriber."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .logging_setup import setup_logging
from .pipeline.orchestrator import Orchestrator
from .ui.tui import TranscriberApp

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Real-time system-audio transcriber + translator.",
    )
    parser.add_argument(
        "--no-translate", action="store_true",
        help="Disable translation even for non-Spanish audio (Hito 5+).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging to file (TUI stays clean).",
    )
    args = parser.parse_args()

    if args.debug:
        import os
        os.environ.setdefault("TRANSCRIBER_LOG_LEVEL", "DEBUG")
    # quiet_console=True because Textual owns the terminal.
    setup_logging(quiet_console=True)

    # Translation hook will be supplied by Hito 5; for now, none.
    translator = None
    if not args.no_translate:
        try:
            from .translation.nllb_engine import NLLBTranslator  # noqa: WPS433
            nllb = NLLBTranslator()
            translator = nllb.translate
        except Exception as e:
            log.warning("Translation backend not available yet (%s) — running ASR only.", e)
            translator = None

    orch = Orchestrator(translator=translator)
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


if __name__ == "__main__":
    sys.exit(main())
