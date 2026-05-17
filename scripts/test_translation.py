"""Hito 5 validator: smoke-test the NLLB translator on canned text.

Usage:
    python scripts/test_translation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcriber.logging_setup import setup_logging  # noqa: E402
from transcriber.translation.nllb_engine import NLLBTranslator  # noqa: E402

CASES = [
    ("en", "Hello, today we will talk about climate change and its impact on biodiversity."),
    ("en", "The quick brown fox jumps over the lazy dog. This is a test of the emergency broadcasting system."),
    ("fr", "Bonjour, comment allez-vous aujourd'hui? Le temps est magnifique à Paris."),
    ("de", "Künstliche Intelligenz verändert die Art, wie wir arbeiten und kommunizieren."),
    ("pt", "Olá, tudo bem? Hoje vamos falar sobre tecnologia e inovação."),
    ("ja", "今日は天気がとても良いので、公園に行きました。"),
    ("es", "Esto ya está en español, así que no debería traducirse."),
]


def main() -> int:
    setup_logging()
    t = NLLBTranslator()
    print()
    for src, text in CASES:
        t0 = time.monotonic()
        out = t.translate(text, src)
        dt = time.monotonic() - t0
        print(f"[{src} -> es]  ({dt*1000:6.0f} ms)")
        print(f"  IN : {text}")
        print(f"  OUT: {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
