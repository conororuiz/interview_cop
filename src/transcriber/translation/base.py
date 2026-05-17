"""Translation backend protocol.

Implementations:
  * `nllb_engine.NLLBTranslator` — offline, Meta's NLLB-200.
  * `deepl_engine.DeepLTranslator` — online, paid API.

Both expose the same `translate(text, src_iso639_1) -> str` signature so the
orchestrator can swap them via configuration.
"""

from __future__ import annotations

from typing import Protocol


class Translator(Protocol):
    target_lang_iso639_1: str

    def translate(self, text: str, src_lang_iso639_1: str) -> str:
        """Translate `text` from `src_lang_iso639_1` to `self.target_lang_iso639_1`.

        Returns the translated string, or the original text if translation is
        unsupported / unnecessary.
        """
        ...
