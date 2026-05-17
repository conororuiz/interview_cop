"""NLLB-200 translator (offline).

We use Meta's `facebook/nllb-200-distilled-1.3B` by default — best quality /
size trade-off for our use-case among the distilled variants. The model is
downloaded on first use into `models/nllb/`.

Notes on the API:
  * NLLB uses BCP-47-ish language codes with a script tag, e.g. `eng_Latn`,
    `spa_Latn`. Whisper emits ISO 639-1 codes (`en`, `es`). We map between
    them with `_WHISPER_TO_NLLB`.
  * The source language is set on the tokenizer (`tokenizer.src_lang = ...`)
    *before* tokenising input. The target language is selected at
    generation time via `forced_bos_token_id`.
  * Long source texts are split on sentence boundaries to stay safely
    below NLLB's 512-token input limit. Live captions rarely exceed that,
    but a long lecture segment can.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from ..config import get_settings
from ..hardware.accel import AccelProfile, detect_accel

log = logging.getLogger(__name__)


# Whisper (ISO 639-1) -> NLLB BCP-47. Covers Whisper's full 100-language set
# with reasonable defaults; unknown codes fall back to `eng_Latn` and we log it.
_WHISPER_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "pl": "pol_Latn",
    "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "cs": "ces_Latn", "sk": "slk_Latn",
    "hu": "hun_Latn", "ro": "ron_Latn", "bg": "bul_Cyrl", "el": "ell_Grek",
    "tr": "tur_Latn", "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn",
    "nn": "nno_Latn", "fi": "fin_Latn", "et": "est_Latn", "lv": "lvs_Latn",
    "lt": "lit_Latn", "is": "isl_Latn", "ca": "cat_Latn", "gl": "glg_Latn",
    "eu": "eus_Latn", "cy": "cym_Latn", "ga": "gle_Latn",
    "ja": "jpn_Jpan", "zh": "zho_Hans", "ko": "kor_Hang", "th": "tha_Thai",
    "vi": "vie_Latn", "id": "ind_Latn", "ms": "zsm_Latn", "fil": "fil_Latn",
    "tl": "tgl_Latn",
    "ar": "arb_Arab", "fa": "pes_Arab", "ur": "urd_Arab", "he": "heb_Hebr",
    "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "mr": "mar_Deva", "gu": "guj_Gujr", "ml": "mal_Mlym", "kn": "kan_Knda",
    "si": "sin_Sinh", "pa": "pan_Guru", "ne": "npi_Deva",
    "sw": "swh_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "am": "amh_Ethi",
    "so": "som_Latn", "af": "afr_Latn", "zu": "zul_Latn",
    "az": "azj_Latn", "kk": "kaz_Cyrl", "ky": "kir_Cyrl", "uz": "uzn_Latn",
    "mn": "khk_Cyrl", "my": "mya_Mymr", "km": "khm_Khmr", "lo": "lao_Laoo",
    "ka": "kat_Geor", "hy": "hye_Armn", "sq": "als_Latn", "mk": "mkd_Cyrl",
    "bs": "bos_Latn", "hr": "hrv_Latn", "sr": "srp_Cyrl", "sl": "slv_Latn",
    "mt": "mlt_Latn", "la": "lat_Latn",
}

_MODEL_CHOICES: dict[str, str] = {
    "nllb-1.3b": "facebook/nllb-200-distilled-1.3B",
    "nllb-600m": "facebook/nllb-200-distilled-600M",
}


def _split_for_nllb(text: str, max_chars: int = 800) -> list[str]:
    """Split a long string at sentence boundaries to stay under NLLB's 512-token limit.
    800 chars ≈ 200-300 tokens for European languages, comfortably safe."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    # Split on sentence-ish boundaries, keeping the delimiter.
    parts = re.split(r"(?<=[.!?¡¿…])\s+", text)
    out: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                out.append(buf)
            if len(p) > max_chars:
                # Hard split very long sentences.
                for i in range(0, len(p), max_chars):
                    out.append(p[i : i + max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        out.append(buf)
    return out


class NLLBTranslator:
    target_lang_iso639_1 = "es"

    def __init__(
        self,
        model_alias: str | None = None,
        target_lang: str | None = None,
        accel: AccelProfile | None = None,
    ):
        import torch  # noqa: WPS433
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        s = get_settings()
        alias = model_alias or s.translation_backend
        if alias not in _MODEL_CHOICES:
            raise ValueError(f"Unknown NLLB alias: {alias!r}. Expected one of {list(_MODEL_CHOICES)}.")
        model_id = _MODEL_CHOICES[alias]

        self.target_lang_iso639_1 = target_lang or s.target_language
        self._target_nllb = _WHISPER_TO_NLLB.get(self.target_lang_iso639_1, "spa_Latn")
        self._accel = accel or detect_accel()
        self._torch = torch

        cache_dir = s.models_dir / "nllb"
        cache_dir.mkdir(parents=True, exist_ok=True)

        log.info("Loading NLLB '%s' on %s (%s)...",
                  model_id, self._accel.torch_device, self._accel.torch_dtype)
        t0 = time.monotonic()

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self._accel.torch_dtype, torch.float32)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=str(cache_dir),
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            torch_dtype=dtype if self._accel.torch_device != "cpu" else torch.float32,
        ).to(self._accel.torch_device)
        self._model.eval()
        # Bump the default max_length (NLLB defaults to 200 tokens) so we don't
        # truncate medium-length sentences. We chunk inputs to 480 src tokens
        # upstream so 512 output tokens is comfortable headroom.
        if getattr(self._model, "generation_config", None) is not None:
            self._model.generation_config.max_length = 512
        log.info("NLLB loaded in %.2fs.", time.monotonic() - t0)

        try:
            self._target_bos_id = self._tokenizer.convert_tokens_to_ids(self._target_nllb)
        except Exception as e:
            raise RuntimeError(f"NLLB tokenizer cannot resolve target lang {self._target_nllb}: {e}") from e

    # --- public API ---

    def translate(self, text: str, src_lang_iso639_1: str) -> str:
        """Translate `text` from src_lang to the configured target language."""
        text = text.strip()
        if not text:
            return ""

        # Same language → no work.
        if src_lang_iso639_1.lower().startswith(self.target_lang_iso639_1.lower()):
            return text

        src_nllb = _WHISPER_TO_NLLB.get(src_lang_iso639_1.lower())
        if src_nllb is None:
            log.warning("Unknown source language %r; falling back to eng_Latn.", src_lang_iso639_1)
            src_nllb = "eng_Latn"

        self._tokenizer.src_lang = src_nllb

        pieces = _split_for_nllb(text)
        translated: list[str] = []
        with self._torch.no_grad():
            for piece in pieces:
                inputs = self._tokenizer(
                    piece, return_tensors="pt", truncation=True, max_length=480,
                )
                inputs = {k: v.to(self._accel.torch_device) for k, v in inputs.items()}
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._target_bos_id,
                    max_length=512,            # use max_length only to silence transformers warning
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                )
                out = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                translated.append(out.strip())
        return " ".join(translated)
