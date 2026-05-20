"""Google Gemini-backed AI responder.

Given a window of recent transcription, produce a thoughtful response in
the same language as the most recent transcript. Streams the response so
the UI can show text as it's generated.

Design choices:
  * `google-generativeai` SDK (official). It supports synchronous streaming
    via `model.generate_content(prompt, stream=True)` returning an iterator
    of chunks, which is exactly what we need from a worker thread.
  * API key is taken from the bare `GEMINI_API_KEY` env var (not prefixed
    with `TRANSCRIBER_`) because that's the convention everyone already
    knows. If absent, the responder reports `is_available == False` and
    the UI disables the button.
  * Heuristic question detection happens here (`detect_question`) but the
    final decision is made by the model — we always send the recent context
    along with the explicit question if found, so the model can use it for
    grounding.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from ..config import get_settings

log = logging.getLogger(__name__)


# Question-mark / inverted-question-mark detector.
_Q_MARK_RE = re.compile(r"[?¿]")

# Languages where we know enough question-starters to be useful.
_QUESTION_STARTERS: dict[str, tuple[str, ...]] = {
    "en": ("what", "who", "when", "where", "why", "how", "which", "whose",
           "do", "does", "did", "is", "are", "was", "were", "can", "could",
           "will", "would", "should", "may", "might", "have", "has", "had"),
    "es": ("qué", "que", "quién", "quien", "cuándo", "cuando", "dónde", "donde",
           "por qué", "porqué", "por que", "porque", "cómo", "como", "cuál",
           "cual", "cuáles", "cuanto", "cuánto", "puede", "podría", "es",
           "son", "está", "están", "tienes", "tiene", "tienen", "hay"),
    "fr": ("quoi", "qui", "quand", "où", "pourquoi", "comment", "quel",
           "quelle", "quels", "quelles", "est-ce", "peux", "peut", "peut-on"),
    "de": ("was", "wer", "wann", "wo", "warum", "wieso", "wie", "welche",
           "welcher", "welches", "kann", "können", "ist", "sind"),
    "pt": ("o que", "quem", "quando", "onde", "por que", "porquê", "como",
           "qual", "quais", "pode", "podem", "é", "são"),
    "it": ("cosa", "chi", "quando", "dove", "perché", "come", "quale",
           "quali", "puoi", "può", "è", "sono"),
}


def detect_question(text: str, lang: str) -> Optional[str]:
    """If `text` looks like a question, return the question sentence; else None.

    Heuristics, in order:
      1. Sentences containing `?` or `¿` → return that sentence.
      2. Sentences starting with a known question word for `lang` → return.
      3. None.
    """
    if not text:
        return None
    # Split on sentence-ish boundaries (keep delimiter).
    parts = re.split(r"(?<=[.!?¡¿…])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    # Prefer the LAST question (most recent intent).
    for p in reversed(parts):
        if _Q_MARK_RE.search(p):
            return p
    starters = _QUESTION_STARTERS.get(lang.lower(), ())
    if starters:
        low = lambda s: s.strip().lstrip("¿¡").lower()
        for p in reversed(parts):
            head = low(p)
            for sw in starters:
                if head.startswith(sw + " ") or head == sw:
                    return p
    return None


@dataclass
class AiPromptContext:
    """Bundle of inputs sent to Gemini."""
    transcript_recent: str       # joined recent transcripts, oldest-first
    detected_question: Optional[str]
    language: str                # ISO 639-1 of the dominant recent language


def _language_name(iso: str) -> str:
    names = {
        "en": "English", "es": "Spanish (Castilian)", "fr": "French",
        "de": "German", "pt": "Portuguese", "it": "Italian",
        "ja": "Japanese", "ko": "Korean", "zh": "Chinese (Simplified)",
        "ar": "Arabic", "ru": "Russian", "hi": "Hindi", "nl": "Dutch",
        "pl": "Polish", "tr": "Turkish", "ca": "Catalan",
    }
    return names.get(iso.lower(), f"language code '{iso}'")


def _build_prompt(ctx: AiPromptContext) -> str:
    """Interview-mode prompt.

    The user is the CANDIDATE being interviewed. The transcript captures
    what the INTERVIEWER is saying. We ghost-write the candidate's reply
    in real time so they can read it (or paraphrase it) back.

    Two cases:
      * Explicit question detected → answer it directly as the candidate.
      * No explicit question (the interviewer is making a comment, framing
        a scenario, or sharing context) → produce a natural interviewee
        reply: a thoughtful acknowledgement, a clarifying question, or a
        short anecdote that demonstrates the relevant skill.
    """
    lang_name = _language_name(ctx.language)

    if ctx.detected_question:
        situation_block = (
            "The interviewer has just asked the candidate this question "
            "(this is what they need to answer):\n"
            f"\"\"\"\n{ctx.detected_question}\n\"\"\"\n"
        )
        action = (
            "Write what the candidate should say in response. Answer the "
            "question DIRECTLY and CONFIDENTLY in 2–4 sentences. If the "
            "question is behavioural (\"tell me about a time when…\", "
            "\"describe a situation…\", \"give me an example of…\") use a "
            "brief STAR structure (Situation → Task → Action → Result) "
            "but keep it tight."
        )
    else:
        situation_block = (
            "There is no explicit question in the transcript yet. The "
            "interviewer is making a comment, setting context, or framing "
            "a scenario. Read the transcript and infer what the candidate "
            "should naturally say back.\n"
        )
        action = (
            "Write a SHORT, natural interviewee reply (1–3 sentences). It "
            "can be a thoughtful acknowledgement that adds substance, a "
            "clarifying question to keep the dialogue flowing, or a brief "
            "relevant anecdote — whatever a strong candidate would say in "
            "that moment. Do NOT invent a full answer to a question that "
            "wasn't asked."
        )

    return (
        "You are a real-time INTERVIEW COPILOT. The person using this app "
        "is the CANDIDATE in a live interview (job, technical, podcast, "
        "or similar). The transcript below is what the INTERVIEWER (or "
        "panel) is saying. You DO NOT play the interviewer — you ghost-"
        "write what the candidate should say next, in the first person.\n\n"
        "Recent transcript (oldest first, most recent at the bottom):\n"
        f"\"\"\"\n{ctx.transcript_recent}\n\"\"\"\n\n"
        f"{situation_block}\n"
        f"Your task: {action}\n\n"
        "Hard rules — follow ALL of them:\n"
        f"- Write ENTIRELY in {lang_name}. Never mix languages.\n"
        "- Speak in the FIRST PERSON as the candidate (\"In my experience\", "
        "\"I usually…\", \"En mi caso…\", etc.). Never refer to the "
        "candidate in the third person.\n"
        "- Be substantive and confident. No filler openings like \"Great "
        "question\", \"Absolutely\", \"That's a great point\".\n"
        "- Do NOT restate or paraphrase the question before answering.\n"
        "- Do NOT mention that you are an AI, do NOT apologise, do NOT "
        "say \"I would say…\" — just say it.\n"
        "- Do NOT invent specific employer names, dates, dollar amounts, "
        "or project names. When a concrete detail would help, use a "
        "placeholder in square brackets like [my current company], "
        "[a recent project], [team size] so the candidate can fill it in.\n"
        "- Output ONLY the answer text. No headings, no labels, no quotes "
        "around the response.\n"
    )


class GeminiResponder:
    def __init__(self, model_name: Optional[str] = None,
                 api_key: Optional[str] = None):
        s = get_settings()
        self.model_name = model_name or s.gemini_model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or ""
        self._configured = False
        self._model = None
        if not self._api_key:
            log.warning(
                "GEMINI_API_KEY not found in environment. The 'Responder' "
                "button will stay disabled. Add it to .env "
                "(GEMINI_API_KEY=...) or set it as a system env var."
            )
            return
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self.model_name)
            self._configured = True
            log.info("Gemini responder ready (model=%s, key=...%s).",
                      self.model_name, self._api_key[-4:])
        except ImportError as e:
            log.warning(
                "google-generativeai is not installed (%s). Run: "
                "pip install google-generativeai", e,
            )
        except Exception as e:
            log.warning("Failed to initialise Gemini: %s", e)

    @property
    def is_available(self) -> bool:
        return self._configured and self._model is not None

    def respond_streaming(self, ctx: AiPromptContext) -> Iterator[str]:
        """Yield response text chunks as they arrive from the model."""
        if not self.is_available:
            raise RuntimeError("Gemini is not configured. Set GEMINI_API_KEY in .env.")
        prompt = _build_prompt(ctx)
        log.debug("Gemini prompt (%d chars).", len(prompt))
        try:
            stream = self._model.generate_content(  # type: ignore[union-attr]
                prompt,
                stream=True,
                generation_config={
                    # Lower temperature → tighter, more focused interview-style
                    # answers. We don't want creative drift in a job context.
                    "temperature": 0.3,
                    "top_p": 0.9,
                    # Enough headroom for a full STAR answer in any language
                    # (longer in Romance languages than English).
                    "max_output_tokens": 900,
                },
            )
            for chunk in stream:
                # chunk.text may be empty for safety / tool chunks; skip those.
                txt = getattr(chunk, "text", None) or ""
                if txt:
                    yield txt
        except Exception as e:
            log.exception("Gemini call failed: %s", e)
            raise

    def respond(self, ctx: AiPromptContext) -> tuple[str, float]:
        """Non-streaming convenience wrapper. Returns (text, seconds)."""
        t0 = time.monotonic()
        parts: list[str] = []
        for chunk in self.respond_streaming(ctx):
            parts.append(chunk)
        return "".join(parts), time.monotonic() - t0
