"""Hito 3 validator: run VAD + segmenter + faster-whisper offline on a WAV.

Usage:
    python scripts/test_offline_asr.py capture_test.wav
    python scripts/test_offline_asr.py capture_test.wav --language es

It runs the full upstream pipeline (VAD + segmenter) over the file in a
streaming-style loop and transcribes each segment as soon as it closes,
the same way the live system will work. Prints a per-segment table with
text, detected language, language probability, and timings.

First run downloads the Whisper large-v3 model (~3 GB) into
./models/faster-whisper/. Subsequent runs are instant.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcriber.asr.whisper_engine import WhisperEngine  # noqa: E402
from transcriber.hardware.accel import detect_accel, describe  # noqa: E402
from transcriber.logging_setup import setup_logging  # noqa: E402
from transcriber.pipeline.segmenter import Segmenter  # noqa: E402
from transcriber.pipeline.vad import SILERO_SR, SileroVAD  # noqa: E402


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    n_out = int(round(len(x) * dst_sr / src_sr))
    idx = np.linspace(0, len(x) - 1, n_out)
    base = idx.astype(int)
    frac = (idx - base).astype(np.float32)
    nxt = np.minimum(base + 1, len(x) - 1)
    return ((1 - frac) * x[base] + frac * x[nxt]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--block-ms", type=int, default=480)
    parser.add_argument("--language", type=str, default=None,
                        help="Force a language code (e.g. 'es'). Default: auto-detect.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override Whisper model (default: from config, large-v3).")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    setup_logging()
    print(describe())

    audio, sr = sf.read(str(args.wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SILERO_SR:
        audio = _resample_linear(audio, sr, SILERO_SR)
        sr = SILERO_SR
    if args.normalize:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = (audio * (0.7 / peak)).astype(np.float32)
    print(f"\nLoaded {args.wav}: {audio.size / sr:.2f}s @ {sr} Hz")

    vad = SileroVAD()
    seg = Segmenter(vad)
    engine = WhisperEngine(model_name=args.model, accel=detect_accel())

    block = int(args.block_ms / 1000 * sr)
    cur_t = 0.0
    pipeline_t0 = time.monotonic()
    total_asr = 0.0
    n_segments = 0

    print()
    print(f"  {'#':>3} | {'start':>7} | {'dur':>5} | {'lang':>4} | {'p_lang':>6} | "
          f"{'rtf':>5} | text")
    print("  " + "-" * 100)

    def handle_segment(s) -> None:
        nonlocal total_asr, n_segments
        t = engine.transcribe(s.pcm, language_hint=args.language)
        total_asr += t.asr_seconds
        n_segments += 1
        text = t.text.replace("\n", " ")[:120]
        print(f"  {n_segments - 1:>3} | {s.start_time:>7.2f} | {s.duration_s:>5.2f} | "
              f"{t.language:>4} | {t.language_prob:>6.2f} | {t.rtf:>5.2f} | {text}")

    for start in range(0, audio.size, block):
        chunk = audio[start : start + block]
        for s in seg.push(chunk, cur_t):
            handle_segment(s)
        cur_t += chunk.size / sr
    for s in seg.flush():
        handle_segment(s)

    total = time.monotonic() - pipeline_t0
    audio_dur = audio.size / sr
    print(f"\nProcessed {audio_dur:.2f}s of audio in {total:.2f}s (RTF total={total/audio_dur:.3f}).")
    print(f"ASR-only time: {total_asr:.2f}s "
          f"({100*total_asr/total:.0f}% of pipeline, RTF_asr={total_asr/audio_dur:.3f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
