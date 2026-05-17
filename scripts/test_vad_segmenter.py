"""Hito 2 validator: run VAD + segmenter over an existing WAV.

Usage:
    python scripts/test_vad_segmenter.py capture_test.wav
    python scripts/test_vad_segmenter.py capture_test.wav --debug-probs
    python scripts/test_vad_segmenter.py capture_test.wav --threshold 0.40
    python scripts/test_vad_segmenter.py capture_test.wav --normalize

Writes each detected segment to ./segments/seg_<n>.wav and prints a
diagnostic table. Use --debug-probs to dump per-chunk VAD probabilities.
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

from transcriber.logging_setup import setup_logging  # noqa: E402
from transcriber.pipeline.segmenter import Segmenter  # noqa: E402
from transcriber.pipeline.vad import SileroVAD, SILERO_SR  # noqa: E402


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
    parser.add_argument("--out-dir", type=Path, default=Path("segments"))
    parser.add_argument("--block-ms", type=int, default=480,
                        help="Simulate streaming with this block size in ms.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override VAD threshold (0..1). Default uses .env setting.")
    parser.add_argument("--debug-probs", action="store_true",
                        help="Log per-chunk speech probabilities (very verbose).")
    parser.add_argument("--normalize", action="store_true",
                        help="Peak-normalize audio to 0.7 before VAD. Use if your "
                             "capture was very quiet (peak < 0.05).")
    args = parser.parse_args()

    # Configure logging level dynamically.
    import os
    if args.debug_probs:
        os.environ.setdefault("TRANSCRIBER_LOG_LEVEL", "DEBUG")
    setup_logging()

    audio, sr = sf.read(str(args.wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SILERO_SR:
        audio = _resample_linear(audio, sr, SILERO_SR)
        sr = SILERO_SR

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    print(f"Loaded {args.wav}: {audio.size / sr:.2f}s @ {sr} Hz, peak={peak:.3f} rms={rms:.4f}")

    if args.normalize and peak > 0:
        gain = 0.7 / peak
        audio = (audio * gain).astype(np.float32)
        print(f"  Normalized by {gain:.2f}x -> new peak={float(np.max(np.abs(audio))):.3f}")

    vad = SileroVAD(threshold=args.threshold)
    seg = Segmenter(vad, debug_probs=args.debug_probs)

    block = int(args.block_ms / 1000 * sr)
    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True)

    all_segs = []
    t0 = time.monotonic()
    cur_t = 0.0
    for start in range(0, audio.size, block):
        chunk = audio[start : start + block]
        for s in seg.push(chunk, cur_t):
            all_segs.append(s)
        cur_t += chunk.size / sr
    all_segs.extend(seg.flush())
    elapsed = time.monotonic() - t0

    print(f"\nVAD diag: total_chunks={seg.total_chunks} speech_chunks={seg.speech_chunks} "
          f"({100*seg.speech_chunks/max(1,seg.total_chunks):.1f}%) max_prob={seg.max_prob:.2f} "
          f"threshold={vad.threshold:.2f}")
    print(f"Processed {audio.size / sr:.2f}s in {elapsed:.2f}s "
          f"(RTF={elapsed / (audio.size / sr):.3f})")
    print(f"\nDetected {len(all_segs)} segments:\n")
    if all_segs:
        print(f"  {'#':>3} | {'start':>7} | {'end':>7} | {'dur':>5} | {'speech%':>7} | {'mean_p':>6}")
        print("  " + "-" * 56)
        for i, s in enumerate(all_segs):
            out_path = out_dir / f"seg_{i:03d}.wav"
            sf.write(str(out_path), s.pcm, sr, subtype="PCM_16")
            print(f"  {i:>3} | {s.start_time:>7.2f} | {s.end_time:>7.2f} | "
                  f"{s.duration_s:>5.2f} | {s.speech_ratio * 100:>6.1f}% | {s.mean_prob:>6.2f}")
        print(f"\nSegments written to {out_dir.absolute()}")
    else:
        print("  (no segments — try --normalize, lower --threshold, or --debug-probs to inspect)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
