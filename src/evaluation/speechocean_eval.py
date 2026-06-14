"""SpeechOcean762 PCC evaluation skeleton (M6).

SpeechOcean762 (OpenSLR #101, HF `mispeech/speechocean762`) is an open-source
**English-L2** pronunciation-assessment dataset with multi-dimensional human
scores.  We use it for **methodological validation**: we want to show that
the *scoring pipeline* (in particular the DTW + alignment + scoring
aggregator) produces numbers that **correlate** with human judgement, by
reporting Pearson Correlation Coefficient (PCC) between our model's
"accuracy" score and the human "accuracy" label.

⚠️ This script intentionally does **not** validate Mandarin tone scoring —
there is no comparable open-source Mandarin set with per-syllable tone labels
at this scale.  We surface this caveat in the README.

Usage
-----
    # 1. Cache the dataset locally (~600 MB)
    python -m src.evaluation.speechocean_eval --download

    # 2. Run the PCC evaluation on N random samples
    python -m src.evaluation.speechocean_eval --n 200

The script saves a CSV of (sample_id, human_accuracy, model_score) and prints
the resulting PCC.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CACHE_DIR


def _load_dataset(split: str = "test"):
    """Stream the SpeechOcean762 HF dataset. Lazy import to keep this file
    cheap to load when the user only wants the pipeline modules."""
    from datasets import load_dataset
    return load_dataset(
        "mispeech/speechocean762", split=split,
        cache_dir=str(CACHE_DIR / "datasets"),
    )


def _score_sample(audio_array, sr: int, reference_text: str) -> float:
    """Run our pipeline on one sample → 0..100 accuracy score.

    For an English-L2 dataset we **cannot** rely on Mandarin alignment, so
    we feed the audio through the parts of the pipeline that are language-
    agnostic: preprocess → MFCC → DTW against a TTS reference of the same
    English text.  The TTS engine is swapped to `edge-tts` with an English
    voice for this script only.
    """
    import numpy as np

    from src.audio.preprocess import preprocess
    from src.features.spectral import extract_mfcc
    from src.feedback.tts import _edge_tts
    from src.scoring.accuracy import _dtw_mean_cost, _cost_to_score

    # User-side features
    user_pre = preprocess((sr, np.asarray(audio_array)))
    user_mfcc = extract_mfcc(user_pre.wav)

    # Reference-side features via English TTS
    ref_path = _edge_tts(reference_text, voice="en-US-AriaNeural")
    ref_pre = preprocess(ref_path)
    ref_mfcc = extract_mfcc(ref_pre.wav)

    cost = _dtw_mean_cost(user_mfcc, ref_mfcc)
    return _cost_to_score(cost)


def evaluate(n_samples: int = 50, split: str = "test",
             output_csv: Path = CACHE_DIR / "speechocean_pcc.csv"
             ) -> Tuple[float, float]:
    """Score `n_samples` from SpeechOcean762 and compute PCC against
    the human "accuracy" label.  Returns `(pcc, p_value)`."""
    from scipy.stats import pearsonr

    ds = _load_dataset(split)
    n_samples = min(n_samples, len(ds))
    rows: List[Tuple[str, float, float]] = []
    for i in range(n_samples):
        sample = ds[i]
        audio = sample["audio"]
        # SpeechOcean762 has per-sample, per-word, per-phone scores.
        # The top-level "accuracy" label is on 0..10 — rescale to 0..100.
        human_accuracy = float(sample["accuracy"]) * 10.0
        text = sample["text"]
        try:
            model_score = _score_sample(audio["array"], audio["sampling_rate"], text)
        except Exception as e:
            print(f"  [skip] sample {i}: {e!r}")
            continue
        rows.append((str(sample.get("speaker", i)), human_accuracy, model_score))
        if (i + 1) % 10 == 0:
            print(f"  scored {i + 1} / {n_samples}")

    if len(rows) < 5:
        raise RuntimeError("Too few successful samples to compute PCC.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "human_accuracy", "model_score"])
        writer.writerows(rows)

    humans = [r[1] for r in rows]
    models = [r[2] for r in rows]
    pcc, p = pearsonr(humans, models)
    return float(pcc), float(p)


def main():
    p = argparse.ArgumentParser(description="SpeechOcean762 PCC validation")
    p.add_argument("--download", action="store_true",
                   help="Just cache the dataset and exit.")
    p.add_argument("--n", type=int, default=50,
                   help="Number of samples to score.")
    p.add_argument("--split", default="test", choices=["train", "test"])
    args = p.parse_args()

    if args.download:
        ds = _load_dataset(args.split)
        print(f"Cached {len(ds)} {args.split} samples → {CACHE_DIR / 'datasets'}")
        return

    print(f"Evaluating {args.n} samples from SpeechOcean762/{args.split} …")
    pcc, p_val = evaluate(args.n, split=args.split)
    print(f"\nPearson Correlation Coefficient: {pcc:.3f}  (p = {p_val:.2e})")
    print(f"CSV written to: {CACHE_DIR / 'speechocean_pcc.csv'}")


if __name__ == "__main__":
    main()
