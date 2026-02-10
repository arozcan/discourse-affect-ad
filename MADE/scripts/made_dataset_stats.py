#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MADE dataset statistics extractor.

Expected folders (customizable):
- dataset/MADE/genai_monologues_with_emo/
- dataset/MADE/gpt_monologues_with_emo/
- dataset/MADE/grok_monologues_with_emo/
- dataset/MADE/mistral_monologues_with_emo/

Each JSON file example:
{
  "id": "monologue_0001",
  "label": "gemini-2.5-flash",
  "segments": [
    {"text": "...", "predicted_emotion": "neutral", "speaker": "PAR"},
    ...
  ]
}
"""

import os
import json
import glob
import math
import argparse
from collections import Counter, defaultdict

EMOTIONS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def safe_std(xs):
    if len(xs) < 2:
        return 0.0
    m = safe_mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def word_count(text: str) -> int:
    # simple whitespace tokenizer (robust + fast)
    return len([w for w in text.strip().split() if w])


def load_monologue(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_files(folder: str):
    # Accept both monologue_0001.json etc.
    return sorted(glob.glob(os.path.join(folder, "*.json")))


def compute_stats_for_folder(folder: str):
    """
    Returns:
      - counts: dict with monologues, utterances
      - lengths: dict with lists for monologue_len, utt_words, utt_chars
      - emotions: Counter
      - bad_files: list of (path, error)
    """
    monologue_lengths = []
    utt_word_lengths = []
    utt_char_lengths = []
    emotion_counter = Counter()
    bad_files = []

    files = iter_json_files(folder)
    for fp in files:
        try:
            data = load_monologue(fp)
            segments = data.get("segments", [])
            # Count only non-empty texts
            texts = []
            emotions = []
            for seg in segments:
                txt = (seg.get("text") or "").strip()
                emo = (seg.get("predicted_emotion") or "").strip().lower()
                if not txt:
                    continue
                texts.append(txt)
                emotions.append(emo)

                # utterance lengths
                utt_word_lengths.append(word_count(txt))
                utt_char_lengths.append(len(txt))

                if emo:
                    emotion_counter[emo] += 1

            monologue_lengths.append(len(texts))

        except Exception as e:
            bad_files.append((fp, str(e)))

    counts = {
        "monologues": len(files) - len(bad_files),
        "utterances": sum(monologue_lengths),
        "files_total": len(files),
        "files_failed": len(bad_files),
    }

    lengths = {
        "monologue_len": monologue_lengths,
        "utt_words": utt_word_lengths,
        "utt_chars": utt_char_lengths,
    }

    return counts, lengths, emotion_counter, bad_files


def summarize_lengths(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}
    return {
        "mean": safe_mean(values),
        "std": safe_std(values),
        "min": min(values),
        "max": max(values),
    }


def emotion_percentages(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return {e: 0.0 for e in EMOTIONS}
    return {e: 100.0 * counter.get(e, 0) / total for e in EMOTIONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        default=[
            "dataset/MADE/genai_monologues_with_emo",
            "dataset/MADE/gpt_monologues_with_emo",
            "dataset/MADE/grok_monologues_with_emo",
            "dataset/MADE/mistral_monologues_with_emo",
        ],
        help="List of dataset folders to scan",
    )
    ap.add_argument(
        "--out_json",
        default="results/made_dataset_stats.json",
        help="Write full stats to this JSON file (set empty to disable)",
    )
    ap.add_argument(
        "--out_csv",
        default="results/made_dataset_stats_by_source.csv",
        help="Write per-source summary CSV (set empty to disable)",
    )
    ap.add_argument(
        "--show_bad_files",
        action="store_true",
        help="Print paths of files that could not be read/parsed",
    )
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    per_source = {}
    total_counts = Counter()
    total_emotions = Counter()
    total_lengths = defaultdict(list)
    all_bad = []

    for root in args.roots:
        source_name = os.path.basename(root.rstrip("/"))
        counts, lengths, emo_ctr, bad = compute_stats_for_folder(root)

        per_source[source_name] = {
            "path": root,
            "counts": counts,
            "lengths_summary": {
                "monologue_sentences": summarize_lengths(lengths["monologue_len"]),
                "utterance_words": summarize_lengths(lengths["utt_words"]),
                "utterance_chars": summarize_lengths(lengths["utt_chars"]),
            },
            "emotion_counts": dict(emo_ctr),
            "emotion_percent": emotion_percentages(emo_ctr),
        }

        total_counts.update({"monologues": counts["monologues"], "utterances": counts["utterances"]})
        total_emotions.update(emo_ctr)
        for k, v in lengths.items():
            total_lengths[k].extend(v)
        all_bad.extend([(source_name, fp, err) for fp, err in bad])

    overall = {
        "counts": dict(total_counts),
        "lengths_summary": {
            "monologue_sentences": summarize_lengths(total_lengths["monologue_len"]),
            "utterance_words": summarize_lengths(total_lengths["utt_words"]),
            "utterance_chars": summarize_lengths(total_lengths["utt_chars"]),
        },
        "emotion_counts": dict(total_emotions),
        "emotion_percent": emotion_percentages(total_emotions),
        "files_failed_total": len(all_bad),
    }

    # -----------------------
    # Pretty print to console
    # -----------------------
    print("\n==============================")
    print("MADE Dataset: Overall Summary")
    print("==============================")
    print(f"Total monologues : {overall['counts'].get('monologues', 0)}")
    print(f"Total utterances : {overall['counts'].get('utterances', 0)}")
    ms = overall["lengths_summary"]["monologue_sentences"]
    uw = overall["lengths_summary"]["utterance_words"]
    print(f"Monologue length (sentences): mean={ms['mean']:.2f}, std={ms['std']:.2f}, min={ms['min']}, max={ms['max']}")
    print(f"Utterance length (words)    : mean={uw['mean']:.2f}, std={uw['std']:.2f}, min={uw['min']}, max={uw['max']}")
    print("\nEmotion distribution (%):")
    for e in EMOTIONS:
        print(f"  {e:10s} {overall['emotion_percent'].get(e, 0.0):6.2f}")

    print("\n--------------------------------")
    print("Per-source (folder) summaries")
    print("--------------------------------")
    for src, info in per_source.items():
        c = info["counts"]
        ms = info["lengths_summary"]["monologue_sentences"]
        print(f"\n[{src}]")
        print(f"  monologues={c['monologues']}  utterances={c['utterances']}  failed_files={c['files_failed']}/{c['files_total']}")
        print(f"  monologue sentences: mean={ms['mean']:.2f} (min={ms['min']}, max={ms['max']})")

    if args.show_bad_files and all_bad:
        print("\nFiles failed to parse/read:")
        for src, fp, err in all_bad[:50]:
            print(f"  [{src}] {fp} :: {err}")
        if len(all_bad) > 50:
            print(f"  ... and {len(all_bad) - 50} more")

    # -----------------------
    # Write outputs
    # -----------------------
    out = {"overall": overall, "per_source": per_source}
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON report: {args.out_json}")

    if args.out_csv:
        # simple CSV with key summary numbers
        import csv

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "source",
                "path",
                "monologues",
                "utterances",
                "mono_sent_mean",
                "mono_sent_std",
                "mono_sent_min",
                "mono_sent_max",
                "utt_words_mean",
                "utt_words_std",
                "utt_words_min",
                "utt_words_max",
            ] + [f"pct_{e}" for e in EMOTIONS])

            for src, info in per_source.items():
                c = info["counts"]
                ms = info["lengths_summary"]["monologue_sentences"]
                uw = info["lengths_summary"]["utterance_words"]
                row = [
                    src,
                    info["path"],
                    c["monologues"],
                    c["utterances"],
                    f"{ms['mean']:.4f}",
                    f"{ms['std']:.4f}",
                    ms["min"],
                    ms["max"],
                    f"{uw['mean']:.4f}",
                    f"{uw['std']:.4f}",
                    uw["min"],
                    uw["max"],
                ] + [f"{info['emotion_percent'].get(e, 0.0):.4f}" for e in EMOTIONS]
                w.writerow(row)

            # overall row
            ms = overall["lengths_summary"]["monologue_sentences"]
            uw = overall["lengths_summary"]["utterance_words"]
            row = [
                "OVERALL",
                "-",
                overall["counts"].get("monologues", 0),
                overall["counts"].get("utterances", 0),
                f"{ms['mean']:.4f}",
                f"{ms['std']:.4f}",
                ms["min"],
                ms["max"],
                f"{uw['mean']:.4f}",
                f"{uw['std']:.4f}",
                uw["min"],
                uw["max"],
            ] + [f"{overall['emotion_percent'].get(e, 0.0):.4f}" for e in EMOTIONS]
            w.writerow(row)

        print(f"Wrote CSV summary: {args.out_csv}")


if __name__ == "__main__":
    main()