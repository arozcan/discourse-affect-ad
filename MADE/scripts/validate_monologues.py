import os
import json
import argparse
import hashlib
import numpy as np
from collections import Counter

VALID_EMOTIONS = {"happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"}

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
def md5_text(text):
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

def validate_file(fpath):
    """Return (is_valid, emotion_counts, sentence_lengths, sentence_hashes, error)"""
    try:
        data = json.load(open(fpath, encoding="utf-8"))
    except Exception as e:
        return False, {}, [], [], f"json_error:{e}"

    if "segments" not in data or not isinstance(data["segments"], list):
        return False, {}, [], [], "no_segments"

    emo_counts = Counter()
    sent_lengths = []
    sent_hashes = []

    for s in data["segments"]:
        txt = s.get("text", "").strip()
        emo = s.get("predicted_emotion", "").strip().lower()

        if not txt:
            return False, {}, [], [], "empty_text"
        if emo not in VALID_EMOTIONS:
            return False, {}, [], [], f"invalid_emotion:{emo}"

        emo_counts[emo] += 1
        sent_lengths.append(len(txt.split()))
        sent_hashes.append(md5_text(txt))

    return True, emo_counts, sent_lengths, sent_hashes, None

# -----------------------------------------------------------
# Main analysis function
# -----------------------------------------------------------
def analyze_folder(folder):
    print(f"\nChecking: {folder}")
    all_emo = Counter()
    all_lengths = []
    file_errors = []
    sentence_hashes = set()
    monologue_hashes = set()
    dupe_sentences = 0
    dupe_monologues = 0

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return None

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            content = open(fpath, encoding="utf-8").read()
            mono_hash = md5_text(content)
            if mono_hash in monologue_hashes:
                dupe_monologues += 1
            monologue_hashes.add(mono_hash)
        except:
            file_errors.append((fname, "read_error"))
            continue

        valid, emos, lens, sent_hashes, err = validate_file(fpath)
        if not valid:
            file_errors.append((fname, err))
            continue

        all_emo.update(emos)
        all_lengths.extend(lens)
        for h in sent_hashes:
            if h in sentence_hashes:
                dupe_sentences += 1
            sentence_hashes.add(h)

    total_sent = sum(all_emo.values())
    total_files = len(files)
    valid_files = total_files - len(file_errors)
    avg_len = np.mean(all_lengths) if all_lengths else 0

    # Emotion distribution — sabit sıralama
    ordered_emotions = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]
    emo_dist = {emo: round(100 * all_emo.get(emo, 0) / total_sent, 1) for emo in ordered_emotions}

    print(f"  Valid: {valid_files}/{total_files} | Invalid: {len(file_errors)}")
    print(f"  Duplicate monologues: {dupe_monologues}")
    print(f"  Duplicate sentences: {dupe_sentences}")
    print(f"  Avg sentence length: {avg_len:.2f} (min={min(all_lengths or [0])}, max={max(all_lengths or [0])})")
    print(f"  Emotion distribution:")
    for emo in ordered_emotions:
        print(f"     {emo:<10} {emo_dist[emo]:>5.1f}%")
    if file_errors:
        print(f"  ⚠️ Sample errors: {file_errors[:3]}")

    return {
        "path": folder,
        "files": total_files,
        "valid": valid_files,
        "invalid": len(file_errors),
        "dupe_mono": dupe_monologues,
        "dupe_sent": dupe_sentences,
        "avg_len": round(avg_len, 2),
        "min_len": int(np.min(all_lengths)) if all_lengths else 0,
        "max_len": int(np.max(all_lengths)) if all_lengths else 0,
        "emo_dist": emo_dist,
        "errors": file_errors[:5],
    }

# -----------------------------------------------------------
# CLI input
# -----------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    import json

    DEFAULT_DIRS = [
        "dataset/MADE/gpt_monologues_with_emo",
        "dataset/MADE/mistral_monologues_with_emo",
        "dataset/MADE/genai_monologues_with_emo",
        "dataset/MADE/grok_monologues_with_emo",
    ]

    parser = argparse.ArgumentParser(description="Validate and analyze synthetic monologue datasets.")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=DEFAULT_DIRS,
        help="List of dataset directories to check (default: 4 main monologue folders)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JSON file for summary"
    )
    args = parser.parse_args()

    all_summaries = {}
    print("Starting validation...\n")

    for folder in args.dirs:
        if not os.path.exists(folder):
            print(f"Skipping missing folder: {folder}")
            continue
        all_summaries[os.path.basename(folder)] = analyze_folder(folder)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to {args.out}")