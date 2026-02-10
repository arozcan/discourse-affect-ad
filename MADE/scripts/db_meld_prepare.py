import os
import pandas as pd
import json
from collections import defaultdict

# --- Input / Output directories ---
MELD_DIR = "download/MELD"
OUTPUT_DIR = "dataset/MELD"

# --- Emotion mappings (IEMOCAP compatible) ---
EMO_MAP = {
    "neutral": "neutral",
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "surprise": "surprised",
    "fear": "fearful",
    "disgust": "disgusted",
}


def normalize_emotion(emo):
    """Normalize the emotion label (or mark as 'other')."""
    if not isinstance(emo, str):
        return "other"
    emo = emo.strip().lower()
    return EMO_MAP.get(emo, "other")


def process_csv(csv_path, split, norm_emotion=False):
    """Converts a CSV file into JSON dialogues."""
    # MELD CSV files can be either Windows-1252 or UTF-8 encoded
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="windows-1252")

    print(f"📂 Processing: {csv_path} ({len(df)} rows)")

    # Group by: Dialogue_ID → utterance list
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        emo = normalize_emotion(row.get("Emotion", "")) if norm_emotion else row.get("Emotion", "")
        utt = {
            "text": str(row.get("Utterance", "")),
            "speaker": str(row.get("Speaker", "")),
            "predicted_emotion": emo,
            "sentiment": str(row.get("Sentiment", "")),
            "utterance_id": int(row.get("Utterance_ID", -1)),
            "season": int(row.get("Season", -1)) if not pd.isna(row.get("Season")) else None,
            "episode": int(row.get("Episode", -1)) if not pd.isna(row.get("Episode")) else None,
            "start_time": str(row.get("StartTime", "")),
            "end_time": str(row.get("EndTime", "")),
        }
        grouped[int(row["Dialogue_ID"])].append(utt)

    # Create output directory for the split
    split_out_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_out_dir, exist_ok=True)

    # Write each dialogue to a separate JSON file
    for dialog_id, segments in grouped.items():
        segments = sorted(segments, key=lambda x: x["utterance_id"])
        data = {"id": f"dialog_{dialog_id}", "segments": segments}

        out_path = os.path.join(split_out_dir, f"dialog_{dialog_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"'{split}' split for {len(grouped)} dialogues saved → {split_out_dir}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = {
        "train": "train_sent_emo.csv",
        "val": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }

    for split, fname in files.items():
        path = os.path.join(MELD_DIR, fname)
        if not os.path.exists(path):
            print(f"{path} not found, skipping.")
            continue
        process_csv(path, split, norm_emotion=True)


if __name__ == "__main__":
    main()