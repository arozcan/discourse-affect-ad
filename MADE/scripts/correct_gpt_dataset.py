import os
import json
from collections import Counter

# Mapping table for extra emotion labels
map_extra = {
    "confused": "surprised",
    "concerned": "fearful",
    "worried": "fearful",
    "anxious": "fearful",
    "frustrated": "angry",
    "conflicted": "sad",
    "hopeful": "happy",
    "curious": "surprised",
    "excited": "happy",
    "bad": "sad",
    "fearful ": "fearful"
}

# Final 7 emotion classes to be used
valid_labels = {"happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"}

def normalize_and_convert(folder_path, label="gpt-4o-mini"):
    label_counter = Counter()

    for fname in os.listdir(folder_path):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(folder_path, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"JSON parse error: {fname}")
            continue

        # Generate ID from filename
        monologue_id = os.path.splitext(fname)[0]

        # Source data → utterances or segments
        if isinstance(data, dict):
            if "utterances" in data:
                items = data["utterances"]
            elif "segments" in data:
                items = data["segments"]
            else:
                # different key → find first list
                items = next((v for v in data.values() if isinstance(v, list)), [])
        elif isinstance(data, list):
            items = data
        else:
            items = [data]

        new_segments = []
        for item in items:
            if isinstance(item, dict):
                text = item.get("text", str(item))
                emo = item.get("predicted_emotion", "neutral")
            else:
                text = str(item)
                emo = "neutral"

            # Mapping extra labels to valid ones
            emo = emo.strip().lower()
            if emo not in valid_labels:
                emo = map_extra.get(emo, "neutral")

            new_segments.append({
                "text": text,
                "predicted_emotion": emo,
                "speaker": "PAR"
            })
            label_counter[emo] += 1

        # New format
        new_data = {
            "id": monologue_id,
            "label": label,
            "segments": new_segments
        }

        # Overwrite file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        print(f"Converted {fname}")

    # Print label distribution
    print("\nLabel distribution after mapping:")
    for label, count in label_counter.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    folder = "dataset/MADE/gpt_monologues_with_emo"
    normalize_and_convert(folder, label="gpt-4o-mini")