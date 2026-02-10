import os
import json
from collections import Counter

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

valid_labels = {"happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"}

def normalize_to_emotions(folder_path):
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

        new_data = {"utterances": []}

        if isinstance(data, dict):
            if "utterances" in data and isinstance(data["utterances"], list):
                items = data["utterances"]
            else:
                items = next((v for v in data.values() if isinstance(v, list)), [])
        elif isinstance(data, list):
            items = data
        else:
            items = [data]

        # Normalize
        for item in items:
            if isinstance(item, str):
                text = item
                emo = "neutral"
            elif isinstance(item, dict):
                text = item.get("text", str(item))
                emo = item.get("predicted_emotion", "neutral")
            else:
                text = str(item)
                emo = "neutral"

            # Mapping
            if emo not in valid_labels:
                emo = map_extra.get(emo, "neutral")

            new_data["utterances"].append({
                "text": text,
                "predicted_emotion": emo
            })
            label_counter[emo] += 1

        with open(path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        print(f"Normalized {fname}")

    print("\nLabel distribution after mapping:")
    for label, count in label_counter.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    folder = "dataset/MADE/gpt_monologues_with_emo"
    normalize_to_emotions(folder)