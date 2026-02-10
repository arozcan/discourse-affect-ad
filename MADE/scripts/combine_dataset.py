import os
import glob
import json

# =========================================================
# 🎭 Valid emotion labels
# =========================================================
valid_labels = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]

# =========================================================
# 📂 Input & Output Config
# =========================================================
folders = [
    "dataset/MADE/gpt_monologues_with_emo",
    "dataset/MADE/genai_monologues_with_emo",
    "dataset/MADE/mistral_monologues_with_emo",
    "dataset/MADE/grok_monologues_with_emo",
]

out_dir = "dataset/MADE/combined_monologues"
os.makedirs(out_dir, exist_ok=True)

# =========================================================

def normalize_monologue(src_path, new_id):
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"JSON error in {src_path}: {e}")
        return None

    if not isinstance(data, dict) or "segments" not in data:
        print(f"Invalid structure: {src_path}")
        return None

    cleaned_segments = []
    for seg in data.get("segments", []):
        txt = seg.get("text", "").strip()
        if not txt:
            continue
        emo = seg.get("predicted_emotion", "neutral").strip().lower()
        if emo not in valid_labels:
            emo = "neutral"
        cleaned_segments.append({
            "text": txt,
            "predicted_emotion": emo,
            "speaker": "PAR"
        })

    if not cleaned_segments:
        return None

    normalized = {
        "id": f"monologue_{new_id}",
        "label": data.get("label", "unknown"),
        "segments": cleaned_segments
    }

    return normalized


# =========================================================
# Main Merge Loop
# =========================================================
if __name__ == "__main__":
    counter = 0
    skipped = 0
    all_files = []

    print("Merging all cleaned datasets into a unified format...\n")

    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        for fname in sorted(glob.glob(os.path.join(folder, "*.json"))):
            counter += 1
            normalized = normalize_monologue(fname, counter)
            if not normalized:
                skipped += 1
                continue

            out_path = os.path.join(out_dir, f"monologue_{counter}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(normalized, f, ensure_ascii=False, indent=2)

    print("\n Merge completed successfully!")
    print(f" Output folder: {out_dir}")
    print(f" Total monologues written: {counter - skipped}")
    print(f" Skipped due to errors: {skipped}")