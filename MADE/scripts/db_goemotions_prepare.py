import csv
import json
import os


EMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

LABEL_MAP = {
    # happy / positive emotions
    "admiration": "happy",
    "amusement": "happy",
    "approval": "happy",
    "caring": "happy",
    "desire": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "joy": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "relief": "happy",
    # negative → sad / angry / fearful / disgusted
    "sadness": "sad",
    "disappointment": "sad",
    "grief": "sad",
    "remorse": "sad",
    "embarrassment": "sad",
    "disgust": "disgusted",
    "disapproval": "angry",
    "anger": "angry",
    "annoyance": "angry",
    "fear": "fearful",
    "nervousness": "fearful",
    # surprise / neutral / mixed → surprised / neutral
    "surprise": "surprised",
    "curiosity": "surprised",
    "confusion": "surprised",
    "realization": "surprised",
}

TARGET_LABELS = ["happy","sad","angry","fearful","disgusted","surprised","neutral"]

def convert_tsv_to_json(tsv_path, json_path):
    out = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            text = row[0].strip()
            labels_str = row[1].strip()
            if not text:
                continue

            lbl_idxs = labels_str.split(",")
            mapped = []
            for idx in lbl_idxs:
                try:
                    emo = EMOTIONS[int(idx)]
                except:
                    continue
                if emo in LABEL_MAP:
                    mapped.append(LABEL_MAP[emo])
            mapped = list(dict.fromkeys(mapped))  # uniq

            if len(mapped) == 0:
                target = "neutral"
            else:
                # single-label
                target = mapped[0]

            out.append({
                "text": text,
                "label": target
            })

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fw:
        json.dump(out, fw, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out)} examples to {json_path}")

def main():
    splits = {
        "train": "download/GoEmotions/train.tsv",
        "val":   "download/GoEmotions/dev.tsv",
        "test":  "download/GoEmotions/test.tsv",
    }
    out_dir = "dataset/GoEmotions"
    for split, tsv in splits.items():
        out_json = os.path.join(out_dir, f"{split}.json")
        convert_tsv_to_json(tsv, out_json)

    # summary
    print("Done converting all splits.")

if __name__ == "__main__":
    main()