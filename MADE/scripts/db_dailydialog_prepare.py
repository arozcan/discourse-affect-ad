#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DailyDialog TXT → JSON
---------------------------------------
Varsayım:
download/DailyDialog/{train,val,test}/
    ├── dialogues_*.txt
    ├── dialogues_act_*.txt
    └── dialogues_emotion_*.txt
"""

import os
import json

# ===============================
INPUT_BASE = "download/DailyDialog"
OUTPUT_BASE = "dataset/DailyDialog"

EMO_MAP = {
    "0": "neutral",
    "1": "angry",
    "2": "disgusted",
    "3": "fearful",
    "4": "happy",
    "5": "sad",
    "6": "surprised",
}


# ===============================
def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def process_split(split):
    print(f"📂 İşleniyor: {split}")
    base_dir = os.path.join(INPUT_BASE, split)
    out_dir = os.path.join(OUTPUT_BASE, split)
    os.makedirs(out_dir, exist_ok=True)

    dialog_file = os.path.join(base_dir, f"dialogues_{split}.txt")
    act_file = os.path.join(base_dir, f"dialogues_act_{split}.txt")
    emo_file = os.path.join(base_dir, f"dialogues_emotion_{split}.txt")

    dialogs = load_lines(dialog_file)
    acts = load_lines(act_file)
    emos = load_lines(emo_file)

    total = min(len(dialogs), len(acts), len(emos))
    saved = 0

    for idx in range(total):
        dialog_line = dialogs[idx]
        act_line = acts[idx]
        emo_line = emos[idx]

        utterances = [u.strip() for u in dialog_line.split("__eou__") if u.strip()]
        act_values = [int(x) for x in act_line.split() if x.strip()]
        emo_values = [EMO_MAP.get(x, "neutral") for x in emo_line.split() if x.strip()]

        if not utterances:
            continue

        segments = []
        for i, utt in enumerate(utterances):
            seg = {
                "text": utt,
                "speaker": f"S{i % 2 + 1}",
                "predicted_emotion": emo_values[i] if i < len(emo_values) else "neutral",
                "utterance_id": i,
            }
            if i < len(act_values):
                seg["dialog_act"] = act_values[i]
            segments.append(seg)

        data = {"id": f"dialog_{idx}", "segments": segments}

        out_path = os.path.join(out_dir, f"dialog_{idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        saved += 1

    print(f"{saved} dialog saved → {out_dir}\n")


# ===============================
def main():
    for split in ["train", "validation", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()