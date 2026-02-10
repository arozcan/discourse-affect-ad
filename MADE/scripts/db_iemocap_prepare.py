import os
import json
import re
from collections import defaultdict

IEMOCAP_DIR = "download/IEMOCAP"     
OUTPUT_DIR = "dataset/IEMOCAP"

TRANSCRIPT_RE = re.compile(r"^(?P<turn>\S+)\s+\[(?P<start>[\d\.]+)-(?P<end>[\d\.]+)\]:\s+(?P<text>.+)$")
EVAL_START_RE = re.compile(r"^\[(?P<start>[\d\.]+)\s*-\s*(?P<end>[\d\.]+)\]\s+(?P<turn>\S+)\s+(?P<emo>\w+)")


EMO_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "exc": "happy",      # excited → happy
    "sad": "sad",
    "ang": "angry",
    "fru": "angry",      # frustration → angry
    "fea": "fearful",
    "sur": "surprised",
    "dis": "disgusted",
    "xxx": "other",
    "oth": "other"
}


def map_emotion(raw):
    return EMO_MAP.get(raw.lower(), "other")  # default other


def parse_transcription(path):
    utterances = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = TRANSCRIPT_RE.match(line.strip())
            if not m:
                continue
            turn = m.group("turn")
            text = m.group("text").strip()
            speaker = "M" if "_M" in turn else "F"
            utterances[turn] = {
                "text": text,
                "speaker": speaker
            }
    return utterances


def parse_emoeval(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = EVAL_START_RE.match(line.strip())
            if not m:
                continue
            turn = m.group("turn")
            emo_raw = m.group("emo").lower()
            emo = map_emotion(emo_raw)
            labels[turn] = emo
    return labels


def process_session(session_dir, out_dir):
    trans_dir = os.path.join(session_dir, "dialog", "transcriptions")
    emo_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")

    for fname in os.listdir(trans_dir):
        if not fname.endswith(".txt"):
            continue

        trans_path = os.path.join(trans_dir, fname)
        emo_path = os.path.join(emo_dir, fname)

        utterances = parse_transcription(trans_path)
        labels = parse_emoeval(emo_path)

        segments = []
        for turn, utt in utterances.items():
            if turn not in labels:
                continue
            emo = labels[turn]
            segments.append({
                "text": utt["text"],
                "speaker": utt["speaker"],
                "predicted_emotion": emo
            })

        if not segments:
            continue

        data = {
            "id": os.path.splitext(fname)[0],
            "segments": segments
        }

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for s in range(1, 6):
        session_dir = os.path.join(IEMOCAP_DIR, f"Session{s}")
        process_session(session_dir, OUTPUT_DIR)


if __name__ == "__main__":
    main()