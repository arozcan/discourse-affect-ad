#!/usr/bin/env python3
"""
Use an ensemble of two SentenceClassifierDeberta models to relabel
MADE-style JSON files with teacher_emotion (+ confidence).

{
  "id": "monologue_1",
  "label": "gpt-4o-mini",
  "segments": [
    {
      "text": "...",
      "predicted_emotion": "happy",   # LLM'in kendi etiketi
      "speaker": "PAR",
      # (opsiyonel) eski teacher_emotion vs.
    },
    ...
  ]
}

Output:
  "teacher_emotion": "<ensemble_label>",
  "teacher_confidence": <float 0-1>

"""

import os
import json
import argparse
from glob import glob

import torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import AutoTokenizer

from model import SentenceClassifierDeberta


EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]


def set_deterministic(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Deterministic mode activated (seed={seed})")


@torch.no_grad()
def run_ensemble_inference(
    texts,
    model1,
    model2,
    tokenizer,
    device,
    max_len=128,
    batch_size=32,
    alpha=0.5,
):
    """
    texts: [str]
    return:
      pred_labels: [str]  (teacher_emotion)
      confidences: [float]  (ensemble probability of chosen class)
    """
    model1.eval()
    model2.eval()

    pred_labels = []
    confidences = []

    n = len(texts)
    print(f"Running ensemble inference on {n} segments...")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        with autocast("cuda"):
            out1 = model1(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=None,
                class_weights=None,
            )
            out2 = model2(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=None,
                class_weights=None,
            )
            logits1 = out1["logits"]  # (B, C)
            logits2 = out2["logits"]  # (B, C)

            p1 = F.softmax(logits1, dim=-1)
            p2 = F.softmax(logits2, dim=-1)

            p_ens = alpha * p1 + (1.0 - alpha) * p2

        probs, ids = torch.max(p_ens, dim=-1)  # (B,)
        for cid, conf in zip(ids.cpu().tolist(), probs.cpu().tolist()):
            pred_labels.append(EMOTION_LABELS[cid])
            confidences.append(float(conf))

    assert len(pred_labels) == n
    assert len(confidences) == n
    return pred_labels, confidences


def main():
    set_deterministic(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of MADE (e.g., dataset/MADE)")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Subfolder under data_dir to read JSONs from "
                             "(e.g., test, train, val, genai_monologues_with_emo)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory (folder structure will mirror input_folder)")

    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--ckpt1", type=str, required=True,
                        help="First teacher checkpoint directory containing model.pt")
    parser.add_argument("--ckpt2", type=str, required=True,
                        help="Second teacher checkpoint directory containing model.pt")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Ensemble weight: p = alpha*p1 + (1-alpha)*p2")

    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dir = os.path.join(args.data_dir, args.input_folder)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input folder not found: {input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_subdir = os.path.join(args.output_dir, args.input_folder)
    os.makedirs(output_subdir, exist_ok=True)

    # --- Models & tokenizer ---
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading model1 from: {args.ckpt1}")
    model1 = SentenceClassifierDeberta(
        model_name=args.model_name,
        num_labels=len(EMOTION_LABELS),
    ).to(device)
    state1 = torch.load(os.path.join(args.ckpt1, "model.pt"), map_location=device)
    model1.load_state_dict(state1)
    print("model1 loaded.")

    print(f"Loading model2 from: {args.ckpt2}")
    model2 = SentenceClassifierDeberta(
        model_name=args.model_name,
        num_labels=len(EMOTION_LABELS),
    ).to(device)
    state2 = torch.load(os.path.join(args.ckpt2, "model.pt"), map_location=device)
    model2.load_state_dict(state2)
    print("model2 loaded.")

    json_files = sorted(glob(os.path.join(input_dir, "*.json")))
    if not json_files:
        raise ValueError(f"No JSON files found under {input_dir}")

    print(f"📁 Found {len(json_files)} JSON files under {input_dir}")

    monologues = []  # list of (filepath, data_dict)
    all_texts = []

    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        monologues.append((fp, data))
        for seg in data.get("segments", []):
            text = seg.get("text", "")
            all_texts.append(text)

    pred_labels, confidences = run_ensemble_inference(
        all_texts,
        model1,
        model2,
        tokenizer,
        device,
        max_len=args.max_len,
        batch_size=args.batch_size,
        alpha=args.alpha,
    )

    idx = 0
    for fp, data in monologues:
        for seg in data.get("segments", []):
            seg["teacher_emotion"] = pred_labels[idx]
            seg["teacher_confidence"] = confidences[idx]
            idx += 1

        base = os.path.basename(fp)
        out_path = os.path.join(output_subdir, base)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done. Labeled {idx} segments.")
    print(f"Output written under: {output_subdir}")


if __name__ == "__main__":
    main()