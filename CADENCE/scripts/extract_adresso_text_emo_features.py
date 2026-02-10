#!/usr/bin/env python3
"""
ADReSSo21_EMO için:
- Her konuşmadaki segmentler için
  * Text embedding (DeBERTa-v3-large CLS, 1024-dim)           -> text_cls
  * Emotion hidden (emo teacher ara katmanı, H-dim)          -> emo_hidden
  * Emotion logits (7-dim)                                   -> emo_logits
  * Emotion probability (softmax(logits), 7-dim)             -> emo_probs

Konuşmacı seçimi parametriktir:
  --speaker_mode par      -> sadece PAR segmentleri
  --speaker_mode par+inv  -> PAR + INV segmentleri

Çıktı formatı (pickle):

{
  "conversations": [
    {
      "conv_id": str,
      "split": "train" | "test",
      "diag_label": "ad" | "cn" | None,
      "speakers":   [ "PAR", "INV", ... ],
      "texts":      [ "utterance 1", "utterance 2", ... ],
      "text_cls":   np.ndarray shape (T, H_text),      # H_text = text_model hidden size
      "emo_hidden": np.ndarray shape (T, H_emo),       # H_emo  = emo_model hidden size
      "emo_logits": np.ndarray shape (T, 7),
      "emo_probs":  np.ndarray shape (T, 7),
    },
    ...
  ],
  "emotion_labels": [...],
  "text_model_name": "...",
  "speaker_mode": "par" | "par+inv",
}
"""

import os
import json
import argparse
from glob import glob
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModel

from model import SentenceClassifierDeberta  # first-stage emotion modeli


EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]
NUM_EMO = len(EMOTION_LABELS)


def set_deterministic(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Deterministic mode activated (seed={seed})")


def load_json_files(split_dir: str) -> List[str]:
    files = sorted(glob(os.path.join(split_dir, "*.json")))
    if not files:
        raise ValueError(f"No JSON files found under {split_dir}")
    # index.json dosyasını at
    return [f for f in files if not os.path.basename(f).startswith("index")]


@torch.no_grad()
def encode_batch(
    texts: List[str],
    tokenizer,
    text_model,
    emo_model,
    device: torch.device,
    max_len: int = 128,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Verilen text listesi için:
      - text_model ile CLS embedding (text_cls)
      - emo_model ile:
          * emotion hidden feat (emo_hidden)
          * emotion logits (emo_logits)
          * softmax(logits) ile olasılık (emo_probs)
    döner.

    text_cls_np:   shape (N, H_text)
    emo_hidden_np: shape (N, H_emo)
    emo_logits_np: shape (N, 7)
    emo_probs_np:  shape (N, 7)
    """
    all_text_cls = []
    all_emo_hidden = []
    all_emo_logits = []
    all_emo_probs = []

    n = len(texts)
    if n == 0:
        h_text = text_model.config.hidden_size
        h_emo = emo_model.backbone.config.hidden_size
        return (
            np.zeros((0, h_text), dtype=np.float32),
            np.zeros((0, h_emo), dtype=np.float32),
            np.zeros((0, NUM_EMO), dtype=np.float32),
            np.zeros((0, NUM_EMO), dtype=np.float32),
        )

    print(f"  🧮 Encoding {n} segments (text + emotion)...")

    use_cuda_amp = (device.type == "cuda")
    if use_cuda_amp:
        ctx = autocast("cuda")
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

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

        with ctx:
            # 1) Text embedding (DeBERTa CLS)
            text_out = text_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            text_cls = text_out.last_hidden_state[:, 0, :]        # (B, H_text)

            # 2) Emotion teacher → hidden feat, logits & probs
            emo_out = emo_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=None,
                class_weights=None,
            )
            # SentenceClassifierDeberta forward:
            #   {"loss": ..., "logits": (B,7), "probs": (B,7), "feat": (B,H_emo)}
            emo_hidden = emo_out["feat"]                          # (B, H_emo)
            logits = emo_out["logits"]                            # (B, 7)
            probs = emo_out["probs"]                              # (B, 7)

        all_text_cls.append(text_cls.detach().cpu().numpy())
        all_emo_hidden.append(emo_hidden.detach().cpu().numpy())
        all_emo_logits.append(logits.detach().cpu().numpy())
        all_emo_probs.append(probs.detach().cpu().numpy())

    text_cls_np   = np.concatenate(all_text_cls,   axis=0).astype(np.float32)
    emo_hidden_np = np.concatenate(all_emo_hidden, axis=0).astype(np.float32)
    emo_logits_np = np.concatenate(all_emo_logits, axis=0).astype(np.float32)
    emo_probs_np  = np.concatenate(all_emo_probs,  axis=0).astype(np.float32)

    assert text_cls_np.shape[0]   == n
    assert emo_hidden_np.shape[0] == n
    assert emo_logits_np.shape[0] == n
    assert emo_probs_np.shape[0]  == n

    return text_cls_np, emo_hidden_np, emo_logits_np, emo_probs_np


def process_split(
    split_dir: str,
    split_name: str,
    tokenizer,
    text_model,
    emo_model,
    device: torch.device,
    max_len: int,
    batch_size: int,
    speaker_mode: str = "par",
) -> List[Dict[str, Any]]:
    """
    Belirli bir split (train/test) altında tüm JSON dosyalarını işler.
    Konuşmacı seçimi speaker_mode ile belirlenir:

      speaker_mode = "par"     -> sadece PAR segmentleri
      speaker_mode = "par+inv" -> PAR + INV segmentleri

    Diğer konuşmacılar (örn. UNK) atlanır.
    """
    files = load_json_files(split_dir)
    convs = []

    print(f"\n🔍 Processing split: {split_name} ({len(files)} files)")
    print(f"   Speaker mode: {speaker_mode}")

    if speaker_mode == "par":
        allowed_speakers = {"PAR"}
    elif speaker_mode == "par+inv":
        allowed_speakers = {"PAR", "INV"}
    else:
        raise ValueError(f"Unknown speaker_mode={speaker_mode}, expected 'par' or 'par+inv'.")

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Bazı pipeline'larda JSON listesi olabilir, ama ADReSSo21'de genelde tek dict bekliyoruz.
        if isinstance(data, list):
            conversations = data
        else:
            conversations = [data]

        for conv in conversations:
            conv_id = conv.get("id") or conv.get("conversation_id") or os.path.splitext(os.path.basename(fp))[0]
            diag_label = conv.get("label") or conv.get("diag_label") or conv.get("diagnosis")
            if isinstance(diag_label, str):
                diag_label_norm = diag_label.lower()
                if diag_label_norm not in ["ad", "cn"]:
                    diag_label_norm = None
            else:
                diag_label_norm = None

            seg_texts = []
            seg_speakers = []

            for seg in conv.get("segments", []):
                spk = (seg.get("speaker") or "").upper()

                # Konuşmacı filtresi
                if spk not in allowed_speakers:
                    continue

                text = (seg.get("text") or "").strip()
                if not text:
                    continue

                seg_speakers.append(spk)
                seg_texts.append(text)

            if len(seg_texts) == 0:
                # Bu konuşma için istenen konuşmacı moduna uyan segment yok → atla
                print(f"  ⚠️ No segments for speaker_mode='{speaker_mode}' in conv {conv_id}, skipping.")
                continue

            # Encode this conversation's segments
            text_cls_np, emo_hidden_np, emo_logits_np, emo_probs_np = encode_batch(
                seg_texts,
                tokenizer,
                text_model,
                emo_model,
                device,
                max_len=max_len,
                batch_size=batch_size,
            )

            conv_obj = {
                "conv_id": conv_id,
                "split": split_name,
                "diag_label": diag_label_norm,   # "ad" / "cn" / None
                "speakers": seg_speakers,        # ["PAR","INV",...]
                "texts": seg_texts,
                "text_cls": text_cls_np,         # (T, H_text)
                "emo_hidden": emo_hidden_np,     # (T, H_emo)
                "emo_logits": emo_logits_np,     # (T, 7)
                "emo_probs": emo_probs_np,       # (T, 7)
            }
            convs.append(conv_obj)

    print(f"  → {len(convs)} conversations in {split_name} (speaker_mode={speaker_mode})")
    return convs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="ADReSSo21_EMO kök klasörü (ör: dataset/ADReSSo21_EMO)",
    )
    parser.add_argument(
        "--emo_ckpt_dir",
        type=str,
        required=True,
        help="First-stage emotion teacher checkpoint (dir or model.pt path)",
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        default="microsoft/deberta-v3-large",
        help="Text encoder model adı (HF, ör: microsoft/deberta-v3-large)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Tokenizer max length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Encoding batch size",
    )
    parser.add_argument(
        "--speaker_mode",
        type=str,
        default="par+inv",
        choices=["par", "par+inv"],
        help="Hangi konuşmacı segmentleri kullanılacak: sadece PAR mı, yoksa PAR+INV mi?",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        required=True,
        help="Çıkış pickle dosyası (örn: features/adresso21_text_emo_segments.pkl)",
    )

    args = parser.parse_args()
    set_deterministic(42)

    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")

    # ---------------------------------------------------
    # 1) Tokenizer & text model (DeBERTa)
    # ---------------------------------------------------
    print(f"📂 Loading tokenizer: {args.text_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    print(f"📂 Loading text encoder model: {args.text_model_name}")
    text_model = AutoModel.from_pretrained(args.text_model_name)
    text_model.to(device)
    text_model.eval()
    print("✅ Text model loaded.")

    # ---------------------------------------------------
    # 2) Emotion model (SentenceClassifierDeberta)
    # ---------------------------------------------------
    print(f"📂 Loading emotion teacher model from: {args.emo_ckpt_dir}")
    emo_model = SentenceClassifierDeberta(
        model_name=args.text_model_name,
        num_labels=NUM_EMO,
    ).to(device)

    ckpt_path = args.emo_ckpt_dir
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Emotion checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    emo_model.load_state_dict(state_dict)
    emo_model.eval()
    print("✅ Emotion model loaded.")

    # ---------------------------------------------------
    # 3) Process splits
    # ---------------------------------------------------
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    convs_all = []

    if os.path.isdir(train_dir):
        convs_train = process_split(
            train_dir,
            "train",
            tokenizer,
            text_model,
            emo_model,
            device,
            max_len=args.max_len,
            batch_size=args.batch_size,
            speaker_mode=args.speaker_mode,
        )
        convs_all.extend(convs_train)
    else:
        print(f"⚠️ Train dir not found: {train_dir}")

    if os.path.isdir(test_dir):
        convs_test = process_split(
            test_dir,
            "test",
            tokenizer,
            text_model,
            emo_model,
            device,
            max_len=args.max_len,
            batch_size=args.batch_size,
            speaker_mode=args.speaker_mode,
        )
        convs_all.extend(convs_test)
    else:
        print(f"⚠️ Test dir not found: {test_dir}")

    data = {
        "conversations": convs_all,
        "emotion_labels": EMOTION_LABELS,
        "text_model_name": args.text_model_name,
        "speaker_mode": args.speaker_mode,
    }

    import pickle
    with open(args.output_pkl, "wb") as f:
        pickle.dump(data, f)

    print(f"\n💾 Saved {len(convs_all)} conversations to: {args.output_pkl}")
    print("✅ Done.")


if __name__ == "__main__":
    main()