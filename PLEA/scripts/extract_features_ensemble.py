#!/usr/bin/env python3


import os
import argparse
import pickle
import torch
from tqdm import tqdm

from model import SentenceClassifierDeberta
from dataloader import (
    get_dataloaders,
)

# ============================================================
#   Helper: deterministic
# ============================================================
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


# ============================================================
#   Save Utility
# ============================================================
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"Saved: {path}")


# ============================================================
#   Model Loader (backbone only)
# ============================================================
def load_model(model_name, checkpoint_path, device, num_labels=7):
    """
    - checkpoint_path varsa: SentenceClassifierDeberta yüklenir, sadece backbone kullanılır.
    - yoksa: HF AutoModel (SentenceClassifierDeberta içinden backbone) kullanılır.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading fine-tuned checkpoint from: {checkpoint_path}")
        model_full = SentenceClassifierDeberta(
            model_name=model_name,
            num_labels=num_labels,
        )
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Sadece backbone parametrelerini al
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }

        missing, unexpected = model_full.backbone.load_state_dict(
            backbone_state, strict=False
        )
        print("Backbone weights loaded.")
        print(f"   Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        model = model_full.backbone
    else:
        # Direkt HF modelini kullanmak istersen (fine-tune edilmemiş)
        from transformers import AutoModel
        print(f"Loading pretrained HuggingFace model: {model_name}")
        model = AutoModel.from_pretrained(model_name)

    model.to(device).eval()
    return model


# ============================================================
#   Contextual Feature Extraction (ENSEMBLE aware)
# ============================================================
@torch.no_grad()
def extract_contextual_features_ensemble(model, dataloader, device):
    """
    Her batch için:

      batch["input_ids"]      : (B, U, L)
      batch["attention_mask"] : (B, U, L)
      batch["target_idx"]     : (B,)
      batch["labels"]         : (B,)

    ENSEMBLE META (varsa):
      batch["teacher_labels"]      : (B,)
      batch["pred_labels"]         : (B,)
      batch["teacher_confidence"]  : (B,)
      batch["agreement"]           : (B,)

    Çıkış: features listesi (dict lerden oluşan):
      {
        "cls_layers": (4, U, H),
        "mean_layers": (4, U, H),
        "target_idx": int,
        "label": int,
        "teacher_label": int,
        "pred_label": int,
        "teacher_confidence": float,
        "agreement": bool,
        "target_sentence": str,
        "all_sentences": List[str],
        "json_name": str,
      }
    """
    model.eval()
    features = []

    for batch in tqdm(dataloader, desc="Extracting contextual embeddings (ensemble)"):
        input_ids = batch["input_ids"].to(device)           # (B, U, L)
        attention_mask = batch["attention_mask"].to(device) # (B, U, L)
        target_idx = batch["target_idx"]                    # (B,)
        labels = batch["labels"]                            # (B,)

        B, U, L = input_ids.shape

        # Transformer'a düzleştir
        ids_flat = input_ids.reshape(B * U, L)              # (B*U, L)
        am_flat = attention_mask.reshape(B * U, L)          # (B*U, L)

        outputs = model(
            input_ids=ids_flat,
            attention_mask=am_flat,
            output_hidden_states=True,
        )

        # Son 4 katman: her biri (B*U, L, H)
        hs = torch.stack(outputs.hidden_states[-4:], dim=0)  # (4, B*U, L, H)
        _, _, _, H = hs.shape

        # CLS embedding: token 0
        cls_emb = hs[:, :, 0, :]                             # (4, B*U, H)

        # MEAN embedding (attention mask'e göre)
        mask = am_flat.unsqueeze(0).unsqueeze(-1).float()    # (1, B*U, L, 1)
        mean_emb = (hs * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)  # (4, B*U, H)

        # (4, B, U, H)'e reshape
        cls_emb = cls_emb.reshape(4, B, U, H).cpu()
        mean_emb = mean_emb.reshape(4, B, U, H).cpu()

        texts = batch.get("texts", None)            # List[List[str]] (context window)
        all_texts = batch.get("all_texts", None)    # List[List[str]] (full convo)
        json_paths = batch.get("json_paths", None)  # List[str]

        has_ensemble_meta = (
            "teacher_labels" in batch
            and "pred_labels" in batch
            and "teacher_confidence" in batch
            and "agreement" in batch
        )

        if has_ensemble_meta:
            teacher_labels = batch["teacher_labels"]
            pred_labels = batch["pred_labels"]
            teacher_conf = batch["teacher_confidence"]
            agreement = batch["agreement"]
        else:
            teacher_labels = pred_labels = teacher_conf = agreement = None

        for i in range(B):
            t_idx = int(target_idx[i])

            sample = {
                "cls_layers":  cls_emb[:, i].numpy(),    # (4, U, H)
                "mean_layers": mean_emb[:, i].numpy(),   # (4, U, H)
                "target_idx":  t_idx,
                "label":       int(labels[i]),
            }

            # --- ENSEMBLE META per-sample ---
            if has_ensemble_meta:
                sample["teacher_label"] = int(teacher_labels[i])
                sample["pred_label"] = int(pred_labels[i])
                sample["teacher_confidence"] = float(teacher_conf[i])
                sample["agreement"] = bool(agreement[i].item())

            if texts is not None and t_idx < len(texts[i]):
                sample["target_sentence"] = texts[i][t_idx]
            else:
                sample["target_sentence"] = ""

            if all_texts is not None:
                sample["all_sentences"] = all_texts[i]

            if json_paths is not None:
                sample["json_name"] = os.path.basename(str(json_paths[i]))
            else:
                sample["json_name"] = ""

            features.append(sample)

    return features


# ============================================================
#   Main
# ============================================================
def main():
    set_deterministic(42)

    parser = argparse.ArgumentParser()

    # --- Genel ---
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    # --- Context ---
    parser.add_argument(
        "--context_window",
        type=int,
        default=0,
        help="ConvDataset ile uyumlu context_window (0, -1, vs.)",
    )
    parser.add_argument("--causal", action="store_true")

    # --- Split ayarları ---
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["folder", "conversation", "session"],
        default="conversation",
        help="folder: MADE_ENSEMBLE/MELD resmi split; conversation/session: random split",
    )
    parser.add_argument("--folder_train", type=str, default=None)
    parser.add_argument("--folder_val", type=str, default=None)
    parser.add_argument("--folder_test", type=str, default=None)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # --- Label ayarları ---
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Alt küme label kullanmak istersen (örn: happy sad neutral)",
    )
    parser.add_argument(
        "--map_others",
        action="store_true",
        help="Label set dışında kalanları 'other' labelına eşle",
    )

    parser.add_argument("--balance_train", action="store_true")
    parser.add_argument("--traindl_shuffle", action="store_true")

    # --- ENSEMBLE FILTER PARAMS (TRAIN DATA İÇİN) ---
    parser.add_argument(
        "--min_teacher_conf",
        type=float,
        default=0.0,
        help="min öğretmen güveni (sadece TRAIN set filtrelemesi için; dataloader_ensemble içinde).",
    )
    parser.add_argument(
        "--require_agreement",
        action="store_true",
        help="Teacher == pred olmasını zorunlu kıl (sadece TRAIN filtrelemesi).",
    )
    parser.add_argument(
        "--disable_ensemble_filters",
        action="store_true",
        help="Ensemble filtrelerini tamamen devre dışı bırak (train_data için).",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model backbone ---
    model = load_model(args.model_name, args.checkpoint_path, device)

    # --- Dataloaders (ENSEMBLE) ---
    folder_paths = (
        {
            "train": args.folder_train,
            "val":   args.folder_val,
            "test":  args.folder_test,
        }
        if args.split_mode == "folder"
        else None
    )

    train_loader, val_loader, test_loader, tokenizer, label2id, id2label = get_dataloaders(
        data_dir=args.data_dir,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        balance_train=args.balance_train,
        context_window=args.context_window,
        classes=args.classes,
        map_others=args.map_others,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_mode=args.split_mode,
        fold_index=None,
        folder_paths=folder_paths,
        causal=args.causal,
        traindl_shuffle=args.traindl_shuffle,
        min_teacher_conf=args.min_teacher_conf,
        require_agreement=args.require_agreement
    )

    splits = [
        ("train", train_loader),
        ("val",   val_loader),
        ("test",  test_loader),
    ]

    for split_name, loader in splits:
        print(f"\nExtracting {split_name} (ensemble-aware) split...")
        feats = extract_contextual_features_ensemble(model, loader, device)

        save_path = os.path.join(args.output_dir, f"{split_name}_context.pkl")
        save_pickle(
            {
                "features":        feats,
                "label2id":        label2id,
                "id2label":        id2label,
                "last_n_layers":   4,
                "context_window":  args.context_window,
                "causal":          args.causal,
                "source_model":    args.model_name,
                "checkpoint_path": args.checkpoint_path,
                "split_mode":      args.split_mode,
                "data_dir":        args.data_dir,
                "ensemble_meta":   True,   # küçük bir bayrak
            },
            save_path,
        )

    print("\nENSEMBLE contextual feature extraction (train/val/test) complete.")


if __name__ == "__main__":
    main()