#!/usr/bin/env python3

import os
import argparse
import pickle
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

from model import SentenceClassifierDeberta
from dataloader import (
    get_dataloaders,
    ConvDataset,
    collate_hierarchical,
    load_dataset_from_dir,
)


# ============================================================
#   Contextual Feature Extraction
# ============================================================
@torch.no_grad()
def extract_contextual_features(model, dataloader, device):
    model.eval()
    features = []

    for batch in tqdm(dataloader, desc="Extracting contextual embeddings"):
        input_ids = batch["input_ids"].to(device)           # (B, U, L)
        attention_mask = batch["attention_mask"].to(device) # (B, U, L)
        target_idx = batch["target_idx"]                    # (B,)
        labels = batch["labels"]                            # (B,)

        B, U, L = input_ids.shape

        ids_flat = input_ids.reshape(B * U, L)              # (B*U, L)
        am_flat = attention_mask.reshape(B * U, L)          # (B*U, L)

        outputs = model(
            input_ids=ids_flat,
            attention_mask=am_flat,
            output_hidden_states=True
        )

        hs = torch.stack(outputs.hidden_states[-4:], dim=0) # (4, B*U, L, H)
        _, _, _, H = hs.shape

        cls_emb = hs[:, :, 0, :]                            # (4, B*U, H)

        mask = am_flat.unsqueeze(0).unsqueeze(-1).float()   # (1, B*U, L, 1)
        mean_emb = (hs * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)  # (4, B*U, H)

        cls_emb = cls_emb.reshape(4, B, U, H).cpu()
        mean_emb = mean_emb.reshape(4, B, U, H).cpu()

        texts = batch.get("texts", None)
        all_texts = batch.get("all_texts", None)
        json_paths = batch.get("json_paths", None)

        for i in range(B):
            t_idx = int(target_idx[i])
            sample = {
                "cls_layers":  cls_emb[:, i].numpy(),   # (4, U, H)
                "mean_layers": mean_emb[:, i].numpy(),  # (4, U, H)
                "target_idx":  t_idx,
                "label":       int(labels[i]),
            }

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
#   Save Utility
# ============================================================
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"Saved: {path}")


# ============================================================
#   Model Loader
# ============================================================
def load_model(model_name, checkpoint_path, device, num_labels=7):
    """
    - checkpoint_path varsa: SentenceClassifierDeberta yüklenir, sadece backbone kullanılır.
    - yoksa: AutoModel.from_pretrained(model_name)
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading fine-tuned checkpoint from: {checkpoint_path}")
        model_full = SentenceClassifierDeberta(model_name=model_name, num_labels=num_labels)
        state_dict = torch.load(checkpoint_path, map_location=device)

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
        print(f"Loading pretrained HuggingFace model: {model_name}")
        model = AutoModel.from_pretrained(model_name)

    model.to(device).eval()
    return model


# ============================================================
#   Single-split DataLoader (no train/val/test)
# ============================================================
def build_single_dataloader(
    data_dir,
    model_name,
    max_len,
    batch_size,
    context_window,
    causal,
    classes,
    map_others,
    num_workers,
    pin_memory,
):
    """
    data_dir altındaki TÜM JSON dosyalarını TEK bir split olarak yükler.
    get_dataloaders ile aynı label mantığını kullanır ama split yapmaz.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 1

    # Label set
    all_labels = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]
    labels = list(classes) if classes else all_labels
    if map_others and "other" not in labels:
        labels.append("other")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    all_data = load_dataset_from_dir(data_dir)  # dict[filename] -> list[examples]

    def flatten_all(data_dict):
        examples = []
        for f in data_dict:
            for ex in data_dict[f]:
                lbl = ex["label"]
                if lbl in label2id:
                    examples.append(ex)
                elif map_others:
                    new_ex = ex.copy()
                    new_ex["label"] = "other"
                    examples.append(new_ex)
        return examples

    all_examples = flatten_all(all_data)

    # Dataset + DataLoader
    dataset = ConvDataset(
        all_examples,
        tokenizer,
        label2id,
        max_len,
        context_window,
        causal
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # eval/extract amaçlı, karıştırmaya gerek yok
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: collate_hierarchical(x, pad_token_id),
    )

    print(f"\nSingle split ({len(dataset)} örnek, klasör={data_dir})")
    return loader, label2id, id2label


# ============================================================
#   Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # Genel
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    # Context
    parser.add_argument(
        "--context_window",
        type=int,
        default=0,
        help="ConvDataset ile uyumlu context_window (0, -1, vs.)"
    )
    parser.add_argument("--causal", action="store_true")

    # Split ayarları (train/val/test için)
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["folder", "conversation", "session"],
        default="conversation",
        help="folder: MELD resmi split; conversation/session: MADE vb. random split"
    )
    parser.add_argument("--folder_train", type=str, default=None)
    parser.add_argument("--folder_val", type=str, default=None)
    parser.add_argument("--folder_test", type=str, default=None)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

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

    parser.add_argument(
        "--single_split_name",
        type=str,
        default=None,
        help=(
            "Eğer verilirse train/val/test oluşturulmaz, data_dir altındaki TÜM veriden "
            "tek bir split üretilir ve {output_dir}/{single_split_name}_context.pkl kaydedilir.\n"
            "Örn: --single_split_name genai"
        ),
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = load_model(args.model_name, args.checkpoint_path, device)

    # ============================================================
    #   SINGLE SPLIT MODU
    # ============================================================
    if args.single_split_name is not None:
        print(f"Single-split mode aktif. Tüm veriler tek split: '{args.single_split_name}'")
        loader, label2id, id2label = build_single_dataloader(
            data_dir=args.data_dir,
            model_name=args.model_name,
            max_len=args.max_len,
            batch_size=args.batch_size,
            context_window=args.context_window,
            causal=args.causal,
            classes=args.classes,
            map_others=args.map_others,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        feats = extract_contextual_features(model, loader, device)

        save_path = os.path.join(args.output_dir, f"{args.single_split_name}_context.pkl")
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
                "split_mode":      "single",
                "data_dir":        args.data_dir,
            },
            save_path,
        )

        print("\nSingle-split contextual feature extraction complete.")
        return

    # ============================================================
    #   KLASİK TRAIN/VAL/TEST MODU
    # ============================================================
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
        folder_paths={
            "train": args.folder_train,
            "val":   args.folder_val,
            "test":  args.folder_test,
        } if args.split_mode == "folder" else None,
        causal=args.causal,
        traindl_shuffle=args.traindl_shuffle,
    )

    splits = [
        ("train", train_loader),
        ("val",   val_loader),
        ("test",  test_loader),
    ]

    for split_name, loader in splits:
        print(f"\nExtracting {split_name} split...")
        feats = extract_contextual_features(model, loader, device)

        save_path = os.path.join(args.output_dir, f"{split_name}_context.pkl")
        save_pickle({
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
        }, save_path)

    print("\nContextual feature extraction (train/val/test) complete.")


if __name__ == "__main__":
    main()