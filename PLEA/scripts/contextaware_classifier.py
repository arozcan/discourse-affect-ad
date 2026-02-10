#!/usr/bin/env python3
"""
Training & evaluation script for ContextAwareDeberta (feature-based).
Uses pre-extracted CLS/Mean embeddings instead of encoder.
Supports optional class-weighted loss (--weighted_loss) and supervised contrastive loss.

Modes:
  --mode train : train + validate + test (full training loop, expects train/val/test_context.pkl)
  --mode eval  : ONLY evaluate a saved checkpoint on chosen split(s)

In eval mode:
  --eval_split can be:
    - "test", "val", "both"
    - OR any custom name, e.g. "made"
    - OR comma-separated list: "made,meld,custom"
  For each split S, the script loads:
    {feature_dir}/{S}_context.pkl
"""

import os
import argparse
import pickle
from datetime import datetime

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- local imports ---
from model import ContextAwareDeberta


# ============================================================
# Utility
# ============================================================
def set_deterministic(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Deterministic mode activated (seed={seed})")


def plot_confusion_matrix(y_true, y_pred, labels, epoch, split, writer=None):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        normalize="true"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{split} Confusion Matrix (Epoch {epoch})")
    if writer:
        writer.add_figure(f"{split}/Confusion_Matrix", fig, epoch)
    plt.close(fig)


# ============================================================
# Dataset & Collate
# ============================================================
class FeatureDataset(Dataset):
    """
    Her örnek:
    {
      "cls_layers":  (4, U, H),
      "mean_layers": (4, U, H),
      "target_idx":  int,
      "label":       int,
    }
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {
            "cls_layers": torch.tensor(x["cls_layers"], dtype=torch.float32),
            "mean_layers": torch.tensor(x["mean_layers"], dtype=torch.float32),
            "target_idx": torch.tensor(x["target_idx"], dtype=torch.long),
            "labels": torch.tensor(x["label"], dtype=torch.long),
        }


# ============================================================
# Sentence pooling function
# ============================================================
def pool_sentence_features(cls_layers, mean_layers, sentence_pool="cls", layer_mode="avg4"):
    """
    Pools sentence-level features.

    Args:
        cls_layers:  (4, U, H)
        mean_layers: (4, U, H)
        sentence_pool: one of ["cls", "mean", "hybrid"]
        layer_mode: one of ["last", "avg4"]
    Returns:
        pooled: (U, H)
    """
    if layer_mode == "last":
        cls_repr = cls_layers[-1]      # (U, H)
        mean_repr = mean_layers[-1]    # (U, H)
    elif layer_mode == "avg4":
        cls_repr = cls_layers.mean(dim=0)   # (U, H)
        mean_repr = mean_layers.mean(dim=0) # (U, H)
    else:
        raise ValueError(f"Unknown layer_mode: {layer_mode}")

    if sentence_pool == "cls":
        return cls_repr
    elif sentence_pool == "mean":
        return mean_repr
    elif sentence_pool == "hybrid":
        return 0.5 * (cls_repr + mean_repr)
    else:
        raise ValueError(f"Unknown sentence_pool: {sentence_pool}")


# ============================================================
# Collate function
# ============================================================
def collate_fn(batch, context_window=None, causal=False,
               sentence_pool="cls", layer_mode="avg4"):
    """
    Collates a batch of conversation-level features.
    Applies sentence-level pooling and context cropping.
    """
    B = len(batch)
    new_batch = []

    for b in batch:
        cls_layers = b["cls_layers"]      # (4, U, H)
        mean_layers = b["mean_layers"]    # (4, U, H)
        t_idx = b["target_idx"].item()
        U = cls_layers.shape[1]

        # --- Context window cropping ---
        if context_window is None:
            start, end = 0, U
        else:
            if causal:
                start = max(0, t_idx - context_window)
                end = t_idx + 1
            else:
                start = max(0, t_idx - context_window)
                end = min(U, t_idx + context_window + 1)

        local_tgt_idx = t_idx - start
        cls_crop = cls_layers[:, start:end]
        mean_crop = mean_layers[:, start:end]

        pooled = pool_sentence_features(
            cls_crop, mean_crop,
            sentence_pool=sentence_pool,
            layer_mode=layer_mode
        )  # (u_i, H)

        new_batch.append({
            "pooled": pooled,
            "target_idx": torch.tensor(local_tgt_idx, dtype=torch.long),
            "labels": b["labels"],
        })

    # --- Dynamic padding ---
    U_max = max(x["pooled"].shape[0] for x in new_batch)
    H = new_batch[0]["pooled"].shape[1]

    feats_tensor = torch.zeros((B, U_max, H))
    tgt_idx = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)
    mask = torch.zeros((B, U_max), dtype=torch.bool)

    for i, b in enumerate(new_batch):
        U_i = b["pooled"].shape[0]
        feats_tensor[i, :U_i] = b["pooled"]
        tgt_idx[i] = b["target_idx"]
        labels[i] = b["labels"]
        mask[i, :U_i] = True

    return {
        "features": feats_tensor,  # (B, U, H)
        "target_idx": tgt_idx,
        "labels": labels,
        "mask": mask,
    }


# ============================================================
# Train / Eval helpers
# ============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer,
                    scaler, class_weights=None, use_amp=True):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        for k in batch:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast("cuda"):
                out = model(
                    features=batch["features"],
                    mask=batch["mask"],
                    target_idx=batch["target_idx"],
                    labels=batch["labels"],
                    class_weights=class_weights,
                )
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(
                features=batch["features"],
                mask=batch["mask"],
                target_idx=batch["target_idx"],
                labels=batch["labels"],
                class_weights=class_weights,
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()
        total_loss += loss.item()

        if writer and step % 50 == 0:
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(dataloader) + step)

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, writer, split, label_names,
             class_weights=None, use_amp=True):
    model.eval()
    preds, labels_true = [], []
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
        for k in batch:
            batch[k] = batch[k].to(device)

        if use_amp:
            with autocast("cuda"):
                out = model(
                    features=batch["features"],
                    mask=batch["mask"],
                    target_idx=batch["target_idx"],
                    labels=batch["labels"],
                    class_weights=class_weights,
                )
                loss = out["loss"]
                logits = out["logits"]
        else:
            out = model(
                features=batch["features"],
                mask=batch["mask"],
                target_idx=batch["target_idx"],
                labels=batch["labels"],
                class_weights=class_weights,
            )
            loss = out["loss"]
            logits = out["logits"]

        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        labels_true.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(labels_true, preds)
    f1 = f1_score(labels_true, preds, average="weighted")

    if writer:
        writer.add_scalar(f"{split}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{split}/Acc", acc, epoch)
        writer.add_scalar(f"{split}/F1", f1, epoch)
        plot_confusion_matrix(labels_true, preds, label_names, epoch, split, writer)

    return avg_loss, acc, f1, preds, labels_true


# ============================================================
# Main
# ============================================================
def main():
    set_deterministic(42)

    parser = argparse.ArgumentParser()

    # === Data & Training ===
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing *_context.pkl files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Training precision mode: fp16 (mixed) or fp32 (full precision)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=2e-4,
        help="Weight decay (L2 regularization strength, default=2e-4)",
    )

    # === Context & pooling ===
    parser.add_argument(
        "--sentence_pool",
        choices=["cls", "mean", "hybrid"],
        default="cls",
        help="Sentence-level representation type",
    )
    parser.add_argument(
        "--layer_mode",
        choices=["last", "avg4"],
        default="last",
        help="Which layers to use: last or average of last 4",
    )
    parser.add_argument(
        "--context_window",
        type=lambda x: None if str(x).lower() in ["none", "null", "all"] else int(x),
        default=None,
        help="If None, use full conversation; otherwise number of context utterances",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="If set, only past context is used (causal mode)",
    )

    # === Loss & balancing ===
    parser.add_argument(
        "--balance_train",
        action="store_true",
        help="Enable weighted sampling for class balance (train mode only)",
    )
    parser.add_argument(
        "--weighted_loss",
        action="store_true",
        help="Use class-weighted CE loss",
    )

    # === Supervised Contrastive Learning ===
    parser.add_argument(
        "--use_supcon",
        action="store_true",
        help="Enable supervised contrastive loss",
    )
    parser.add_argument(
        "--supcon_weight",
        type=float,
        default=0.1,
        help="Weight of SupCon loss component",
    )
    parser.add_argument(
        "--supcon_temp",
        type=float,
        default=0.07,
        help="Temperature for SupCon loss",
    )

    # === Run mode & model IO ===
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="eval",
        help="Run mode: 'train' to train+eval, 'eval' to only evaluate a saved model (default: eval).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained ContextAwareDeberta checkpoint (.pt) "
             "OR a directory containing model.pt (required in eval mode).",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="run",
        help="Base name for saved model directory in train mode "
             "(timestamp will be prepended automatically).",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        help=(
            "Which split(s) to evaluate in eval mode.\n"
            "Options:\n"
            "  - 'test', 'val', or 'both'\n"
            "  - OR any custom name (e.g., 'made', 'meld')\n"
            "  - OR multiple splits separated by comma (e.g., 'made,meld,custom')\n"
            "For each split S, the script loads {feature_dir}/{S}_context.pkl"
        ),
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (args.precision == "fp16")
    scaler = GradScaler("cuda") if use_amp else None

    # Helper: load one split
    def load_split(name):
        path = os.path.join(args.feature_dir, f"{name}_context.pkl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected feature file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    # ========================================================
    #   MODE = TRAIN
    # ========================================================
    if args.mode == "train":
        # --- load all splits ---
        train_data = load_split("train")
        val_data = load_split("val")
        test_data = load_split("test")

        label2id = train_data["label2id"]
        id2label = {v: k for k, v in label2id.items()}

        train_ds = FeatureDataset(train_data["features"])
        val_ds = FeatureDataset(val_data["features"])
        test_ds = FeatureDataset(test_data["features"])

        # --- class weights from train ---
        labels_np = np.array([x["label"] for x in train_data["features"]])
        class_counts = np.bincount(labels_np, minlength=len(label2id))
        weights_np = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        weights_np = weights_np / weights_np.sum() * len(weights_np)
        class_weights = (
            torch.tensor(weights_np, dtype=torch.float32, device=device)
            if args.weighted_loss
            else None
        )

        # collate
        collate = lambda b: collate_fn(
            b,
            context_window=args.context_window,
            causal=args.causal,
            sentence_pool=args.sentence_pool,
            layer_mode=args.layer_mode,
        )

        # loaders
        if args.balance_train:
            print("WeightedRandomSampler aktif...")
            sample_weights = [weights_np[l] for l in labels_np]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                collate_fn=collate,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate,
            )

        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
        )

        # model
        model = ContextAwareDeberta(
            input_dim=1024,
            num_labels=len(label2id),
            sentence_pool=args.sentence_pool,
            gru_layers=2,
            gru_hidden_size=512,
            dropout=0.1,
            use_supcon=args.use_supcon,
            supcon_weight=args.supcon_weight,
            supcon_temp=args.supcon_temp,
            causal=args.causal,
        ).to(device)

        # optional init from checkpoint
        if args.checkpoint is not None:
            ckpt_path = args.checkpoint
            if os.path.isdir(ckpt_path):
                ckpt_path = os.path.join(ckpt_path, "model.pt")
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            print(f"Loading checkpoint into ContextAwareDeberta from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Checkpoint loaded.")

        # logging dirs
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"{timestamp}_{args.model_out}"
        save_dir = os.path.join("saved_model", run_name)
        best_dir = os.path.join(save_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))

        best_f1 = 0.0

        # optimizer & scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(0.06 * total_steps)
        print(f"LR Scheduler: cosine_with_warmup | total_steps={total_steps}, warmup_steps={warmup_steps}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
            num_cycles=1,
        )

        # training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                writer,
                scaler,
                class_weights,
                use_amp,
            )
            val_loss, val_acc, val_f1, _, _ = evaluate(
                model,
                val_loader,
                device,
                epoch,
                writer,
                "Val",
                list(label2id.keys()),
                class_weights,
                use_amp,
            )
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

            torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pt"))
            torch.save(
                {"epoch": epoch + 1, "val_f1": val_f1},
                os.path.join(save_dir, "last_state.pt"),
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
                torch.save(
                    {"epoch": epoch + 1, "best_f1": best_f1},
                    os.path.join(best_dir, "training_state.pt"),
                )
                print(f"Best model saved @ epoch {epoch+1} (F1={best_f1:.4f})")

        # final test
        print("\nEvaluating best model on test split...")
        best_model_path = os.path.join(best_dir, "model.pt")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_loss, test_acc, test_f1, preds, labels = evaluate(
            model,
            test_loader,
            device,
            args.epochs,
            writer,
            "Test",
            list(label2id.keys()),
            class_weights,
            use_amp,
        )

        print(f"\nTest: Loss={test_loss:.4f}, Acc={test_acc:.4f}, F1={test_f1:.4f}")
        print(
            classification_report(
                labels,
                preds,
                labels=list(range(len(label2id))),
                target_names=list(label2id.keys()),
                zero_division=0,
            )
        )
        writer.close()
        return

    # ========================================================
    #   MODE = EVAL
    # ========================================================
    if args.checkpoint is None:
        raise ValueError(
            "In eval mode, you must provide --checkpoint pointing to a trained "
            "ContextAwareDeberta .pt file or a directory containing model.pt."
        )

    # --- Determine which splits to evaluate ---
    if args.eval_split.lower() == "both":
        eval_splits = ["val", "test"]
    else:
        # allow comma-separated custom splits
        eval_splits = [s.strip() for s in args.eval_split.split(",") if s.strip()]

    print(f"Eval splits: {eval_splits}")

    datasets = {}
    loaders = {}
    label2id = None
    id2label = None
    class_weights = None

    # collate (common)
    collate = lambda b: collate_fn(
        b,
        context_window=args.context_window,
        causal=args.causal,
        sentence_pool=args.sentence_pool,
        layer_mode=args.layer_mode,
    )

    # load splits; infer label2id and class_weights from first split
    for idx, split_name in enumerate(eval_splits):
        data = load_split(split_name)
        ds = FeatureDataset(data["features"])
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        datasets[split_name] = ds
        loaders[split_name] = dl

        if idx == 0:
            label2id = data["label2id"]
            id2label = {v: k for k, v in label2id.items()}

            if args.weighted_loss:
                labels_np = np.array([x["label"] for x in data["features"]])
                class_counts = np.bincount(labels_np, minlength=len(label2id))
                weights_np = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
                weights_np = weights_np / weights_np.sum() * len(weights_np)
                class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
            else:
                class_weights = None
        else:
            if data["label2id"] != label2id:
                raise ValueError(
                    f"Label mapping mismatch between splits: first={label2id}, {split_name}={data['label2id']}"
                )

    num_labels = len(label2id)
    model = ContextAwareDeberta(
        input_dim=1024,
        num_labels=num_labels,
        sentence_pool=args.sentence_pool,
        gru_layers=2,
        gru_hidden_size=512,
        dropout=0.1,
        use_supcon=args.use_supcon,
        supcon_weight=args.supcon_weight,
        supcon_temp=args.supcon_temp,
        causal=args.causal,
    ).to(device)

    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint into ContextAwareDeberta from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Checkpoint loaded.")

    writer = None  # eval-only, no TB logging

    for split_name in eval_splits:
        print(f"\nEvaluating on '{split_name}' split...")
        loss, acc, f1, preds, labels = evaluate(
            model,
            loaders[split_name],
            device,
            0,
            writer,
            split_name.capitalize(),
            list(label2id.keys()),
            class_weights,
            use_amp,
        )
        print(f"\n{split_name.capitalize()}: Loss={loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
        print(
            classification_report(
                labels,
                preds,
                labels=list(range(len(label2id))),
                target_names=list(label2id.keys()),
                zero_division=0,
            )
        )


if __name__ == "__main__":
    main()