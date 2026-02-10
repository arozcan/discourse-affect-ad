#!/usr/bin/env python3
"""
Training & evaluation script for SentenceClassifierDeberta (Stage-1).
Uses get_dataloaders() as data source with context_window=0
(i.e., single-sentence classification).

Supports:
- MELD: split_mode="folder" + folder_train/val/test
- MADE (ve diğerleri): split_mode="conversation" veya "session"
  + train/val/test ratio (random split)
"""

import os
import argparse
import torch
import numpy as np
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_polynomial_decay_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataloader import get_dataloaders
from model import SentenceClassifierDeberta


# ============================================================
# Utility functions
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
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))), normalize="true")
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
# Training / Evaluation
# ============================================================
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    writer,
    scaler,
    class_weights=None,
    grad_accum_steps=1
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"][:, 0, :].to(device)
        attention_mask = batch["attention_mask"][:, 0, :].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                class_weights=class_weights,
            )
            loss = out["loss"] / grad_accum_steps
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * grad_accum_steps

        if writer and step % 50 == 0:
            writer.add_scalar(
                "Train/Loss",
                loss.item() * grad_accum_steps,
                epoch * len(dataloader) + step,
            )

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    epoch=0,
    writer=None,
    split="Eval",
    label_names=None,
    class_weights=None,
):
    model.eval()
    preds, labels_true = [], []
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
        input_ids = batch["input_ids"][:, 0, :].to(device)
        attention_mask = batch["attention_mask"][:, 0, :].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                class_weights=class_weights,
            )
            loss = out["loss"]
            logits = out["logits"]

        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        labels_true.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(labels_true, preds)
    f1 = f1_score(labels_true, preds, average="weighted")

    if writer and split not in ["Eval"]:
        writer.add_scalar(f"{split}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{split}/Acc", acc, epoch)
        writer.add_scalar(f"{split}/F1", f1, epoch)
        if label_names:
            plot_confusion_matrix(labels_true, preds, label_names, epoch, split, writer)

    print(f"\n{split} Results — Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

    if split in ["Eval", "Test"] and label_names is not None:
        label_ids = list(range(len(label_names)))
        print(
            classification_report(
                labels_true,
                preds,
                labels=label_ids,
                target_names=label_names,
                zero_division=0,
            )
        )

    return avg_loss, acc, f1, preds, labels_true


# ============================================================
# Main
# ============================================================
def main():
    set_deterministic(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")

    # Data & model
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay (L2 regularization strength, default=1e-2)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum_steps)")

    # Dataloader / splitting
    parser.add_argument("--split_mode", choices=["conversation", "session", "folder", "single_file"],
                    default="folder")
    parser.add_argument("--folder_train", type=str, default=None,
                        help="If split_mode='folder', name of train subfolder")
    parser.add_argument("--folder_val", type=str, default=None,
                        help="If split_mode='folder', name of val subfolder")
    parser.add_argument("--folder_test", type=str, default=None,
                        help="If split_mode='folder', name of test subfolder")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="If split_mode!='folder', random train ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="If split_mode!='folder', random val ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="If split_mode!='folder', random test ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    # Loss & label options
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--balance_train", action="store_true")
    parser.add_argument("--classes", type=str, default=None,
                        help='Optional comma-separated subset of labels, e.g. "happy,sad,neutral"')
    parser.add_argument("--map_others", action="store_true",
                        help="Map labels outside given classes to 'other'")

    # Logging / eval
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--eval_model_dir", type=str, default=None,
                        help="Directory with best_model/model.pt for eval mode")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # folder_paths (only if split_mode == folder)
    folder_paths = (
        {
            "train": args.folder_train,
            "val": args.folder_val,
            "test": args.folder_test,
        }
        if args.split_mode == "folder"
        else None
    )

    # ============================================================
    # --- EVALUATION MODE ---
    # ============================================================
    if args.mode == "eval":
        if not args.eval_model_dir:
            raise ValueError("--eval_model_dir gerekli (ör: saved_model/.../best_model)")

        print(f"Evaluating model from: {args.eval_model_dir}")
        model = SentenceClassifierDeberta(
            model_name=args.model_name,
            num_labels=7,
        ).to(device)
        model.load_state_dict(
            torch.load(os.path.join(args.eval_model_dir, "model.pt"), map_location=device)
        )
        model.eval()

        _, _, test_loader, tokenizer, label2id, id2label = get_dataloaders(
            data_dir=args.data_dir,
            model_name=args.model_name,
            max_len=args.max_len,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            balance_train=False,
            context_window=0,
            classes=[c.strip() for c in args.classes.split(",")] if args.classes else None,
            map_others=args.map_others,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_mode=args.split_mode,
            fold_index=None,
            folder_paths=folder_paths,
            causal=False,
            traindl_shuffle=False,
        )

        print(label2id)
        print(id2label)

        evaluate(
            model,
            test_loader,
            device,
            split="Eval",
            label_names=list(label2id.keys()),
        )
        return

    # ============================================================
    # --- TRAIN MODE ---
    # ============================================================
    scaler = GradScaler("cuda")

    print(f"Loading dataset via get_dataloaders() (context_window=0)")
    train_loader, val_loader, test_loader, tokenizer, label2id, id2label = get_dataloaders(
        data_dir=args.data_dir,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        balance_train=args.balance_train,
        context_window=0,
        classes=[c.strip() for c in args.classes.split(",")] if args.classes else None,
        map_others=args.map_others,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_mode=args.split_mode,
        fold_index=None,
        folder_paths=folder_paths,
        causal=False,
        traindl_shuffle=True,
    )

    # --- Class weights ---
    labels_all = [label2id[b["label"]] for b in train_loader.dataset.data]
    counts = np.bincount(labels_all, minlength=len(label2id))
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights.sum() * len(weights)
    class_weights = (
        torch.tensor(weights, dtype=torch.float32, device=device)
        if args.weighted_loss
        else None
    )

    # --- Model ---
    model = SentenceClassifierDeberta(
        model_name=args.model_name,
        num_labels=len(label2id),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs

    warmup_steps = int(0.06 * total_steps)
    print(
        f"LR Scheduler: polynomial_decay | total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}"
    )

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        lr_end=0.0,
        power=1.0,  # linear decay
    )

    # --- Logging ---
    tag_parts = [
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        f"data={os.path.basename(args.data_dir)}",
        f"model={os.path.basename(args.model_name)}",
        f"epochs={args.epochs}",
        f"bs={args.batch_size}",
        f"lr={args.lr}",
        f"maxlen={args.max_len}",
        f"split={args.split_mode}",
        f"wd={args.weight_decay}",
        f"accum={args.grad_accum_steps}",
    ]
    if args.weighted_loss:
        tag_parts.append("wloss")
    if args.balance_train:
        tag_parts.append("balanced")

    run_name = "_".join(tag_parts)
    save_dir = os.path.join("saved_model", run_name)
    best_dir = os.path.join(save_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))
    best_f1 = 0.0

    print(f"\nLogging run: {run_name}")
    print(f"Model save dir: {save_dir}")
    print(f"TensorBoard log dir: {args.log_dir}/{run_name}")

    # --- Training loop ---
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
            grad_accum_steps=args.grad_accum_steps,
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
        )
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            tokenizer.save_pretrained(best_dir)
            torch.save(
                {"epoch": epoch + 1, "best_f1": best_f1},
                os.path.join(best_dir, "training_state.pt"),
            )
            print(f"Best model saved @ epoch {epoch+1} (F1={best_f1:.4f})")

    # --- Final test ---
    print("\nEvaluating best model...")
    best_model = SentenceClassifierDeberta(
        model_name=args.model_name,
        num_labels=len(label2id),
    ).to(device)
    best_model.load_state_dict(
        torch.load(os.path.join(best_dir, "model.pt"), map_location=device)
    )
    print(f"Model weights loaded from {best_dir}/model.pt")
    test_loss, test_acc, test_f1, preds, labels = evaluate(
        best_model,
        test_loader,
        device,
        args.epochs,
        writer,
        "Test",
        list(label2id.keys()),
        class_weights,
    )
    writer.close()


if __name__ == "__main__":
    main()