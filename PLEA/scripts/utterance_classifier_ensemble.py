#!/usr/bin/env python3
"""
Training & evaluation script for SentenceClassifierDeberta with ENSEMBLE META.

- get_dataloaders() → dataloader_ensemble.get_dataloaders
- context_window=0 → single-sentence classification
- MADE_ENSEMBLE gibi datasetlerde:
    * teacher_emotion   → teacher_label
    * predicted_emotion → label (primary LLM target)
    * teacher_confidence + agreement → ağırlıklı loss

Supports:
- MELD: split_mode="folder" + folder_train/val/test
- MADE / MADE_ENSEMBLE: split_mode="folder" veya "conversation"
"""

import os
import argparse
import torch
import numpy as np
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch import nn
from transformers import get_polynomial_decay_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# --- local imports ---
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
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(labels))), normalize="true"
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
# Helper: ensemble weight hesaplama
# ============================================================
def compute_ensemble_weights(batch, weighting: str):
    """
    batch:
      - teacher_confidence: (B,)
      - agreement: (B,) 0/1 veya bool
    weighting:
      - "none"            : hepsi 1.0
      - "confidence"      : conf (clamp [0.2, 1.0])
      - "agreement_conf"  : conf * 1.5 (agree) / 0.5 (disagree)
    """
    conf = batch["teacher_confidence"]  # (B,)
    base_conf = torch.clamp(conf, 0.2, 1.0)

    if weighting == "none":
        w = torch.ones_like(base_conf)
    elif weighting == "confidence":
        w = base_conf
    elif weighting == "agreement_conf":
        agree = batch["agreement"].float()
        w = base_conf * (1.5 * agree + 0.5 * (1.0 - agree))
        w = torch.clamp(w, 0.1, 2.0)
    else:
        w = torch.ones_like(base_conf)

    return w


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
    grad_accum_steps=1,
    use_ensemble_meta: bool = False,
    ensemble_weighting: str = "none",
    lambda_llm: float = 0.0,
    min_teacher_conf: float = 0.0, 
    train_target: str = "llm"   # "llm" | "teacher"
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    # Manual CE (örnek bazlı loss için)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        # context_window=0 → sadece hedef cümle: [:, 0, :]
        input_ids = batch["input_ids"][:, 0, :].to(device)
        attention_mask = batch["attention_mask"][:, 0, :].to(device)


        if train_target == "teacher":
            labels = batch["teacher_labels"].to(device)
        else:
            labels = batch["labels"].to(device)

        has_ensemble = (
            use_ensemble_meta
            and "teacher_confidence" in batch
            and "teacher_labels" in batch
            and "agreement" in batch
        )

        with autocast("cuda"):
            if has_ensemble:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    class_weights=None,
                )
                logits = out["logits"]  # (B, C)

                teacher_ids = batch["teacher_labels"].to(device)
                conf = batch["teacher_confidence"].to(device)

                # --- Hard target switching ---
                # labels already are the primary (LLM) targets; optionally switch to teacher when confident.
                final_targets = labels.clone()

                if min_teacher_conf is not None and min_teacher_conf > 0.0:
                    teacher_valid = teacher_ids >= 0
                    teacher_ok = teacher_valid & (conf >= min_teacher_conf)
                    final_targets[teacher_ok] = teacher_ids[teacher_ok]

                # final_targets
                ce_main = ce_loss(logits, final_targets)  # (B,)

                per_example_loss = ce_main

                if lambda_llm > 0.0:
                    # labels are the LLM targets; auxiliary loss encourages consistency with LLM supervision.
                    ce_llm = ce_loss(logits, labels)  # (B,)
                    per_example_loss = per_example_loss + lambda_llm * ce_llm

                sample_weights = compute_ensemble_weights(
                    batch, ensemble_weighting
                ).to(device)
                loss_vec = per_example_loss * sample_weights
                loss = loss_vec.sum() / sample_weights.sum()
            else:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    class_weights=class_weights,
                )
                loss = out["loss"]

            loss = loss / grad_accum_steps

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
    use_ensemble_meta: bool = False,
    min_teacher_conf: float = 0.0,
    eval_target: str = "labels",   # "labels" | "llm" | "teacher" | "hybrid"
):
    """
    eval_target:
      - "labels" : batch["labels"] (LLM targets)
      - "llm"    : same as "labels" (alias)
      - "teacher": batch["teacher_labels"] (where available), else labels
      - "hybrid" : conf>=thr -> teacher, else labels
    """
    assert eval_target in ["labels", "llm", "teacher", "hybrid"], \
        f"Invalid eval_target={eval_target}"

    model.eval()
    preds, labels_true = [], []
    total_loss = 0.0

    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
        input_ids = batch["input_ids"][:, 0, :].to(device)
        attention_mask = batch["attention_mask"][:, 0, :].to(device)

        # default eval targets: dataset labels
        labels = batch["labels"].to(device)
        targets = labels

        # ensemble meta varsa hedef seçimini değiştir
        has_ensemble = (
            use_ensemble_meta
            and ("teacher_confidence" in batch)
            and ("teacher_labels" in batch)
        )

        if has_ensemble and eval_target != "labels":
            teacher_ids = batch["teacher_labels"].to(device)      # (B,)
            conf = batch["teacher_confidence"].to(device)         # (B,)

            teacher_valid = teacher_ids >= 0

            if eval_target == "llm":
                targets = labels

            elif eval_target == "teacher":
                targets = labels.clone()
                targets[teacher_valid] = teacher_ids[teacher_valid]

            elif eval_target == "hybrid":
                targets = labels.clone()
                thr = float(min_teacher_conf) if min_teacher_conf is not None else 0.0

                if thr > 0.0:
                    teacher_ok = teacher_valid & (conf >= thr)
                    targets[teacher_ok] = teacher_ids[teacher_ok]
                else:
                    # thr=0 ise teacher varsa teacher, yoksa labels
                    targets[teacher_valid] = teacher_ids[teacher_valid]

        with autocast("cuda"):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                class_weights=None,
            )
            logits = out["logits"]
            loss = ce_loss(logits, targets)

        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        labels_true.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(labels_true, preds)
    f1 = f1_score(labels_true, preds, average="weighted")

    if writer and split not in ["Eval"]:
        writer.add_scalar(f"{split}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{split}/Acc", acc, epoch)
        writer.add_scalar(f"{split}/F1", f1, epoch)
        if label_names:
            plot_confusion_matrix(labels_true, preds, label_names, epoch, split, writer)

    print(
        f"\n{split} ({eval_target}) Results — "
        f"Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}"
    )

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
    parser.add_argument("--model_prefixes", type=str, default=None,
                        help='Only keep JSON files with these filename prefixes (comma-separated). '
                        'Example: "genai,gpt" or "mistral". Works for split_mode="folder" and "conversation".')

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay (L2 regularization strength, default=1e-2)",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size × grad_accum_steps)",
    )

    # Dataloader / splitting
    parser.add_argument(
        "--split_mode",
        choices=["conversation", "session", "folder", "single_file"],
        default="folder",
    )
    parser.add_argument(
        "--folder_train",
        type=str,
        default=None,
        help="If split_mode='folder', name of train subfolder",
    )
    parser.add_argument(
        "--folder_val",
        type=str,
        default=None,
        help="If split_mode='folder', name of val subfolder",
    )
    parser.add_argument(
        "--folder_test",
        type=str,
        default=None,
        help="If split_mode='folder', name of test subfolder",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="If split_mode!='folder', random train ratio",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="If split_mode!='folder', random val ratio",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="If split_mode!='folder', random test ratio",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    # Loss & label options
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--balance_train", action="store_true")
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help='Optional comma-separated subset of labels, e.g. "happy,sad,neutral"',
    )
    parser.add_argument(
        "--map_others",
        action="store_true",
        help="Map labels outside given classes to 'other'",
    )
    parser.add_argument(
        "--use_ensemble_meta",
        action="store_true",
        help="If set, use teacher_confidence/agreement + optional LLM pred for weighted training",
    )
    parser.add_argument(
        "--ensemble_weighting",
        type=str,
        default="agreement_conf",
        choices=["none", "confidence", "agreement_conf"],
        help="How to weight samples when use_ensemble_meta is True",
    )
    parser.add_argument(
        "--lambda_llm",
        type=float,
        default=0.0,
        help="Auxiliary loss weight towards LLM predicted labels (0=disabled)",
    )
    parser.add_argument(
        "--min_teacher_conf",
        type=float,
        default=0.0,
        help="Threshold for teacher_confidence. Train tarafında: "
             "conf >= thr → teacher label, aksi durumda (varsa) LLM label kullanılır.",
    )
    parser.add_argument(
        "--require_agreement",
        action="store_true",
        help="If set and filters enabled, keep only segments where teacher_label == pred_label "
             "(applied in dataloader).",
    )
    parser.add_argument(
        "--enable_ensemble_filters",
        action="store_true",
        help=(
            "Enable ensemble filtering in dataloader. "
            "When set, min_teacher_conf and/or require_agreement "
            "are applied to TRAIN data before training."
        ),
    )
    parser.add_argument(
        "--eval_target",
        type=str,
        default="llm",
        choices=["llm", "teacher", "hybrid"],
        help=(
            "Evaluation target selection:\n"
            "  llm     = LLM predicted labels\n"
            "  teacher = ensemble teacher labels\n"
            "  hybrid  = confidence-based teacher/LLM switching"
        ),
    )
    parser.add_argument(
        "--train_target",
        type=str,
        default="labels",
        choices=["llm", "teacher", "hybrid"],
        help=(
            "Training target selection:\n"
            "  llm     = LLM predicted labels\n"
            "  teacher = ensemble teacher labels\n"
        ),
    )
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument(
        "--eval_model_dir",
        type=str,
        default=None,
        help="Directory with best_model/model.pt for eval mode",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            num_labels=7,  # MADE/MELD için 7 emosyon sınıfı
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
            classes=[c.strip() for c in args.classes.split(",")]
            if args.classes
            else None,
            map_others=args.map_others,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_mode=args.split_mode,
            fold_index=None,
            folder_paths=folder_paths,
            causal=False,
            traindl_shuffle=False,
            min_teacher_conf=args.min_teacher_conf,
            require_agreement=args.require_agreement,
            model_prefixes=args.model_prefixes
        )

        print(label2id)
        print(id2label)

        evaluate(
            model,
            test_loader,
            device,
            split="Eval",
            label_names=list(label2id.keys()),
            use_ensemble_meta=args.use_ensemble_meta,
            min_teacher_conf=args.min_teacher_conf,
            eval_target=args.eval_target,
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
        min_teacher_conf=args.min_teacher_conf,
        require_agreement=args.require_agreement,
        apply_ensemble_filters=args.enable_ensemble_filters,
        model_prefixes=args.model_prefixes
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
        f"model={os.path.basename(args.model_name)}",
        f"epochs={args.epochs}",
        f"bs={args.batch_size}",
        f"lr={args.lr}",
        f"maxlen={args.max_len}",
        f"split={args.split_mode}",
        f"wd={args.weight_decay}",
        f"accum={args.grad_accum_steps}",
    ]

    if args.model_prefixes:
        # "genai,gpt" gibi -> daha kısa bir tag
        mp = args.model_prefixes.replace(",", "+").replace(" ", "")
        tag_parts.append(f"src={mp}")

    if args.weighted_loss:
        tag_parts.append("wloss")
    if args.balance_train:
        tag_parts.append("balanced")
    if args.use_ensemble_meta:
        tag_parts.append(f"ens-{args.ensemble_weighting}")
        if args.lambda_llm > 0:
            tag_parts.append(f"lamllm={args.lambda_llm}")
    if args.min_teacher_conf > 0.0:
        tag_parts.append(f"minconf={args.min_teacher_conf}")
    if args.require_agreement:
        tag_parts.append("agree_only")
    if args.enable_ensemble_filters:
        tag_parts.append("filter")

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
            use_ensemble_meta=args.use_ensemble_meta,
            ensemble_weighting=args.ensemble_weighting,
            lambda_llm=args.lambda_llm,
            min_teacher_conf=args.min_teacher_conf,
            train_target=args.train_target
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
            use_ensemble_meta=args.use_ensemble_meta,
            min_teacher_conf=args.min_teacher_conf,
            eval_target=args.eval_target,
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
        use_ensemble_meta=args.use_ensemble_meta,
        min_teacher_conf=args.min_teacher_conf,
        eval_target=args.eval_target,
    )
    writer.close()


if __name__ == "__main__":
    main()