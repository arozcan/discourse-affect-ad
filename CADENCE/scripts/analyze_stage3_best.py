#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-3 Best Model Analysis (ADReSSo21, AD vs CN)

- Loads features pkl produced by extract_adresso_text_emo_features.py
- Recreates StratifiedKFold splits identically to train_eval_adresso_seq.py
- Loads best_model_{model_type}_fold{K}.pt for each fold and runs inference on that fold's val set
- Produces:
  * per-conversation predictions CSV
  * CV summary text file
  * t-SNE plots for conversation-level embeddings (emo_hidden / text_cls / fused)
  * Emotion distribution comparison (mean emo_probs) AD vs CN
  * Temporal dynamics: volatility (L2 diff), entropy of emo_probs
  * Attention diagnostics: entropy / concentration, peak position (for text & emo streams)

Usage:
python scripts/analyze_stage3_best.py \
  --features_pkl features/adresso21_text_emo_segments_made_teacher.pkl \
  --model_dir saved_model/adresso_seq_cv_par_both_made_teacher \
  --model_type bilstm \
  --speaker_mode all \
  --feature_type text+emo_hidden \
  --max_segments 128 \
  --hidden_dim 256 \
  --num_layers 1 \
  --seed 42 \
  --out_dir analysis/stage3_best_made_teacher
"""

import os
import math
import argparse
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Emotion labels for plotting
EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]

import numpy as np

from scipy.spatial.distance import jensenshannon

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ----------------------------
# Utilities
# ----------------------------
def set_deterministic(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_mean_pool(x: np.ndarray) -> np.ndarray:
    # x: (T, D)
    if x is None or len(x) == 0:
        return None
    return np.mean(x, axis=0)


def softmax_np(x: np.ndarray, axis: int = -1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-12, None)


def entropy_probs(p: np.ndarray, axis: int = -1) -> np.ndarray:
    # p: (..., C)
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


# ----------------------------
# Causal/structural analysis helpers
# ----------------------------
def normalize_rows(M: np.ndarray) -> np.ndarray:
    M = M.astype(float)
    row_sum = M.sum(axis=1, keepdims=True)
    return np.divide(M, row_sum, out=np.zeros_like(M), where=row_sum > 0)


def js_divergence_matrix(M1: np.ndarray, M2: np.ndarray) -> float:
    """
    Jensen–Shannon divergence between two row-normalized transition matrices.
    """
    M1n = normalize_rows(M1)
    M2n = normalize_rows(M2)

    js_vals = []
    for i in range(M1n.shape[0]):
        if M1n[i].sum() > 0 and M2n[i].sum() > 0:
            js = jensenshannon(M1n[i], M2n[i])
            if np.isfinite(js):
                js_vals.append(js)
    return float(np.mean(js_vals)) if js_vals else float("nan")


def resample_curve(x: np.ndarray, T: int = 50) -> np.ndarray:
    """
    Resample a 1D temporal curve to fixed length T using linear interpolation.
    """
    if len(x) < 2:
        return np.full(T, np.nan)
    xp = np.linspace(0, 1, len(x))
    xq = np.linspace(0, 1, T)
    return np.interp(xq, xp, x)


# ----------------------------
# Try importing training script to guarantee same preprocessing/model
# ----------------------------
IMPORTED_FROM_TRAIN_SCRIPT = False
try:
    # If you keep analyze_stage3_best.py under scripts/, this relative import usually works:
    # from scripts import train_eval_adresso_seq as train_mod
    import train_eval_adresso_seq as train_mod  # run from same folder
    IMPORTED_FROM_TRAIN_SCRIPT = True
except Exception:
    train_mod = None
    IMPORTED_FROM_TRAIN_SCRIPT = False


# ----------------------------
# Minimal fallbacks (used only if import fails)
# ----------------------------
class ConversationSeqDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], max_segments: int):
        self.samples = samples
        self.max_segments = max_segments

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, max_segments: int):
    B = len(batch)
    L = max_segments

    has_text = "text_features" in batch[0]
    has_emo = "emo_features" in batch[0]

    text_dim = batch[0]["text_features"].shape[1] if has_text else None
    emo_dim = batch[0]["emo_features"].shape[1] if has_emo else None

    text_feats = torch.zeros(B, L, text_dim, dtype=torch.float32) if has_text else None
    emo_feats = torch.zeros(B, L, emo_dim, dtype=torch.float32) if has_emo else None

    mask = torch.zeros(B, L, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    conv_ids = []

    for i, item in enumerate(batch):
        labels[i] = item["label"]
        conv_ids.append(item["conv_id"])

        # determine T
        x = item["text_features"] if has_text else item["emo_features"]
        T = x.shape[0]
        tlen = min(T, L)
        mask[i, :tlen] = True

        if has_text:
            text_feats[i, :tlen, :] = torch.from_numpy(item["text_features"][:tlen])
        if has_emo:
            emo_feats[i, :tlen, :] = torch.from_numpy(item["emo_features"][:tlen])

    return {
        "text_features": text_feats,
        "emo_features": emo_feats,
        "mask": mask,
        "labels": labels,
        "conv_ids": conv_ids,
    }


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        text_dim: Optional[int],
        emo_dim: Optional[int],
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.2,
        attn_tau: float = 1.5,
    ):
        super().__init__()
        assert text_dim is not None or emo_dim is not None

        self.use_text = text_dim is not None
        self.use_emo = emo_dim is not None
        self.attn_tau = attn_tau

        if self.use_text:
            self.text_lstm = nn.LSTM(
                text_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, bidirectional=True
            )
            self.text_attn = nn.Linear(hidden_dim * 2, 1)

        if self.use_emo:
            self.emo_lstm = nn.LSTM(
                emo_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, bidirectional=True
            )
            self.emo_attn = nn.Linear(hidden_dim * 2, 1)

        fusion_dim = 0
        if self.use_text:
            fusion_dim += hidden_dim * 2
        if self.use_emo:
            fusion_dim += hidden_dim * 2

        self.dropout = nn.Dropout(dropout)
        self.temporal_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def _apply_attention(self, seq_out, mask, attn_layer):
        scores = attn_layer(seq_out).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores / self.attn_tau, dim=-1)
        ctx = torch.sum(w.unsqueeze(-1) * seq_out, dim=1)
        return ctx, w

    def forward(self, text_features=None, emo_features=None, mask=None):
        contexts = []
        attn_weights = {}

        if self.use_text:
            text_out, _ = self.text_lstm(text_features)
            text_out = self.temporal_dropout(text_out)
            text_ctx, text_attn = self._apply_attention(text_out, mask, self.text_attn)
            contexts.append(text_ctx)
            attn_weights["text"] = text_attn

        if self.use_emo:
            emo_out, _ = self.emo_lstm(emo_features)
            emo_out = self.temporal_dropout(emo_out)
            emo_ctx, emo_attn = self._apply_attention(emo_out, mask, self.emo_attn)
            contexts.append(emo_ctx)
            attn_weights["emo"] = emo_attn

        fused = torch.cat(contexts, dim=-1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits, attn_weights


def build_model_fallback(
    model_type: str,
    text_dim: Optional[int],
    emo_dim: Optional[int],
    hidden_dim: int,
    num_layers: int,
    num_classes: int,
    dropout: float,
):
    mt = model_type.lower()
    if mt != "bilstm":
        raise ValueError("Fallback builder only supports bilstm here. Please ensure train script import works.")
    return BiLSTMAttentionClassifier(
        text_dim=text_dim,
        emo_dim=emo_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        attn_tau=1.5,
    )


# ----------------------------
# Data preparation (speaker + feature filtering) with alignment to original conv objects
# ----------------------------
@dataclass
class SampleWithMeta:
    conv_id: str
    label: int
    idx: List[int]  # selected segment indices within original conv
    text_features: Optional[np.ndarray]
    emo_features: Optional[np.ndarray]


def build_samples_with_meta(
    convs_train: List[Dict[str, Any]],
    label2id: Dict[str, int],
    speaker_mode: str,
    feature_type: str,
) -> Tuple[List[SampleWithMeta], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      samples_meta: list with chosen segment indices per conv (to reuse emo_probs, etc.)
      conv_by_id:   mapping conv_id -> original conv object
    """
    conv_by_id = {c["conv_id"]: c for c in convs_train}

    samples_meta: List[SampleWithMeta] = []
    for conv in convs_train:
        diag = conv.get("diag_label")
        if diag not in label2id:
            continue

        speakers = conv.get("speakers", None)
        if speakers is None:
            continue
        speakers = [str(s).upper() for s in speakers]
        T = len(speakers)

        text_cls = conv.get("text_cls", None)
        emo_hidden = conv.get("emo_hidden", None)

        need_text = feature_type in ["text", "text+emo_hidden"]
        need_emo = feature_type in ["emo_hidden", "text+emo_hidden"]

        if need_text and text_cls is None:
            continue
        if need_emo and emo_hidden is None:
            continue

        # length consistency
        if need_text and np.asarray(text_cls).shape[0] != T:
            continue
        if need_emo and np.asarray(emo_hidden).shape[0] != T:
            continue

        if speaker_mode == "par":
            idx = [i for i, s in enumerate(speakers) if s == "PAR"]
        elif speaker_mode == "all":
            idx = list(range(T))
        else:
            raise ValueError(f"Unknown speaker_mode: {speaker_mode}")

        if len(idx) == 0:
            continue

        tf = np.asarray(text_cls)[idx].astype(np.float32) if need_text else None
        ef = np.asarray(emo_hidden)[idx].astype(np.float32) if need_emo else None

        samples_meta.append(
            SampleWithMeta(
                conv_id=conv["conv_id"],
                label=label2id[diag],
                idx=idx,
                text_features=tf,
                emo_features=ef,
            )
        )

    return samples_meta, conv_by_id


def to_plain_samples(samples_meta: List[SampleWithMeta]) -> List[Dict[str, Any]]:
    out = []
    for s in samples_meta:
        d = {"conv_id": s.conv_id, "label": s.label}
        if s.text_features is not None:
            d["text_features"] = s.text_features
        if s.emo_features is not None:
            d["emo_features"] = s.emo_features
        out.append(d)
    return out


# ----------------------------
# Inference collection
# ----------------------------
@torch.no_grad()
def run_inference_collect(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    all_conv_ids: List[str] = []
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    # attention weights per conv (variable length): store as python list of np arrays
    all_attn_text: List[Optional[np.ndarray]] = []
    all_attn_emo: List[Optional[np.ndarray]] = []

    for batch in loader:
        text_feats = batch.get("text_features", None)
        emo_feats = batch.get("emo_features", None)
        if text_feats is not None:
            text_feats = text_feats.to(device)
        if emo_feats is not None:
            emo_feats = emo_feats.to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        conv_ids = batch["conv_ids"]

        logits, attn = model(text_features=text_feats, emo_features=emo_feats, mask=mask)
        prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred = np.argmax(prob, axis=-1)

        # attn: dict with keys "text"/"emo" -> (B, L)
        # Convert to per-sample arrays trimmed by mask length
        attn_text = attn.get("text", None) if isinstance(attn, dict) else None
        attn_emo = attn.get("emo", None) if isinstance(attn, dict) else None
        if attn_text is not None:
            attn_text = attn_text.detach().cpu().numpy()
        if attn_emo is not None:
            attn_emo = attn_emo.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        for i in range(len(conv_ids)):
            L = int(mask_np[i].sum())
            all_conv_ids.append(conv_ids[i])
            all_labels.append(int(labels[i].cpu().item()))
            all_preds.append(int(pred[i]))
            all_probs.append(prob[i])

            if attn_text is not None:
                all_attn_text.append(attn_text[i, :L].copy())
            else:
                all_attn_text.append(None)

            if attn_emo is not None:
                all_attn_emo.append(attn_emo[i, :L].copy())
            else:
                all_attn_emo.append(None)

    return {
        "conv_ids": all_conv_ids,
        "labels": np.array(all_labels, dtype=int),
        "preds": np.array(all_preds, dtype=int),
        "probs": np.stack(all_probs, axis=0) if len(all_probs) else None,
        "attn_text": all_attn_text,
        "attn_emo": all_attn_emo,
    }


# ----------------------------
# Plotting (matplotlib only, no seaborn)
# ----------------------------
def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_tsne(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: str,
    perplexity: int = 20,
    seed: int = 42,
    metric: str = "euclidean",
):
    plt = _import_matplotlib()
    if X.shape[0] < 5:
        return

    # TSNE stability: cap perplexity
    perp = min(perplexity, max(2, (X.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        metric=metric,
    )
    Z = tsne.fit_transform(X)

    # labels: 0=cn, 1=ad
    idx0 = (y == 0)
    idx1 = (y == 1)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z[idx0, 0], Z[idx0, 1], alpha=0.8, label="CN")
    plt.scatter(Z[idx1, 0], Z[idx1, 1], alpha=0.8, label="AD")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar_with_err(
    mean_a: np.ndarray,
    se_a: np.ndarray,
    mean_b: np.ndarray,
    se_b: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
):
    plt = _import_matplotlib()
    x = np.arange(len(labels))
    width = 0.4

    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, mean_a, width, yerr=se_a, capsize=3, label="CN")
    plt.bar(x + width/2, mean_b, width, yerr=se_b, capsize=3, label="AD")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_box(
    data_cn: np.ndarray,
    data_ad: np.ndarray,
    ylabel: str,
    title: str,
    out_path: str,
):
    plt = _import_matplotlib()
    plt.figure(figsize=(6, 4))
    plt.boxplot([data_cn, data_ad], labels=["CN", "AD"], showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---- UMAP plotting helper ----
def plot_umap(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: str,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
):
    if not HAS_UMAP or X.shape[0] < 5:
        return
    plt = _import_matplotlib()
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    Z = reducer.fit_transform(X)
    idx0 = (y == 0)
    idx1 = (y == 1)
    plt.figure(figsize=(7, 6))
    plt.scatter(Z[idx0, 0], Z[idx0, 1], alpha=0.8, label="CN")
    plt.scatter(Z[idx1, 0], Z[idx1, 1], alpha=0.8, label="AD")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Attention metrics
# ----------------------------
def attn_entropy(attn_w: np.ndarray) -> float:
    # attn_w: (L,), sums to 1
    p = np.clip(attn_w, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def attn_peak_pos(attn_w: np.ndarray) -> float:
    # normalized peak position in [0,1]
    if attn_w is None or len(attn_w) == 0:
        return float("nan")
    idx = int(np.argmax(attn_w))
    if len(attn_w) == 1:
        return 0.0
    return float(idx / (len(attn_w) - 1))


# ----------------------------
# Temporal signature plots
# ----------------------------
def plot_temporal_signature(curves_cn, curves_ad, ylabel, title, out_path):
    plt = _import_matplotlib()
    cn = np.stack(curves_cn)
    ad = np.stack(curves_ad)

    x = np.linspace(0, 1, cn.shape[1])
    plt.figure(figsize=(12, 4))

    plt.plot(x, cn.mean(axis=0), label="CN", color="blue")
    plt.fill_between(
        x,
        cn.mean(axis=0) - cn.std(axis=0),
        cn.mean(axis=0) + cn.std(axis=0),
        alpha=0.2,
    )

    plt.plot(x, ad.mean(axis=0), label="AD", color="red")
    plt.fill_between(
        x,
        ad.mean(axis=0) - ad.std(axis=0),
        ad.mean(axis=0) + ad.std(axis=0),
        alpha=0.2,
    )

    plt.xlabel("Normalized time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_pkl", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True, help="Folder containing best_model_{model_type}_foldK.pt")
    ap.add_argument("--model_type", type=str, default="bilstm", choices=["bilstm", "bigru", "transformer", "pooling"])
    ap.add_argument("--speaker_mode", type=str, default="all", choices=["par", "all"])
    ap.add_argument("--feature_type", type=str, default="text+emo_hidden", choices=["text", "emo_hidden", "text+emo_hidden"])
    ap.add_argument("--max_segments", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="analysis/stage3_best")
    ap.add_argument("--tsne_perplexity", type=int, default=20)
    ap.add_argument("--tsne_metric", type=str, default="euclidean",
                    choices=["euclidean", "cosine", "manhattan"])
    ap.add_argument("--tsne_perplexities", type=int, nargs="+", default=[20],
                    help="List of perplexities for t-SNE (e.g., 5 10 20 30)")
    ap.add_argument("--use_umap", action="store_true",
                    help="Also generate UMAP plots (recommended)")
    ap.add_argument("--umap_neighbors", type=int, default=10)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--umap_metric", type=str, default="cosine")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    set_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load features
    with open(args.features_pkl, "rb") as f:
        data = pickle.load(f)

    conversations = data["conversations"]
    convs_train = [
        c for c in conversations
        if c.get("split") == "train" and c.get("diag_label") in ["ad", "cn"]
    ]

    label2id = {"cn": 0, "ad": 1}
    id2label = {0: "cn", 1: "ad"}

    # 2) Build samples + meta indices for emotion probs/logits
    samples_meta, conv_by_id = build_samples_with_meta(
        convs_train=convs_train,
        label2id=label2id,
        speaker_mode=args.speaker_mode,
        feature_type=args.feature_type,
    )
    if len(samples_meta) == 0:
        raise RuntimeError("No samples after filtering. Check speaker_mode / feature_type.")

    # Plain samples for DataLoader
    all_samples_plain = to_plain_samples(samples_meta)
    labels_np = np.array([s.label for s in samples_meta], dtype=int)

    # dims
    sample0 = samples_meta[0]
    text_dim = sample0.text_features.shape[1] if sample0.text_features is not None else None
    emo_dim = sample0.emo_features.shape[1] if sample0.emo_features is not None else None

    # 3) Recreate KFold splits
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # Collect outputs across folds (each conv appears exactly once in val across folds)
    global_rows = []
    fold_results = []

    # For embedding plots
    conv_embed_text = []
    conv_embed_emo = []
    conv_embed_fused = []
    conv_embed_y = []

    # Emotion distribution stats (mean over segments within conv, using idx selection)
    emo_prob_means_cn = []
    emo_prob_means_ad = []

    # Temporal stats
    vol_cn, vol_ad = [], []
    ent_cn, ent_ad = [], []

    # Temporal curves (normalized time)
    entropy_curves_cn, entropy_curves_ad = [], []
    confidence_curves_cn, confidence_curves_ad = [], []
    alignment_curves_cn, alignment_curves_ad = [], []

    # Global affective transition stats
    C = len(EMOTION_LABELS)
    transition_cn = np.zeros((C, C), dtype=np.int64)
    transition_ad = np.zeros((C, C), dtype=np.int64)

    # Dwell time and switching rate
    dwell_cn, dwell_ad = [], []
    switch_rate_cn, switch_rate_ad = [], []

    # Attention stats
    attnE_text_cn, attnE_text_ad = [], []
    attnE_emo_cn, attnE_emo_ad = [], []
    peak_text_cn, peak_text_ad = [], []
    peak_emo_cn, peak_emo_ad = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np), start=1):
        set_deterministic(args.seed + fold_idx)

        val_samples_plain = [all_samples_plain[i] for i in val_idx]
        val_samples_meta = [samples_meta[i] for i in val_idx]

        val_ds = ConversationSeqDataset(val_samples_plain, args.max_segments)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, args.max_segments),
        )

        best_model_path = os.path.join(args.model_dir, f"best_model_{args.model_type}_fold{fold_idx}.pt")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Missing: {best_model_path}")

        # Build model
        if IMPORTED_FROM_TRAIN_SCRIPT and train_mod is not None:
            model = train_mod.build_model(
                model_type=args.model_type,
                text_dim=text_dim,
                emo_dim=emo_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                num_classes=2,
                max_segments=args.max_segments,
                dropout=0.2,
            ).to(device)
        else:
            # Minimal fallback (bilstm only)
            model = build_model_fallback(
                model_type=args.model_type,
                text_dim=text_dim,
                emo_dim=emo_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=2,
                dropout=0.2,
            ).to(device)

        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Inference collection
        out = run_inference_collect(model, val_loader, device)
        y_true = out["labels"]
        y_pred = out["preds"]

        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="weighted")
        fold_results.append({"fold": fold_idx, "acc": val_acc, "f1": val_f1})

        # Per-conv rows + analysis features
        # Map conv_id -> meta (to access original conv emo_probs/hidden/text)
        meta_by_cid = {m.conv_id: m for m in val_samples_meta}

        for i, cid in enumerate(out["conv_ids"]):
            yt = int(out["labels"][i])
            yp = int(out["preds"][i])
            prob_ad = float(out["probs"][i, 1]) if out["probs"] is not None else float("nan")

            global_rows.append({
                "conv_id": cid,
                "fold": fold_idx,
                "true_id": yt,
                "true_label": id2label[yt],
                "pred_id": yp,
                "pred_label": id2label[yp],
                "p_ad": prob_ad,
            })

            m = meta_by_cid[cid]
            conv = conv_by_id[cid]
            idx = m.idx

            # --- Embeddings (conv-level mean pool) ---
            if m.text_features is not None:
                emb_text = safe_mean_pool(m.text_features)
            else:
                emb_text = None
            if m.emo_features is not None:
                emb_emo = safe_mean_pool(m.emo_features)
            else:
                emb_emo = None

            # fused for visualization (concat)
            if emb_text is not None and emb_emo is not None:
                emb_fused = np.concatenate([emb_text, emb_emo], axis=0)
            elif emb_text is not None:
                emb_fused = emb_text.copy()
            elif emb_emo is not None:
                emb_fused = emb_emo.copy()
            else:
                emb_fused = None

            if emb_text is not None:
                conv_embed_text.append(emb_text)
            if emb_emo is not None:
                conv_embed_emo.append(emb_emo)
            if emb_fused is not None:
                conv_embed_fused.append(emb_fused)
            conv_embed_y.append(yt)

            # --- Emotion distribution from emo_probs if exists ---
            emo_probs = conv.get("emo_probs", None)
            if emo_probs is not None:
                emo_probs_sel = np.asarray(emo_probs)[idx]
                p_mean = np.mean(emo_probs_sel, axis=0)  # (C,)
                if yt == 0:
                    emo_prob_means_cn.append(p_mean)
                else:
                    emo_prob_means_ad.append(p_mean)

                # temporal entropy (mean)
                ent = float(np.mean(entropy_probs(emo_probs_sel, axis=-1)))
                if yt == 0:
                    ent_cn.append(ent)
                else:
                    ent_ad.append(ent)

                # Temporal entropy & confidence curves
                entropy_curve = entropy_probs(emo_probs_sel, axis=-1)
                confidence_curve = np.max(emo_probs_sel, axis=-1)

                entropy_curve_r = resample_curve(entropy_curve)
                conf_curve_r = resample_curve(confidence_curve)

                if yt == 0:
                    entropy_curves_cn.append(entropy_curve_r)
                    confidence_curves_cn.append(conf_curve_r)
                else:
                    entropy_curves_ad.append(entropy_curve_r)
                    confidence_curves_ad.append(conf_curve_r)
            # Emotion–Text alignment drift (cosine similarity)
            if m.text_features is not None and m.emo_features is not None:
                Tm = min(len(m.text_features), len(m.emo_features))
                sims = []
                for t in range(Tm):
                    a = m.text_features[t]
                    b = m.emo_features[t]
                    na = np.linalg.norm(a)
                    nb = np.linalg.norm(b)
                    if na > 0 and nb > 0:
                        sims.append(np.dot(a, b) / (na * nb))
                if len(sims) >= 2:
                    sims_r = resample_curve(np.array(sims))
                    if yt == 0:
                        alignment_curves_cn.append(sims_r)
                    else:
                        alignment_curves_ad.append(sims_r)

            # --- Emotion transition / dwell / switching analysis ---
            if emo_probs is not None:
                emo_seq = np.argmax(emo_probs_sel, axis=-1)  # (T,)

                # transitions
                for a, b in zip(emo_seq[:-1], emo_seq[1:]):
                    if yt == 0:
                        transition_cn[a, b] += 1
                    else:
                        transition_ad[a, b] += 1

                # switch rate
                switches = np.sum(emo_seq[:-1] != emo_seq[1:])
                rate = switches / max(1, (len(emo_seq) - 1))
                if yt == 0:
                    switch_rate_cn.append(rate)
                else:
                    switch_rate_ad.append(rate)

                # dwell time (run-length encoding)
                run_len = 1
                for t in range(1, len(emo_seq)):
                    if emo_seq[t] == emo_seq[t - 1]:
                        run_len += 1
                    else:
                        if yt == 0:
                            dwell_cn.append(run_len)
                        else:
                            dwell_ad.append(run_len)
                        run_len = 1
                # last run
                if yt == 0:
                    dwell_cn.append(run_len)
                else:
                    dwell_ad.append(run_len)

            # --- Temporal volatility from emo_hidden ---
            emo_hid = m.emo_features  # (T, D_emo) already speaker-filtered
            if emo_hid is not None and emo_hid.shape[0] >= 2:
                diffs = emo_hid[1:] - emo_hid[:-1]
                v = float(np.mean(np.linalg.norm(diffs, axis=-1)))
            else:
                v = float("nan")
            if yt == 0:
                vol_cn.append(v)
            else:
                vol_ad.append(v)

            # --- Attention diagnostics (if present) ---
            w_text = out["attn_text"][i]
            w_emo = out["attn_emo"][i]

            if w_text is not None:
                e = attn_entropy(w_text)
                ppos = attn_peak_pos(w_text)
                (attnE_text_cn if yt == 0 else attnE_text_ad).append(e)
                (peak_text_cn if yt == 0 else peak_text_ad).append(ppos)

            if w_emo is not None:
                e = attn_entropy(w_emo)
                ppos = attn_peak_pos(w_emo)
                (attnE_emo_cn if yt == 0 else attnE_emo_ad).append(e)
                (peak_emo_cn if yt == 0 else peak_emo_ad).append(ppos)

    # ----------------------------
    # Save per-conv predictions
    # ----------------------------
    import csv
    pred_csv = os.path.join(args.out_dir, "per_conv_predictions.csv")
    with open(pred_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(global_rows[0].keys()))
        w.writeheader()
        for r in global_rows:
            w.writerow(r)

    # ----------------------------
    # CV Summary file
    # ----------------------------
    mean_acc = float(np.mean([fr["acc"] for fr in fold_results]))
    mean_f1 = float(np.mean([fr["f1"] for fr in fold_results]))
    std_acc = float(np.std([fr["acc"] for fr in fold_results]))
    std_f1 = float(np.std([fr["f1"] for fr in fold_results]))

    summary_txt = os.path.join(args.out_dir, "cv_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=====================================\n")
        f.write(f"📊 {args.k_folds}-FOLD CV SUMMARY (model_type={args.model_type})\n")
        f.write("=====================================\n")
        for fr in fold_results:
            f.write(f"Fold {fr['fold']}: Acc={fr['acc']:.4f}, F1={fr['f1']:.4f}\n")
        f.write("\n📌 Overall (CV):\n")
        f.write(f"  Acc  mean={mean_acc:.4f}, std={std_acc:.4f}\n")
        f.write(f"  F1   mean={mean_f1:.4f}, std={std_f1:.4f}\n")

    # ----------------------------
    # Plots
    # ----------------------------
    y_all = np.array(conv_embed_y, dtype=int)

    # t-SNE / UMAP: emo / text / fused
    for perp in args.tsne_perplexities:
        tag = f"p{perp}_{args.tsne_metric}"

        if len(conv_embed_emo) > 0:
            X = np.stack(conv_embed_emo, axis=0)
            plot_tsne(
                X, y_all,
                title=f"t-SNE (emo_hidden) — perp={perp}, metric={args.tsne_metric}",
                out_path=os.path.join(args.out_dir, f"embedding_tsne_emo_hidden_{tag}.png"),
                perplexity=perp,
                seed=args.seed,
                metric=args.tsne_metric,
            )
            if args.use_umap:
                plot_umap(
                    X, y_all,
                    title="UMAP (emo_hidden)",
                    out_path=os.path.join(args.out_dir, "embedding_umap_emo_hidden.png"),
                    n_neighbors=args.umap_neighbors,
                    min_dist=args.umap_min_dist,
                    metric=args.umap_metric,
                    seed=args.seed,
                )

        if len(conv_embed_text) > 0:
            X = np.stack(conv_embed_text, axis=0)
            plot_tsne(
                X, y_all,
                title=f"t-SNE (text_cls) — perp={perp}, metric={args.tsne_metric}",
                out_path=os.path.join(args.out_dir, f"embedding_tsne_text_cls_{tag}.png"),
                perplexity=perp,
                seed=args.seed,
                metric=args.tsne_metric,
            )
            if args.use_umap:
                plot_umap(
                    X, y_all,
                    title="UMAP (text_cls)",
                    out_path=os.path.join(args.out_dir, "embedding_umap_text_cls.png"),
                    n_neighbors=args.umap_neighbors,
                    min_dist=args.umap_min_dist,
                    metric=args.umap_metric,
                    seed=args.seed,
                )

        if len(conv_embed_fused) > 0:
            X = np.stack(conv_embed_fused, axis=0)
            plot_tsne(
                X, y_all,
                title=f"t-SNE (fused text+emo) — perp={perp}, metric={args.tsne_metric}",
                out_path=os.path.join(args.out_dir, f"embedding_tsne_fused_{tag}.png"),
                perplexity=perp,
                seed=args.seed,
                metric=args.tsne_metric,
            )
            if args.use_umap:
                plot_umap(
                    X, y_all,
                    title="UMAP (fused text+emo)",
                    out_path=os.path.join(args.out_dir, "embedding_umap_fused.png"),
                    n_neighbors=args.umap_neighbors,
                    min_dist=args.umap_min_dist,
                    metric=args.umap_metric,
                    seed=args.seed,
                )

    # Emotion distribution bar plot
    if len(emo_prob_means_cn) > 0 and len(emo_prob_means_ad) > 0:
        P_cn = np.stack(emo_prob_means_cn, axis=0)
        P_ad = np.stack(emo_prob_means_ad, axis=0)
        mean_cn = P_cn.mean(axis=0)
        mean_ad = P_ad.mean(axis=0)
        se_cn = P_cn.std(axis=0) / np.sqrt(P_cn.shape[0])
        se_ad = P_ad.std(axis=0) / np.sqrt(P_ad.shape[0])

        C = mean_cn.shape[0]
        if len(EMOTION_LABELS) == C:
            emo_labels = EMOTION_LABELS
        else:
            emo_labels = [f"e{i}" for i in range(C)]
        plot_bar_with_err(
            mean_cn, se_cn, mean_ad, se_ad,
            labels=emo_labels,
            title="Emotion distribution (mean emo_probs) — CN vs AD",
            out_path=os.path.join(args.out_dir, "emotion_distribution_ad_vs_cn.png"),
        )

    # Temporal stats: volatility + entropy
    vol_cn_np = np.array([v for v in vol_cn if np.isfinite(v)], dtype=float)
    vol_ad_np = np.array([v for v in vol_ad if np.isfinite(v)], dtype=float)
    if len(vol_cn_np) > 0 and len(vol_ad_np) > 0:
        plot_box(
            vol_cn_np, vol_ad_np,
            ylabel="Mean ||Δ emo_hidden||",
            title="Temporal volatility of emo_hidden (per conversation)",
            out_path=os.path.join(args.out_dir, "temporal_volatility.png"),
        )

    ent_cn_np = np.array([v for v in ent_cn if np.isfinite(v)], dtype=float)
    ent_ad_np = np.array([v for v in ent_ad if np.isfinite(v)], dtype=float)
    if len(ent_cn_np) > 0 and len(ent_ad_np) > 0:
        plot_box(
            ent_cn_np, ent_ad_np,
            ylabel="Mean entropy(emo_probs)",
            title="Temporal uncertainty of emotion (per conversation)",
            out_path=os.path.join(args.out_dir, "temporal_entropy.png"),
        )

    # Attention stats
    # entropy
    def _arr(a): return np.array([x for x in a if np.isfinite(x)], dtype=float)

    if len(attnE_emo_cn) > 0 and len(attnE_emo_ad) > 0:
        plot_box(
            _arr(attnE_emo_cn), _arr(attnE_emo_ad),
            ylabel="Entropy(attn)",
            title="Attention entropy (emo stream) — CN vs AD",
            out_path=os.path.join(args.out_dir, "attention_entropy_emo.png"),
        )
    if len(attnE_text_cn) > 0 and len(attnE_text_ad) > 0:
        plot_box(
            _arr(attnE_text_cn), _arr(attnE_text_ad),
            ylabel="Entropy(attn)",
            title="Attention entropy (text stream) — CN vs AD",
            out_path=os.path.join(args.out_dir, "attention_entropy_text.png"),
        )

    # peak position
    if len(peak_emo_cn) > 0 and len(peak_emo_ad) > 0:
        plot_box(
            _arr(peak_emo_cn), _arr(peak_emo_ad),
            ylabel="Peak position (0=start, 1=end)",
            title="Attention peak position (emo stream) — CN vs AD",
            out_path=os.path.join(args.out_dir, "attention_peakpos_emo.png"),
        )
    if len(peak_text_cn) > 0 and len(peak_text_ad) > 0:
        plot_box(
            _arr(peak_text_cn), _arr(peak_text_ad),
            ylabel="Peak position (0=start, 1=end)",
            title="Attention peak position (text stream) — CN vs AD",
            out_path=os.path.join(args.out_dir, "attention_peakpos_text.png"),
        )

    # ----------------------------
    # Transition matrix divergence (JS)
    # ----------------------------
    js_div = js_divergence_matrix(transition_cn, transition_ad)
    with open(os.path.join(args.out_dir, "transition_js_divergence.txt"), "w") as f:
        f.write(f"Jensen–Shannon divergence (CN vs AD): {js_div:.4f}\n")

    # ----------------------------
    # Global affective transition matrices
    def plot_transition_matrix(M: np.ndarray, labels: List[str], title: str, out_path: str):
        plt = _import_matplotlib()
        # row-normalize
        M = M.astype(float)
        row_sum = M.sum(axis=1, keepdims=True)
        M = np.divide(M, row_sum, out=np.zeros_like(M), where=row_sum > 0)

        plt.figure(figsize=(7, 6))
        im = plt.imshow(M, cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)

        plt.xlabel("Next Emotion")
        plt.ylabel("Current Emotion")


        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    plot_transition_matrix(
        transition_cn,
        EMOTION_LABELS,
        title="Global Affective Transition Matrix (CN)",
        out_path=os.path.join(args.out_dir, "transition_matrix_cn.png"),
    )

    plot_transition_matrix(
        transition_ad,
        EMOTION_LABELS,
        title="Global Affective Transition Matrix (AD)",
        out_path=os.path.join(args.out_dir, "transition_matrix_ad.png"),
    )

    # Switch rate
    if len(switch_rate_cn) > 0 and len(switch_rate_ad) > 0:
        plot_box(
            np.array(switch_rate_cn),
            np.array(switch_rate_ad),
            ylabel="Emotion switching rate",
            title="Emotion switching rate per monologue",
            out_path=os.path.join(args.out_dir, "emotion_switch_rate.png"),
        )

    # Dwell time
    if len(dwell_cn) > 0 and len(dwell_ad) > 0:
        plot_box(
            np.array(dwell_cn),
            np.array(dwell_ad),
            ylabel="Dwell length (segments)",
            title="Emotion dwell time distribution",
            out_path=os.path.join(args.out_dir, "emotion_dwell_time.png"),
        )


    if len(entropy_curves_cn) > 0 and len(entropy_curves_ad) > 0:
        plot_temporal_signature(
            entropy_curves_cn,
            entropy_curves_ad,
            ylabel="Entropy(emo_probs)",
            title="Temporal Emotion Uncertainty Signature",
            out_path=os.path.join(args.out_dir, "temporal_entropy_signature.png"),
        )

    if len(confidence_curves_cn) > 0 and len(confidence_curves_ad) > 0:
        plot_temporal_signature(
            confidence_curves_cn,
            confidence_curves_ad,
            ylabel="Max emotion probability",
            title="Temporal Emotion Confidence Signature",
            out_path=os.path.join(args.out_dir, "temporal_confidence_signature.png"),
        )

    if len(alignment_curves_cn) > 0 and len(alignment_curves_ad) > 0:
        plot_temporal_signature(
            alignment_curves_cn,
            alignment_curves_ad,
            ylabel="Cosine(text, emotion)",
            title="Emotion–Text Alignment Drift",
            out_path=os.path.join(args.out_dir, "emotion_text_alignment.png"),
        )

    # ----------------------------
    # Numeric analysis summary file
    # ----------------------------
    stats_txt = os.path.join(args.out_dir, "analysis_numeric_summary.txt")
    with open(stats_txt, "w", encoding="utf-8") as f:
        # Header
        f.write("Stage-3 Best Model – Numeric Analysis Summary\n")
        f.write("===========================================\n\n")

        # CV performance
        f.write("CV PERFORMANCE\n")
        f.write(f"Acc mean={mean_acc:.4f}, std={std_acc:.4f}\n")
        f.write(f"F1  mean={mean_f1:.4f}, std={std_f1:.4f}\n\n")

        # Emotion distribution (means per class)
        if len(emo_prob_means_cn) > 0 and len(emo_prob_means_ad) > 0:
            P_cn = np.stack(emo_prob_means_cn, axis=0)
            P_ad = np.stack(emo_prob_means_ad, axis=0)
            mean_cn = P_cn.mean(axis=0)
            mean_ad = P_ad.mean(axis=0)
            f.write("EMOTION DISTRIBUTION (mean emo_probs)\n")
            for i, lab in enumerate(EMOTION_LABELS):
                f.write(f"{lab}: CN={mean_cn[i]:.4f}, AD={mean_ad[i]:.4f}\n")
            f.write("\n")

        # Temporal volatility & entropy
        vol_cn_np = np.array([v for v in vol_cn if np.isfinite(v)], dtype=float)
        vol_ad_np = np.array([v for v in vol_ad if np.isfinite(v)], dtype=float)
        if len(vol_cn_np) > 0 and len(vol_ad_np) > 0:
            f.write("TEMPORAL VOLATILITY (emo_hidden)\n")
            f.write(f"CN mean={vol_cn_np.mean():.4f}, std={vol_cn_np.std():.4f}\n")
            f.write(f"AD mean={vol_ad_np.mean():.4f}, std={vol_ad_np.std():.4f}\n\n")

        ent_cn_np = np.array([v for v in ent_cn if np.isfinite(v)], dtype=float)
        ent_ad_np = np.array([v for v in ent_ad if np.isfinite(v)], dtype=float)
        if len(ent_cn_np) > 0 and len(ent_ad_np) > 0:
            f.write("TEMPORAL EMOTION ENTROPY\n")
            f.write(f"CN mean={ent_cn_np.mean():.4f}, std={ent_cn_np.std():.4f}\n")
            f.write(f"AD mean={ent_ad_np.mean():.4f}, std={ent_ad_np.std():.4f}\n\n")

        # Switching rate & dwell time
        if len(switch_rate_cn) > 0 and len(switch_rate_ad) > 0:
            f.write("EMOTION SWITCH RATE\n")
            f.write(f"CN mean={np.mean(switch_rate_cn):.4f}, std={np.std(switch_rate_cn):.4f}\n")
            f.write(f"AD mean={np.mean(switch_rate_ad):.4f}, std={np.std(switch_rate_ad):.4f}\n\n")

        if len(dwell_cn) > 0 and len(dwell_ad) > 0:
            f.write("EMOTION DWELL TIME (segments)\n")
            f.write(f"CN mean={np.mean(dwell_cn):.4f}, std={np.std(dwell_cn):.4f}\n")
            f.write(f"AD mean={np.mean(dwell_ad):.4f}, std={np.std(dwell_ad):.4f}\n\n")

        # Attention diagnostics
        if len(attnE_emo_cn) > 0 and len(attnE_emo_ad) > 0:
            f.write("ATTENTION ENTROPY (EMO)\n")
            f.write(f"CN mean={np.mean(attnE_emo_cn):.4f}\n")
            f.write(f"AD mean={np.mean(attnE_emo_ad):.4f}\n\n")

        if len(attnE_text_cn) > 0 and len(attnE_text_ad) > 0:
            f.write("ATTENTION ENTROPY (TEXT)\n")
            f.write(f"CN mean={np.mean(attnE_text_cn):.4f}\n")
            f.write(f"AD mean={np.mean(attnE_text_ad):.4f}\n\n")

        # Transition matrix divergence
        f.write("GLOBAL AFFECTIVE TRANSITION\n")
        f.write(f"JS divergence (CN vs AD) = {js_div:.4f}\n\n")

        # Footer
        f.write("END OF SUMMARY\n")

    print(f"📄 Numeric analysis saved to: {stats_txt}")

    # ----------------------------
    # Print quick console summary
    # ----------------------------
    print(f"✅ Saved: {pred_csv}")
    print(f"✅ Saved: {summary_txt}")
    print(f"📌 Overall (CV): Acc mean={mean_acc:.4f}±{std_acc:.4f}, F1 mean={mean_f1:.4f}±{std_f1:.4f}")
    print(f"🖼 Figures saved under: {args.out_dir}")

    # Optional: print classification report using aggregated predictions
    y_true_all = np.array([r["true_id"] for r in global_rows], dtype=int)
    y_pred_all = np.array([r["pred_id"] for r in global_rows], dtype=int)
    print("\n📈 Aggregated (all folds, val-only) report:")
    print(classification_report(y_true_all, y_pred_all, target_names=["cn", "ad"], digits=4))


if __name__ == "__main__":
    main()