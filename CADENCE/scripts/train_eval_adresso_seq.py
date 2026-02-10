#!/usr/bin/env python3
"""
ADReSSo21 için konuşma-seviyesinde AD / CN sınıflandırma (Stratified K-fold CV).

Girdi:
  extract_adresso_text_emo_features.py çıktısı (.pkl), örn:
    features/adresso21_text_emo_segments.pkl

Her konuşma kaydı (önerilen format):
  {
    "conv_id": str,
    "split": "train" | "test",
    "diag_label": "ad" | "cn" | None,
    "speakers":   [ "PAR", "INV", ... ],
    "texts":      [ "...", "..." ],

    # --- Text stream ---
    "text_cls":   np.ndarray (T, H_text),     # Pretrained language model (örn. DeBERTa) CLS temsilleri

    # --- Emotion stream ---
    "emo_hidden": np.ndarray (T, H_emo),      # Emotion teacher modelinden gizli temsiller (CLS / pooled)
    "emo_logits": np.ndarray (T, C),          # (opsiyonel) emotion logits
    "emo_probs":  np.ndarray (T, C),          # (opsiyonel) softmax(logits)
  }

Bu script:
  - Sadece split='train' ve diag_label ∈ {ad, cn} olan konuşmaları kullanır
  - Stratified K-fold cross-validation uygular
  - Konuşma-seviyesinde sınıflandırma yapar (segment-level input, conversation-level decision)

Temel tasarım ilkeleri:
  - Multi-stream mimari:
      * Text ve emotion özellikleri ayrı kanallar (stream) olarak işlenir
      * Her stream kendi temporal encoder’ına (BiGRU / BiLSTM / Transformer) sahiptir
  - Late fusion:
      * Modalite-temsilleri zaman boyutunda değil, konuşma seviyesinde birleştirilir
      * Erken füzyon (segment-level concat) bilinçli olarak kullanılmaz
  - Modüler ve ablation-dostu yapı:
      * Text-only
      * Emotion-only
      * Text + Emotion (dual-stream)

Parametreler:
  * speaker_mode:
        "par"   → sadece hasta (PAR) segmentleri
        "all"   → PAR + INV segmentleri
  * feature_type:
        "text"              → sadece text stream
        "emo_hidden"        → sadece emotion stream
        "text+emo_hidden"   → dual-stream (text + emotion)
  * model_type:
        "bigru"         → Multi-stream BiGRU + attention (varsayılan)
        "bilstm"        → Multi-stream BiLSTM + attention
        "transformer"   → Multi-stream Transformer encoder + attention pooling
        "pooling"       → Multi-stream mean pooling + MLP (baseline)

Eğitim:
  - Varsayılan loss: CrossEntropy
  - Optimizer: AdamW
  - Opsiyonel öğrenme oranı zamanlaması:
      * cosine decay + linear warmup
"""

import os
import argparse
import pickle
import math
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report


# -------------------------------------------------
# Deterministiklik
# -------------------------------------------------
def set_deterministic(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Deterministic mode activated (seed={seed})")


# -------------------------------------------------
# Dataset & collate
# -------------------------------------------------
class ConversationSeqDataset(Dataset):
    """
    samples: list of dict
      {
        "conv_id": str,
        "label": int,
        "text_features": np.ndarray (T, D_text)  [optional]
        "emo_features":  np.ndarray (T, D_emo)   [optional]
      }
    """
    def __init__(self, samples: List[Dict[str, Any]], max_segments: int):
        self.samples = samples
        self.max_segments = max_segments
        print(f"📦 ConversationSeqDataset: {len(self.samples)} samples, max_segments={self.max_segments}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, max_segments: int):
    """
    Multi-stream collate:
      Returns:
        text_features: (B, L, D_text) or None
        emo_features:  (B, L, D_emo)  or None
        mask:          (B, L)
        labels:        (B,)
    """
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

        if has_text:
            x = item["text_features"]
        else:
            x = item["emo_features"]

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


# -------------------------------------------------
# Model: Multi-Stream BiGRU + Attention (TEXT / EMO)
# -------------------------------------------------
class BiGRUAttentionClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int | None,
        emo_dim: int | None,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
        attn_tau: float = 1.5,
    ):
        super().__init__()

        assert text_dim is not None or emo_dim is not None, \
            "At least one of text_dim or emo_dim must be provided."

        self.use_text = text_dim is not None
        self.use_emo = emo_dim is not None

        if self.use_text:
            self.text_gru = nn.GRU(
                text_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.text_attn = nn.Linear(hidden_dim * 2, 1)

        if self.use_emo:
            self.emo_gru = nn.GRU(
                emo_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.emo_attn = nn.Linear(hidden_dim * 2, 1)

        fusion_dim = 0
        if self.use_text:
            fusion_dim += hidden_dim * 2
        if self.use_emo:
            fusion_dim += hidden_dim * 2

        self.dropout = nn.Dropout(dropout)
        self.temporal_dropout = nn.Dropout(dropout)
        # Gated fusion (conversation-level)
        if self.use_text and self.use_emo:
            self.gate = nn.Linear(fusion_dim, hidden_dim * 2)
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        else:
            # single-stream fallback
            self.classifier = nn.Linear(fusion_dim, num_classes)

        self.attn_tau = attn_tau

    def _apply_attention(self, seq_out, mask, attn_layer):
        """
        seq_out: (B, L, 2H)
        mask:    (B, L)
        """
        scores = attn_layer(seq_out).squeeze(-1)          # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores / self.attn_tau, dim=-1)   # (B, L)
        context = torch.sum(weights.unsqueeze(-1) * seq_out, dim=1)  # (B, 2H)
        return context, weights

    def forward(self, text_features=None, emo_features=None, mask=None):
        """
        text_features: (B, L, D_text) or None
        emo_features:  (B, L, D_emo)  or None
        mask:          (B, L)
        """
        contexts = []
        attn_weights = {}

        if self.use_text:
            assert text_features is not None
            text_out, _ = self.text_gru(text_features)
            text_out = self.temporal_dropout(text_out)
            text_ctx, text_attn = self._apply_attention(text_out, mask, self.text_attn)
            contexts.append(text_ctx)
            attn_weights["text"] = text_attn

        if self.use_emo:
            assert emo_features is not None
            emo_out, _ = self.emo_gru(emo_features)
            emo_out = self.temporal_dropout(emo_out)
            emo_ctx, emo_attn = self._apply_attention(emo_out, mask, self.emo_attn)
            contexts.append(emo_ctx)
            attn_weights["emo"] = emo_attn

        if self.use_text and self.use_emo:
            text_ctx, emo_ctx = contexts
            concat_ctx = torch.cat([text_ctx, emo_ctx], dim=-1)
            alpha = torch.sigmoid(self.gate(concat_ctx))      # (B, 2H)
            fused = alpha * text_ctx + (1.0 - alpha) * emo_ctx
        else:
            fused = contexts[0]

        fused = self.dropout(fused)
        logits = self.classifier(fused)

        return logits, attn_weights


# -------------------------------------------------
# Model: Multi-Stream BiLSTM + Attention (TEXT / EMO)
# -------------------------------------------------
class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int | None,
        emo_dim: int | None,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
        attn_tau=1.5,
    ):
        super().__init__()

        assert text_dim is not None or emo_dim is not None, \
            "At least one of text_dim or emo_dim must be provided."

        self.use_text = text_dim is not None
        self.use_emo = emo_dim is not None

        if self.use_text:
            self.text_lstm = nn.LSTM(
                text_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.text_attn = nn.Linear(hidden_dim * 2, 1)

        if self.use_emo:
            self.emo_lstm = nn.LSTM(
                emo_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
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
        self.attn_tau = attn_tau

    def _apply_attention(self, seq_out, mask, attn_layer):
        """
        seq_out: (B, L, 2H)
        mask:    (B, L)
        """
        scores = attn_layer(seq_out).squeeze(-1)          # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores / self.attn_tau, dim=-1)         # (B, L)
        context = torch.sum(weights.unsqueeze(-1) * seq_out, dim=1)  # (B, 2H)
        return context, weights

    def forward(self, text_features=None, emo_features=None, mask=None):
        """
        text_features: (B, L, D_text) or None
        emo_features:  (B, L, D_emo)  or None
        mask:          (B, L)
        """
        contexts = []
        attn_weights = {}

        if self.use_text:
            assert text_features is not None
            text_out, _ = self.text_lstm(text_features)
            text_out = self.temporal_dropout(text_out)
            text_ctx, text_attn = self._apply_attention(text_out, mask, self.text_attn)
            contexts.append(text_ctx)
            attn_weights["text"] = text_attn

        if self.use_emo:
            assert emo_features is not None
            emo_out, _ = self.emo_lstm(emo_features)
            emo_out = self.temporal_dropout(emo_out)
            emo_ctx, emo_attn = self._apply_attention(emo_out, mask, self.emo_attn)
            contexts.append(emo_ctx)
            attn_weights["emo"] = emo_attn

        fused = torch.cat(contexts, dim=-1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        return logits, attn_weights


# -------------------------------------------------
# Model: Multi-Stream Transformer Encoder + Attention Pooling
# -------------------------------------------------
class TransformerSeqClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int | None,
        emo_dim: int | None,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        max_len: int,
        dropout: float = 0.1,
        attn_tau: float = 1.5,
    ):
        super().__init__()

        assert text_dim is not None or emo_dim is not None, \
            "At least one of text_dim or emo_dim must be provided."

        self.use_text = text_dim is not None
        self.use_emo = emo_dim is not None

        # -------- TEXT STREAM --------
        if self.use_text:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.text_pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden_dim))

            text_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=num_layers)
            self.text_attn = nn.Linear(hidden_dim, 1)

        # -------- EMO STREAM --------
        if self.use_emo:
            self.emo_proj = nn.Linear(emo_dim, hidden_dim)
            self.emo_pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden_dim))

            emo_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.emo_encoder = nn.TransformerEncoder(emo_layer, num_layers=num_layers)
            self.emo_attn = nn.Linear(hidden_dim, 1)

        fusion_dim = 0
        if self.use_text:
            fusion_dim += hidden_dim
        if self.use_emo:
            fusion_dim += hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.temporal_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.attn_tau = attn_tau

    def _attn_pool(self, enc_out, mask, attn_layer):
        """
        enc_out: (B, L, H)
        mask:    (B, L)
        """
        scores = attn_layer(enc_out).squeeze(-1)          # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores / self.attn_tau, dim=-1)   # (B, L)
        context = torch.sum(weights.unsqueeze(-1) * enc_out, dim=1)  # (B, H)
        return context, weights

    def forward(self, text_features=None, emo_features=None, mask=None):
        """
        text_features: (B, L, D_text) or None
        emo_features:  (B, L, D_emo)  or None
        mask:          (B, L)
        """
        contexts = []
        attn_weights = {}

        src_key_padding_mask = ~mask  # Transformer convention

        if self.use_text:
            assert text_features is not None
            x = self.text_proj(text_features)
            x = x + self.text_pos_emb[:, : x.size(1), :]
            enc = self.text_encoder(x, src_key_padding_mask=src_key_padding_mask)
            enc = self.temporal_dropout(enc)
            ctx, attn = self._attn_pool(enc, mask, self.text_attn)
            contexts.append(ctx)
            attn_weights["text"] = attn

        if self.use_emo:
            assert emo_features is not None
            x = self.emo_proj(emo_features)
            x = x + self.emo_pos_emb[:, : x.size(1), :]
            enc = self.emo_encoder(x, src_key_padding_mask=src_key_padding_mask)
            enc = self.temporal_dropout(enc)
            ctx, attn = self._attn_pool(enc, mask, self.emo_attn)
            contexts.append(ctx)
            attn_weights["emo"] = attn

        fused = torch.cat(contexts, dim=-1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        return logits, attn_weights

# -------------------------------------------------
# Model: Multi-Stream Mean Pooling + MLP (baseline)
# -------------------------------------------------
class PoolingClassifier(nn.Module):
    """
    Multi-stream pooling baseline:
      - Text ve emo stream'ler ayrı ayrı mean pooling
      - Late fusion (concat)
      - 2 katmanlı MLP
    """
    def __init__(
        self,
        text_dim: int | None,
        emo_dim: int | None,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        attn_tau: float = 1.5,
    ):
        super().__init__()

        assert text_dim is not None or emo_dim is not None, \
            "At least one of text_dim or emo_dim must be provided."

        self.use_text = text_dim is not None
        self.use_emo = emo_dim is not None

        fusion_dim = 0
        if self.use_text:
            fusion_dim += text_dim
        if self.use_emo:
            fusion_dim += emo_dim

        self.fc1 = nn.Linear(fusion_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.temporal_dropout = nn.Dropout(dropout)
        self.attn_tau = attn_tau

    def _mean_pool(self, feats, mask):
        """
        feats: (B, L, D)
        mask:  (B, L)
        """
        mask_f = mask.unsqueeze(-1).float()           # (B, L, 1)
        summed = torch.sum(feats * mask_f, dim=1)    # (B, D)
        counts = torch.clamp(mask_f.sum(dim=1), min=1e-6)  # (B, 1)
        return summed / counts                        # (B, D)

    def forward(self, text_features=None, emo_features=None, mask=None):
        """
        text_features: (B, L, D_text) or None
        emo_features:  (B, L, D_emo)  or None
        mask:          (B, L)
        """
        pooled_feats = []

        # Apply temporal dropout before pooling (for regularization consistency)
        if self.use_text:
            assert text_features is not None
            text_features = self.temporal_dropout(text_features)
            pooled_text = self._mean_pool(text_features, mask)
            pooled_feats.append(pooled_text)

        if self.use_emo:
            assert emo_features is not None
            emo_features = self.temporal_dropout(emo_features)
            pooled_emo = self._mean_pool(emo_features, mask)
            pooled_feats.append(pooled_emo)

        fused = torch.cat(pooled_feats, dim=-1)
        x = self.fc1(fused)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        # pooling modelinde attention yok
        return logits, None


# -------------------------------------------------
# Train / eval helpers
# -------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        text_feats = batch.get("text_features", None)
        emo_feats = batch.get("emo_features", None)

        if text_feats is not None:
            text_feats = text_feats.to(device)
        if emo_feats is not None:
            emo_feats = emo_feats.to(device)

        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, _ = model(
            text_features=text_feats,
            emo_features=emo_feats,
            mask=mask,
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(list(preds))
        all_labels.extend(list(labels.cpu().numpy()))

    from sklearn.metrics import precision_score, recall_score
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1, precision, recall


@torch.no_grad()
def eval_one_epoch(model, loader, device, return_details: bool = False):
    """
    return_details=False:
        -> avg_loss, acc, f1, all_preds, all_labels
    return_details=True:
        -> avg_loss, acc, f1, all_preds, all_labels, all_conv_ids
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_conv_ids = []

    criterion = nn.CrossEntropyLoss()

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

        logits, _ = model(
            text_features=text_feats,
            emo_features=emo_feats,
            mask=mask,
        )
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(list(preds))
        all_labels.extend(list(labels.cpu().numpy()))
        all_conv_ids.extend(list(conv_ids))

    from sklearn.metrics import precision_score, recall_score
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

    if return_details:
        return avg_loss, acc, f1, precision, recall, all_preds, all_labels, all_conv_ids
    else:
        return avg_loss, acc, f1, precision, recall, all_preds, all_labels


# -------------------------------------------------
# LR scheduler: cosine + warmup
# -------------------------------------------------
def build_cosine_warmup_scheduler(optimizer, num_training_steps: int, warmup_ratio: float = 0.1):
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


# -------------------------------------------------
# Yardımcı: features & speaker filtresi
# -------------------------------------------------
def build_samples_from_conversations(
    conversations,
    label2id,
    speaker_mode: str = "par",
    feature_type: str = "both",
) -> List[Dict[str, Any]]:
    """
    speaker_mode: "par" (sadece PAR) veya "all" (PAR + INV vs.)
    feature_type:
        "text"
        "emo_hidden"
        "text+emo_hidden"
    """
    samples = []

    for conv in conversations:
        diag = conv.get("diag_label")
        if diag not in label2id:
            continue

        speakers = conv.get("speakers", None)
        text_cls = conv.get("text_cls", None)
        emo_hidden = conv.get("emo_hidden", None)

        if speakers is None:
            continue

        speakers = [str(s).upper() for s in speakers]

        # Sadeleştirilmiş feature kullanımı (multi-stream hazırlığı)
        ft = feature_type
        need_text = ft in ["text", "text+emo_hidden"]
        need_emo_hidden = ft in ["emo_hidden", "text+emo_hidden"]

        # Gerekli alanlardan biri yoksa bu konuşmayı atla
        if need_text and text_cls is None:
            continue
        if need_emo_hidden and emo_hidden is None:
            continue

        # T uzunluklarını kontrol et
        T = len(speakers)
        arrays = []
        if need_text:
            arrays.append(np.asarray(text_cls))
        if need_emo_hidden:
            arrays.append(np.asarray(emo_hidden))

        if any(a.shape[0] != T for a in arrays):
            # Tutarsız uzunluk varsa konuşmayı at
            continue

        # 1) Speaker filtresi
        if speaker_mode == "par":
            idx = [i for i, s in enumerate(speakers) if s == "PAR"]
        elif speaker_mode == "all":
            idx = list(range(T))
        else:
            raise ValueError(f"Unknown speaker_mode: {speaker_mode}")

        if len(idx) == 0:
            continue

        # 2) Seçilen segmentleri al (multi-stream: concat YOK)
        sample = {
            "conv_id": conv["conv_id"],
            "label": label2id[diag],
        }

        if need_text:
            sample["text_features"] = np.asarray(text_cls)[idx].astype(np.float32)

        if need_emo_hidden:
            sample["emo_features"] = np.asarray(emo_hidden)[idx].astype(np.float32)

        samples.append(sample)

    print(
        f"🧾 Built {len(samples)} samples with "
        f"speaker_mode='{speaker_mode}', feature_type='{feature_type}'"
    )
    return samples


# -------------------------------------------------
# Model builder (parametrik)
# -------------------------------------------------
def build_model(
    model_type: str,
    text_dim: int | None,
    emo_dim: int | None,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    num_classes: int,
    max_segments: int,
    dropout: float = 0.2,
):
    model_type = model_type.lower()

    if model_type == "bigru":
        print("🧠 Using Dual-Stream BiGRU + attention model.")
        return BiGRUAttentionClassifier(
            text_dim=text_dim,
            emo_dim=emo_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            attn_tau=1.3,
        )

    elif model_type == "bilstm":
        print("🧠 Using Dual-Stream BiLSTM + attention model.")
        return BiLSTMAttentionClassifier(
            text_dim=text_dim,
            emo_dim=emo_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            attn_tau=1.5,
        )

    elif model_type == "transformer":
        print("🧠 Using Dual-Stream Transformer encoder + attention pooling model.")
        return TransformerSeqClassifier(
            text_dim=text_dim,
            emo_dim=emo_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            max_len=max_segments,
            dropout=dropout,
        )

    elif model_type == "pooling":
        print("🧠 Using Dual-Stream mean-pooling + MLP classifier.")
        return PoolingClassifier(
            text_dim=text_dim,
            emo_dim=emo_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# -------------------------------------------------
# main (K-fold CV)
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train (default) or eval. In eval mode, skips training and only evaluates existing best_model_*.pt files.",
    )
    parser.add_argument(
        "--features_pkl",
        type=str,
        required=True,
        help="extract_adresso_features.py çıktısı (örn: features/adresso21_text_emo_segments.pkl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_model/adresso_seq_cv",
        help="Modellerin kayıt klasörü",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="RNN/Transformer katman sayısı",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Transformer encoder için head sayısı (sadece model_type=transformer)",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=64,
        help="Konuşma başına maksimum segment sayısı (pad/trim)",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Stratified K-Fold sayısı (default: 5)",
    )
    parser.add_argument(
        "--speaker_mode",
        type=str,
        default="all",
        choices=["par", "all"],
        help="Hangi konuşmacı segmentleri kullanılacak: 'par' veya 'all'",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="both",
        choices=[
            "text",
            "emo_hidden",
            "text+emo_hidden"
        ],
        help="Hangi feature kullanılacak",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bigru",
        choices=["bigru", "bilstm", "transformer", "pooling"],
        help="Kullanılacak model tipi",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="LR scheduler tipi (none veya cosine)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Cosine scheduler için warmup oranı (0-1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    # ⬇⬇⬇ YENİ: fold bazlı prediction'ları CSV'ye kaydet
    parser.add_argument(
        "--save_fold_predictions",
        action="store_true",
        help="Verilirse her fold için validation conv_id/true/pred CSV'ye kaydedilir.",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Validation F1 tabanlı early stopping kullan",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epoch sayısı)",
    )

    args = parser.parse_args()
    if args.mode == "eval":
        assert os.path.isdir(args.output_dir), "output_dir must exist for eval mode"
    set_deterministic(args.seed)

    if args.mode == "train":
        os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")
    print(f"🔧 model_type={args.model_type}, feature_type={args.feature_type}, speaker_mode={args.speaker_mode}")

    # ---------------------------------------------------
    # 1) Özellikleri yükle
    # ---------------------------------------------------
    print(f"📂 Loading features from: {args.features_pkl}")
    with open(args.features_pkl, "rb") as f:
        data = pickle.load(f)

    conversations = data["conversations"]
    print(f"  → Total conversations in pkl: {len(conversations)}")

    # Sadece train split ve AD/CN etiketli konuşmaları al
    convs_train = [
        c for c in conversations
        if c.get("split") == "train" and c.get("diag_label") in ["ad", "cn"]
    ]
    print(f"  → Labeled train conversations (ad/cn): {len(convs_train)}")

    label2id = {"cn": 0, "ad": 1}
    id2label = {0: "cn", 1: "ad"}

    # ---------------------------------------------------
    # 2) Speaker & feature filtresi ile sample listesi
    # ---------------------------------------------------
    all_samples = build_samples_from_conversations(
        convs_train,
        label2id=label2id,
        speaker_mode=args.speaker_mode,
        feature_type=args.feature_type,
    )
    if len(all_samples) == 0:
        print("❌ No samples after filtering; check speaker_mode / feature_type.")
        return

    labels = [s["label"] for s in all_samples]
    labels_np = np.array(labels)

    # input_dim tespiti (multi-stream)
    sample0 = all_samples[0]

    # Multi-stream input dimensions
    text_dim = sample0["text_features"].shape[1] if "text_features" in sample0 else None
    emo_dim  = sample0["emo_features"].shape[1]  if "emo_features" in sample0 else None

    if args.feature_type == "text":
        print(f"🔧 Using TEXT stream only, text_dim={text_dim}")
    elif args.feature_type == "emo_hidden":
        print(f"🔧 Using EMO stream only, emo_dim={emo_dim}")
    elif args.feature_type == "text+emo_hidden":
        print(f"🔧 Using TEXT+EMO (dual-stream), text_dim={text_dim}, emo_dim={emo_dim}")
    else:
        raise ValueError(f"Unsupported feature_type for multi-stream: {args.feature_type}")

    # ---------------------------------------------------
    # 3) K-fold stratified CV
    # ---------------------------------------------------
    skf = StratifiedKFold(
        n_splits=args.k_folds,
        shuffle=True,
        random_state=args.seed,
    )

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np), start=1):
        print(f"\n======================")
        print(f"   📁 Fold {fold_idx}/{args.k_folds}")
        print(f"======================")

        # Fold'a özel determinism (opsiyonel ama tutarlı olur)
        set_deterministic(args.seed + fold_idx)

        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        train_dataset = ConversationSeqDataset(train_samples, args.max_segments)
        val_dataset = ConversationSeqDataset(val_samples, args.max_segments)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, args.max_segments),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, args.max_segments),
        )

        # Multi-stream input dimensions
        text_dim = sample0["text_features"].shape[1] if "text_features" in sample0 else None
        emo_dim  = sample0["emo_features"].shape[1]  if "emo_features" in sample0 else None

        best_model_path = os.path.join(
            args.output_dir,
            f"best_model_{args.model_type}_fold{fold_idx}.pt"
        )

        if args.mode == "train":
            # 🔨 Model oluştur (parametrik)
            model = build_model(
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

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

            # Scheduler (isteğe bağlı)
            if args.scheduler == "cosine":
                num_training_steps = args.epochs * max(1, len(train_loader))
                scheduler = build_cosine_warmup_scheduler(
                    optimizer,
                    num_training_steps=num_training_steps,
                    warmup_ratio=args.warmup_ratio,
                )
                print(f"📉 Using cosine LR scheduler with warmup_ratio={args.warmup_ratio}")
            else:
                scheduler = None
                print("📉 No LR scheduler used (scheduler=none).")

            best_val_f1 = 0.0
            best_epoch = -1

            # Early stopping (parametrik, val F1 based)
            if args.early_stopping:
                patience = args.patience
                patience_counter = 0
                print(f"⏹ Early stopping enabled (patience={patience})")
            else:
                patience = None
                patience_counter = None
                print("⏹ Early stopping disabled")

            # -----------------------------
            # Eğitim döngüsü (fold bazında)
            # -----------------------------
            for epoch in range(1, args.epochs + 1):
                print(f"\n🧭 Fold {fold_idx} - Epoch {epoch}/{args.epochs}")

                train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(
                    model, train_loader, optimizer, device, scheduler=scheduler
                )
                val_loss, val_acc, val_f1, val_precision, val_recall, val_preds, val_labels = eval_one_epoch(
                    model, val_loader, device, return_details=False
                )

                print(
                    f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f}, "
                    f"precision={train_precision:.4f}, recall={train_recall:.4f}"
                )
                print(
                    f"  Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}, "
                    f"precision={val_precision:.4f}, recall={val_recall:.4f}"
                )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    if args.early_stopping:
                        patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  ✅ Best model updated at epoch {epoch} (Val F1={best_val_f1:.4f})")
                else:
                    if args.early_stopping:
                        patience_counter += 1
                        print(f"  ⏳ EarlyStopping patience {patience_counter}/{patience}")

                if args.early_stopping and patience_counter >= patience:
                    print(
                        f"  🛑 Early stopping triggered at epoch {epoch}. "
                        f"Best epoch was {best_epoch} (Val F1={best_val_f1:.4f})"
                    )
                    break

            print(f"\n🏁 Fold {fold_idx} finished. Best epoch: {best_epoch}, Best Val F1={best_val_f1:.4f}")
            print(f"💾 Best model for fold {fold_idx} saved to: {best_model_path}")

            # En iyi modeli yükleyip fold final raporu + (opsiyonel) CSV kayıt
            best_model = build_model(
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
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))

            val_loss, val_acc, val_f1, val_precision, val_recall, val_preds, val_labels, val_conv_ids = eval_one_epoch(
                best_model, val_loader, device, return_details=True
            )
            print("\n📈 Final VAL results (best model, this fold):")
            print(
                f"  Loss = {val_loss:.4f}, Acc = {val_acc:.4f}, F1 = {val_f1:.4f}, "
                f"Precision = {val_precision:.4f}, Recall = {val_recall:.4f}"
            )
            print(
                classification_report(
                    val_labels,
                    val_preds,
                    labels=[0, 1],
                    target_names=[id2label[0], id2label[1]],
                    digits=4,
                )
            )

            # ⬇⬇⬇ YENİ: fold bazlı prediction CSV'leri
            if args.save_fold_predictions:
                import csv
                out_csv = os.path.join(args.output_dir, f"{args.model_type}_fold{fold_idx}_val_predictions.csv")
                with open(out_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["conv_id", "true_id", "true_label", "pred_id", "pred_label"])
                    for cid, y_true, y_pred in zip(val_conv_ids, val_labels, val_preds):
                        writer.writerow([
                            cid,
                            int(y_true),
                            id2label[int(y_true)],
                            int(y_pred),
                            id2label[int(y_pred)],
                        ])
                print(f"📝 Saved fold-{fold_idx} validation predictions to: {out_csv}")

            fold_results.append(
                {
                    "fold": fold_idx,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                }
            )
        else:
            # EVAL MODE: skip training, just load and evaluate best_model
            print(f"🔎 [EVAL MODE] Loading best model from: {best_model_path}")
            assert os.path.exists(best_model_path), f"Best model not found: {best_model_path}"
            # Build model
            model = build_model(
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
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            # Only run eval_one_epoch
            val_loss, val_acc, val_f1, val_precision, val_recall, val_preds, val_labels, val_conv_ids = eval_one_epoch(
                model, val_loader, device, return_details=True
            )
            print("\n📈 [EVAL MODE] Final VAL results (best model, this fold):")
            print(
                f"  Loss = {val_loss:.4f}, Acc = {val_acc:.4f}, F1 = {val_f1:.4f}, "
                f"Precision = {val_precision:.4f}, Recall = {val_recall:.4f}"
            )
            print(
                classification_report(
                    val_labels,
                    val_preds,
                    labels=[0, 1],
                    target_names=[id2label[0], id2label[1]],
                    digits=4,
                )
            )
            # Optionally save predictions
            if args.save_fold_predictions:
                import csv
                out_csv = os.path.join(args.output_dir, f"{args.model_type}_fold{fold_idx}_val_predictions.csv")
                with open(out_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["conv_id", "true_id", "true_label", "pred_id", "pred_label"])
                    for cid, y_true, y_pred in zip(val_conv_ids, val_labels, val_preds):
                        writer.writerow([
                            cid,
                            int(y_true),
                            id2label[int(y_true)],
                            int(y_pred),
                            id2label[int(y_pred)],
                        ])
                print(f"📝 [EVAL MODE] Saved fold-{fold_idx} validation predictions to: {out_csv}")
            fold_results.append(
                {
                    "fold": fold_idx,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                }
            )

    # ---------------------------------------------------
    # 4) Fold sonuçlarının özeti
    # ---------------------------------------------------
    print("\n==============================")
    print(f"   📊 {args.k_folds}-FOLD CV SUMMARY (model_type={args.model_type})")
    print("==============================")
    for fr in fold_results:
        print(
            f"Fold {fr['fold']}: "
            f"Loss={fr['val_loss']:.4f}, "
            f"Acc={fr['val_acc']:.4f}, "
            f"F1={fr['val_f1']:.4f}, "
            f"Precision={fr['val_precision']:.4f}, "
            f"Recall={fr['val_recall']:.4f}"
        )

    mean_acc = np.mean([fr["val_acc"] for fr in fold_results])
    mean_f1 = np.mean([fr["val_f1"] for fr in fold_results])
    mean_precision = np.mean([fr["val_precision"] for fr in fold_results])
    mean_recall = np.mean([fr["val_recall"] for fr in fold_results])
    std_acc = np.std([fr["val_acc"] for fr in fold_results])
    std_f1 = np.std([fr["val_f1"] for fr in fold_results])
    std_precision = np.std([fr["val_precision"] for fr in fold_results])
    std_recall = np.std([fr["val_recall"] for fr in fold_results])

    print("\n📌 Overall (CV):")
    print(f"  Acc      mean={mean_acc:.4f}, std={std_acc:.4f}")
    print(f"  F1       mean={mean_f1:.4f}, std={std_f1:.4f}")
    print(f"  Precision mean={mean_precision:.4f}, std={std_precision:.4f}")
    print(f"  Recall    mean={mean_recall:.4f}, std={std_recall:.4f}")

    summary_path = os.path.join(args.output_dir, f"{args.model_type}_cv_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model type   : {args.model_type}\n")
        f.write(f"Feature type : {args.feature_type}\n")
        f.write(f"Speaker mode : {args.speaker_mode}\n")
        f.write(f"Num folds    : {args.k_folds}\n\n")

        f.write("Per-fold results:\n")
        for fr in fold_results:
            f.write(
                f"Fold {fr['fold']}: "
                f"Loss={fr['val_loss']:.4f}, "
                f"Acc={fr['val_acc']:.4f}, "
                f"F1={fr['val_f1']:.4f}, "
                f"Precision={fr['val_precision']:.4f}, "
                f"Recall={fr['val_recall']:.4f}\n"
            )

        f.write("\nOverall (CV):\n")
        f.write(f"Acc       mean={mean_acc:.4f}, std={std_acc:.4f}\n")
        f.write(f"F1        mean={mean_f1:.4f}, std={std_f1:.4f}\n")
        f.write(f"Precision mean={mean_precision:.4f}, std={std_precision:.4f}\n")
        f.write(f"Recall    mean={mean_recall:.4f}, std={std_recall:.4f}\n")

    print(f"📝 CV summary saved to: {summary_path}")


if __name__ == "__main__":
    main()