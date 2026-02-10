import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel, DebertaV2Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ContextAwareDeberta(nn.Module):
    """
    Context-aware classifier trained on pre-extracted contextual features.
    Now expects pooled features (after sentence-level pooling applied externally).

    Girdi (forward):
      - features:  (B, U, H)
      - target_idx: (B,)
      - labels: (B,)
      - mask: (B, U)
      - class_weights: (C,)
    """

    def __init__(self,
                 input_dim: int = 768,
                 num_labels: int = 7,
                 sentence_pool: str = "last_hidden",  # sadece bilgi için tutuluyor
                 gru_hidden_size: int = 512,
                 gru_layers: int = 1,
                 dropout: float = 0.1,
                 use_supcon: bool = False,
                 supcon_weight: float = 0.1,
                 supcon_temp: float = 0.07,
                 causal: bool = True):
        super().__init__()

        self.sentence_pool = sentence_pool
        self.use_supcon = use_supcon
        self.supcon_weight = supcon_weight
        self.supcon_temp = supcon_temp
        self.causal = causal

        # 🔹 Utterance-level context encoder (bidirectional GRU)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_size // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        gru_out_dim = gru_hidden_size

        # 🔹 Context attention projections
        self.ctx_key = nn.Linear(gru_out_dim, gru_out_dim, bias=False)
        self.ctx_val = nn.Linear(gru_out_dim, gru_out_dim, bias=False)
        self.tgt_qry = nn.Linear(gru_out_dim, gru_out_dim, bias=False)

        # 🔹 Gated fusion + classifier
        self.gate = nn.Linear(gru_out_dim * 2, gru_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_out_dim, num_labels)

        # 🔹 Optional projection head for SupCon
        if self.use_supcon:
            self.proj_head = nn.Sequential(
                nn.Linear(gru_out_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )

    # ------------------------------------------------------------
    def forward(self,
                features: torch.Tensor,
                target_idx: torch.Tensor,
                labels: torch.Tensor = None,
                mask: torch.Tensor = None,
                class_weights: torch.Tensor = None):
        """
        Girdi:
            features: (B, U, H)
            target_idx: (B,)
            mask: (B, U)
        """
        device = features.device
        B, U, H = features.shape
        if mask is None:
            mask = torch.ones(B, U, dtype=torch.bool, device=device)

        # --------------------------------------------------------
        # 1️⃣ GRU: contextual encoding
        # --------------------------------------------------------
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(features, lengths.clamp_min(1),
                                                   batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        ctx_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=U)

        # --------------------------------------------------------
        # 2️⃣ Target and context attention
        # --------------------------------------------------------
        batch_idx = torch.arange(B, device=device)
        tgt_states = ctx_out[batch_idx, target_idx]  # (B,H)

        pos = torch.arange(U, device=device).unsqueeze(0).expand(B, U)
        if self.causal:
            attn_mask = (pos < target_idx.unsqueeze(1)) & mask
        else:
            attn_mask = mask

        if attn_mask.sum() == 0:
            fused = tgt_states
            attn_weights = None
        else:
            Q = self.tgt_qry(tgt_states)
            K = self.ctx_key(ctx_out)
            V = self.ctx_val(ctx_out)

            attn_logits = torch.einsum("bh,buh->bu", Q, K) / (K.size(-1) ** 0.5)
            attn_logits = attn_logits.masked_fill(~attn_mask, torch.finfo(attn_logits.dtype).min)
            attn = torch.softmax(attn_logits, dim=-1)
            ctx_vec = torch.einsum("bu,buh->bh", attn, V)

            g = torch.sigmoid(self.gate(torch.cat([tgt_states, ctx_vec], dim=-1)))
            fused = g * tgt_states + (1.0 - g) * ctx_vec
            attn_weights = attn

        # --------------------------------------------------------
        # 3️⃣ Classification + optional SupCon
        # --------------------------------------------------------
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)

        loss, supcon_loss = None, None
        if labels is not None:
            # ⚖️ Weighted or normal CE loss
            if class_weights is not None:
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            else:
                loss = F.cross_entropy(logits, labels)

            # Supervised Contrastive Loss (optional)
            if self.use_supcon and labels.dtype == torch.long:
                z = F.normalize(self.proj_head(fused), dim=-1)
                sim = torch.matmul(z, z.T) / self.supcon_temp
                eye = torch.eye(B, device=device)
                sim = sim - 1e9 * eye
                exp_sim = torch.exp(sim)

                labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
                mask_pos = labels_eq.float() * (1 - eye)

                pos_sum = (exp_sim * mask_pos).sum(dim=1)
                all_sum = exp_sim.sum(dim=1)
                valid = pos_sum > 0
                supcon_loss = torch.zeros(B, device=device)
                supcon_loss[valid] = -torch.log((pos_sum[valid] + 1e-8) / (all_sum[valid] + 1e-8))
                supcon_loss = supcon_loss.mean()

                loss = loss + self.supcon_weight * supcon_loss

        return {
            "loss": loss,
            "logits": logits,
            "probs": probs,
            "supcon_loss": (supcon_loss.detach() if supcon_loss is not None else None),
            "attn_weights": attn_weights,
        }



class SentenceClassifierDeberta(nn.Module):
    """
    Tek cümle (sentence-level) duygu sınıflandırıcısı.
    DeBERTa-v3-base backbone üzerinde CLS vektörünü alır
    ve softmax çıktısı üretir.
    """

    def __init__(self,
                 model_name: str = "microsoft/deberta-v3-base",
                 num_labels: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor | None = None,
                class_weights: torch.Tensor | None = None):
        """
        Girdi:
          input_ids: (B,L)
          attention_mask: (B,L)

        Çıktı:
          {
            "loss": loss or None,
            "logits": (B,7),
            "probs":  (B,7),
            "feat":   (B,hidden_size)   # <--- yeni! duygusal ara katman vektörü
          }
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )

        # CLS embedding → bu bizim "emo_hidden"
        cls_emb = outputs.last_hidden_state[:, 0, :]   # (B, H)

        x = self.dropout(cls_emb)
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)

        # Hesaplanabiliyorsa loss
        loss = None
        if labels is not None:
            if class_weights is not None:
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            else:
                loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "probs": probs,
            "feat": cls_emb      # 🔥 AD/CN modeline vereceğimiz H-dim continuous emotion embedding
        }