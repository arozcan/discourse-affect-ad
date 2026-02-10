import os, json, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


# ============================================================
#   Prefix filtering helpers (genai / gpt / grok / mistral ...)
# ============================================================
def _normalize_prefixes(model_prefixes):
    """
    Accepts:
      - None
      - list[str]
      - "genai,gpt" (comma-separated str)
    Returns:
      - None or list[str] (lowercased)
    """
    if model_prefixes is None:
        return None
    if isinstance(model_prefixes, str):
        model_prefixes = [p.strip() for p in model_prefixes.split(",") if p.strip()]
    model_prefixes = [p.lower().strip() for p in model_prefixes if str(p).strip()]
    return model_prefixes if len(model_prefixes) > 0 else None


def _match_prefix(fname: str, model_prefixes):
    """
    fname example: "genai__monologue_0016.json"
    """
    if model_prefixes is None:
        return True
    f = fname.lower()
    # more strict first, then fallback
    for p in model_prefixes:
        if f.startswith(p + "__") or f.startswith(p + "_") or f.startswith(p):
            return True
    return False


# ============================================================
#   Dataset (B,U,L + target_idx + context window)
# ============================================================
class ConvDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=128, context_window=None, causal=False):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.context_window = context_window
        self.causal = causal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        all_sents = item["all_sentences"]
        target_idx = item["index"]
        label = item["label"]
        json_name = item.get("json_name", "")

        # --- Context window logic ---
        if self.context_window is None or self.context_window < 0:
            # full conversation mode
            start, end = 0, len(all_sents)
        else:
            start = max(0, target_idx - self.context_window)
            end = target_idx + 1 if self.causal else min(len(all_sents), target_idx + self.context_window + 1)

        sentences = [s["text"] for s in all_sents[start:end]]
        if len(sentences) == 0:
            sentences = [all_sents[target_idx]["text"]]
            start = target_idx
            end = target_idx + 1

        local_target_idx = target_idx - start

        # --- Tokenize ---
        encoded = [
            self.tokenizer(
                s,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_tensors="pt"
            )
            for s in sentences
        ]

        label_id = torch.tensor(self.label2id[label], dtype=torch.long)

        out = {
            "encoded_utterances": encoded,
            "target_idx": torch.tensor(local_target_idx, dtype=torch.long),
            "label": label_id,
            "json_name": json_name,
            "texts": sentences,                          
            "all_texts": [s["text"] for s in all_sents],
        }

        # ===== ENSEMBLE META =====
        if ("teacher_label" in item) or ("teacher_confidence" in item):
            t_str = item.get("teacher_label", None)
            t_id = self.label2id[t_str] if (t_str is not None and t_str in self.label2id) else -1
            out["teacher_label_id"] = torch.tensor(t_id, dtype=torch.long)

            conf = item.get("teacher_confidence", None)
            if conf is None:
                conf = 1.0
            out["teacher_confidence"] = torch.tensor(float(conf), dtype=torch.float32)

        return out


# ============================================================
#   Collate: batch → (B,U,L)
# ============================================================
def collate_hierarchical(batch, pad_token_id=1):
    B = len(batch)
    U_max = max(len(b["encoded_utterances"]) for b in batch)
    L_max = max(
        enc["input_ids"].size(1)
        for b in batch
        for enc in b["encoded_utterances"]
    )

    input_ids = torch.full((B, U_max, L_max), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros((B, U_max, L_max), dtype=torch.long)
    target_idx = torch.zeros((B,), dtype=torch.long)
    labels = torch.zeros((B,), dtype=torch.long)

    for i, b in enumerate(batch):
        for j, enc in enumerate(b["encoded_utterances"]):
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            input_ids[i, j, :ids.size(0)] = ids
            attn_mask[i, j, :mask.size(0)] = mask
        target_idx[i] = b["target_idx"]
        labels[i] = b["label"]

    batch_jsons = [b.get("json_name", "") for b in batch]
    batch_texts = [b.get("texts", []) for b in batch]
    batch_all_texts = [b.get("all_texts", []) for b in batch]  # ✅ tüm konuşmalar

    out = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "target_idx": target_idx,
        "labels": labels,
        "json_paths": batch_jsons,
        "texts": batch_texts,
        "all_texts": batch_all_texts,
    }

    # ===== ENSEMBLE META =====
    if "teacher_label_id" in batch[0]:
        teacher_labels = torch.full((B,), -1, dtype=torch.long)
        teacher_conf = torch.zeros((B,), dtype=torch.float32)

        for i, b in enumerate(batch):
            teacher_labels[i] = b["teacher_label_id"]
            teacher_conf[i] = b["teacher_confidence"]

        # labels already contain the primary (LLM) targets; agreement is teacher == labels when teacher exists.
        valid = teacher_labels >= 0
        agreement = (teacher_labels == labels) & valid

        out["teacher_labels"] = teacher_labels
        out["teacher_confidence"] = teacher_conf
        out["agreement"] = agreement

    return out


# ============================================================
def parse_conversation(json_path):
    """Backward compatible parser.

    Supports both:
    - base format: segments[*].predicted_emotion
    - ensemble format: segments[*].teacher_emotion + teacher_confidence (+ predicted_emotion)

    Primary llm label:
    - use predicted_emotion

    Also stores optional meta:
    - teacher_label, teacher_confidence
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = data.get("segments", data.get("utterances", []))
    examples = []

    for i, sent in enumerate(sentences):
        pred = sent.get("predicted_emotion")
        teacher = sent.get("teacher_emotion")
        conf = sent.get("teacher_confidence")

        if pred is None and teacher is None:
            continue

        label = pred  # default

        ex = {
            "all_sentences": sentences,
            "index": i,
            "label": label,
            "json_name": os.path.basename(json_path),
        }

        # ensemble meta
        if teacher is not None or conf is not None:
            ex["teacher_label"] = teacher
            ex["teacher_confidence"] = conf

        examples.append(ex)

    return examples


def load_dataset_from_dir(data_dir, model_prefixes=None):
    """
    Loads all *.json under data_dir.
    Optionally filters by filename prefix.
    """
    model_prefixes = _normalize_prefixes(model_prefixes)
    all_data = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        if not _match_prefix(fname, model_prefixes):
            continue
        path = os.path.join(data_dir, fname)
        all_data[fname] = parse_conversation(path)
    return all_data


# ============================================================
def get_dataloaders(
    data_dir,
    model_name="microsoft/deberta-v3-base",
    max_len=128,
    batch_size=8,
    seed=42,
    num_workers=4,
    pin_memory=True,
    balance_train=False,
    context_window=0,
    classes=None,
    map_others=False,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    split_mode="conversation",
    fold_index=None,
    folder_paths=None,
    causal=False,
    traindl_shuffle=True,
    min_teacher_conf=0.0,
    require_agreement=False,
    apply_ensemble_filters=False,
    model_prefixes=None, 
):
    model_prefixes = _normalize_prefixes(model_prefixes)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 1

    # Label set
    all_labels = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]
    labels = list(classes) if classes else all_labels
    if map_others and "other" not in labels:
        labels.append("other")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # --- Split (folder / session / conversation) ---
    if split_mode == "folder":
        if folder_paths is None:
            raise ValueError("split_mode='folder' requires folder_paths={'train':..., 'val':..., 'test':...}")

        def read_folder(subdir):
            path = os.path.join(data_dir, subdir)
            data = {}

            files = [f for f in os.listdir(path) if f.endswith(".json")]
            if model_prefixes is not None:
                files = [f for f in files if _match_prefix(f, model_prefixes)]
                print(f"Folder '{subdir}': keeping {len(files)} files for prefixes={model_prefixes}")

            for f in files:
                data[f] = parse_conversation(os.path.join(path, f))
            return data

        train_dict = read_folder(folder_paths["train"])
        val_dict = read_folder(folder_paths["val"])
        test_dict = read_folder(folder_paths["test"])

        def flatten(data_dict):
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

        train_data = flatten(train_dict)
        val_data = flatten(val_dict)
        test_data = flatten(test_dict)

    elif split_mode == "single_file":
        import json

        def load_split_json(split_name: str):
            """
            format: data_dir/{split_name}.json
            [
              {"text": "...", "label": "happy"},
              {"text": "...", "label": "sad"},
              ...
            ]
            """
            path = os.path.join(data_dir, f"{split_name}.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"SINGLE_FILE split missing: {path}")

            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)

            examples = []
            for i, it in enumerate(items):
                txt = it.get("text", "")
                lbl = it.get("label")

                if not txt:
                    continue

                if lbl in label2id:
                    mapped_lbl = lbl
                elif map_others:
                    mapped_lbl = "other"
                else:
                    continue

                sentences = [{"text": txt, "predicted_emotion": mapped_lbl}]

                examples.append({
                    "all_sentences": sentences,
                    "index": 0,
                    "label": mapped_lbl,
                    "json_name": f"{split_name}.json",
                })

            return examples

        train_data = load_split_json("train")
        val_data   = load_split_json("val")
        test_data  = load_split_json("test")

    else:
        # conversation/session split: directory scan + (optional) prefix filter
        all_data = load_dataset_from_dir(data_dir, model_prefixes=model_prefixes)
        files = list(all_data.keys())

        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        train_files, temp = train_test_split(files, test_size=(1 - train_ratio), random_state=seed)
        val_size = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(temp, test_size=(1 - val_size), random_state=seed)

        def flatten(filelist):
            examples = []
            for f in filelist:
                for ex in all_data[f]:
                    lbl = ex["label"]
                    if lbl in label2id:
                        examples.append(ex)
                    elif map_others:
                        new_ex = ex.copy()
                        new_ex["label"] = "other"
                        examples.append(new_ex)
            return examples

        train_data = flatten(train_files)
        val_data = flatten(val_files)
        test_data = flatten(test_files)


    # ------------------------------------------------------------
    def _apply_ensemble_filters(examples):
        if not apply_ensemble_filters:
            return examples

        if (min_teacher_conf is None or min_teacher_conf <= 0.0) and not require_agreement:
            return examples

        filtered = []
        for ex in examples:
            conf = ex.get("teacher_confidence", None)
            t_lbl = ex.get("teacher_label", None)
            p_lbl = ex.get("label", None)  # primary label (LLM)

            if conf is None and t_lbl is None and p_lbl is None:
                filtered.append(ex)
                continue

            if min_teacher_conf is not None and min_teacher_conf > 0.0:
                if conf is None or conf < min_teacher_conf:
                    continue

            if require_agreement:
                if t_lbl is None or p_lbl is None or t_lbl != p_lbl:
                    continue

            filtered.append(ex)

        return filtered

    if len(train_data) > 0:
        original_train_len = len(train_data)
        train_data = _apply_ensemble_filters(train_data)
        if apply_ensemble_filters:
            print(
                f"\nENSEMBLE FILTERS → Train: {original_train_len} → {len(train_data)} "
                f"(min_conf={min_teacher_conf}, require_agreement={require_agreement})"
            )
        else:
            print(
                f"\nENSEMBLE FILTERS DISABLED (apply_ensemble_filters=False) — "
                f"Train examples kept: {len(train_data)}"
            )

    # Dataset + DataLoader
    train_ds = ConvDataset(train_data, tokenizer, label2id, max_len, context_window, causal)
    val_ds = ConvDataset(val_data, tokenizer, label2id, max_len, context_window, causal)
    test_ds = ConvDataset(test_data, tokenizer, label2id, max_len, context_window, causal)

    if balance_train:
        print("WeightedRandomSampler aktif...")
        targets = [label2id[d["label"]] for d in train_data]
        counts = np.bincount(targets, minlength=len(label2id))
        weights = np.where(counts > 0, 1.0 / counts, 0.0)
        sample_weights = [weights[t] for t in targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda x: collate_hierarchical(x, pad_token_id),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=traindl_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda x: collate_hierarchical(x, pad_token_id),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: collate_hierarchical(x, pad_token_id),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: collate_hierarchical(x, pad_token_id),
    )

    # Özet
    def summarize(name, dataset):
        dist = Counter([ex["label"] for ex in dataset.data])
        print(f"\n{name} set ({len(dataset)} örnek):")
        for lbl, n in dist.items():
            print(f"  {lbl}: {n}")

    summarize("Train", train_ds)
    summarize("Val", val_ds)
    summarize("Test", test_ds)

    return train_loader, val_loader, test_loader, tokenizer, label2id, id2label