#!/usr/bin/env python3
"""
Stage 1 — Dataset Similarity Analysis (MADE vs Reference)
---------------------------------------------------------
Includes:
- Jensen–Shannon distance (emotion distribution)
- Three semantic similarity metrics:
    1) Centroid cosine similarity
    2) Nearest-Neighbor Average similarity
    3) Maximum Mean Discrepancy (MMD, RBF)
- Affective transition divergences (global + per-dialog)
"""

import os
import json
import argparse
import random

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)  # fixed seed for deterministic sampling

# ======================================================
# CONFIG
# ======================================================
EMOTIONS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]
MODEL_NAME = "all-mpnet-base-v2"

# ======================================================
# LOAD
# ======================================================
def load_sentences(folder_list):
    """
    Loads sentence texts and emotion labels from JSON files.

    Supports:
    1) MADE / MADE_ENSEMBLE format:
       {
         "segments": [
           { "text": "...", "predicted_emotion": "happy" }
         ]
       }

    2) GoEmotions-like flat format:
       [
         { "text": "...", "label": "sad" }
       ]
    """
    sentences, labels = [], []

    for folder in folder_list:
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"JSON error: {fpath} ({e})")
                continue

            # --------------------------------------------------
            # CASE 1: MADE / MADE_ENSEMBLE (dict + segments)
            # --------------------------------------------------
            if isinstance(data, dict) and "segments" in data:
                for seg in data.get("segments", []):
                    txt = (seg.get("text", "") or "").strip()
                    emo = (
                        seg.get("predicted_emotion")
                        or seg.get("label")
                        or ""
                    ).lower()

                    if txt and emo in EMOTIONS:
                        sentences.append(txt)
                        labels.append(emo)

            # --------------------------------------------------
            # CASE 2: GoEmotions-style (list of {text, label})
            # --------------------------------------------------
            elif isinstance(data, list):
                for item in data:
                    txt = (item.get("text", "") or "").strip()
                    emo = (item.get("label", "") or "").lower()

                    if txt and emo in EMOTIONS:
                        sentences.append(txt)
                        labels.append(emo)

            else:
                print(f"Unknown JSON format: {fpath}")

    return sentences, labels

# ======================================================
# METRICS
# ======================================================
def emotion_distribution(labels):
    counts = np.array([labels.count(e) for e in EMOTIONS], dtype=float)
    s = counts.sum()
    return counts / s if s > 0 else counts

def kl_divergence(p, q):
    """KL(P || Q). Note: not symmetric."""
    eps = 1e-9
    return float(np.sum(p * np.log((p + eps) / (q + eps))))

def rbf_mmd2(x, y, gamma=1.0):
    """Maximum Mean Discrepancy (RBF kernel). Lower = more similar."""
    def pdist2(a, b):
        return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)

    kxx = torch.exp(-gamma * pdist2(x, x)).mean()
    kyy = torch.exp(-gamma * pdist2(y, y)).mean()
    kxy = torch.exp(-gamma * pdist2(x, y)).mean()
    return float(kxx + kyy - 2 * kxy)

# ======================================================
# TRANSITIONS
# ======================================================
def transition_counts(labels):
    """Raw transition counts for a single label sequence (no normalization)."""
    mat = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        if a in EMOTIONS and b in EMOTIONS:
            i, j = EMOTIONS.index(a), EMOTIONS.index(b)
            mat[i, j] += 1.0
    return mat

def transition_matrix_global_from_folders(folder_list):
    """
    GLOBAL transition matrix:
    - sums raw transition counts per file (dialog/monologue)
    - normalizes once at the end
    - avoids artificial transitions across file boundaries.

    Supports:
    1) MADE / MADE_ENSEMBLE dict format with "segments"
       - uses seg["predicted_emotion"] (fallback seg["label"])

    2) GoEmotions-like list format:
       - uses item["label"]
    """
    mat_total = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=float)

    def _extract_labels_from_json(data):
        # CASE 1: dict with segments
        if isinstance(data, dict) and "segments" in data:
            labels = []
            for seg in data.get("segments", []):
                emo = (seg.get("predicted_emotion") or seg.get("label") or "").lower()
                if emo in EMOTIONS:
                    labels.append(emo)
            return labels

        # CASE 2: list of {text, label} (GoEmotions style)
        if isinstance(data, list):
            labels = []
            for item in data:
                emo = (item.get("label") or "").lower()
                if emo in EMOTIONS:
                    labels.append(emo)
            return labels

        return []

    for folder in folder_list:
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            labels = _extract_labels_from_json(data)
            if len(labels) < 2:
                continue

            mat_total += transition_counts(labels)

    s = mat_total.sum()
    return mat_total / s if s > 0 else mat_total

def transition_matrix_per_dialog_folders(folder_list):
    """
    PER-DIALOG average transition matrix:
    - for each file: compute transitions, normalize within file
    - average normalized matrices across files (each file weighted equally)

    Supports:
    1) MADE / MADE_ENSEMBLE dict format with "segments"
       - uses seg["predicted_emotion"] (fallback seg["label"])
    2) GoEmotions-like list format
       - uses item["label"]
    """
    mat_sum = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=float)
    count = 0

    def _extract_labels_from_json(data):
        # CASE 1: MADE-style dict with segments
        if isinstance(data, dict) and "segments" in data:
            labels = []
            for seg in data.get("segments", []):
                emo = (seg.get("predicted_emotion") or seg.get("label") or "").lower()
                if emo in EMOTIONS:
                    labels.append(emo)
            return labels

        # CASE 2: GoEmotions-style list
        if isinstance(data, list):
            labels = []
            for item in data:
                emo = (item.get("label") or "").lower()
                if emo in EMOTIONS:
                    labels.append(emo)
            return labels

        return []

    for folder in folder_list:
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            labels = _extract_labels_from_json(data)
            if len(labels) < 2:
                continue

            mat = transition_counts(labels)
            s = mat.sum()
            if s > 0:
                mat_sum += mat / s
                count += 1

    return mat_sum / count if count > 0 else mat_sum

def dist_dict(dist_vec):
    return {EMOTIONS[i]: float(dist_vec[i]) for i in range(len(EMOTIONS))}

def mat_to_list(mat):
    return [[float(x) for x in row] for row in mat]

# ======================================================
# MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser(description="Compare MADE dataset to a reference dataset (e.g., MELD or IEMOCAP).")
    parser.add_argument("--made_dir", required=True, help="Path to MADE dataset folder")
    parser.add_argument("--ref_dirs", nargs="+", required=True, help="List of reference dataset directories")
    parser.add_argument("--ref_name", required=True, help="Reference dataset name (for output naming)")
    parser.add_argument("--out_dir", default="results", help="Output directory for results and plots")
    parser.add_argument("--tag", default="", help="Optional tag to avoid overwriting outputs (e.g., gpt, grok, genai).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"{args.ref_name}" + (f"_{args.tag}" if args.tag else "")
    results_json = os.path.join(args.out_dir, f"stage1_results_{suffix}.json")
    results_fig = os.path.join(args.out_dir, f"stage1_comparison_{suffix}.png")

    print(f"Loading datasets: MADE vs {args.ref_name}")
    made_sents, made_labels = load_sentences([args.made_dir])
    ref_sents, ref_labels = load_sentences(args.ref_dirs)
    print(f"MADE: {len(made_sents)} sentences | {args.ref_name}: {len(ref_sents)} sentences")

    # Guard
    n = min(1000, len(made_sents), len(ref_sents))
    if n == 0:
        raise ValueError("No valid sentences found in one of the datasets. Check JSON format and labels.")

    # 1) Emotion distribution (SciPy returns Jensen–Shannon *distance* by default)
    p, q = emotion_distribution(made_labels), emotion_distribution(ref_labels)
    jsd = float(jensenshannon(p, q))
    print(f"\nJensen–Shannon distance (Emotion Distribution): {jsd:.4f}")

    # 2) Semantic similarities
    print("\nComputing semantic similarities...")
    model = SentenceTransformer(MODEL_NAME)

    made_samp = random.sample(made_sents, n)
    ref_samp = random.sample(ref_sents, n)

    emb1 = model.encode(made_samp, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
    emb2 = model.encode(ref_samp, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)

    # 2a) Centroid cosine
    c1 = emb1.mean(dim=0, keepdim=True)
    c2 = emb2.mean(dim=0, keepdim=True)
    centroid_cos = float(util.cos_sim(c1, c2))

    # 2b) Nearest-neighbor average
    S = util.cos_sim(emb1, emb2)
    rowmax = S.max(dim=1).values.mean()
    colmax = S.max(dim=0).values.mean()
    nn_avg = float((rowmax + colmax) / 2)

    # 2c) MMD (RBF)
    mmd_val = rbf_mmd2(emb1, emb2, gamma=1.0)

    print(f"   • Centroid cosine similarity:      {centroid_cos:.4f}")
    print(f"   • Nearest-neighbor avg similarity: {nn_avg:.4f}")
    print(f"   • MMD² distance:                   {mmd_val:.4f}")

    # 3) Affective transition divergences
    print("\n Computing Affective Transition Divergences...")
    mat_made_global = transition_matrix_global_from_folders([args.made_dir])
    mat_ref_global = transition_matrix_global_from_folders(args.ref_dirs)
    kld_global = kl_divergence(mat_made_global, mat_ref_global)

    mat_made_local = transition_matrix_per_dialog_folders([args.made_dir])
    mat_ref_local = transition_matrix_per_dialog_folders(args.ref_dirs)
    kld_local = kl_divergence(mat_made_local, mat_ref_local)

    print(f"   • Global-level KL (MADE || REF):        {kld_global:.4f}")
    print(f"   • Per-dialog/monologue KL (MADE || REF): {kld_local:.4f}")

    # Visualization
    x = np.arange(len(EMOTIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x - width/2, p, width, label="MADE", color="#4C72B0")
    ax.bar(x + width/2, q, width, label=args.ref_name, color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=30)
    ax.set_ylabel("Proportion")
    ax.set_title("Emotion Distribution Comparison")
    ax.legend(frameon=False)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    sns.heatmap(
        np.abs(mat_made_local - mat_ref_local),
        cmap="magma",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        ax=ax[1]
    )
    ax[1].set_title("Δ Affective Transition Matrix (Per-dialog)")

    plt.tight_layout()
    plt.savefig(results_fig, dpi=300)
    plt.show()

    # Save results JSON
    results = {
        "Reference": args.ref_name,
        "Jensen_Shannon_Distance": jsd,
        "Semantic_Cosine_Centroid": centroid_cos,
        "Semantic_Cosine_NNAvg": nn_avg,
        "Semantic_MMD": mmd_val,
        "Affective_Transition_KL_Global_(MADE||REF)": kld_global,
        "Affective_Transition_KL_PerDialog/Monologue_(MADE||REF)": kld_local,
        "MADE_sentence_count": len(made_sents),
        f"{args.ref_name}_sentence_count": len(ref_sents),
        "Model": MODEL_NAME,
        "Semantic_sample_size": n,
        "Semantic_sampling_seed": 42,
        "Emotion_Distribution_MADE": dist_dict(p),
        "Emotion_Distribution_REF": dist_dict(q),

        "Transition_Global_MADE": mat_to_list(mat_made_global),
        "Transition_Global_REF": mat_to_list(mat_ref_global),

        "Transition_PerDialog_MADE": mat_to_list(mat_made_local),
        "Transition_PerDialog_REF": mat_to_list(mat_ref_local),
    }

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n DONE: Results saved")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()