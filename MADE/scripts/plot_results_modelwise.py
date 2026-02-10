#!/usr/bin/env python3
"""
plot_results_modelwise.py
------------------------

Model-wise visualization for Stage-1 (MADE subsets vs MELD).

Inputs:
  - 4 model-wise Stage-1 JSON files (genai/gpt/grok/mistral), each containing:
      Emotion_Distribution_MADE
      Emotion_Distribution_REF      (MELD reference distribution; same across files)
      Transition_Global_MADE
      Transition_Global_REF

Outputs:
  1) Emotion distribution plot (single figure, 5 curves):
        MELD(REF) + {GENAI,GPT,GROK,MISTRAL}(MADE subsets)
  2) Global transition delta heatmaps (single figure, 4 panels):
        Δ = |T_global(MADE subset) - T_global(MELD)|

Example:
  python scripts/plot_results_modelwise.py \
    --jsons \
      results/model/stage1_results_MELD_genai.json \
      results/model/stage1_results_MELD_gpt.json \
      results/model/stage1_results_MELD_grok.json \
      results/model/stage1_results_MELD_mistral.json \
    --out_dir results/model

Notes:
  - Model name is inferred from the filename (since JSON "Reference" is always MELD).
  - Transition heatmaps use independent color scales (no shared scale) for readability.
"""

import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EMOTIONS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]

MODEL_PATTERNS = [
    ("GENAI", re.compile(r"(?:^|[_\-])genai(?:[_\-]|$)", re.IGNORECASE)),
    ("GPT", re.compile(r"(?:^|[_\-])gpt(?:[_\-]|$)", re.IGNORECASE)),
    ("GROK", re.compile(r"(?:^|[_\-])grok(?:[_\-]|$)", re.IGNORECASE)),
    ("MISTRAL", re.compile(r"(?:^|[_\-])mistral(?:[_\-]|$)", re.IGNORECASE)),
]

MODEL_ORDER = {"GENAI": 0, "GPT": 1, "GROK": 2, "MISTRAL": 3}


def load_json(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dist_to_list(dist: dict):
    return [float(dist[e]) for e in EMOTIONS]


def infer_model_name(path: str) -> str:
    stem = Path(path).stem
    low = stem.lower()
    for name, rx in MODEL_PATTERNS:
        if rx.search(low):
            return name
    return stem  # fallback


def plot_emotion_distributions(items, out_path: Path):
    """
    Single plot with 5 curves:
      - MELD (REF) once
      - 4 model MADE distributions
    """
    # take REF distribution from first JSON
    ref_dist = items[0][1].get("Emotion_Distribution_REF", None)
    if not isinstance(ref_dist, dict):
        raise SystemExit("Emotion_Distribution_REF not found in the first JSON (expected MELD reference distribution).")

    plt.figure(figsize=(10, 4))

    # REF curve
    plt.plot(
        EMOTIONS,
        dist_to_list(ref_dist),
        marker="s",
        linestyle="--",
        linewidth=2.8,
        label="MELD (REF)"
    )

    # Model curves
    for model, d, p in items:
        made_dist = d.get("Emotion_Distribution_MADE", None)
        if not isinstance(made_dist, dict):
            raise SystemExit(f"Emotion_Distribution_MADE not found in: {p}")

        plt.plot(
            EMOTIONS,
            dist_to_list(made_dist),
            marker="o",
            linestyle="-",
            linewidth=2.2,
            label=model
        )

    plt.title("Emotion Distributions (MADE Subsets vs MELD Reference)")
    plt.ylabel("Probability")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"✅ Saved: {out_path}")


def plot_global_transition_deltas(items, out_path: Path):
    """
    1x4 heatmaps:
      Δ = |Transition_Global_MADE - Transition_Global_REF|
    Independent color scale per subplot.
    """
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)

    for idx, (model, d, p) in enumerate(items):
        mat_made = np.array(d["Transition_Global_MADE"], dtype=float)
        mat_ref = np.array(d["Transition_Global_REF"], dtype=float)
        delta = np.abs(mat_made - mat_ref)

        ax = axes[0, idx]
        im = ax.imshow(delta, aspect="auto")

        ax.set_title(model, fontsize=12)
        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_yticks(range(len(EMOTIONS)))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(EMOTIONS, fontsize=9)
        ax.set_xlabel("Next emotion", fontsize=9)
        ax.set_ylabel("Current emotion", fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("|T(MADE) − T(MELD)|", fontsize=9)

    fig.suptitle(
        "Global Affective Transition Differences (Δ = |MADE-subset − MELD|)",
        fontsize=14,
        y=1.05
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsons",
        nargs="+",
        required=True,
        help="4 model-wise Stage-1 result JSON files (genai/gpt/grok/mistral)."
    )
    ap.add_argument(
        "--out_dir",
        default="results/model",
        help="Output directory for figures"
    )
    ap.add_argument(
        "--prefix",
        default="modelwise",
        help="Filename prefix for outputs"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all JSONs and infer model names
    items = []
    for p in args.jsons:
        d = load_json(p)
        model = infer_model_name(p)
        items.append((model, d, p))

    # Sort in stable order: GENAI, GPT, GROK, MISTRAL
    items.sort(key=lambda x: MODEL_ORDER.get(x[0], 999))

    # Output paths
    out_dist = out_dir / f"{args.prefix}_emotion_distributions.png"
    out_heat = out_dir / f"{args.prefix}_global_transition_deltas.png"

    # Make plots
    plot_emotion_distributions(items, out_dist)
    plot_global_transition_deltas(items, out_heat)


if __name__ == "__main__":
    main()