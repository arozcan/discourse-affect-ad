#!/usr/bin/env python3
"""
plot_results.py
---------------

Dataset-wise Stage-1 visualization (MADE vs human datasets).

Inputs:
  - Stage-1 result JSON files (e.g., MELD / IEMOCAP / DailyDialog), each containing:
      Emotion_Distribution_MADE
      Emotion_Distribution_REF
      Transition_Global_MADE
      Transition_Global_REF

Outputs (two figures):
  1) Emotion distributions (single plot, 1 MADE curve + REF curves per dataset)
  2) Global transition delta heatmaps (one per dataset):
        Δ = |T_global(MADE) - T_global(REF)|
     Independent color scale per dataset (no shared scale)

Examples:
  python scripts/plot_results.py \
    --jsons results/stage1_results_MELD.json results/stage1_results_IEMOCAP.json results/stage1_results_DailyDialog.json \
    --out_dir results \
    --made_from MELD

Notes:
  - If --made_from is not given, MADE distribution is taken from the first JSON.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EMOTIONS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]


def load_json(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dist_to_list(dist: dict):
    return [float(dist[e]) for e in EMOTIONS]


def plot_emotion_distributions(items, out_path: Path, made_from: str = ""):
    """
    Single plot with 1 MADE curve (once) + REF curve for each dataset JSON.
    """
    # Choose which JSON provides the MADE distribution
    made_source = None
    if made_from:
        for ref_name, d in items:
            if str(ref_name).lower() == made_from.lower():
                made_source = (ref_name, d)
                break
        if made_source is None:
            raise SystemExit(f"--made_from '{made_from}' not found among JSON Reference names: {[x[0] for x in items]}")
    else:
        made_source = items[0]

    made_ref_name, made_d = made_source
    made_dist = made_d.get("Emotion_Distribution_MADE", None)
    if not isinstance(made_dist, dict):
        raise SystemExit(f"Emotion_Distribution_MADE not found (or not a dict) in the selected MADE source JSON: {made_ref_name}")

    plt.figure(figsize=(10, 4))

    # 1) MADE curve (once)
    plt.plot(
        EMOTIONS,
        dist_to_list(made_dist),
        marker="o",
        linestyle="-",
        linewidth=2.5,
        label="MADE"
    )

    # 2) REF curves (one per JSON)
    for ref_name, d in items:
        ref_dist = d.get("Emotion_Distribution_REF", None)
        if not isinstance(ref_dist, dict):
            raise SystemExit(f"Emotion_Distribution_REF not found (or not a dict) in JSON: {ref_name}")

        plt.plot(
            EMOTIONS,
            dist_to_list(ref_dist),
            marker="s",
            linestyle="--",
            linewidth=2.0,
            label=str(ref_name)
        )

    #plt.title("Emotion Distributions (MADE vs Human Datasets)")
    plt.ylabel("Probability")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved: {out_path}")


def plot_global_transition_deltas(items, out_path: Path):
    """
    Global transition delta heatmaps:
      Δ = |Transition_Global_MADE - Transition_Global_REF|
    Independent color scale per dataset (no shared scale).
    """
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)

    for idx, (label, d) in enumerate(items):
        mat_made = np.array(d["Transition_Global_MADE"], dtype=float)
        mat_ref = np.array(d["Transition_Global_REF"], dtype=float)
        delta = np.abs(mat_made - mat_ref)

        ax = axes[0, idx]
        im = ax.imshow(delta, aspect="auto")

        ax.set_title(str(label), fontsize=12)
        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_yticks(range(len(EMOTIONS)))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(EMOTIONS, fontsize=9)
        ax.set_xlabel("Next emotion", fontsize=9)
        ax.set_ylabel("Current emotion", fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("|T(MADE) − T(REF)|", fontsize=9)

    # fig.suptitle(
    #     "Global Affective Transition Differences (Δ = |MADE − Reference|)",
    #     fontsize=14,
    #     y=1.05
    # )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", nargs="+", required=True, help="Stage-1 result JSON files (dataset-wise).")
    ap.add_argument("--out_dir", default="results", help="Output directory for figures.")
    ap.add_argument("--prefix", default="stage1", help="Filename prefix for outputs.")
    ap.add_argument("--made_from", default="", help="Reference name to choose MADE distribution from (e.g., MELD).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load JSONs
    items = []
    for p in args.jsons:
        d = load_json(p)
        ref_name = d.get("Reference", Path(p).stem)
        items.append((ref_name, d))

    # Output paths
    out_dist = out_dir / f"{args.prefix}_emotion_distributions.png"
    out_heat = out_dir / f"{args.prefix}_global_transition_deltas.png"

    # Plots
    plot_emotion_distributions(items, out_dist, made_from=args.made_from)
    plot_global_transition_deltas(items, out_heat)


if __name__ == "__main__":
    main()