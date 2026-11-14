#!/usr/bin/env python
"""Inference only: evaluate existing five-fold BiLSTM checkpoints on official test."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from prepare_adresso_test_labels import DX_TO_LABEL
from train_eval_adresso_seq import build_model, build_samples_from_conversations, collate_fn


CONFIGS = {
    "dual_stream": ("text+emo_hidden", "adresso_seq_cv_par_both_made_teacher"),
    "text_only": ("text", "adresso_seq_cv_par_both_made_teacher_text"),
    "emotion_only": ("emo_hidden", "adresso_seq_cv_par_both_made_teacher_emo_hidden"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_labels(path):
    labels = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"ID", "Dx"}.issubset(reader.fieldnames or []):
            raise ValueError("Labels must have ID,Dx columns")
        for row in reader:
            rid, dx = row["ID"].strip(), row["Dx"].strip()
            if not rid or rid in labels or dx not in DX_TO_LABEL:
                raise ValueError(f"Invalid/duplicate ID or diagnosis: {rid!r}, {dx!r}")
            labels[rid] = int(DX_TO_LABEL[dx] == "ad")
    if not labels or set(labels.values()) != {0, 1}:
        raise ValueError("Both CN and AD labels are required")
    return labels


def validate_features(data, labels):
    conversations = data["conversations"]
    ids = [c["conv_id"] for c in conversations]
    if len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError("Feature IDs must uniquely and exactly match official test IDs")
    if any(c.get("split") != "test" for c in conversations):
        raise ValueError("Only test-split features are accepted; training data must remain separate")
    for c in conversations:
        length = len(c["speakers"])
        if length == 0:
            raise ValueError(f"Empty conversation: {c['conv_id']}")
        for key in ("text_cls", "emo_hidden"):
            array = np.asarray(c[key])
            if array.ndim != 2 or array.shape[0] != length or not np.isfinite(array).all():
                raise ValueError(f"Invalid features: {c['conv_id']} / {key}")
        expected = "ad" if labels[c["conv_id"]] else "cn"
        if c.get("diag_label") not in (None, expected):
            raise ValueError(f"Feature diagnosis disagrees with official CSV: {c['conv_id']}")
    return sorted(conversations, key=lambda c: c["conv_id"])


def metric_arrays(y, prediction):
    """Confusion counts and metrics; works for one or many bootstrap rows."""
    tn = np.sum((y == 0) & (prediction == 0), axis=-1)
    fp = np.sum((y == 0) & (prediction == 1), axis=-1)
    fn = np.sum((y == 1) & (prediction == 0), axis=-1)
    tp = np.sum((y == 1) & (prediction == 1), axis=-1)
    n0, n1 = tn + fp, tp + fn
    f0 = 2 * tn / np.maximum(2 * tn + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {
        "accuracy": (tn + tp) / (n0 + n1),
        "weighted_f1": (n0 * f0 + n1 * f1) / (n0 + n1),
        "sensitivity_ad": tp / np.maximum(n1, 1),
        "specificity_cn": tn / np.maximum(n0, 1),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def bootstrap_indices(y, repeats, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    return np.concatenate([rng.choice(g, (repeats, len(g)), replace=True) for g in groups], axis=1)


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_pkl", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--checkpoint_root", default="CADENCE/saved_model")
    parser.add_argument("--output_dir", default="CADENCE/revision_tracking/results/official_test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bootstrap_repeats", type=int, default=2000)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_repeats < 1:
        parser.error("batch_size and bootstrap_repeats must be positive")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; use a new directory to preserve existing results")
    labels = read_labels(args.labels_csv)
    with Path(args.features_pkl).open("rb") as handle:
        conversations = validate_features(pickle.load(handle), labels)
    # Labels are attached for compatibility with the original collate function;
    # they are never passed into the network or used to select checkpoints.
    for c in conversations:
        c["diag_label"] = "ad" if labels[c["conv_id"]] else "cn"
    checkpoints = {
        name: [Path(args.checkpoint_root) / folder / f"best_model_bilstm_fold{k}.pt" for k in range(1, 6)]
        for name, (_, folder) in CONFIGS.items()
    }
    for paths in checkpoints.values():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    if args.dry_run:
        print(f"PASS: {len(conversations)} test records; 15 existing checkpoints; outputs: {output}")
        return
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "features": str(Path(args.features_pkl).resolve()), "features_sha256": sha256(args.features_pkl),
        "labels": str(Path(args.labels_csv).resolve()), "labels_sha256": sha256(args.labels_csv),
        "checkpoints": {name: [{"path": str(p.resolve()), "sha256": sha256(p)} for p in paths]
                        for name, paths in checkpoints.items()},
        "model": "bilstm", "hidden_dim": 256, "num_layers": 1,
        "max_segments": 128, "speaker_mode": "all", "threshold": 0.5,
        "ensemble": "equal average of all five fold softmax AD probabilities",
        "bootstrap": {"method": "participant bootstrap stratified by diagnosis; fixed models",
                      "repeats": args.bootstrap_repeats, "seed": 42},
        "device": device, "torch_version": torch.__version__,
        "training_performed": False,
    }
    # Persist the complete rule before producing any test predictions or metrics.
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    ids = [c["conv_id"] for c in conversations]
    y = np.array([labels[rid] for rid in ids])
    bootstrap = bootstrap_indices(y, args.bootstrap_repeats, 42)
    predictions, metrics, summaries = [], [], []
    for name, (feature_type, _) in CONFIGS.items():
        samples = build_samples_from_conversations(conversations, {"cn": 0, "ad": 1}, "all", feature_type)
        if [s["conv_id"] for s in samples] != ids:
            raise ValueError(f"Dropped or reordered samples for {name}")
        text_dim = samples[0]["text_features"].shape[1] if "text_features" in samples[0] else None
        emo_dim = samples[0]["emo_features"].shape[1] if "emo_features" in samples[0] else None
        fold_probs = []
        for k, path in enumerate(checkpoints[name], 1):
            model = build_model("bilstm", text_dim, emo_dim, 256, 1, 4, 2, 128, 0.2).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
            model.eval()
            probabilities = []
            with torch.inference_mode():
                for offset in range(0, len(samples), args.batch_size):
                    batch = collate_fn(samples[offset:offset + args.batch_size], 128)
                    text = batch["text_features"]
                    emo = batch["emo_features"]
                    logits, _ = model(text_features=text.to(device) if text is not None else None,
                                      emo_features=emo.to(device) if emo is not None else None,
                                      mask=batch["mask"].to(device))
                    probabilities.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
            fold_probs.append(np.asarray(probabilities))
            del model
            print(f"{name}: fold {k} inference complete", flush=True)
        all_probs = list(enumerate(fold_probs, 1)) + [("ensemble", np.mean(fold_probs, axis=0))]
        for fold, probs in all_probs:
            if not np.isfinite(probs).all():
                raise ValueError(f"Nonfinite probabilities: {name}/{fold}")
            predicted = (probs >= 0.5).astype(int)
            predictions.extend(dict(model_id=name, fold=fold, conv_id=rid, y_true=int(yt),
                                    p_ad=float(p), y_pred=int(yp))
                               for rid, yt, p, yp in zip(ids, y, probs, predicted))
            values = metric_arrays(y, predicted)
            row = dict(model_id=name, fold=fold, n=len(y), **{key: float(v) for key, v in values.items()})
            boot = metric_arrays(y[bootstrap], predicted[bootstrap])
            for metric in ("accuracy", "weighted_f1", "sensitivity_ad", "specificity_cn"):
                row[metric + "_ci_low"], row[metric + "_ci_high"] = np.quantile(boot[metric], [0.025, 0.975]).tolist()
            metrics.append(row)
        for metric in ("accuracy", "weighted_f1", "sensitivity_ad", "specificity_cn"):
            values = [r[metric] for r in metrics if r["model_id"] == name and r["fold"] != "ensemble"]
            summaries.append(dict(model_id=name, metric=metric, mean=float(np.mean(values)),
                                  sd=float(np.std(values)), n_models=5,
                                  interpretation="between-model variation on the same test participants"))
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "fold_model_summary.csv", summaries)
    print(f"Saved official test evaluation: {output}. No training performed.")


if __name__ == "__main__":
    main()
