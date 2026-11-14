#!/usr/bin/env python
"""Validate official test labels and copy audio into labeled AD/CN folders."""

import argparse
import csv
import json
import filecmp
import os
import shutil
from collections import Counter
from pathlib import Path


DX_TO_LABEL = {"Control": "cn", "ProbableAD": "ad"}


def load_test_labels(labels_csv, audio_dir):
    """Require unique IDs, known diagnoses and exact coverage of test WAV files."""
    audio_dir = Path(audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        raise ValueError(f"No test WAV files in {audio_dir}")
    labels = {}
    with Path(labels_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"ID", "Dx"}.issubset(reader.fieldnames or []):
            raise ValueError("Test label CSV must contain ID,Dx columns")
        for row in reader:
            recording_id = row["ID"].strip()
            diagnosis = row["Dx"].strip()
            if not recording_id or Path(recording_id).name != recording_id:
                raise ValueError(f"Invalid recording ID: {recording_id!r}")
            if recording_id in labels:
                raise ValueError(f"Duplicate test ID: {recording_id}")
            if diagnosis not in DX_TO_LABEL:
                raise ValueError(f"Unknown diagnosis for {recording_id}: {diagnosis!r}")
            labels[recording_id] = DX_TO_LABEL[diagnosis]
    audio_ids = {path.stem for path in audio_files}
    missing = audio_ids - labels.keys()
    extra = labels.keys() - audio_ids
    if missing or extra:
        raise ValueError(
            f"Test ID mismatch: missing labels={sorted(missing)}, "
            f"labels without audio={sorted(extra)}"
        )
    return labels


def create_labeled_view(test_dir, labels_csv, output_dir=None):
    test_dir = Path(test_dir).resolve()
    labels_csv = Path(labels_csv).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else test_dir / "labeled"
    labels = load_test_labels(labels_csv, test_dir / "audio")
    copies = []
    for recording_id, label in sorted(labels.items()):
        source = test_dir / "audio" / f"{recording_id}.wav"
        target = output_dir / "audio" / label / source.name
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f"Existing link conflicts with source: {target}")
        elif target.exists():
            if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
                raise ValueError(f"Existing file conflicts with source: {target}")
        copies.append((recording_id, label, source, target))
    for _, _, source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            shutil.copy2(source, target)
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "split", "label", "audio", "labeled_audio"])
        writer.writeheader()
        for recording_id, label, source, target in copies:
            writer.writerow(dict(ID=recording_id, split="test", label=label,
                                 audio=os.path.relpath(source, output_dir),
                                 labeled_audio=os.path.relpath(target, output_dir)))
    (output_dir / "provenance.json").write_text(json.dumps({
        "labels_csv": str(labels_csv), "test_dir": str(test_dir),
        "split": "test", "counts": dict(Counter(labels.values())),
        "audio_storage": "regular file copies of original test WAV files",
        "manifest_path_base": "directory containing manifest.csv",
    }, indent=2) + "\n", encoding="utf-8")
    return output_dir, Counter(labels.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    output_dir, counts = create_labeled_view(args.test_dir, args.labels_csv, args.output_dir)
    print(f"Validated labeled test copies: {output_dir} ({dict(counts)})")


if __name__ == "__main__":
    main()
