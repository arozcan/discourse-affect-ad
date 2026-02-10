import os
import shutil
import random
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_DIRS = [
    "dataset/MADE/gpt_monologues_with_emo",
    "dataset/MADE/genai_monologues_with_emo",
    "dataset/MADE/mistral_monologues_with_emo",
    "dataset/MADE/grok_monologues_with_emo",
]

OUTPUT_BASE = Path("dataset/MADE")
SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

SEED = 42
EXT = ".json"

# =========================
# UTILS
# =========================
def collect_json_files(input_dirs):
    files = []
    for d in input_dirs:
        model_name = Path(d).name.replace("_monologues_with_emo", "")
        for f in Path(d).glob(f"*{EXT}"):
            files.append((model_name, f))
    return files


def make_dirs(base, splits):
    for s in splits:
        (base / s).mkdir(parents=True, exist_ok=True)


def split_files(files, splits):
    random.shuffle(files)
    n = len(files)

    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def copy_with_prefix(files_by_split, output_base):
    for split, items in files_by_split.items():
        out_dir = output_base / split
        for model_name, src_path in items:
            new_name = f"{model_name}__{src_path.name}"
            dst_path = out_dir / new_name
            shutil.copy2(src_path, dst_path)


# =========================
# MAIN
# =========================
def main():
    random.seed(SEED)

    print("Collecting JSON files...")
    files = collect_json_files(INPUT_DIRS)
    print(f"  → Found {len(files)} JSON files")

    print("Creating output directories...")
    make_dirs(OUTPUT_BASE, SPLITS)

    print("Splitting dataset...")
    split_map = split_files(files, SPLITS)

    for k, v in split_map.items():
        print(f"  {k}: {len(v)} files")

    print("Copying files...")
    copy_with_prefix(split_map, OUTPUT_BASE)

    print("Done.")
    print("Dataset written to:")
    for s in SPLITS:
        print(f"  - {OUTPUT_BASE / s}")


if __name__ == "__main__":
    main()