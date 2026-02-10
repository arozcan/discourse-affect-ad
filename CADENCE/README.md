

# CADENCE: Clinical Affective Discourse Modeling

CADENCE is **Stage 3** of the *discourse-affect-ad* framework.
It transfers discourse-level affective representations learned from synthetic
monologues (MADE) and ensemble-supervised utterance models (PLEA) to a
clinically grounded Alzheimer’s disease detection task.

Rather than relying on acoustic features or manually annotated emotion labels,
CADENCE models how affective structure unfolds across extended spontaneous
speech and integrates affective and semantic information for robust clinical
inference.

---

## Overview

CADENCE operates on **transcribed spontaneous speech** and consists of four
main steps:

1. Utterance segmentation using WhisperX  
2. Text + affective feature extraction using pretrained utterance-level
   emotion encoders  
3. Discourse-level sequence modeling with multiple temporal architectures  
4. Clinical evaluation and analysis on the ADReSSo benchmark  

---

## Environment Setup

```bash
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are recommended.

### WhisperX GPU Environment (Recommended)

For GPU-accelerated utterance segmentation with WhisperX, the environment can
be created using the provided Conda specification:

```bash
conda env create -f whisperx-gpu.yml
conda activate whisperx-gpu
```

This environment includes CUDA-enabled PyTorch, WhisperX, and all required
dependencies for fast and reproducible speech segmentation on NVIDIA GPUs.

---

## 1. Utterance Segmentation (WhisperX)

Speech recordings from ADReSSo are segmented into utterances using WhisperX:

```bash
python scripts/prepare_adresso_whisperx_segments.py \
  --adresso_root download/ADReSSo21/diagnosis \
  --output_root dataset/ADReSSo21_WHISPERX_SEG
```

This step produces time-aligned utterance-level transcripts for each speaker.

---

## 2. Feature Extraction

Each utterance is encoded using **textual features** and **affective hidden
representations** obtained from pretrained PLEA models.

### Single-teacher models

```bash
python scripts/extract_adresso_text_emo_features.py \
  --data_dir dataset/ADReSSo21_WHISPERX_SEG \
  --emo_ckpt_dir ../PLEA/saved_model/utterance_classifier_made_teacher/best_model \
  --output_pkl features/adresso21_text_emo_segments_made_teacher.pkl
```

Available teachers include:
- `made_teacher`
- `made_llm`
- `meld`
- `goemotions`

---

### Ensemble-based hybrid supervision (τ-sweep)

```bash
for tau in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  python scripts/extract_adresso_text_emo_features.py \
    --data_dir dataset/ADReSSo21_WHISPERX_SEG \
    --emo_ckpt_dir ../PLEA/saved_model/utterance_classifier_ensemble_made_hybrid_mtc_${tau}/best_model \
    --output_pkl features/adresso21_text_emo_segments_hybrid_mtc_${tau}.pkl
done
```

The parameter `τ` controls the interpolation between teacher-only and
generator-aware supervision.

---

## 3. Discourse-Level Sequence Modeling

Discourse-level inference is performed using multiple temporal architectures:

- **BiLSTM**
- **BiGRU**
- **Transformer**
- **Temporal pooling**

Example (BiLSTM):

```bash
python scripts/train_eval_adresso_seq.py \
  --mode train \
  --features_pkl features/adresso21_text_emo_segments_made_teacher.pkl \
  --output_dir saved_model/adresso_seq_cv_par_both_made_teacher \
  --model_type bilstm \
  --speaker_mode all \
  --feature_type text+emo_hidden \
  --epochs 100 \
  --batch_size 8 \
  --hidden_dim 256 \
  --lr 1e-5 \
  --max_segments 128 \
  --scheduler none \
  --save_fold_predictions
```

All experiments are evaluated using **5-fold cross-validation** under
demographically matched protocols.

---

## Feature Ablations

CADENCE supports systematic ablations over representation type:

- `text`
- `emo_hidden`
- `text+emo_hidden`

Late fusion of semantic and affective representations consistently yields the
strongest performance.

---

## 4. Analysis and Visualization

Best-performing models can be analyzed using dimensionality reduction
and embedding visualization:

```bash
python scripts/analyze_stage3_best.py \
  --features_pkl features/adresso21_text_emo_segments_made_teacher.pkl \
  --model_dir saved_model/adresso_seq_cv_par_both_made_teacher \
  --model_type bilstm \
  --feature_type text+emo_hidden \
  --max_segments 128 \
  --hidden_dim 256 \
  --seed 42 \
  --out_dir analysis/stage3_best_made_teacher \
  --tsne_metric cosine \
  --use_umap
```

This step enables qualitative inspection of discourse-level affective structure
and class separability.

---

## Key Findings

- Discourse-level affective structure provides diagnostic information beyond
  lexical content alone  
- Affective representations learned without clinical labels generalize
  effectively to human-authored spontaneous speech  
- Ensemble-supervised affective learning yields more stable and transferable
  representations than single-teacher supervision  
- Best-performing configurations exceed **91% accuracy and F1** on ADReSSo
  using text-only inputs  

---

## Experimental Results on ADReSSo

### Comparison with Existing Methods

| Work | Modality | Protocol | Acc | F1 | Precision / Recall |
|-----|--------|----------|-----|----|--------------------|
| Wang et al. (2021) | Multimodal | 10CV | 77.20 | 77.60 | 78.70 / 74.00 |
| Rohanian et al. (2021) | Multimodal | — | 84.00 | — | — |
| Cui et al. (2023) | Multimodal | 80/20 | 89.40 | — | — |
| Ying et al. (2023) | Multimodal | 10CV | 83.70 | 83.60 | 85.30 / 84.00 |
| Priyadarshinee et al. (2023) | Text | 70/30 | 88.70 | — | — |
| CogniAlign (2025) | Multimodal | 5CV | 90.36 | 90.11 | 90.15 / 90.76 |
| **CADENCE (Ours)** | **Text (Affective-Aware)** | **5CV** | **91.59** | **91.57** | **91.88 / 91.59** |

---

### Effect of Supervision Source  
*(BiLSTM downstream, 5-fold CV)*

| Supervision Source | Acc | F1 | Precision | Recall |
|-------------------|-----|----|-----------|--------|
| MELD (single teacher) | 82.57 ± 4.66 | 82.49 ± 4.68 | 82.82 ± 4.85 | 82.57 ± 4.66 |
| GoEmotions (single teacher) | 82.66 ± 11.13 | 82.15 ± 11.96 | 83.47 ± 10.87 | 82.66 ± 11.13 |
| **MADE (ensemble)** | **91.59 ± 4.40** | **91.57 ± 4.41** | **91.88 ± 4.41** | **91.59 ± 4.40** |

---

### Effect of Sequence Modeling Architecture  
*(MADE supervision, 5-fold CV)*

| Sequence Model | Acc | F1 | Precision | Recall |
|---------------|-----|----|-----------|--------|
| Mean Pooling (MLP) | 82.57 ± 6.03 | 82.48 ± 6.07 | 83.01 ± 5.80 | 82.57 ± 6.03 |
| Transformer + Attn | 83.80 ± 5.96 | 83.79 ± 5.94 | 84.00 ± 6.02 | 83.80 ± 5.96 |
| BiGRU + Attn | 86.79 ± 5.10 | 86.75 ± 5.10 | 87.33 ± 5.01 | 86.79 ± 5.10 |
| **BiLSTM + Attn** | **91.59 ± 4.40** | **91.57 ± 4.41** | **91.88 ± 4.41** | **91.59 ± 4.40** |

---

### Effect of Utterance-Level Representation Type  
*(BiLSTM downstream, 5-fold CV)*

| Representation Type | Acc | F1 | Precision | Recall |
|--------------------|-----|----|-----------|--------|
| Text-only | 72.87 ± 8.41 | 72.57 ± 8.62 | 73.63 ± 8.80 | 72.87 ± 8.41 |
| Emotion-only | 87.99 ± 5.64 | 87.91 ± 5.69 | 88.77 ± 5.41 | 87.99 ± 5.64 |
| **Text + Emotion (Dual-stream)** | **91.59 ± 4.40** | **91.57 ± 4.41** | **91.88 ± 4.41** | **91.59 ± 4.40** |

---
---

## Relation to the Full Framework

CADENCE corresponds to **Stage 3** of the pipeline:

```
MADE  →  PLEA  →  CADENCE
```

---

## Citation

If you use CADENCE, please cite the accompanying paper:

```bibtex
@article{ozcan2026discourseaffect,
  title   = {Discourse-Level Affective Structure in Speech for Alzheimer’s Detection},
  author  = {Ozcan, Ahmet Remzi},
  year    = {2026}
}
```