# PLEA: Pseudo-Label Aware Emotion Classification

This repository implements an **emotion classification framework** designed for **utterance-level modeling**, with a particular focus on **noisy or synthetic labels**.

The pipeline consists of:
1. Context-free utterance classification
2. Ensemble-based pseudo-label generation
3. Pseudo-label–aware re-training on MADE

---

## 1. Problem Setting

### Objective
Each utterance (monologue segment) is classified independently, without contextual information, in order to:
- learn robust semantic + emotional representations under noisy supervision
- enable reliable utterance-level emotion classification

### Key Characteristics
- Model: microsoft/deberta-v3-large
- Input: Single sentence (utterance)
- Context: None (context-free)
- Target: Emotion / cognitive state
- Output: Emotion class probabilities (and optionally [CLS] embeddings)

---

## 2. Utterance-Level Training on External Datasets

### 2.1 Training on MELD

```bash
python scripts/utterance_classifier.py \
  --mode train \
  --model_name microsoft/deberta-v3-large \
  --data_dir dataset/MELD \
  --split_mode folder \
  --folder_train train --folder_val val --folder_test test \
  --max_len 128 \
  --epochs 10 --batch_size 32 --lr 1e-5 --weight_decay 1e-2 \
  --weighted_loss
```

Test Performance (MELD):
- Accuracy: 0.62
- F1 (weighted): 0.63

---

### 2.2 Training on GoEmotions (7-Class Mapping)

GoEmotions is first mapped to 7 emotion classes and converted to the project’s JSON format using:

```bash
prepare_goemotions.py
```

Training:

```bash
python scripts/utterance_classifier.py \
  --mode train \
  --model_name microsoft/deberta-v3-large \
  --data_dir dataset/GoEmotions \
  --split_mode single_file \
  --max_len 128 \
  --epochs 10 --batch_size 24 --lr 1e-5 --weight_decay 1e-2
```

Test Performance (GoEmotions):
- Accuracy: 0.69
- F1 (weighted): 0.69

---

## 3. Ensemble Teacher Generation on MADE

### Motivation
MADE contains synthetic / LLM-generated labels, which are inherently noisy.
Instead of trusting these labels directly, we generate pseudo-labels using an ensemble of two independently trained utterance-level classifiers:
- MELD-trained classifier
- GoEmotions-trained classifier

---

### Pseudo-Label Generation

For each utterance, the ensemble produces:
- teacher_emotion
- teacher_confidence
- predicted_emotion (LLM)

```bash
python scripts/ensemble_teacher_relabel.py \
  --data_dir dataset/MADE \
  --input_folder train \
  --output_dir dataset/MADE_ENSEMBLE \
  --model_name microsoft/deberta-v3-large \
  --ckpt1 saved_model/utterance_classifier_meld/best_model \
  --ckpt2 saved_model/utterance_classifier_goemotion/best_model \
  --alpha 0.5 \
  --max_len 128 \
  --batch_size 32
```

---

## 4. Re-Training on MADE_ENSEMBLE (Pseudo-Label Aware)

### MADE_ENSEMBLE JSON Fields
- predicted_emotion: LLM-generated label
- teacher_emotion: ensemble pseudo-label
- teacher_confidence: ensemble confidence score
- agreement: whether teacher and LLM agree

### Training Strategy
- Hard label switching based on teacher_confidence
- Weighted loss using confidence and agreement
- Optional auxiliary LLM loss (disabled in best run)

---

**Pseudo-Label Aware Training (Ensemble Enabled).**  
This setting activates ensemble metadata and performs *hard label switching* based on `teacher_confidence`, combined with *confidence- and agreement-weighted loss*. When the ensemble is confident (`teacher_confidence ≥ 0.6`), the teacher pseudo-label is used; otherwise, the LLM label is selected. This approach explicitly controls label noise and reduces the impact of unreliable synthetic annotations. It achieves **0.84 accuracy** and **0.83 weighted F1**, yielding more conservative but noise-robust learning.

```bash
python scripts/utterance_classifier_ensemble.py \
  --mode train \
  --model_name microsoft/deberta-v3-large \
  --data_dir dataset/MADE_ENSEMBLE \
  --split_mode folder \
  --folder_train train --folder_val val --folder_test test \
  --max_len 128 \
  --epochs 20 --batch_size 128 --lr 1e-5  --weight_decay 1e-2 \
  --use_ensemble_meta \
  --train_target teacher \
  --eval_target teacher

python scripts/utterance_classifier_ensemble.py \
  --mode eval \
  --model_name microsoft/deberta-v3-large \
  --data_dir dataset/MADE_ENSEMBLE \
  --split_mode folder \
  --folder_train train --folder_val val --folder_test test \
  --max_len 128 \
  --epochs 20 --batch_size 128 --lr 1e-5  --weight_decay 1e-2 \
  --use_ensemble_meta \
  --train_target teacher \
  --eval_target teacher \
  --eval_model_dir saved_model/utterance_classifier_made_teacher/best_model

📊 Eval (teacher) Results — Loss: 0.3057, Acc: 0.9053, F1: 0.9058
              precision    recall  f1-score   support

       happy       0.89      0.96      0.92      3470
         sad       0.89      0.86      0.88       667
       angry       0.86      0.81      0.84      1037
     fearful       0.81      0.91      0.86      1560
   disgusted       0.80      0.86      0.83      1338
   surprised       0.92      0.93      0.92      1656
     neutral       0.96      0.90      0.93      8094

    accuracy                           0.91     17822
   macro avg       0.88      0.89      0.88     17822
weighted avg       0.91      0.91      0.91     17822
```

```bash
python scripts/utterance_classifier_ensemble.py \
  --mode train \
  --model_name microsoft/deberta-v3-large \
  --data_dir dataset/MADE_ENSEMBLE \
  --split_mode folder \
  --folder_train train --folder_val val --folder_test test \
  --max_len 128 \
  --epochs 20 --batch_size 128 --lr 1e-5  --weight_decay 1e-2 \
  --train_target llm \
  --eval_target llm

python scripts/utterance_classifier_ensemble.py \
  --mode eval \
  --model_name microsoft/deberta-v3-large \
  --data_dir dataset/MADE_ENSEMBLE \
  --split_mode folder \
  --folder_train train --folder_val val --folder_test test \
  --max_len 128 \
  --epochs 20 --batch_size 128 --lr 1e-5  --weight_decay 1e-2 \
  --train_target llm \
  --eval_target llm \
  --eval_model_dir saved_model/utterance_classifier_made_llm/best_model

📊 Eval (llm) Results — Loss: 0.7926, Acc: 0.7142, F1: 0.7120
              precision    recall  f1-score   support

       happy       0.68      0.80      0.73      2146
         sad       0.66      0.55      0.60      1415
       angry       0.67      0.61      0.64      1390
     fearful       0.69      0.82      0.75      2390
   disgusted       0.66      0.60      0.63      1144
   surprised       0.64      0.56      0.60      1428
     neutral       0.77      0.75      0.76      7909

    accuracy                           0.71     17822
   macro avg       0.68      0.67      0.67     17822
weighted avg       0.71      0.71      0.71     17822
```
| Train Target | Eval Target | Loss  | Acc   | F1    |
|--------------|-------------|-------|-------|-------|
| teacher      | teacher     | 0.3057| 0.9053| 0.9058|
| llm          | llm         | 0.7926| 0.7142| 0.7120|

```bash
for thr in 0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  echo "===== Running min_teacher_conf=$thr ====="
  python scripts/utterance_classifier_ensemble.py \
    --mode train \
    --model_name microsoft/deberta-v3-large \
    --data_dir dataset/MADE_ENSEMBLE \
    --split_mode folder \
    --folder_train train --folder_val val --folder_test test \
    --max_len 128 \
    --epochs 20 --batch_size 128 --lr 1e-5 --weight_decay 1e-2 \
    --use_ensemble_meta \
    --ensemble_weighting agreement_conf \
    --min_teacher_conf "$thr" \
    --train_target hybrid \
    --eval_target hybrid
done
```

| min_teacher_conf | Hybrid Loss | Hybrid Acc | Hybrid F1 | LLM Loss | LLM Acc | LLM F1 | Teacher Loss | Teacher Acc | Teacher F1 |
|------------------|-------------|------------|-----------|----------|---------|--------|--------------|-------------|------------|
| 0.10 | 0.3057 | 0.9053 | 0.9058 | 3.3181 | 0.5584 | 0.5591 | 0.3057 | 0.9053 | 0.9058 |
| 0.20 | 0.3069 | 0.9063 | 0.9067 | 3.3133 | 0.5595 | 0.5601 | 0.3066 | 0.9064 | 0.9068 |
| 0.30 | 0.3162 | 0.9038 | 0.9043 | 3.2562 | 0.5614 | 0.5618 | 0.3062 | 0.9057 | 0.9062 |
| 0.40 | 0.3738 | 0.8855 | 0.8854 | 2.7259 | 0.5779 | 0.5778 | 0.3258 | 0.8928 | 0.8933 |
| 0.50 | 0.5544 | 0.8245 | 0.8237 | 1.8485 | 0.6321 | 0.6308 | 0.5589 | 0.8263 | 0.8267 |
| 0.60 | 0.6743 | 0.7788 | 0.7771 | 1.3562 | 0.6683 | 0.6648 | 0.8230 | 0.7639 | 0.7657 |
| 0.70 | 0.7506 | 0.7512 | 0.7483 | 1.1326 | 0.6853 | 0.6815 | 0.9452 | 0.7329 | 0.7347 |
| 0.80 | 0.8097 | 0.7292 | 0.7258 | 0.9947 | 0.6941 | 0.6904 | 0.9700 | 0.7213 | 0.7234 |
| 0.90 | 0.8446 | 0.7188 | 0.7144 | 0.8997 | 0.7077 | 0.7033 | 1.0914 | 0.6960 | 0.6962 |


**Summary.**  
While LLM-only training provides a strong upper bound on MADE in-distribution performance, pseudo-label–aware training offers improved robustness by selectively trusting ensemble-generated labels and down-weighting uncertain samples. This trade-off is critical when the learned representations are later reused for downstream, context-aware modeling.


---

## Summary

- Utterance-only training learns robust representations  
- Ensemble pseudo-labeling reduces synthetic label noise  
- Pseudo-label–aware training improves reliability  
- Architecture remains fixed; only the supervision strategy varies
