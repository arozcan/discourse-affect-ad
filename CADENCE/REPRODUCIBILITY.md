# Official test and statistical analysis

Run all commands below from the repository root. Activate the CADENCE environment first (`conda env create -f CADENCE/whisperx-gpu.yml`). The code uses NumPy, SciPy, pandas, scikit-learn, matplotlib, PyTorch, Transformers and WhisperX. Environment specifications are provided per module; checkpoints are supplied by the user, not downloaded by these commands.

## Inputs

- Authorized ADReSSo diagnosis audio under `CADENCE/download/ADReSSo21/diagnosis/`.
- Official test labels `task1.csv` with `ID,Dx` columns (`Control` / `ProbableAD`), obtained from the data provider. Set `TEST_LABELS` to its local path.
- Development MMSE CSV and official source `dx-mmse.csv`, obtained through the source access routes in the root README.
- PLEA MADE-teacher encoder checkpoint under `PLEA/saved_model/utterance_classifier_made_teacher/best_model/`.
- Existing five-fold CADENCE checkpoints under `CADENCE/saved_model/`, in `adresso_seq_cv_par_both_made_teacher`, `adresso_seq_cv_par_both_made_teacher_text`, and `adresso_seq_cv_par_both_made_teacher_emo_hidden`. The evaluator expects `best_model_bilstm_fold1.pt` through `best_model_bilstm_fold5.pt` in each directory, using the original model architecture. These weights are not bundled with this release. Without them, inference cannot reproduce the published predictions; source training commands are provided separately in README.md.
- Development feature cache `CADENCE/features/adresso21_text_emo_segments_made_teacher.pkl`, created by the original extraction pipeline.

Only load trusted pickle/checkpoint files. No new clinical datasets, individual prediction tables, or model weights are included in this update. Aggregated reference results are in `CADENCE/results/`.

## 1. Prepare official test segments

```bash
export TEST_LABELS=/path/to/task1.csv
python CADENCE/scripts/prepare_adresso_whisperx_segments.py \
  --adresso_root CADENCE/download/ADReSSo21/diagnosis \
  --output_root CADENCE/dataset/ADReSSo21_WHISPERX_SEG \
  --split test --test_labels_csv "$TEST_LABELS"
```

`--dry_run` validates IDs before ASR. Test labels are metadata; they do not guide transcription. Full-recording ASR is used without verified patient/interviewer separation. Minimum segment duration is 0.8 s, pause threshold is 0.9 s and the 6 s duration is a soft split criterion. Existing development outputs are retained with `--split test`.

The optional `prepare_adresso_test_labels.py` copies audio into labeled AD/CN directories and writes a relative-path manifest; it is not required by ASR:

```bash
python CADENCE/scripts/prepare_adresso_test_labels.py \
  --test_dir CADENCE/download/ADReSSo21/diagnosis/test-dist \
  --labels_csv "$TEST_LABELS"
```

## 2. Extract test features

```bash
python CADENCE/scripts/extract_adresso_text_emo_features.py \
  --data_dir CADENCE/dataset/ADReSSo21_WHISPERX_SEG \
  --emo_ckpt_dir PLEA/saved_model/utterance_classifier_made_teacher/best_model \
  --text_model_name microsoft/deberta-v3-large \
  --split test --speaker_mode par+inv --max_len 128 --batch_size 8 \
  --output_pkl CADENCE/revision_tracking/features/adresso21_test_text_emo_made_teacher.pkl
```

`--dry_run` checks inputs without loading encoders. Existing feature outputs are protected unless `--overwrite` is explicitly given. A metadata sidecar records input hashes and extraction settings. Speaker tags are not verified diarization; the published evaluation uses all cached segments.

## 3. Evaluate existing fold models (no training)

```bash
python CADENCE/scripts/evaluate_adresso_official_test.py \
  --features_pkl CADENCE/revision_tracking/features/adresso21_test_text_emo_made_teacher.pkl \
  --labels_csv "$TEST_LABELS" --checkpoint_root CADENCE/saved_model \
  --output_dir CADENCE/revision_tracking/results/official_test \
  --batch_size 8 --bootstrap_repeats 2000
```

Append `--dry_run` to validate feature/label coverage and checkpoint availability. Use a new empty output directory when repeating an evaluation. Five fold models are evaluated separately and their AD probabilities are equally averaged at a fixed 0.5 threshold. No model selection or tuning uses the official test. Outputs include predictions, per-fold/ensemble metrics, descriptive model variation and input/checkpoint hashes.

## 4. Paired comparisons and recording summaries

```bash
python CADENCE/scripts/analyze_revision_statistics.py \
  --labels_csv "$TEST_LABELS" \
  --mmse_csv CADENCE/download/ADReSSo21/diagnosis/train/adresso-train-mmse-scores.csv \
  --output_dir CADENCE/revision_tracking/analysis/reviewer_statistics \
  --bootstrap_repeats 10000 --permutations 10000
```

Input paths can be overridden with `--predictions`, `--development_features`, and `--test_features`. Exact McNemar tests compare accuracy; paired within-record prediction swaps compare weighted F1. Holm correction covers the three model pairs separately for each metric. Bootstrap resampling preserves diagnosis counts. These intervals condition on fixed models and do not measure training uncertainty.

This script also produces recording-level affective summaries and preliminary group tests. **Use the grouped analysis below for the manuscript's affective inference**, rather than the preliminary recording-level group tests.

## 5. Split/source identity audit

```bash
python CADENCE/scripts/audit_adresso_splits.py \
  --labels_csv "$TEST_LABELS" \
  --source_metadata /path/to/dx-mmse.csv \
  --output_dir CADENCE/revision_tracking/analysis/split_audit
```

The audit reads original saved `*_foldN_val_predictions.csv` files for the three BiLSTM representation settings and dual-stream BiGRU/Transformer/pooling settings. It reconstructs the five stratified folds using seed 42 and the original feature-cache order, verifies labels and recording coverage, checks byte-identical audio, and audits source Pitt participant IDs. `--features_pkl`, `--checkpoint_root`, and `--adresso_root` override the default paths. If `--source_metadata` is omitted, the script attempts to retrieve the official source metadata URL.

## 6. Group known repeated development visits

```bash
python CADENCE/scripts/analyze_revision_participant_groups.py \
  --record_metrics CADENCE/revision_tracking/analysis/reviewer_statistics/patient_affective_metrics.csv \
  --source_metadata /path/to/dx-mmse.csv \
  --output_dir CADENCE/revision_tracking/analysis/participant_grouped
```

Per-record metrics are averaged across known Pitt visits before inference, giving equal weight to visits and then analysis units. There are 162 units from 166 development recordings: 146 Pitt participants plus 16 source-record proxies, 87 AD and 75 CN. Five two-sided 10,000-permutation tests use Holm adjustment, with pointwise 95% grouped bootstrap CIs (10,000 repeats). Dwell is not significant after correction (p≈0.0519); neutral probability, entropy, embedding volatility and switching have adjusted p<0.001. Entropy is order-invariant; volatility is the mean L2 distance between consecutive affective embeddings.

## Evaluation scope and reference results

Development accuracy is 91.59% (SD 4.40), weighted F1 91.57% (SD 4.41). The same held-out development fold selected each checkpoint and supplied its reported score: these are validation-selected folds, without an inner validation set.

| Official-test ensemble | Accuracy (%) | Weighted F1 (%) |
|---|---:|---:|
| Dual-stream | 87.32 | 87.29 |
| Emotion-only | 83.10 | 82.96 |
| Text-only | 64.79 | 63.84 |

Both dual-stream and emotion-only exceed text-only after paired-test Holm correction. Dual-stream versus emotion-only is not significant (p=0.25 for both metrics).

Official-test results concern all 71 retained benchmark recordings (35 AD, 36 CN), not an external clinical cohort. Source metadata reveals repeated Pitt participants across some development folds and official development/test partitions. Original source test IDs could not be reliably mapped to local anonymized IDs, so uncertain matches were not removed. Recording IDs are disjoint, but participant-independent generalization is not established. No new training was performed for these evaluations. Affective analyses were exploratory/post hoc, not preregistered clinical effects.
