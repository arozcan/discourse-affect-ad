# Discourse-Level Affective Structure Framework: Code, Data, and Reproducibility Guide

This repository contains the code, processed artifacts, and documentation for a
three-stage framework that studies discourse-level affective structure in
speech and applies it to Alzheimer's disease detection.

The framework has three components:

- `MADE`: construction and validation of a synthetic monologue emotion corpus
- `PLEA`: pseudo-label-aware utterance-level affective learning
- `CADENCE`: discourse-level clinical modeling for AD detection on ADReSSo

## Dataset Information

### Datasets used in this study

| Dataset | Role in study | Source type | Access / URL |
|---|---|---|---|
| MADE corpus | Synthetic monologue corpus created in this study and used as the main affective pretraining resource | This study | https://github.com/arozcan/llm-emotion-audit |
| ADReSSo 2021 | Clinical downstream benchmark for Alzheimer's disease detection | Third-party controlled-access dataset | https://talkbank.org/dementia/ADReSSo-2021/ |
| MELD | Human-annotated emotion benchmark used for MADE validation and PLEA teacher training | Third-party dataset | https://github.com/declare-lab/MELD |
| IEMOCAP | Human-annotated emotion benchmark used for MADE validation | Third-party dataset | https://sail.usc.edu/iemocap/ |
| DailyDialog | Human-annotated dialogue benchmark used for MADE validation | Third-party dataset | https://yanran.li/dailydialog |
| GoEmotions | Human-annotated utterance benchmark used for MADE validation and PLEA teacher training | Third-party dataset | https://github.com/google-research/google-research/tree/master/goemotions |

### Data availability and restrictions

- The MADE corpus was constructed in this study and its processing pipeline is
  documented in the public repository listed above.
- ADReSSo is a controlled-access third-party dataset and must be requested from
  its original provider through DementiaBank/TalkBank.
- External benchmark datasets such as MELD, IEMOCAP, DailyDialog, and
  GoEmotions remain subject to their own licenses and access conditions.

### ADReSSo source details

- Official challenge/data page: https://talkbank.org/dementia/ADReSSo-2021/
- Challenge paper DOI: https://doi.org/10.21437/Interspeech.2021-1220
- Access note: ADReSSo is distributed via DementiaBank/TalkBank and requires
  controlled-access registration and agreement with the data use rules.

### MADE corpus source details

- Repository URL: https://github.com/arozcan/llm-emotion-audit
- MADE is not a third-party corpus. It is the synthetic corpus constructed in
  this study from model-generated monologues and the accompanying processing
  pipeline is included in this repository.

## Code Information

### Repository layout

- `MADE/`: scripts for preparing reference datasets, generating synthetic
  monologues, validating outputs, and comparing MADE with human datasets
- `PLEA/`: scripts for utterance-level emotion classification and
  pseudo-label-aware training
- `CADENCE/`: scripts for ADReSSo segmentation, feature extraction, and
  discourse-level sequence modeling

### Main scripts

- `MADE/scripts/db_*_prepare.py`: convert external datasets into the unified
  JSON format
- `MADE/scripts/*gen_monolog_with_emo.py`: generate synthetic monologues
- `MADE/scripts/validate_monologues.py`: structural and statistical quality
  control
- `MADE/scripts/compare_datasets.py`: dataset-level similarity analysis
- `PLEA/scripts/utterance_classifier.py`: train teacher models
- `PLEA/scripts/ensemble_teacher_relabel.py`: generate ensemble pseudo-labels
- `PLEA/scripts/utterance_classifier_ensemble.py`: train pseudo-label-aware
  utterance classifiers
- `CADENCE/scripts/prepare_adresso_whisperx_segments.py`: segment ADReSSo audio
  with WhisperX
- `CADENCE/scripts/extract_adresso_text_emo_features.py`: extract text and
  affective features
- `CADENCE/scripts/train_eval_adresso_seq.py`: discourse-level model training
  and evaluation

## Usage Instructions

### 1. Clone repository

```bash
git clone https://github.com/arozcan/discourse-affect-ad.git
cd discourse-affect-ad
```

### 2. Create environments

For MADE:

```bash
cd MADE
conda env create -f environment_linux.yml
conda activate made
```

For PLEA:

```bash
cd ../PLEA
conda env create -f plea.yml
conda activate plea
```

For CADENCE/WhisperX:

```bash
cd ../CADENCE
conda env create -f whisperx-gpu.yml
conda activate whisperx-gpu
```

These environment files are the authoritative dependency specifications for
reproducing the experiments:

- `MADE/environment_linux.yml`
- `MADE/environment_macos.yml`
- `PLEA/plea.yml`
- `CADENCE/whisperx-gpu.yml`

### 3. Prepare or obtain datasets

- Place external datasets under each module's `download/` directory as expected
  by the scripts.
- For ADReSSo, request access through DementiaBank/TalkBank and place the
  released files under `CADENCE/download/ADReSSo21/`.

### 4. Run the pipeline

#### MADE

```bash
cd MADE
python scripts/db_meld_prepare.py
python scripts/db_dailydialog_prepare.py
python scripts/db_iemocap_prepare.py
python scripts/db_goemotions_prepare.py
python scripts/validate_monologues.py --dirs dataset/MADE/*_monologues_with_emo
python scripts/combine_dataset.py
python scripts/compare_datasets.py
```

#### PLEA

```bash
cd ../PLEA
python scripts/utterance_classifier.py \
  --mode train \
  --model_name microsoft/deberta-v3-large \
  --data_dir dataset/MELD \
  --split_mode folder \
  --folder_train train --folder_val val --folder_test test \
  --max_len 128 \
  --epochs 10 --batch_size 32 --lr 1e-5 --weight_decay 1e-2 \
  --weighted_loss

python scripts/utterance_classifier.py \
  --mode train \
  --model_name microsoft/deberta-v3-large \
  --data_dir dataset/GoEmotions \
  --split_mode single_file \
  --max_len 128 \
  --epochs 10 --batch_size 24 --lr 1e-5 --weight_decay 1e-2

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

python scripts/utterance_classifier_ensemble.py \
  --mode train \
  --model_name microsoft/deberta-v3-large \
  --data_dir dataset/MADE_ENSEMBLE \
  --split_mode folder \
  --folder_train train --folder_val val --folder_test test \
  --max_len 128 \
  --epochs 20 --batch_size 128 --lr 1e-5 --weight_decay 1e-2 \
  --use_ensemble_meta \
  --train_target teacher \
  --eval_target teacher
```

#### CADENCE

```bash
cd ../CADENCE
python scripts/prepare_adresso_whisperx_segments.py \
  --adresso_root download/ADReSSo21/diagnosis \
  --output_root dataset/ADReSSo21_WHISPERX_SEG

python scripts/extract_adresso_text_emo_features.py \
  --data_dir dataset/ADReSSo21_WHISPERX_SEG \
  --emo_ckpt_dir ../PLEA/saved_model/utterance_classifier_made_teacher/best_model \
  --output_pkl features/adresso21_text_emo_segments_made_teacher.pkl

python scripts/train_eval_adresso_seq.py \
  --mode train \
  --features_pkl features/adresso21_text_emo_segments_made_teacher.pkl \
  --output_dir saved_model/adresso_seq_cv_par_both_made_teacher \
  --model_type bilstm \
  --speaker_mode all \
  --feature_type text+emo_hidden
```

For full command examples, see:

- `MADE/README.md`
- `PLEA/README.md`
- `CADENCE/README.md`

## Requirements

### Software

- Python 3.9+
- Conda or Anaconda/Miniconda
- PyTorch 2.0+
- Hugging Face Transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- WhisperX for ADReSSo segmentation

### Dependency files

- `MADE/environment_linux.yml` and `MADE/environment_macos.yml` define the MADE
  environment for Linux and macOS, respectively.
- `PLEA/plea.yml` defines the PLEA environment.
- `CADENCE/whisperx-gpu.yml` defines the WhisperX/CADENCE GPU environment.
- Reproducing the full pipeline should be done from these environment files
  rather than by manual package installation.

### Runtime notes

- GPU is recommended for WhisperX segmentation and for the larger transformer
  models used in PLEA/CADENCE.
- Some scripts expect a Hugging Face token:

```bash
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

## Methodology

### Stage 1: MADE

Human-annotated emotion datasets are first normalized into a shared 7-class
emotion taxonomy. Synthetic monologues are then generated with multiple LLMs
using a common discourse prompt. The generated files are cleaned, structurally
validated, combined, and compared against human datasets using emotion
distribution similarity, semantic embedding similarity, and affective
transition divergence.

### Stage 2: PLEA

Utterance-level classifiers are trained on external human-annotated datasets
and used as teachers. Their outputs are combined into pseudo-labels that
provide a more robust supervisory signal than raw synthetic labels alone. A
pseudo-label-aware classifier is then trained on the MADE corpus.

### Stage 3: CADENCE

ADReSSo speech recordings are segmented into utterances with WhisperX. Each
utterance is encoded with semantic and affective representations, and the
resulting sequences are modeled with discourse-level neural architectures such
as BiLSTM, BiGRU, and Transformer-based variants. Evaluation is carried out
with cross-validation on the ADReSSo diagnosis benchmark.

## Citations

If you use the framework or repository, please cite:

```bibtex
@misc{ozcan_llm_emotion_audit_2026,
  author       = {Ozcan, Ahmet Remzi},
  title        = {Discourse-Level Affective Structure Framework},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/arozcan/discourse-affect-ad}
}
```

For ADReSSo, please also cite:

```bibtex
@inproceedings{luz2021adresso,
  author    = {Luz, Saturnino and Haider, Fasih and de la Fuente, Sofia and Fromm, Davida and MacWhinney, Brian},
  title     = {Detecting Cognitive Decline Using Speech Only: The ADReSSo Challenge},
  booktitle = {Proc. Interspeech 2021},
  year      = {2021},
  doi       = {10.21437/Interspeech.2021-1220}
}
```

## License & Contribution Guidelines

- This repository is provided for research and reproducibility purposes.
- External datasets remain subject to their original licenses and access rules.
- ADReSSo data must not be redistributed outside the permissions granted by
  DementiaBank/TalkBank.
- Contributions should preserve dataset access restrictions and clearly
  document any preprocessing or model changes.
