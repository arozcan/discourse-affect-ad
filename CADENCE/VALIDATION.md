# Release validation — 2026-11-14

Checks were performed in the author's existing whisperx-gpu environment against the saved manuscript inputs. No training or model inference was run.

- Parsed all Python scripts under MADE, PLEA and CADENCE.
- Ran `--help` for the seven added/updated preprocessing, feature-extraction, evaluation and statistical/audit CLIs.
- Recomputed accuracy and weighted F1 for all 18 saved official-test metric rows from the saved predictions; values agreed with both the release CSV and scikit-learn.
- Recomputed all six paired model comparisons with the original seeds and 10,000 resamples; differences, intervals and raw/Holm p-values matched saved reference outputs.
- Ran official-test evaluation with `--dry_run`: all 71 test records and 15 original checkpoints passed input validation.
- Ran the grouped affective analysis in a temporary output directory: all five differences, intervals and p-values matched saved reference outputs (162 analysis units).
- Ran the split/source audit using local authorized audio, feature cache, metadata and saved fold predictions. Saved memberships matched reconstructed folds; source participant overlap remained detected.

These are checks of code portability and agreement with saved analyses, not an independent reproduction of upstream training or clinical validation. Model checkpoints and clinical inputs must be obtained/generated separately. Per-record clinical artifacts used for validation are not newly distributed here.
