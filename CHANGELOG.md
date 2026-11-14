# Changelog

## 1.1.0 — 2026-11-14

- Add official-test audio-label preparation, split-specific ASR and feature extraction, and inference from existing five-fold checkpoints.
- Add paired model comparisons, grouped repeated-visit affective analysis, and saved split/source-identity auditing.
- Expose analysis input paths for use outside the manuscript workspace.
- Include aggregate official-test, paired-comparison, grouped-analysis and split-count results.
- Document checkpoint-selection bias, participant overlap, full-recording ASR and the scope of held-out benchmark evaluation.
- Remove third-party datasets, ADReSSo-derived transcripts and feature caches from the tracked release; retain local copies and document source access. Earlier Git history is unchanged.
- Preserve MADE synthetic data and the existing training implementations. Model weights are not included.
