# Aggregate reference results

These are copies of the saved outputs used in the revised manuscript, not freshly generated results. Values in CSV files are proportions unless explicitly labeled otherwise.

- `official_test/metrics.csv`: 71-record official-test per-fold and fixed-ensemble performance and bootstrap CIs.
- `official_test/fold_model_summary.csv`: descriptive variation of five trained fold models on the same test records, not repeated-seed uncertainty.
- `official_test/paired_model_tests.csv`: exact McNemar / paired prediction-swap comparisons, Holm-adjusted within metric.
- `participant_grouped/affective_group_tests.csv`: final affective inference after averaging known repeated development visits; use this instead of recording-level preliminary tests.
- `sample_summary.csv`: historical sample summary; its `n_participants` column counts recordings and does not establish unique participants. New code uses `n_recordings` for this field.
- `segment_histogram_counts.csv`: historical histogram; its `n_participants` field counts recordings. New code uses `n_recordings`.
- `split_audit/fold_distribution.csv`: recording counts for the saved stratified folds and official test.

Source statistics: 166 development recordings, 71 official test recordings; grouped development analysis has 162 units. Source participant overlap remains unresolved for anonymized test IDs. See ../REPRODUCIBILITY.md for protocol and limitations. Individual clinical records, predictions and source identity crosswalks are not added to this release.
