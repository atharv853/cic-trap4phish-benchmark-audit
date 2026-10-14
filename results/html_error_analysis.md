# HTML Specialist Model Error Analysis (LIME, SHAP & Permutation Audit)

## Error Statistics
- **Total Test Samples**: 3081
- **Total Test Errors**: 223 (False Positives: 117, False Negatives: 106)
- **Test Accuracy**: 0.9276
- **Test F1 Score**: 0.9276

## Top-1 LIME Predictors by Error Type
- **False Positives (FP Top-1 Features)**: `mailto_link_count` (112), `keywords_to_words_ratio` (2), `url_avg_length` (2)
- **False Negatives (FN Top-1 Features)**: `mailto_link_count` (104), `tag_count` (1), `form_count` (1)

## Methodological Comparison on Error Samples
- **LIME Top-5 Features**: `mailto_link_count`, `keywords_to_words_ratio`, `url_avg_length`, `tag_count`, `form_count`
- **SHAP Top-5 Features (Global Error Impact)**: `url_punct_char_count`, `form_count`, `keywords_to_words_ratio`, `script_entropy`, `embedded_js_count`
- **Permutation Importance Top-5 Features**: `object_tag_count`, `external_link_count`, `eval_in_script_blocks`, `redirect_mechanism_count`, `escaped_char_count`

## Top-5 Ranking Overlaps Across Methods
- **LIME & SHAP Overlap**: 2 of 5 features
- **LIME & Permutation Overlap**: 0 of 5 features
- **SHAP & Permutation Overlap**: 0 of 5 features
- **Features Shared Across All Three Methods**: 0 of 5 features

## Diagnostic Summary
- Local tabular perturbation in LIME highlights instance-level boundary features for error cases.
- Cross-method feature audit reveals 0 features shared between SHAP global impact and Permutation Importance on error instances, while LIME shares 2 top feature(s) with SHAP and 0 with Permutation Importance.

## Error Pattern Clustering (K-Means)
- **Optimal Error Clusters (K)**: 3
- **Silhouette Score**: 0.8840

Detailed per-sample signed LIME weights are recorded in `results/html_errors_lime.csv`.
