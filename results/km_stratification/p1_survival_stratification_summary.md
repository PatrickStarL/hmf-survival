# P1 Survival Stratification Summary

- reuse_mode: direct reuse of completed mainline 5-seed x 5-fold test predictions
- official_display: 5-seed mean-risk ensemble OOF KM plot for each cancer
- split_rule: median risk split with balanced fallback when needed
- time_unit_for_plots: years

## Ensemble Figures

| Dataset | Cases | Low risk | High risk | Ensemble OOF c-index | Log-rank p | Median survival low | Median survival high | Figure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| STAD | 253 | 126 | 127 | 0.6249 | 0.0017 | NR | 3.54 | km_STAD_ensemble_oof.png |
| CRC | 269 | 134 | 135 | 0.7847 | 0.000117 | NR | NR | km_CRC_ensemble_oof.png |
| KIRC | 328 | 164 | 164 | 0.7582 | 4.3e-09 | NR | NR | km_KIRC_ensemble_oof.png |
| LUAD | 391 | 195 | 196 | 0.6670 | 4.7e-05 | NR | 3.89 | km_LUAD_ensemble_oof.png |
| HNSC | 385 | 192 | 193 | 0.5845 | 0.0703 | NR | 7.04 | km_HNSC_ensemble_oof.png |

## Combined Panel

- panel_png: km_panel_5cancer_ensemble.png

## Seed-Level Reference

| Dataset | Seed | Mean fold c-index | OOF c-index | Log-rank p | Merge note |
| --- | ---: | ---: | ---: | ---: | --- |
| STAD | 1 | 0.5587 | 0.5571 | 0.3213 | unique_sample_ids |
| STAD | 7 | 0.6327 | 0.6265 | 0.0043 | unique_sample_ids |
| STAD | 13 | 0.6115 | 0.6082 | 0.0095 | unique_sample_ids |
| STAD | 21 | 0.5711 | 0.5801 | 0.0515 | unique_sample_ids |
| STAD | 42 | 0.6148 | 0.6087 | 0.0280 | unique_sample_ids |

| CRC | 1 | 0.7776 | 0.7695 | 0.0014 | unique_sample_ids |
| CRC | 7 | 0.7621 | 0.7658 | 0.000804 | unique_sample_ids |
| CRC | 13 | 0.8004 | 0.7749 | 7.3e-05 | unique_sample_ids |
| CRC | 21 | 0.7788 | 0.7529 | 0.0019 | unique_sample_ids |
| CRC | 42 | 0.7594 | 0.6928 | 0.0282 | unique_sample_ids |

| KIRC | 1 | 0.7477 | 0.7347 | 4.6e-06 | unique_sample_ids |
| KIRC | 7 | 0.7654 | 0.7558 | 1.7e-08 | unique_sample_ids |
| KIRC | 13 | 0.7355 | 0.7121 | 1.7e-06 | unique_sample_ids |
| KIRC | 21 | 0.7489 | 0.7387 | 1.1e-08 | unique_sample_ids |
| KIRC | 42 | 0.7751 | 0.7503 | 6.2e-06 | unique_sample_ids |

| LUAD | 1 | 0.6486 | 0.6494 | 0.0012 | unique_sample_ids |
| LUAD | 7 | 0.6832 | 0.6691 | 7.5e-05 | unique_sample_ids |
| LUAD | 13 | 0.6489 | 0.6449 | 0.000625 | unique_sample_ids |
| LUAD | 21 | 0.6695 | 0.6593 | 2.2e-05 | unique_sample_ids |
| LUAD | 42 | 0.6518 | 0.6468 | 0.0013 | unique_sample_ids |

| HNSC | 1 | 0.5826 | 0.5718 | 0.0454 | unique_sample_ids |
| HNSC | 7 | 0.5785 | 0.5713 | 0.0086 | unique_sample_ids |
| HNSC | 13 | 0.6116 | 0.6075 | 0.0423 | unique_sample_ids |
| HNSC | 21 | 0.5779 | 0.5792 | 0.0259 | unique_sample_ids |
| HNSC | 42 | 0.5621 | 0.5670 | 0.0024 | unique_sample_ids |
