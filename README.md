# Foundation-Model-Driven Two-Stage Hierarchical Multimodal Cancer Survival Prediction

This repository contains the experiment code for an undergraduate thesis, built on top of PS3 (PS3: A Multimodal Transformer Integrating Pathology Reports with Histology Images and Biological Pathways for Cancer Survival Prediction, ICCV 2025). It redesigns the modality tokenization and fusion structure for cancer survival prediction on TCGA cohorts, integrating whole-slide histology images, pathology reports, and transcriptomic (RNA) data.

## Method Overview

For each case, prognostic signal is extracted from three heterogeneous evidence sources, unified into tokens, and fused in two stages:

1. **Modality tokenization**
   - **Histology**: a frozen mSTAR encoder extracts patch-level features, compressed into 32 pathology prototype tokens via PANTHER prototype-based statistical modeling;
   - **Pathology report**: a frozen CONCH encoder embeds the report text; a self-attention-based length-reshaping module adaptively selects Top-K text tokens based on per-fold statistics;
   - **RNA / transcriptomics**: expression is reorganized along the 50 MSigDB Hallmark pathways, encoded by a per-pathway self-normalizing MLP, then compressed from 50 pathway tokens into 16 omics tokens;
   - All three modalities are linearly projected into a shared space of dimension d=256.
2. **Two-stage hierarchical fusion**
   - **Stage 1 (token-level alignment)**: the concatenated multimodal token sequence goes through a co-attention block for fine-grained cross-modal interaction;
   - **Stage 2 (modality-level summarization)**: each modality is attention-pooled into a summary token; the three summary tokens plus a learnable CLS token are fed into a shallow Transformer (depth=2, heads=4) to produce the case-level joint representation.
3. **Training objective**: Cox partial likelihood (primary supervision) plus a bidirectional histo-text InfoNCE contrastive loss (weight 0.002, temperature 0.07) to mitigate representation-scale discrepancy between the two independently pretrained foundation models (mSTAR, CONCH).

## Results

All numbers below are computed directly from the raw per-fold/per-seed logs under `results/` (see `results/*/cv_summary_*.json` for the exact source). They are reported as-is, including where they do not favor this work.

### Mainline (5 TCGA cohorts, 5-fold x 5-seed, 125 runs)

| Model | STAD | CRC | KIRC | LUAD | HNSC | Mean |
|---|---:|---:|---:|---:|---:|---:|
| PS3 (as reported in the original paper) [1] | 0.638 | 0.826 | 0.776 | 0.640 | 0.627 | 0.701 |
| This work | 0.598 +/- 0.073 | 0.776 +/- 0.078 | 0.755 +/- 0.059 | 0.660 +/- 0.059 | 0.583 +/- 0.056 | **0.674** |

The PS3 row is quoted directly from the original paper's own published table (5 of their 6 reported cohorts, excluding BLCA which this repo does not use), not an independent reproduction in this repo. On this protocol, the redesigned tokenization/fusion in this work does **not** outperform PS3.

### Ablation: modality combination (CRC/LUAD/STAD, 3 seeds x 5 folds)

| Modalities | CRC | LUAD | STAD | Mean |
|---|---:|---:|---:|---:|
| Histology only | 0.697 +/- 0.089 | 0.583 +/- 0.052 | 0.605 +/- 0.069 | 0.628 |
| Histology + text | 0.717 +/- 0.097 | 0.602 +/- 0.064 | 0.545 +/- 0.066 | 0.622 |
| Histology + RNA | 0.764 +/- 0.074 | 0.616 +/- 0.051 | 0.635 +/- 0.071 | 0.672 |
| Histology + text + RNA (full) | 0.778 +/- 0.082 | 0.650 +/- 0.057 | 0.601 +/- 0.060 | 0.676 |

RNA is the modality that most consistently helps; adding text on top of histology is not consistently beneficial (it helps CRC, hurts STAD, where reports are more structured/templated).

### Ablation: fusion structure (CRC/LUAD/STAD, 3 seeds x 5 folds)

| Fusion structure | CRC | LUAD | STAD | Mean |
|---|---:|---:|---:|---:|
| Flat co-attention (single-stage, PS3-style) | 0.787 +/- 0.062 | 0.631 +/- 0.063 | 0.636 +/- 0.050 | **0.685** |
| Shallow CLS (late fusion) | 0.767 +/- 0.067 | 0.622 +/- 0.068 | 0.585 +/- 0.089 | 0.658 |
| Two-stage hierarchical (this work) | 0.778 +/- 0.082 | 0.650 +/- 0.057 | 0.601 +/- 0.060 | 0.676 |

**On this data, the two-stage hierarchical structure does not outperform flat single-stage co-attention fusion.** This is consistent with PS3's own ablation study, which tested a hierarchical fusion variant and found it 5.58% worse than their single-stage joint attention [1]. The result here should be read as an independent replication of that finding under a different encoder/data setup, not as evidence that staged fusion helps.

### KM survival stratification (5-cancer ensemble out-of-fold)

| Cohort | Cases | Low/High risk | Ensemble OOF C-index | Log-rank p |
|---|---:|---:|---:|---:|
| STAD | 253 | 126/127 | 0.625 | 1.7e-03 |
| CRC | 269 | 134/135 | 0.785 | 1.2e-04 |
| KIRC | 328 | 164/164 | 0.758 | 4.3e-09 |
| LUAD | 391 | 195/196 | 0.667 | 4.7e-05 |
| HNSC | 385 | 192/193 | 0.584 | 0.070 |

Log-rank p < 0.01 (strongly significant) for KIRC, CRC, LUAD, STAD; HNSC reaches p = 0.070, near but below the conventional significance threshold. This part of the analysis is independent of the C-index numbers above and holds regardless of them.

Raw result artifacts live under results/: mainline_5cancer/ (main protocol), ablation_modality/, ablation_structure/ (two ablation studies), and km_stratification/ (KM curves, log-rank tests, figures and tables). Every number in this section can be recomputed from those files.

[1] Raza et al., *PS3: A Multimodal Transformer Integrating Pathology Reports with Histology Images and Biological Pathways for Cancer Survival Prediction*, ICCV 2025. [arXiv:2509.20022](https://arxiv.org/abs/2509.20022)

## Repository Layout

```text
hmf-survival/
|-- requirements.txt
|-- configs/
|   `-- PANTHER_default/config.json      # default PANTHER prototype-aggregation config
|-- results/                             # aggregated metrics + KM figures (no model weights)
|   |-- mainline_5cancer/
|   |-- ablation_modality/
|   |-- ablation_structure/
|   `-- km_stratification/
`-- src/
    |-- mil_models/                      # model definitions
    |   |-- model_multimodal_hierarchical_fusion.py   # two-stage fusion model (top-level module)
    |   |-- model_multimodal_encoders.py              # per-modality encoders / token construction
    |   |-- model_multimodal_cls_fusion.py            # shallow CLS-Transformer summary fusion
    |   |-- model_factory_hierarchical_fusion.py      # model builder / entry point
    |   |-- components.py                # attention (CoAttention/CoAttentionLayer) / FFN / survival head modules
    |   |-- model_PANTHER.py, PANTHER/   # PANTHER prototype-based statistical modeling
    |   |-- model_OT.py, OT/             # optional optimal-transport aggregator (not wired up by default, see note below)
    |   |-- model_h2t.py, model_protocount.py, model_configs.py
    |   |-- text_processing.py, tokenizer.py
    |   `-- models/                       # ABMIL, text baseline, shared building blocks
    |-- training/
    |   |-- main_survival_hierarchical_fusion.py      # train/eval entry point
    |   `-- trainer_hierarchical_fusion.py            # training loop, checkpoint, metrics
    |-- wsi_datasets/                     # datasets and batch assembly
    |-- utils/                            # loss, scheduler, split I/O, misc utilities
    |-- scripts/survival/                 # batch run scripts for the mainline and both ablations
    |-- splits/                           # 5-fold train/test splits for 5 cancers (case_id, survival labels, no images/features)
    `-- data_csvs/rna/                    # Hallmark-pathway RNA expression matrices (derived from public TCGA data)
```

Not included in this repository: raw WSI slides, pre-extracted patch/text features (.pt), mSTAR/CONCH model weights, or any training checkpoint (.pth). These range from hundreds of MB to several GB and must be prepared separately (see below).

Note on `model_OT.py`: the mainline model always uses `model_histo_type=PANTHER`; the alternative `OT` aggregator path depends on an optimal-transport-kernel submodule that is not vendored in this repository, so it is only imported lazily if you explicitly request `model_type='OT'`. It is not needed to reproduce any of the results reported here.

## Setup

Python 3.10+ and PyTorch 2.1+ are recommended (experiments in the thesis used PyTorch 2.8.0 + CUDA 12.8 on a single RTX 3090).

```bash
pip install -r requirements.txt
```

Common issues:

- `ModuleNotFoundError: ot`: install POT via `pip install POT` (the import name is `ot`).
- `ModuleNotFoundError: faiss`: if you use the original PS3 offline prototype pipeline, `main_prototype` can fall back to `--mode kmeans`.
- `torch.load(weights_only=...)` errors on PyTorch 2.6+: feature/text loading in this repo already handles this.

## Preparing Data

Three kinds of offline features must be prepared and organized per case before training:

1. WSI patch features: extract patch-level features from TCGA slides with mSTAR (or another pathology foundation model), producing one `.pt` file per slide (shape `(num_patches, 1024)`) under the directory passed as `--data_source` (a `feats_pt/` folder).
2. Pathology report embeddings: encode report text with CONCH, producing one `.pt` file per case under `--reports_dir`.
3. RNA expression matrices: already provided in `src/data_csvs/rna/hallmarks/<CANCER>/rna_clean.csv` (5 cancers, Hallmark-pathway reorganized), ready to use.
4. Splits: already provided in `src/splits/tcga-<cancer>/TCGA_<CANCER>_overall_survival_k=<fold>/` (train.csv, test.csv).

`splits/` covers the five cancers reported in the thesis: STAD, CRC(COADREAD), KIRC, LUAD, HNSC. WSI and report data originate from TCGA via the NCI GDC portal; cohort sizes (from thesis Table 4.1):

| Cohort | Cases | Slides |
|---|---:|---:|
| STAD | 253 | 253 |
| CRC | 269 | 272 |
| KIRC | 328 | 331 |
| LUAD | 391 | 441 |
| HNSC | 385 | 405 |

## Running Experiments

Scripts under `src/scripts/survival/` correspond to every experiment in the thesis (Chapter 5). Local paths inside the scripts have been replaced with placeholders (`/path/to/hmf-survival`, `/path/to/your/data_root`) -- override them via environment variables, or edit the variables at the top of each script directly:

```bash
export PROJECT_DIR=/your/local/path/to/hmf-survival
export DATA_ROOT=/your/local/path/to/extracted_features   # root holding feats_pt / reports_pt

bash src/scripts/survival/run_mainline_panther_attnpool_genehisto_5seed5fold.sh   # mainline: 5 cancers x 5 folds x 5 seeds
bash src/scripts/survival/run_p1_modality_contribution_ablation_3cancer_3seed5fold.sh  # ablation: modality combinations
bash src/scripts/survival/run_p1_fusion_structure_compare_3cancer_3seed5fold.sh        # ablation: fusion structures
bash src/scripts/survival/run_p1_survival_stratification_5cancer.sh                    # KM stratification + log-rank tests
```

Or call the training entry point directly with custom arguments:

```bash
python -m src.training.main_survival_hierarchical_fusion \
  --split_dir "$PROJECT_DIR/src/splits/tcga-stad/TCGA_STAD_overall_survival_k=0" \
  --data_source "$DATA_ROOT/tcga_stad/features_mstar_stad/feats_pt" \
  --reports_dir "$DATA_ROOT/tcga_stad/features_conch_stad/reports_pt" \
  --omics_dir "$PROJECT_DIR/src/data_csvs/rna" \
  --type_of_path hallmarks \
  --n_proto 32 --pathway_token_count 16 \
  --loss_fn cox --max_epochs 40
```

## Citation and Acknowledgments

This project builds on:

- Raza et al., PS3: A Multimodal Transformer Integrating Pathology Reports with Histology Images and Biological Pathways for Cancer Survival Prediction, ICCV 2025.
- Song et al., Morphological prototyping for unsupervised slide representation learning in computational pathology (PANTHER), CVPR 2024.
- Xu et al., A multimodal knowledge-enhanced whole-slide pathology foundation model (mSTAR), Nature Communications 2025.
- Lu et al., A visual-language foundation model for computational pathology (CONCH), Nature Medicine 2024.
- TCGA (The Cancer Genome Atlas): source of WSI and RNA data; report text sourced from the TCGA-Reports machine-readable dataset.

## Disclaimer

This repository is for academic research and reproducibility only and contains no patient-identifiable data. Everything under `src/data_csvs/rna/` and `src/splits/` consists of derived statistics from public TCGA data -- no imaging, raw sequencing reads, or patient-identifying information.
