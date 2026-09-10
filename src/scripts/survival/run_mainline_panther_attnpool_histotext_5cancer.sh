#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/hmf-survival}"
PREP_SCRIPT="${PREP_SCRIPT:-${PROJECT_DIR}/src/scripts/survival/prepare_crc_mstar_casefold_alias.py}"
BASE_RUN_SCRIPT="${BASE_RUN_SCRIPT:-${PROJECT_DIR}/src/scripts/survival/run_mainline_panther_attnpool_genehisto_5seed5fold.sh}"
SPLIT_ROOT="${SPLIT_ROOT:-${PROJECT_DIR}/src/splits/tcga-coadread}"
SOURCE_CRC_VISION_DIR="${SOURCE_CRC_VISION_DIR:-/path/to/your/data_root/tcga_crc/features_mstar_crc/feats_pt}"
CRC_ALIAS_ROOT="${CRC_ALIAS_ROOT:-${PROJECT_DIR}/src/results/crc_mstar_casefold_alias_latest}"
CRC_ALIAS_VISION_DIR="${CRC_ALIAS_VISION_DIR:-${CRC_ALIAS_ROOT}/feats_pt}"
CRC_ALIAS_SUMMARY_JSON="${CRC_ALIAS_SUMMARY_JSON:-${CRC_ALIAS_ROOT}/summary.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-/path/to/your/data_root/mainline_twostage_panther_mstar_conch_5cancer_${TS}}"
DATASET_LIST="${DATASET_LIST:-STAD CRC KIRC LUAD HNSC}"

echo "[5cancer] preparing CRC casefold alias: ${CRC_ALIAS_VISION_DIR}"
"${PYTHON_BIN}" "${PREP_SCRIPT}" \
  --split-root "${SPLIT_ROOT}" \
  --source-dir "${SOURCE_CRC_VISION_DIR}" \
  --out-dir "${CRC_ALIAS_VISION_DIR}" \
  --summary-json "${CRC_ALIAS_SUMMARY_JSON}"

echo "[5cancer] CRC alias summary: ${CRC_ALIAS_SUMMARY_JSON}"
echo "[5cancer] launching five-dataset mainline with DATASET_LIST=${DATASET_LIST}"

RUN_ROOT="${RUN_ROOT}" \
DATASET_LIST="${DATASET_LIST}" \
CRC_VISION_DIR="${CRC_ALIAS_VISION_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${BASE_RUN_SCRIPT}"
