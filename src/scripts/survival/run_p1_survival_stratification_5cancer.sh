#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/hmf-survival}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-${PROJECT_DIR}/src/scripts/survival/run_p1_survival_stratification_5cancer.py}"
DATASET_LIST="${DATASET_LIST:-STAD CRC KIRC LUAD HNSC}"
FOLD_LIST="${FOLD_LIST:-0 1 2 3 4}"
TARGET_COL="${TARGET_COL:-dss_survival_days}"
TIME_UNIT="${TIME_UNIT:-years}"
MIN_GROUP_SIZE="${MIN_GROUP_SIZE:-5}"
GENERATE_SEED_PLOTS="${GENERATE_SEED_PLOTS:-1}"
GENERATE_ENSEMBLE_PLOTS="${GENERATE_ENSEMBLE_PLOTS:-1}"
GENERATE_PANEL="${GENERATE_PANEL:-1}"
DPI="${DPI:-220}"
MAINLINE_RUN_ROOT="${MAINLINE_RUN_ROOT:-$(ls -dt /path/to/your/data_root/mainline_twostage_panther_mstar_conch_5cancer_* 2>/dev/null | head -n 1 || true)}"
TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-/path/to/your/data_root/formal_p1_survival_stratification_5cancer_${TS}}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "runner not found: ${RUNNER}" >&2
  exit 1
fi

if [[ -z "${MAINLINE_RUN_ROOT}" ]]; then
  echo "MAINLINE_RUN_ROOT is empty and no latest 5-cancer mainline run was found." >&2
  exit 1
fi

if [[ ! -d "${MAINLINE_RUN_ROOT}" ]]; then
  echo "mainline run root not found: ${MAINLINE_RUN_ROOT}" >&2
  exit 1
fi

read -r -a DATASETS <<< "${DATASET_LIST}"
read -r -a FOLDS <<< "${FOLD_LIST}"

echo "[p1-stratification] run_root=${RUN_ROOT}"
echo "[p1-stratification] mainline_run_root=${MAINLINE_RUN_ROOT}"
echo "[p1-stratification] datasets=${DATASET_LIST}"
echo "[p1-stratification] folds=${FOLD_LIST}"
echo "[p1-stratification] target_col=${TARGET_COL}"
echo "[p1-stratification] time_unit=${TIME_UNIT}"
echo "[p1-stratification] min_group_size=${MIN_GROUP_SIZE}"
echo "[p1-stratification] generate_seed_plots=${GENERATE_SEED_PLOTS}"
echo "[p1-stratification] generate_ensemble_plots=${GENERATE_ENSEMBLE_PLOTS}"
echo "[p1-stratification] generate_panel=${GENERATE_PANEL}"

"${PYTHON_BIN}" "${RUNNER}" \
  --run-root "${RUN_ROOT}" \
  --mainline-run-root "${MAINLINE_RUN_ROOT}" \
  --datasets "${DATASETS[@]}" \
  --folds "${FOLDS[@]}" \
  --target-col "${TARGET_COL}" \
  --time-unit "${TIME_UNIT}" \
  --min-group-size "${MIN_GROUP_SIZE}" \
  --generate-seed-plots "${GENERATE_SEED_PLOTS}" \
  --generate-ensemble-plots "${GENERATE_ENSEMBLE_PLOTS}" \
  --generate-panel "${GENERATE_PANEL}" \
  --dpi "${DPI}"
