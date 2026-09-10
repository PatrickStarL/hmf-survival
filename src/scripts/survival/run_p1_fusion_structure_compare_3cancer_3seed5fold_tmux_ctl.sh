#!/usr/bin/env bash
set -euo pipefail

SELF_SCRIPT="${SELF_SCRIPT:-/path/to/hmf-survival/src/scripts/survival/run_p1_fusion_structure_compare_3cancer_3seed5fold_tmux_ctl.sh}"
PROJECT_DIR="${PROJECT_DIR:-/path/to/hmf-survival}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/src/scripts/survival/run_p1_fusion_structure_compare_3cancer_3seed5fold.sh}"
BASE_SCRIPT="${BASE_SCRIPT:-${PROJECT_DIR}/src/scripts/survival/run_p1_fusion_structure_compare_3cancer_5seed5fold.sh}"
SESSION="${SESSION:-hmf_p1_fusion_structure_compare_3cancer_3seed5fold}"
GUARD_SESSION="${GUARD_SESSION:-hmf_p1_fusion_structure_compare_3cancer_3seed5fold_guard}"
STATE_DIR="${STATE_DIR:-/path/to/your/data_root/.tmux_state}"
RUN_ROOT_FILE="${RUN_ROOT_FILE:-${STATE_DIR}/${SESSION}_run_root.txt}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_SCRIPT="${PREP_SCRIPT:-${PROJECT_DIR}/src/scripts/survival/prepare_crc_mstar_casefold_alias.py}"

NUM_WORKERS="${NUM_WORKERS:-8}"
RETRY="${RETRY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PREP_CRC_ALIAS="${PREP_CRC_ALIAS:-1}"

STRUCTURE_LIST="${STRUCTURE_LIST:-twostage coattn_text shallow_cls}"
DATASET_LIST="${DATASET_LIST:-LUAD STAD CRC}"
SEED_LIST="${SEED_LIST:-1 7 13}"
FOLD_LIST="${FOLD_LIST:-0 1 2 3 4}"
TARGET_COL="${TARGET_COL:-dss_survival_days}"

LR="${LR:-0.0004}"
WD="${WD:-1e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
BAG_SIZE="${BAG_SIZE:-16}"
TRAIN_BAG_SIZE="${TRAIN_BAG_SIZE:-16}"
VAL_BAG_SIZE="${VAL_BAG_SIZE:-0}"
ES_MIN_EPOCHS="${ES_MIN_EPOCHS:-5}"
ES_PATIENCE="${ES_PATIENCE:-6}"
ES_METRIC="${ES_METRIC:-loss}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.002}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.07}"
PATHWAY_DROPOUT="${PATHWAY_DROPOUT:-0.0}"
OPT="${OPT:-adamW}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_PROTO="${N_PROTO:-32}"
PATHWAY_TOKEN_COUNT="${PATHWAY_TOKEN_COUNT:-16}"
TAU="${TAU:-0.001}"
OT_EPS="${OT_EPS:-0.1}"
EM_ITER="${EM_ITER:-1}"
VISION_IN_DIM="${VISION_IN_DIM:-1024}"
TEXT_IN_DIM="${TEXT_IN_DIM:-512}"
PATH_PROJ_DIM="${PATH_PROJ_DIM:-256}"
CLS_DEPTH="${CLS_DEPTH:-2}"
CLS_NUM_HEADS="${CLS_NUM_HEADS:-4}"
CLS_MLP_RATIO="${CLS_MLP_RATIO:-2.0}"
CLS_DROPOUT="${CLS_DROPOUT:-0.1}"
APPEND_EMBED="${APPEND_EMBED:-random}"

SPLIT_ROOT_CRC="${SPLIT_ROOT_CRC:-${PROJECT_DIR}/src/splits/tcga-coadread}"
SOURCE_CRC_VISION_DIR="${SOURCE_CRC_VISION_DIR:-/path/to/your/data_root/tcga_crc/features_mstar_crc/feats_pt}"
CRC_ALIAS_ROOT="${CRC_ALIAS_ROOT:-${PROJECT_DIR}/src/results/crc_mstar_casefold_alias_latest}"
CRC_ALIAS_VISION_DIR="${CRC_ALIAS_VISION_DIR:-${CRC_ALIAS_ROOT}/feats_pt}"
CRC_ALIAS_SUMMARY_JSON="${CRC_ALIAS_SUMMARY_JSON:-${CRC_ALIAS_ROOT}/summary.json}"

PASS_ENV_VARS=(
  PROJECT_DIR
  PYTHON_BIN
  PREP_SCRIPT
  BASE_SCRIPT
  NUM_WORKERS
  RETRY
  SKIP_EXISTING
  PREP_CRC_ALIAS
  STRUCTURE_LIST
  DATASET_LIST
  SEED_LIST
  FOLD_LIST
  TARGET_COL
  LR
  WD
  MAX_EPOCHS
  BAG_SIZE
  TRAIN_BAG_SIZE
  VAL_BAG_SIZE
  ES_MIN_EPOCHS
  ES_PATIENCE
  ES_METRIC
  CONTRASTIVE_WEIGHT
  CONTRASTIVE_TEMP
  PATHWAY_DROPOUT
  OPT
  LR_SCHEDULER
  WARMUP_EPOCHS
  BATCH_SIZE
  N_PROTO
  PATHWAY_TOKEN_COUNT
  TAU
  OT_EPS
  EM_ITER
  VISION_IN_DIM
  TEXT_IN_DIM
  PATH_PROJ_DIM
  CLS_DEPTH
  CLS_NUM_HEADS
  CLS_MLP_RATIO
  CLS_DROPOUT
  APPEND_EMBED
  SPLIT_ROOT_CRC
  SOURCE_CRC_VISION_DIR
  CRC_ALIAS_ROOT
  CRC_ALIAS_VISION_DIR
  CRC_ALIAS_SUMMARY_JSON
)

mkdir -p "${STATE_DIR}"

shell_quote() {
  printf '%q' "$1"
}

new_run_root() {
  date -u +"/path/to/your/data_root/formal_p1_fusion_structure_compare_3cancer_3seed5fold_%Y%m%d_%H%M%S"
}

get_run_root() {
  if [[ -f "${RUN_ROOT_FILE}" ]]; then
    cat "${RUN_ROOT_FILE}"
  fi
}

latest_run_root() {
  local rr
  rr="$(get_run_root || true)"
  if [[ -n "${rr}" && -d "${rr}" ]]; then
    echo "${rr}"
    return 0
  fi
  ls -dt /path/to/your/data_root/formal_p1_fusion_structure_compare_3cancer_3seed5fold_* 2>/dev/null | head -n 1 || true
}

launcher_cmd() {
  local rr="$1"
  local launcher_log="${rr}/launcher.log"
  local cmd
  local var

  cmd="mkdir -p $(shell_quote "${rr}") && cd $(shell_quote "${PROJECT_DIR}") && env RUN_ROOT=$(shell_quote "${rr}")"
  for var in "${PASS_ENV_VARS[@]}"; do
    cmd+=" $(shell_quote "${var}=${!var}")"
  done
  cmd+=" bash $(shell_quote "${RUN_SCRIPT}") >> $(shell_quote "${launcher_log}") 2>&1"
  printf '%s' "${cmd}"
}

cmd_start() {
  local rr="${1:-}"

  if [[ -z "${rr}" ]]; then
    rr="$(get_run_root || true)"
  fi
  if [[ -z "${rr}" ]]; then
    rr="$(new_run_root)"
  fi

  mkdir -p "${rr}"
  echo "${rr}" > "${RUN_ROOT_FILE}"

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[start] tmux session already exists: ${SESSION}"
    echo "run_root=${rr}"
    return 0
  fi

  tmux new-session -d -s "${SESSION}" "$(launcher_cmd "${rr}")"
  sleep 1
  echo "[start] started ${SESSION}"
  echo "run_root=${rr}"
}

cmd_status() {
  local rr
  rr="$(latest_run_root || true)"

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "session=${SESSION} status=up"
  else
    echo "session=${SESSION} status=down"
  fi

  if tmux has-session -t "${GUARD_SESSION}" 2>/dev/null; then
    echo "guard_session=${GUARD_SESSION} status=up"
  else
    echo "guard_session=${GUARD_SESSION} status=down"
  fi

  echo "run_script=${RUN_SCRIPT}"
  echo "structures=${STRUCTURE_LIST}"
  echo "datasets=${DATASET_LIST}"
  echo "seeds=${SEED_LIST}"
  echo "folds=${FOLD_LIST}"
  echo "target_col=${TARGET_COL}"
  echo "max_epochs=${MAX_EPOCHS}"
  echo "lr=${LR}"
  echo "wd=${WD}"

  echo "--- processes ---"
  if [[ -n "${rr}" ]]; then
    pgrep -af "${rr}" || true
  else
    pgrep -af "run_p1_fusion_structure_compare_3cancer_3seed5fold.sh" || true
  fi

  if [[ -z "${rr}" ]]; then
    echo "run_root=NA"
    return 0
  fi

  echo "run_root=${rr}"

  local status_tsv="${rr}/reports/run_status.tsv"
  local progress_log="${rr}/reports/progress.log"
  local summary_md="${rr}/reports/cv_summary_structure_compare.md"
  local summary_json="${rr}/reports/cv_summary_structure_compare.json"
  local launcher_log="${rr}/launcher.log"

  if [[ -f "${status_tsv}" ]]; then
    awk -F '\t' 'NR>1{tot++; if($2=="DONE")d++; else if($2=="SKIP")s++; else if($2=="FAIL")f++} END{printf("progress total=%d done=%d skip=%d fail=%d\n", tot+0, d+0, s+0, f+0)}' "${status_tsv}"
    echo "status_tsv=${status_tsv}"
  fi
  if [[ -f "${summary_md}" ]]; then
    echo "summary_md=${summary_md}"
  fi
  if [[ -f "${summary_json}" ]]; then
    echo "summary_json=${summary_json}"
  fi
  if [[ -f "${launcher_log}" ]]; then
    echo "launcher_log=${launcher_log}"
  fi
  if [[ -f "${progress_log}" ]]; then
    echo "--- progress tail ---"
    tail -n 20 "${progress_log}"
  fi
  if [[ -f "${status_tsv}" ]]; then
    echo "--- status tail ---"
    tail -n 20 "${status_tsv}"
  fi
}

cmd_logs() {
  local rr
  rr="$(latest_run_root || true)"
  if [[ -z "${rr}" ]]; then
    echo "[logs] no run root found"
    exit 1
  fi

  local launcher_log="${rr}/launcher.log"
  local progress_log="${rr}/reports/progress.log"
  local status_tsv="${rr}/reports/run_status.tsv"

  echo "run_root=${rr}"
  if [[ -f "${launcher_log}" ]]; then
    echo "--- launcher.log ---"
    tail -n 80 "${launcher_log}"
  fi
  if [[ -f "${progress_log}" ]]; then
    echo "--- progress.log ---"
    tail -n 80 "${progress_log}"
  fi
  if [[ -f "${status_tsv}" ]]; then
    echo "--- run_status.tsv ---"
    tail -n 80 "${status_tsv}"
  fi
}

cmd_attach() {
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[attach] session not running: ${SESSION}"
    exit 1
  fi
  tmux attach -t "${SESSION}"
}

cmd_stop() {
  local rr
  rr="$(get_run_root || true)"

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-session -t "${SESSION}"
    echo "[stop] killed ${SESSION}"
  else
    echo "[stop] session not running: ${SESSION}"
  fi

  pkill -f "run_p1_fusion_structure_compare_3cancer_3seed5fold.sh" 2>/dev/null || true
  if [[ -n "${rr}" ]]; then
    pkill -f "${rr}" 2>/dev/null || true
  fi
}

guard_env_prefix() {
  local prefix="SELF_SCRIPT=$(shell_quote "${SELF_SCRIPT}") SESSION=$(shell_quote "${SESSION}")"
  local var
  prefix+=" GUARD_SESSION=$(shell_quote "${GUARD_SESSION}")"
  prefix+=" RUN_SCRIPT=$(shell_quote "${RUN_SCRIPT}")"
  prefix+=" STATE_DIR=$(shell_quote "${STATE_DIR}")"
  prefix+=" RUN_ROOT_FILE=$(shell_quote "${RUN_ROOT_FILE}")"
  for var in "${PASS_ENV_VARS[@]}"; do
    prefix+=" $(shell_quote "${var}=${!var}")"
  done
  printf '%s' "${prefix}"
}

cmd_guard_start() {
  if tmux has-session -t "${GUARD_SESSION}" 2>/dev/null; then
    echo "[guard-start] guard already running: ${GUARD_SESSION}"
    return 0
  fi

  tmux new-session -d -s "${GUARD_SESSION}" "bash -lc 'while true; do if ! tmux has-session -t $(shell_quote "${SESSION}") 2>/dev/null; then $(guard_env_prefix) $(shell_quote "${SELF_SCRIPT}") start; fi; sleep 30; done'"
  echo "[guard-start] started ${GUARD_SESSION}"
}

cmd_guard_stop() {
  if tmux has-session -t "${GUARD_SESSION}" 2>/dev/null; then
    tmux kill-session -t "${GUARD_SESSION}"
    echo "[guard-stop] killed ${GUARD_SESSION}"
  else
    echo "[guard-stop] guard not running: ${GUARD_SESSION}"
  fi
}

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") start [RUN_ROOT]
  $(basename "$0") status
  $(basename "$0") logs
  $(basename "$0") attach
  $(basename "$0") stop
  $(basename "$0") guard-start
  $(basename "$0") guard-stop

Defaults:
  STRUCTURE_LIST=${STRUCTURE_LIST}
  DATASET_LIST=${DATASET_LIST}
  SEED_LIST=${SEED_LIST}
  FOLD_LIST=${FOLD_LIST}
USAGE
}

main() {
  local cmd="${1:-status}"
  case "${cmd}" in
    start)
      shift || true
      cmd_start "${1:-}"
      ;;
    status)
      cmd_status
      ;;
    logs)
      cmd_logs
      ;;
    attach)
      cmd_attach
      ;;
    stop)
      cmd_stop
      ;;
    guard-start)
      cmd_guard_start
      ;;
    guard-stop)
      cmd_guard_stop
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
