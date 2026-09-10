#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/hmf-survival}"
BASE_SCRIPT="${BASE_SCRIPT:-${PROJECT_DIR}/src/scripts/survival/run_p1_fusion_structure_compare_3cancer_5seed5fold.sh}"
SEED_LIST="${SEED_LIST:-1 7 13}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${RUN_ROOT:-}" ]]; then
  TS="$(date -u +%Y%m%d_%H%M%S)"
  export RUN_ROOT="/path/to/your/data_root/formal_p1_fusion_structure_compare_3cancer_3seed5fold_${TS}"
fi

export SEED_LIST
exec bash "${BASE_SCRIPT}"
