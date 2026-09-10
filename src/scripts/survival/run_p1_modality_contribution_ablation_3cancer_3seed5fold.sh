#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/hmf-survival}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_SCRIPT="${PREP_SCRIPT:-${PROJECT_DIR}/src/scripts/survival/prepare_crc_mstar_casefold_alias.py}"

TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-/path/to/your/data_root/formal_p1_modality_contribution_ablation_3cancer_3seed5fold_${TS}}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
REPORT_ROOT="${REPORT_ROOT:-${RUN_ROOT}/reports}"
STATUS_TSV="${STATUS_TSV:-${REPORT_ROOT}/run_status.tsv}"

NUM_WORKERS="${NUM_WORKERS:-8}"
RETRY="${RETRY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PREP_CRC_ALIAS="${PREP_CRC_ALIAS:-1}"

MODALITY_LIST="${MODALITY_LIST:-histo histo_text histo_rna full}"
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

read -r -a MODALITIES <<< "${MODALITY_LIST}"
read -r -a DATASETS <<< "${DATASET_LIST}"
read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a FOLDS <<< "${FOLD_LIST}"

mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"

if [[ ! -f "${STATUS_TSV}" ]]; then
  printf "timestamp_utc\tstatus\tablation\tdataset\tseed\tfold\ttask\tc_index_test\tsummary_csv\tlog_file\tnote\n" > "${STATUS_TSV}"
fi

contains_word() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "${PREP_CRC_ALIAS}" == "1" ]] && contains_word "CRC" "${DATASETS[@]}"; then
  echo "[p1-modality] preparing CRC casefold alias: ${CRC_ALIAS_VISION_DIR}"
  "${PYTHON_BIN}" "${PREP_SCRIPT}" \
    --split-root "${SPLIT_ROOT_CRC}" \
    --source-dir "${SOURCE_CRC_VISION_DIR}" \
    --out-dir "${CRC_ALIAS_VISION_DIR}" \
    --summary-json "${CRC_ALIAS_SUMMARY_JSON}"
fi

cat > "${REPORT_ROOT}/run_config.txt" <<CFG
start_utc=$(date -u '+%F %T UTC')
project_dir=${PROJECT_DIR}
run_root=${RUN_ROOT}
profile=P1_modality_contribution_ablation_retained_mainline_3cancer
notes=Strict formal P1 modality-contribution ablation for LUAD/STAD/CRC at 3 seeds x 5 folds. All runs share the same retained-mainline twostage training entry, preprocessing, token budgets, contrastive settings, and optimization protocol. The only intentional change is active modality subset in {histo, histo+text, histo+rna, full}.
modalities=${MODALITY_LIST}
datasets=${DATASET_LIST}
seeds=${SEED_LIST}
folds=${FOLD_LIST}
retry=${RETRY}
skip_existing=${SKIP_EXISTING}
prep_crc_alias=${PREP_CRC_ALIAS}
num_workers=${NUM_WORKERS}
target_col=${TARGET_COL}
model_histo_type=PANTHER
model_histo_config=PANTHER_default
base_model_mm_type=twostage_cls
contrastive_pairs=histo_text
force_gene_histo_contrastive=false
use_gated_fusion=false
use_path_qformer=false
use_pathway_attnpool=true
freeze_conch=true
append_embed=${APPEND_EMBED}
hparams=max_epochs=${MAX_EPOCHS},lr=${LR},wd=${WD},opt=${OPT},lr_scheduler=${LR_SCHEDULER},warmup_epochs=${WARMUP_EPOCHS},batch_size=${BATCH_SIZE},bag_size=${BAG_SIZE},train_bag_size=${TRAIN_BAG_SIZE},val_bag_size=${VAL_BAG_SIZE},es_min_epochs=${ES_MIN_EPOCHS},es_patience=${ES_PATIENCE},es_metric=${ES_METRIC},contrastive_weight=${CONTRASTIVE_WEIGHT},contrastive_temp=${CONTRASTIVE_TEMP},pathway_dropout=${PATHWAY_DROPOUT},n_proto=${N_PROTO},pathway_token_count=${PATHWAY_TOKEN_COUNT},vision_in_dim=${VISION_IN_DIM},text_in_dim=${TEXT_IN_DIM},path_proj_dim=${PATH_PROJ_DIM},tau=${TAU},ot_eps=${OT_EPS},em_iter=${EM_ITER},cls_depth=${CLS_DEPTH},cls_num_heads=${CLS_NUM_HEADS},cls_mlp_ratio=${CLS_MLP_RATIO},cls_dropout=${CLS_DROPOUT}
crc_alias_root=${CRC_ALIAS_ROOT}
crc_alias_vision_dir=${CRC_ALIAS_VISION_DIR}
crc_alias_summary_json=${CRC_ALIAS_SUMMARY_JSON}
CFG

resolve_dataset() {
  local dataset="$1"
  case "${dataset}" in
    CRC)
      SPLIT_PREFIX="${PROJECT_DIR}/src/splits/tcga-coadread/TCGA_COADREAD_overall_survival_k="
      VISION_DIR="${CRC_ALIAS_VISION_DIR}"
      REPORT_DIR="/path/to/your/data_root/tcga_crc/features_conch_crc/reports_pt"
      OMICS_DIR="${PROJECT_DIR}/src/data_csvs/rna"
      OMICS_CHECK="${PROJECT_DIR}/src/data_csvs/rna/hallmarks/COADREAD/rna_clean.csv"
      ;;
    LUAD)
      SPLIT_PREFIX="${PROJECT_DIR}/src/splits/tcga-luad/TCGA_LUAD_overall_survival_k="
      VISION_DIR="/path/to/your/data_root/tcga_luad/features_mstar_luad/feats_pt"
      REPORT_DIR="/path/to/your/data_root/tcga_luad/features_conch_luad/reports_pt"
      OMICS_DIR="${PROJECT_DIR}/src/data_csvs/rna"
      OMICS_CHECK="${PROJECT_DIR}/src/data_csvs/rna/hallmarks/LUAD/rna_clean.csv"
      ;;
    STAD)
      SPLIT_PREFIX="${PROJECT_DIR}/src/splits/tcga-stad/TCGA_STAD_overall_survival_k="
      VISION_DIR="/path/to/your/data_root/tcga_stad/features_mstar_stad/feats_pt"
      REPORT_DIR="/path/to/your/data_root/tcga_stad/features_conch_stad/reports_pt"
      OMICS_DIR="${PROJECT_DIR}/src/data_csvs/rna"
      OMICS_CHECK="${PROJECT_DIR}/src/data_csvs/rna/hallmarks/STAD/rna_clean.csv"
      ;;
    *)
      echo "[ERROR] Unknown dataset: ${dataset}" >&2
      return 1
      ;;
  esac

  [[ -d "${VISION_DIR}" ]] || { echo "[ERROR] Missing VISION_DIR: ${VISION_DIR}" >&2; return 1; }
  [[ -d "${REPORT_DIR}" ]] || { echo "[ERROR] Missing REPORT_DIR: ${REPORT_DIR}" >&2; return 1; }
  [[ -f "${OMICS_CHECK}" ]] || { echo "[ERROR] Missing RNA file: ${OMICS_CHECK}" >&2; return 1; }
}

resolve_ablation() {
  local ablation="$1"
  case "${ablation}" in
    histo|histo_only|wsi)
      ABLATION_KEY="histo"
      ABLATION_LABEL="histo"
      ENTRY_MODULE="src.training.main_survival_hierarchical_fusion"
      MODEL_MM_TYPE="histo_only_twostage"
      EXP_CODE="p1_modality_histo_formal"
      ABLATION_NOTE="strict_modality_only_histo"
      ;;
    histo_text|histo+text)
      ABLATION_KEY="histo_text"
      ABLATION_LABEL="histo_text"
      ENTRY_MODULE="src.training.main_survival_hierarchical_fusion"
      MODEL_MM_TYPE="histo_text_twostage"
      EXP_CODE="p1_modality_histo_text_formal"
      ABLATION_NOTE="strict_modality_only_histo_text"
      ;;
    histo_rna|histo+rna|histo_gene)
      ABLATION_KEY="histo_rna"
      ABLATION_LABEL="histo_rna"
      ENTRY_MODULE="src.training.main_survival_hierarchical_fusion"
      MODEL_MM_TYPE="histo_rna_twostage"
      EXP_CODE="p1_modality_histo_rna_formal"
      ABLATION_NOTE="strict_modality_only_histo_rna"
      ;;
    full|tri_modal|trimodal)
      ABLATION_KEY="full"
      ABLATION_LABEL="full"
      ENTRY_MODULE="src.training.main_survival_hierarchical_fusion"
      MODEL_MM_TYPE="full_twostage"
      EXP_CODE="p1_modality_full_formal"
      ABLATION_NOTE="strict_modality_only_full"
      ;;
    *)
      echo "[ERROR] Unknown ablation: ${ablation}" >&2
      return 1
      ;;
  esac
}

extract_cindex() {
  local summary_csv="$1"
  python3 - "$summary_csv" <<'PY'
import csv
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
print(row.get("c_index_test", "NA"))
PY
}

total=0
done=0
skipped=0
failed=0
for _ablation in "${MODALITIES[@]}"; do
  for _dataset in "${DATASETS[@]}"; do
    for _seed in "${SEEDS[@]}"; do
      for _fold in "${FOLDS[@]}"; do
        total=$((total + 1))
      done
    done
  done
done

run_one() {
  local ablation="$1"
  local dataset="$2"
  local seed="$3"
  local fold="$4"

  resolve_ablation "${ablation}"
  resolve_dataset "${dataset}"

  local split_dir="${SPLIT_PREFIX}${fold}"
  local task="${dataset}_${ABLATION_KEY}_panther_modality_k${fold}_s${seed}"
  local result_root="${RUN_ROOT}/results/${ABLATION_KEY}/${dataset}/seed_${seed}/k_${fold}"
  local log_file="${LOG_ROOT}/${task}.log"

  mkdir -p "${result_root}"

  local existing
  existing="$(find "${result_root}" -type f -name summary.csv | head -n 1 || true)"
  if [[ "${SKIP_EXISTING}" == "1" && -n "${existing}" ]]; then
    skipped=$((skipped + 1))
    printf "%s\tSKIP\t%s\t%s\t%s\t%s\t%s\tNA\t%s\t%s\tresume_skip_existing\n" \
      "$(date -u '+%F %T UTC')" "${ABLATION_LABEL}" "${dataset}" "${seed}" "${fold}" "${task}" "${existing}" "${log_file}" >> "${STATUS_TSV}"
    return 0
  fi

  local -a cmd=(
    "${PYTHON_BIN}" -m "${ENTRY_MODULE}"
    --task "${task}"
    --exp_code "${EXP_CODE}"
    --split_dir "${split_dir}"
    --split_names train,test
    --data_source "${VISION_DIR}"
    --reports_dir "${REPORT_DIR}"
    --results_dir "${result_root}"
    --model_histo_type PANTHER
    --model_histo_config PANTHER_default
    --n_proto "${N_PROTO}"
    --in_dim "${VISION_IN_DIM}"
    --text_in_dim "${TEXT_IN_DIM}"
    --path_proj_dim "${PATH_PROJ_DIM}"
    --out_type allcat
    --em_iter "${EM_ITER}"
    --tau "${TAU}"
    --ot_eps "${OT_EPS}"
    --model_mm_type "${MODEL_MM_TYPE}"
    --loss_fn cox
    --omics_dir "${OMICS_DIR}"
    --omics_modality pathway
    --type_of_path hallmarks
    --attn_mode full_all_es
    --append_embed "${APPEND_EMBED}"
    --bag_size "${BAG_SIZE}"
    --train_bag_size "${TRAIN_BAG_SIZE}"
    --val_bag_size "${VAL_BAG_SIZE}"
    --max_epochs "${MAX_EPOCHS}"
    --lr "${LR}"
    --wd "${WD}"
    --opt "${OPT}"
    --lr_scheduler "${LR_SCHEDULER}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --seed "${seed}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --early_stopping 1
    --es_min_epochs "${ES_MIN_EPOCHS}"
    --es_patience "${ES_PATIENCE}"
    --es_metric "${ES_METRIC}"
    --pathway_dropout "${PATHWAY_DROPOUT}"
    --use_gated_fusion false
    --contrastive_weight "${CONTRASTIVE_WEIGHT}"
    --contrastive_temp "${CONTRASTIVE_TEMP}"
    --contrastive_pairs histo_text
    --force_gene_histo_contrastive false
    --use_pathway_attnpool true
    --use_path_qformer false
    --freeze_non_qformer false
    --freeze_conch true
    --target_col "${TARGET_COL}"
    --overwrite True
    --pathway_token_count "${PATHWAY_TOKEN_COUNT}"
    --cls_depth "${CLS_DEPTH}"
    --cls_num_heads "${CLS_NUM_HEADS}"
    --cls_mlp_ratio "${CLS_MLP_RATIO}"
    --cls_dropout "${CLS_DROPOUT}"
  )

  echo "[$(date -u '+%F %T UTC')] START ${task}" | tee -a "${log_file}"
  echo "ablation=${ABLATION_LABEL}" | tee -a "${log_file}"
  echo "dataset=${dataset}" | tee -a "${log_file}"
  echo "split_dir=${split_dir}" | tee -a "${log_file}"
  echo "results_dir=${result_root}" | tee -a "${log_file}"
  echo "note=${ABLATION_NOTE}" | tee -a "${log_file}"

  local attempt rc
  rc=1
  for attempt in $(seq 1 $((RETRY + 1))); do
    if (cd "${PROJECT_DIR}" && "${cmd[@]}") >> "${log_file}" 2>&1; then
      rc=0
      break
    fi
    echo "[$(date -u '+%F %T UTC')] RETRY ${task} attempt=${attempt}" | tee -a "${log_file}"
    sleep 5
  done

  local summary_csv cidx
  summary_csv="$(find "${result_root}" -type f -name summary.csv | head -n 1 || true)"

  if [[ ${rc} -eq 0 && -n "${summary_csv}" ]]; then
    cidx="$(extract_cindex "${summary_csv}" || echo NA)"
    done=$((done + 1))
    printf "%s\tDONE\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date -u '+%F %T UTC')" "${ABLATION_LABEL}" "${dataset}" "${seed}" "${fold}" "${task}" "${cidx}" "${summary_csv}" "${log_file}" "${ABLATION_NOTE}" >> "${STATUS_TSV}"
    echo "[$(date -u '+%F %T UTC')] DONE ${task} c_index=${cidx}" | tee -a "${log_file}"
  else
    failed=$((failed + 1))
    printf "%s\tFAIL\t%s\t%s\t%s\t%s\t%s\tNA\t%s\t%s\t%s;rc=%s\n" \
      "$(date -u '+%F %T UTC')" "${ABLATION_LABEL}" "${dataset}" "${seed}" "${fold}" "${task}" "${summary_csv}" "${log_file}" "${ABLATION_NOTE}" "${rc}" >> "${STATUS_TSV}"
    echo "[$(date -u '+%F %T UTC')] FAIL ${task} rc=${rc}" | tee -a "${log_file}"
  fi
}

for ablation in "${MODALITIES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for fold in "${FOLDS[@]}"; do
        run_one "${ablation}" "${dataset}" "${seed}" "${fold}"
        echo "[progress] done=${done} skipped=${skipped} failed=${failed} / total=${total}" | tee -a "${REPORT_ROOT}/progress.log"
      done
    done
  done
done

python3 - "${STATUS_TSV}" "${REPORT_ROOT}" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

status_tsv = Path(sys.argv[1])
report_root = Path(sys.argv[2])

with status_tsv.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

done_rows = [r for r in rows if r.get("status") == "DONE"]
fail_rows = [r for r in rows if r.get("status") == "FAIL"]
skip_rows = [r for r in rows if r.get("status") == "SKIP"]

fold_items = []
group_scores = defaultdict(list)
group_counts = defaultdict(lambda: {"done": 0, "fail": 0, "skip": 0})
ablation_scores = defaultdict(list)

for row in rows:
    key = (row["ablation"], row["dataset"])
    if row["status"] == "DONE":
        group_counts[key]["done"] += 1
    elif row["status"] == "FAIL":
        group_counts[key]["fail"] += 1
    elif row["status"] == "SKIP":
        group_counts[key]["skip"] += 1

for row in done_rows:
    try:
        score = float(row["c_index_test"])
    except Exception:
        score = None
    item = {
        "ablation": row["ablation"],
        "dataset": row["dataset"],
        "seed": int(row["seed"]),
        "fold": int(row["fold"]),
        "task": row["task"],
        "c_index_test": score,
        "summary_csv": row["summary_csv"],
        "log_file": row["log_file"],
        "note": row["note"],
    }
    fold_items.append(item)
    if score is not None:
        group_scores[(row["ablation"], row["dataset"])].append(score)
        ablation_scores[row["ablation"]].append(score)

def summarize_scores(scores):
    if not scores:
        return {"n": 0, "mean": None, "std": None}
    if len(scores) == 1:
        return {"n": 1, "mean": scores[0], "std": 0.0}
    return {
        "n": len(scores),
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores),
    }

by_ablation_dataset = []
for key in sorted(group_counts.keys()):
    ablation, dataset = key
    stat = summarize_scores(group_scores[key])
    by_ablation_dataset.append({
        "ablation": ablation,
        "dataset": dataset,
        "done": group_counts[key]["done"],
        "fail": group_counts[key]["fail"],
        "skip": group_counts[key]["skip"],
        **stat,
    })

by_ablation = []
for ablation in sorted({r["ablation"] for r in rows}):
    stat = summarize_scores(ablation_scores[ablation])
    done = sum(1 for r in done_rows if r["ablation"] == ablation)
    fail = sum(1 for r in fail_rows if r["ablation"] == ablation)
    skip = sum(1 for r in skip_rows if r["ablation"] == ablation)
    by_ablation.append({
        "ablation": ablation,
        "done": done,
        "fail": fail,
        "skip": skip,
        **stat,
    })

all_scores = [x["c_index_test"] for x in fold_items if x["c_index_test"] is not None]
overall = summarize_scores(all_scores)

payload = {
    "status_tsv": str(status_tsv),
    "done_runs": len(done_rows),
    "failed_runs": len(fail_rows),
    "skipped_runs": len(skip_rows),
    "all_rows": len(rows),
    "overall": overall,
    "by_ablation": by_ablation,
    "by_ablation_dataset": by_ablation_dataset,
    "fold_results": sorted(
        fold_items,
        key=lambda x: (x["ablation"], x["dataset"], x["seed"], x["fold"]),
    ),
}

(report_root / "cv_summary_modality_ablation.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# P1 Modality Contribution Ablation Summary",
    "",
    f"- Done runs: {payload['done_runs']} / {payload['all_rows']}",
    f"- Failed runs: {payload['failed_runs']}",
    f"- Skipped runs: {payload['skipped_runs']}",
    f"- Overall mean c-index(test): {payload['overall']['mean'] if payload['overall']['mean'] is not None else 'NA'}",
    f"- Overall std c-index(test): {payload['overall']['std'] if payload['overall']['std'] is not None else 'NA'}",
    "",
    "## By Ablation",
    "",
    "| Ablation | Done | Fail | Skip | Mean c-index(test) | Std |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for item in payload["by_ablation"]:
    lines.append(
        f"| {item['ablation']} | {item['done']} | {item['fail']} | {item['skip']} | "
        f"{item['mean'] if item['mean'] is not None else 'NA'} | "
        f"{item['std'] if item['std'] is not None else 'NA'} |"
    )

lines.extend([
    "",
    "## By Ablation + Dataset",
    "",
    "| Ablation | Dataset | Done | Fail | Skip | Mean c-index(test) | Std |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
])
for item in payload["by_ablation_dataset"]:
    lines.append(
        f"| {item['ablation']} | {item['dataset']} | {item['done']} | {item['fail']} | {item['skip']} | "
        f"{item['mean'] if item['mean'] is not None else 'NA'} | "
        f"{item['std'] if item['std'] is not None else 'NA'} |"
    )

lines.extend([
    "",
    "## Fold Results",
    "",
    "| Ablation | Dataset | Seed | Fold | c-index(test) | Summary | Log | Note |",
    "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
])
for item in payload["fold_results"]:
    lines.append(
        f"| {item['ablation']} | {item['dataset']} | {item['seed']} | {item['fold']} | "
        f"{item['c_index_test'] if item['c_index_test'] is not None else 'NA'} | "
        f"{item['summary_csv']} | {item['log_file']} | {item['note']} |"
    )

(report_root / "cv_summary_modality_ablation.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

{
  echo "end_utc=$(date -u '+%F %T UTC')"
  echo "total=${total}"
  echo "done=${done}"
  echo "skipped=${skipped}"
  echo "failed=${failed}"
  echo "status_tsv=${STATUS_TSV}"
  echo "summary_json=${REPORT_ROOT}/cv_summary_modality_ablation.json"
  echo "summary_md=${REPORT_ROOT}/cv_summary_modality_ablation.md"
} >> "${REPORT_ROOT}/run_config.txt"

echo "[finished] done=${done} skipped=${skipped} failed=${failed} / total=${total}" | tee -a "${REPORT_ROOT}/progress.log"
