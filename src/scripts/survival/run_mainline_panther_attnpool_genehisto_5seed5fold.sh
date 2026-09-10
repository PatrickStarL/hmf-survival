#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/hmf-survival}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-/path/to/your/data_root/mainline_twostage_panther_mstar_conch_5seed5fold_${TS}}"
LOG_ROOT="${RUN_ROOT}/logs"
REPORT_ROOT="${RUN_ROOT}/reports"
STATUS_TSV="${REPORT_ROOT}/run_status.tsv"

NUM_WORKERS="${NUM_WORKERS:-8}"
RETRY="${RETRY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SEED_LIST="${SEED_LIST:-1 7 13 21 42}"
FOLD_LIST="${FOLD_LIST:-0 1 2 3 4}"
DATASET_LIST="${DATASET_LIST:-HNSC KIRC LUAD STAD}"
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
CRC_VISION_DIR="${CRC_VISION_DIR:-/path/to/your/data_root/tcga_crc/features_mstar_crc/feats_pt}"

read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a FOLDS <<< "${FOLD_LIST}"
read -r -a DATASETS <<< "${DATASET_LIST}"
mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"

if [[ ! -f "${STATUS_TSV}" ]]; then
  printf "timestamp_utc\tstatus\tdataset\tseed\tfold\ttask\tc_index_test\tsummary_csv\tlog_file\tnote\n" > "${STATUS_TSV}"
fi

cat > "${REPORT_ROOT}/run_config.txt" <<CFG
start_utc=$(date -u '+%F %T UTC')
project_dir=${PROJECT_DIR}
run_root=${RUN_ROOT}
model_entry=python -m src.training.main_survival_hierarchical_fusion
profile=mainline_twostage_panther_mstar_conch_histotext
notes=Official retained mainline: mSTAR WSI -> PANTHER -> 32 histology tokens -> token-level co-attention with CONCH report tokens and RNA 50->16 tokens -> per-modality attention pooling -> shallow CLS Transformer -> Cox. Q-Former and gated fusion are disabled. Contrastive alignment is retained for histo_text.
datasets=${DATASET_LIST}
seeds=${SEED_LIST}
folds=${FOLD_LIST}
retry=${RETRY}
skip_existing=${SKIP_EXISTING}
num_workers=${NUM_WORKERS}
target_col=${TARGET_COL}
model_histo_type=PANTHER
model_histo_config=PANTHER_default
model_mm_type=twostage_cls
use_path_qformer=false
use_pathway_attnpool=true
contrastive_pairs=histo_text
force_gene_histo_contrastive=false
use_gated_fusion=false
freeze_conch=true
append_embed=${APPEND_EMBED}
hparams=max_epochs=${MAX_EPOCHS},lr=${LR},wd=${WD},opt=${OPT},lr_scheduler=${LR_SCHEDULER},warmup_epochs=${WARMUP_EPOCHS},batch_size=${BATCH_SIZE},bag_size=${BAG_SIZE},train_bag_size=${TRAIN_BAG_SIZE},val_bag_size=${VAL_BAG_SIZE},es_min_epochs=${ES_MIN_EPOCHS},es_patience=${ES_PATIENCE},es_metric=${ES_METRIC},contrastive_weight=${CONTRASTIVE_WEIGHT},contrastive_temp=${CONTRASTIVE_TEMP},pathway_dropout=${PATHWAY_DROPOUT},n_proto=${N_PROTO},pathway_token_count=${PATHWAY_TOKEN_COUNT},vision_in_dim=${VISION_IN_DIM},text_in_dim=${TEXT_IN_DIM},path_proj_dim=${PATH_PROJ_DIM},tau=${TAU},ot_eps=${OT_EPS},em_iter=${EM_ITER},cls_depth=${CLS_DEPTH},cls_num_heads=${CLS_NUM_HEADS},cls_mlp_ratio=${CLS_MLP_RATIO},cls_dropout=${CLS_DROPOUT}
CFG

resolve_dataset() {
  local dataset="$1"
  case "${dataset}" in
    BLCA)
      SPLIT_PREFIX="/path/to/hmf-survival/src/splits/tcga-blca/TCGA_BLCA_overall_survival_k="
      VISION_DIR="/path/to/your/data_root/tcga_blca/features_mstar_blca/feats_pt"
      REPORT_DIR="/path/to/your/data_root/tcga_blca/features_conch_blca/reports_pt"
      OMICS_DIR="/path/to/hmf-survival/src/data_csvs/rna"
      OMICS_CHECK="/path/to/hmf-survival/src/data_csvs/rna/hallmarks/BLCA/rna_clean.csv"
      ;;
    CRC)
      SPLIT_PREFIX="/path/to/hmf-survival/src/splits/tcga-coadread/TCGA_COADREAD_overall_survival_k="
      VISION_DIR="${CRC_VISION_DIR}"
      REPORT_DIR="/path/to/your/data_root/tcga_crc/features_conch_crc/reports_pt"
      OMICS_DIR="/path/to/hmf-survival/src/data_csvs/rna"
      OMICS_CHECK="/path/to/hmf-survival/src/data_csvs/rna/hallmarks/COADREAD/rna_clean.csv"
      ;;
    HNSC)
      SPLIT_PREFIX="/path/to/hmf-survival/src/splits/tcga-hnsc/TCGA_HNSC_overall_survival_k="
      VISION_DIR="/path/to/your/data_root/tcga_hnsc/features_mstar_hnsc/feats_pt"
      REPORT_DIR="/path/to/your/data_root/tcga_hnsc/features_conch_hnsc/reports_pt"
      OMICS_DIR="/path/to/hmf-survival/src/data_csvs/rna"
      OMICS_CHECK="/path/to/hmf-survival/src/data_csvs/rna/hallmarks/HNSC/rna_clean.csv"
      ;;
    KIRC)
      SPLIT_PREFIX="/path/to/hmf-survival/src/splits/tcga-kirc/TCGA_KIRC_overall_survival_k="
      VISION_DIR="/path/to/your/data_root/tcga_kirc/features_mstar_kirc/feats_pt"
      REPORT_DIR="/path/to/your/data_root/tcga_kirc/features_conch_kirc/reports_pt"
      OMICS_DIR="/path/to/hmf-survival/src/data_csvs/rna"
      OMICS_CHECK="/path/to/hmf-survival/src/data_csvs/rna/hallmarks/KIRC/rna_clean.csv"
      ;;
    LUAD)
      SPLIT_PREFIX="/path/to/hmf-survival/src/splits/tcga-luad/TCGA_LUAD_overall_survival_k="
      VISION_DIR="/path/to/your/data_root/tcga_luad/features_mstar_luad/feats_pt"
      REPORT_DIR="/path/to/your/data_root/tcga_luad/features_conch_luad/reports_pt"
      OMICS_DIR="/path/to/hmf-survival/src/data_csvs/rna"
      OMICS_CHECK="/path/to/hmf-survival/src/data_csvs/rna/hallmarks/LUAD/rna_clean.csv"
      ;;
    STAD)
      SPLIT_PREFIX="/path/to/hmf-survival/src/splits/tcga-stad/TCGA_STAD_overall_survival_k="
      VISION_DIR="/path/to/your/data_root/tcga_stad/features_mstar_stad/feats_pt"
      REPORT_DIR="/path/to/your/data_root/tcga_stad/features_conch_stad/reports_pt"
      OMICS_DIR="/path/to/hmf-survival/src/data_csvs/rna"
      OMICS_CHECK="/path/to/hmf-survival/src/data_csvs/rna/hallmarks/STAD/rna_clean.csv"
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

extract_cindex() {
  local summary_csv="$1"
  python3 - "$summary_csv" <<'PY'
import csv, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    row = next(csv.DictReader(f))
print(row.get('c_index_test', 'NA'))
PY
}

total=0
done=0
skipped=0
failed=0
for _d in "${DATASETS[@]}"; do
  for _s in "${SEEDS[@]}"; do
    for _k in "${FOLDS[@]}"; do
      total=$((total+1))
    done
  done
done

run_one() {
  local dataset="$1"
  local seed="$2"
  local fold="$3"

  resolve_dataset "${dataset}"

  local split_dir="${SPLIT_PREFIX}${fold}"
  local task="${dataset}_twostage_panther_histotext_k${fold}_s${seed}"
  local result_root="${RUN_ROOT}/results/${dataset}/seed_${seed}/k_${fold}"
  local log_file="${LOG_ROOT}/${task}.log"

  mkdir -p "${result_root}"
  local existing
  existing="$(find "${result_root}" -type f -name summary.csv | head -n 1 || true)"
  if [[ "${SKIP_EXISTING}" == "1" && -n "${existing}" ]]; then
    skipped=$((skipped+1))
    printf "%s\tSKIP\t%s\t%s\t%s\t%s\tNA\t%s\t%s\tresume_skip_existing\n" \
      "$(date -u '+%F %T UTC')" "${dataset}" "${seed}" "${fold}" "${task}" "${existing}" "${log_file}" >> "${STATUS_TSV}"
    return 0
  fi

  local -a cmd=(
    "${PYTHON_BIN}" -m src.training.main_survival_hierarchical_fusion
    --task "${task}"
    --exp_code twostage_formal
    --split_dir "${split_dir}"
    --split_names train,test
    --data_source "${VISION_DIR}"
    --reports_dir "${REPORT_DIR}"
    --results_dir "${result_root}"
    --model_histo_type PANTHER
    --model_histo_config PANTHER_default
    --n_proto "${N_PROTO}"
    --pathway_token_count "${PATHWAY_TOKEN_COUNT}"
    --in_dim "${VISION_IN_DIM}"
    --text_in_dim "${TEXT_IN_DIM}"
    --path_proj_dim "${PATH_PROJ_DIM}"
    --out_type allcat
    --em_iter "${EM_ITER}"
    --tau "${TAU}"
    --ot_eps "${OT_EPS}"
    --model_mm_type twostage_cls
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
    --cls_depth "${CLS_DEPTH}"
    --cls_num_heads "${CLS_NUM_HEADS}"
    --cls_mlp_ratio "${CLS_MLP_RATIO}"
    --cls_dropout "${CLS_DROPOUT}"
    --target_col "${TARGET_COL}"
    --overwrite True
  )

  echo "[$(date -u '+%F %T UTC')] START ${task}" | tee -a "${log_file}"
  echo "split_dir=${split_dir}" | tee -a "${log_file}"
  echo "results_dir=${result_root}" | tee -a "${log_file}"

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
    done=$((done+1))
    printf "%s\tDONE\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tok\n" \
      "$(date -u '+%F %T UTC')" "${dataset}" "${seed}" "${fold}" "${task}" "${cidx}" "${summary_csv}" "${log_file}" >> "${STATUS_TSV}"
  else
    failed=$((failed+1))
    printf "%s\tFAIL\t%s\t%s\t%s\t%s\tNA\t%s\t%s\trc=%s\n" \
      "$(date -u '+%F %T UTC')" "${dataset}" "${seed}" "${fold}" "${task}" "${summary_csv}" "${log_file}" "${rc}" >> "${STATUS_TSV}"
  fi
}

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      run_one "${dataset}" "${seed}" "${fold}"
      echo "[progress] done=${done} skipped=${skipped} failed=${failed} / total=${total}" | tee -a "${REPORT_ROOT}/progress.log"
    done
  done
done

python3 - "${STATUS_TSV}" "${REPORT_ROOT}" <<'PY'
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

status_tsv, report_root = sys.argv[1], sys.argv[2]
rows = []
with open(status_tsv, 'r', encoding='utf-8') as f:
    header = f.readline().rstrip('\n').split('\t')
    for line in f:
        vals = line.rstrip('\n').split('\t')
        if len(vals) < len(header):
            vals += [''] * (len(header) - len(vals))
        rows.append(dict(zip(header, vals)))

summary = {
    'total_rows': len(rows),
    'done': sum(r['status'] == 'DONE' for r in rows),
    'skip': sum(r['status'] == 'SKIP' for r in rows),
    'fail': sum(r['status'] == 'FAIL' for r in rows),
    'by_dataset': {},
    'by_dataset_seed': {},
    'by_dataset_fold': {},
}

bucket_dataset = defaultdict(list)
bucket_seed = defaultdict(list)
bucket_fold = defaultdict(list)
for r in rows:
    if r['status'] != 'DONE':
        continue
    try:
        c = float(r['c_index_test'])
    except Exception:
        continue
    bucket_dataset[r['dataset']].append(c)
    bucket_seed[(r['dataset'], r['seed'])].append(c)
    bucket_fold[(r['dataset'], r['fold'])].append(c)

for dataset, vals in sorted(bucket_dataset.items()):
    summary['by_dataset'][dataset] = {
        'n': len(vals),
        'mean_c_index_test': statistics.mean(vals),
        'std_c_index_test': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        'min_c_index_test': min(vals),
        'max_c_index_test': max(vals),
    }
for (dataset, seed), vals in sorted(bucket_seed.items()):
    summary['by_dataset_seed'].setdefault(dataset, {})[seed] = {
        'n': len(vals),
        'mean_c_index_test': statistics.mean(vals),
        'std_c_index_test': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }
for (dataset, fold), vals in sorted(bucket_fold.items(), key=lambda x: (x[0][0], int(x[0][1]))):
    summary['by_dataset_fold'].setdefault(dataset, {})[fold] = {
        'n': len(vals),
        'mean_c_index_test': statistics.mean(vals),
        'std_c_index_test': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }

os.makedirs(report_root, exist_ok=True)
json_path = os.path.join(report_root, 'cv_summary_mainline_twostage_5seed5fold.json')
md_path = os.path.join(report_root, 'cv_summary_mainline_twostage_5seed5fold.md')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

lines = [
    '# Mainline Two-Stage PANTHER + mSTAR + CONCH + RNA16 + histo_text contrastive',
    '',
    f"- done: {summary['done']}",
    f"- skip: {summary['skip']}",
    f"- fail: {summary['fail']}",
    '',
]
for dataset in sorted(summary['by_dataset'].keys()):
    s = summary['by_dataset'][dataset]
    lines.append(f"## {dataset}")
    lines.append(
        f"- overall: n={s['n']}, mean={s['mean_c_index_test']:.6f}, std={s['std_c_index_test']:.6f}, min={s['min_c_index_test']:.6f}, max={s['max_c_index_test']:.6f}"
    )
    for fold in sorted(summary['by_dataset_fold'].get(dataset, {}).keys(), key=int):
        ss = summary['by_dataset_fold'][dataset][fold]
        lines.append(f"- fold {fold}: n={ss['n']}, mean={ss['mean_c_index_test']:.6f}, std={ss['std_c_index_test']:.6f}")
    for seed in sorted(summary['by_dataset_seed'].get(dataset, {}).keys(), key=lambda x: int(x)):
        ss = summary['by_dataset_seed'][dataset][seed]
        lines.append(f"- seed {seed}: n={ss['n']}, mean={ss['mean_c_index_test']:.6f}, std={ss['std_c_index_test']:.6f}")
    lines.append('')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(json_path)
print(md_path)
PY

cat >> "${REPORT_ROOT}/run_config.txt" <<CFG
end_utc=$(date -u '+%F %T UTC')
total=${total}
done=${done}
skipped=${skipped}
failed=${failed}
status_tsv=${STATUS_TSV}
summary_json=${REPORT_ROOT}/cv_summary_mainline_twostage_5seed5fold.json
summary_md=${REPORT_ROOT}/cv_summary_mainline_twostage_5seed5fold.md
CFG

echo "run_root=${RUN_ROOT}"
echo "status_tsv=${STATUS_TSV}"
echo "summary_json=${REPORT_ROOT}/cv_summary_mainline_twostage_5seed5fold.json"
echo "summary_md=${REPORT_ROOT}/cv_summary_mainline_twostage_5seed5fold.md"
