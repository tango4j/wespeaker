#!/bin/bash
# Evaluate pretrained WeSpeaker ResNet293_LM on VoxCeleb1 cleaned trials.
#
# Scoring matches examples/voxceleb/v2/local/score.sh: subtract the mean of
# VoxCeleb2 **dev** embeddings, then cosine similarity (same as published v2 recipe).
#
# Download checkpoint (if needed):
#   https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM
#
# Usage:
#   cd /path/to/wespeaker/wespeaker
#   bash examples/voxceleb/v2/eval_resnet293_lm/run_eval.sh
#
# Environment:
#   MODEL_DIR            Directory with avg_model.pt + config.yaml
#   VOX1_TEST_WAV        VoxCeleb1 test wav root
#   VOXCELEB2_WAV_ROOT   VoxCeleb2 dev wav tree (same as prepare_data.sh)
#   CAL_MEAN             true | false  (default true — ResNet293_LM standard)
#   SMOKE_MEAN_FROM_VOX1 1 for pipeline test only (duplicate vox1 utts; not for benchmarks)
#
# Flags: --no-cal-mean  --smoke-mean-vox1  --stage / --stop-stage / --gpus / --nj / --nj-vox2

set -euo pipefail

VOX1_TEST_WAV="${VOX1_TEST_WAV:-/disk_f_nvd/datasets/voxceleb1/test_wav}"
MODEL_DIR="${MODEL_DIR:-/disk_f_nvd/datasets/Yodas/wespeaker/models/voxceleb_resnet293_LM}"
MODEL_CKPT="${MODEL_DIR}/avg_model.pt"
MODEL_CONFIG="${MODEL_DIR}/config.yaml"

VOXCELEB2_WAV_ROOT="${VOXCELEB2_WAV_ROOT:-/disk_f_nvd/datasets/voxceleb2_wav}"
CAL_MEAN="${CAL_MEAN:-true}"
SMOKE_MEAN_FROM_VOX1="${SMOKE_MEAN_FROM_VOX1:-0}"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
WESPEAKER_ROOT="$(cd "${EVAL_DIR}/../../../.." && pwd)"
DATA_DIR="${EVAL_DIR}/data/vox1"
DATA_VOX2="${EVAL_DIR}/data/vox2_dev"
EXP_DIR="${EVAL_DIR}/exp"
TITANET_LOCAL="${EVAL_DIR}/../titanet/local"

GPUS="${GPUS:-[0]}"
NJ="${NJ:-1}"
NJ_VOX2="${NJ_VOX2:-4}"
BATCH_VOX1="${BATCH_VOX1:-1}"
BATCH_VOX2="${BATCH_VOX2:-16}"

stage=1
stop_stage=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)        stage=$2;        shift 2 ;;
    --stop-stage)   stop_stage=$2;   shift 2 ;;
    --gpus)         GPUS="$2";       shift 2 ;;
    --nj)           NJ=$2;           shift 2 ;;
    --nj-vox2)      NJ_VOX2=$2;      shift 2 ;;
    --no-cal-mean)  CAL_MEAN=false;  shift ;;
    --smoke-mean-vox1) SMOKE_MEAN_FROM_VOX1=1; shift ;;
    *)              echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export PYTHONPATH="${WESPEAKER_ROOT}:${PYTHONPATH:-}"
export PYTHONIOENCODING=UTF-8
cd "${WESPEAKER_ROOT}"

if [[ ! -f "${MODEL_CKPT}" ]] || [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "ERROR: Missing checkpoint or config under MODEL_DIR=${MODEL_DIR}"
  echo "  Expected: ${MODEL_CKPT}  and  ${MODEL_CONFIG}"
  exit 1
fi

echo "============================================="
echo "  ResNet293_LM evaluation (WeSpeaker)"
echo "============================================="
echo "  MODEL_DIR       : ${MODEL_DIR}"
echo "  VOX1_TEST_WAV   : ${VOX1_TEST_WAV}"
echo "  CAL_MEAN        : ${CAL_MEAN}"
if [[ "${CAL_MEAN}" == "true" ]]; then
  if [[ "${SMOKE_MEAN_FROM_VOX1}" == "1" ]]; then
    echo "  Vox2 cohort     : SMOKE (prefixed vox1 dup — not for benchmarks)"
  else
    echo "  VOXCELEB2_WAV_ROOT: ${VOXCELEB2_WAV_ROOT}"
  fi
fi
echo "  DATA_DIR        : ${DATA_DIR}"
echo "  EXP_DIR         : ${EXP_DIR}"
echo "============================================="

# ----- Stage 1: lists -----
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[Stage 1a] Vox1 test wav.scp + trials ..."
  mkdir -p "${DATA_DIR}/trials"

  find "${VOX1_TEST_WAV}" -name "*.wav" \
    | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF, $0}' \
    | sort > "${DATA_DIR}/wav.scp"

  awk '{print $1}' "${DATA_DIR}/wav.scp" \
    | awk -F"/" '{print $0, $1}' > "${DATA_DIR}/utt2spk"

  echo "  wav.scp lines: $(wc -l < "${DATA_DIR}/wav.scp")"

  for trial_name in veri_test2 list_test_hard2 list_test_all2; do
    url="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/${trial_name}.txt"
    dest="${DATA_DIR}/trials/${trial_name}.txt"
    if [ ! -f "${dest}" ]; then
      wget --no-check-certificate -q "${url}" -O "${dest}"
    fi
  done

  awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
    "${DATA_DIR}/trials/veri_test2.txt" > "${DATA_DIR}/trials/vox1_O_cleaned.kaldi"
  awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
    "${DATA_DIR}/trials/list_test_hard2.txt" > "${DATA_DIR}/trials/vox1_H_cleaned.kaldi"
  awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
    "${DATA_DIR}/trials/list_test_all2.txt" > "${DATA_DIR}/trials/vox1_E_cleaned.kaldi"

  if [[ "${CAL_MEAN}" == "true" ]]; then
    echo "[Stage 1b] Vox2 dev wav.scp (cohort mean, cf. v2/local/score.sh) ..."
    mkdir -p "${DATA_VOX2}"
    if [[ "${SMOKE_MEAN_FROM_VOX1}" == "1" ]]; then
      awk '{print "vox2mean_" $1, $2}' "${DATA_DIR}/wav.scp" > "${DATA_VOX2}/wav.scp"
      echo "  (smoke) vox2 lines: $(wc -l < "${DATA_VOX2}/wav.scp")"
    else
      if [[ ! -d "${VOXCELEB2_WAV_ROOT}" ]]; then
        echo "ERROR: CAL_MEAN=true but VOXCELEB2_WAV_ROOT is not a directory: ${VOXCELEB2_WAV_ROOT}"
        echo "  Use --no-cal-mean, or SMOKE_MEAN_FROM_VOX1=1 / --smoke-mean-vox1 for a smoke test."
        exit 1
      fi
      find "${VOXCELEB2_WAV_ROOT}" -name "*.wav" \
        | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF, $0}' \
        | sort > "${DATA_VOX2}/wav.scp"
      echo "  vox2_dev wav.scp lines: $(wc -l < "${DATA_VOX2}/wav.scp")"
    fi
  else
    echo "[Stage 1b] Skipped (no cohort mean)."
  fi
fi

# ----- Stage 2: raw.list -----
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "[Stage 2a] raw.list (vox1) ..."
  python tools/make_raw_list.py \
    "${DATA_DIR}/wav.scp" \
    "${DATA_DIR}/utt2spk" \
    "${DATA_DIR}/raw.list"
  if [[ "${CAL_MEAN}" == "true" ]]; then
    echo "[Stage 2b] raw.list (vox2_dev) ..."
    awk '{print $1}' "${DATA_VOX2}/wav.scp" \
      | awk -F"/" '{print $0, $1}' > "${DATA_VOX2}/utt2spk"
    python tools/make_raw_list.py \
      "${DATA_VOX2}/wav.scp" \
      "${DATA_VOX2}/utt2spk" \
      "${DATA_VOX2}/raw.list"
  fi
fi

# ----- Stage 3: extract -----
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "[Stage 3] Extract embeddings (ResNet293_LM) ..."
  mkdir -p "${EXP_DIR}/embeddings/vox1" "${EXP_DIR}/embeddings/vox2_dev" "${EXP_DIR}"
  cp -f "${MODEL_CONFIG}" "${EXP_DIR}/config.yaml"

  wavs_v1=$(wc -l < "${DATA_DIR}/wav.scp")
  bash tools/extract_embedding.sh \
    --exp_dir "${EXP_DIR}" \
    --model_path "${MODEL_CKPT}" \
    --data_type raw \
    --data_list "${DATA_DIR}/raw.list" \
    --wavs_num "${wavs_v1}" \
    --store_dir vox1 \
    --batch_size "${BATCH_VOX1}" \
    --num_workers 1 \
    --nj "${NJ}" \
    --gpus "${GPUS}" \
    --aug-prob 0.0

  echo "  Vox1 xvector.scp: $(wc -l < "${EXP_DIR}/embeddings/vox1/xvector.scp") lines"

  if [[ "${CAL_MEAN}" == "true" ]]; then
    wavs_v2=$(wc -l < "${DATA_VOX2}/wav.scp")
    bash tools/extract_embedding.sh \
      --exp_dir "${EXP_DIR}" \
      --model_path "${MODEL_CKPT}" \
      --data_type raw \
      --data_list "${DATA_VOX2}/raw.list" \
      --wavs_num "${wavs_v2}" \
      --store_dir vox2_dev \
      --batch_size "${BATCH_VOX2}" \
      --num_workers 4 \
      --nj "${NJ_VOX2}" \
      --gpus "${GPUS}" \
      --aug-prob 0.0
    echo "  Vox2 dev xvector.scp: $(wc -l < "${EXP_DIR}/embeddings/vox2_dev/xvector.scp") lines"
  fi
fi

# ----- Stage 4: score -----
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "[Stage 4] Cosine scoring + EER / minDCF / threshold @ EER ..."
  mkdir -p "${EXP_DIR}/scores"
  trials="vox1_O_cleaned.kaldi"

  if [[ "${CAL_MEAN}" == "true" ]]; then
    CAL_ARGS=(--cal_mean True --cal_mean_dir "${EXP_DIR}/embeddings/vox2_dev")
  else
    CAL_ARGS=(--cal_mean False --cal_mean_dir "none")
  fi

  for trial_file in ${trials}; do
    python wespeaker/bin/score.py \
      --exp_dir "${EXP_DIR}" \
      --eval_scp_path "${EXP_DIR}/embeddings/vox1/xvector.scp" \
      "${CAL_ARGS[@]}" \
      "${DATA_DIR}/trials/${trial_file}"
  done

  {
    echo ""
    echo "========== RESULTS (ResNet293_LM) =========="
    echo "  CAL_MEAN=${CAL_MEAN}  SMOKE_MEAN_FROM_VOX1=${SMOKE_MEAN_FROM_VOX1}"
    echo "============================================="
  } | tee -a "${EXP_DIR}/scores/vox1_cos_result"

  if [[ -f "${TITANET_LOCAL}/report_metrics.py" ]]; then
    for trial_file in ${trials}; do
      python "${TITANET_LOCAL}/report_metrics.py" \
        --p_target 0.01 --c_miss 1 --c_fa 1 \
        "${EXP_DIR}/scores/${trial_file}.score" \
        2>&1 | tee -a "${EXP_DIR}/scores/vox1_cos_result"
      echo "" | tee -a "${EXP_DIR}/scores/vox1_cos_result"
    done
  else
    for trial_file in ${trials}; do
      python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 --c_fa 1 --c_miss 1 \
        "${EXP_DIR}/scores/${trial_file}.score" \
        2>&1 | tee -a "${EXP_DIR}/scores/vox1_cos_result"
      echo "" | tee -a "${EXP_DIR}/scores/vox1_cos_result"
    done
  fi

  echo "Full log: ${EXP_DIR}/scores/vox1_cos_result"
fi
