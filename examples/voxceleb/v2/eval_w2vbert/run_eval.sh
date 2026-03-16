#!/bin/bash
# Evaluate pretrained W2V-BERT 2.0 (w2vbert2_mfa)
# on VoxCeleb1-O (cleaned) test set.
#
# Usage:
#   cd /home/taejinp/projects/wespeaker/wespeaker
#   bash examples/voxceleb/v2/eval_w2vbert/run_eval.sh
#
# Prerequisites:
#   conda activate nemo012826

set -euo pipefail

# ──────────────────────── configurable paths ────────────────────────
VOX1_TEST_WAV="/disk_f_nvd/datasets/voxceleb1/test_wav"
MODEL_DIR="/disk_f_nvd/datasets/Yodas/wespeaker/models/w2vbert2_mfa"
MODEL_CKPT="${MODEL_DIR}/avg_model.pt"
MODEL_CONFIG="${MODEL_DIR}/config.yaml"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
WESPEAKER_ROOT="$(cd "${EVAL_DIR}/../../../.." && pwd)"

# Reuse data prepared by the SimAMResNet100 eval (same vox1 test set)
SAMRESNET_DATA="${EVAL_DIR}/../eval_samresnet100/data/vox1"
DATA_DIR="${EVAL_DIR}/data/vox1"
EXP_DIR="${EVAL_DIR}/exp"

GPUS="[0]"
NJ=1
BATCH_SIZE=1

stage=1
stop_stage=4

# ──────────────────────── parse CLI overrides ────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)       stage=$2;       shift 2 ;;
    --stop-stage)  stop_stage=$2;  shift 2 ;;
    --gpus)        GPUS="$2";      shift 2 ;;
    --nj)          NJ=$2;          shift 2 ;;
    *)             echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export PYTHONPATH="${WESPEAKER_ROOT}:${PYTHONPATH:-}"
export PYTHONIOENCODING=UTF-8
cd "${WESPEAKER_ROOT}"

echo "============================================="
echo "  W2V-BERT 2.0 evaluation on VoxCeleb1-O"
echo "============================================="
echo "  VOX1_TEST_WAV : ${VOX1_TEST_WAV}"
echo "  MODEL_CKPT    : ${MODEL_CKPT}"
echo "  DATA_DIR      : ${DATA_DIR}"
echo "  EXP_DIR       : ${EXP_DIR}"
echo "  WESPEAKER_ROOT: ${WESPEAKER_ROOT}"
echo "============================================="

# ===================== Stage 1: Prepare / symlink data =====================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[Stage 1] Setting up data directory ..."

  if [ -d "${SAMRESNET_DATA}" ] && [ -f "${SAMRESNET_DATA}/wav.scp" ]; then
    echo "  Reusing data from SimAMResNet100 eval ..."
    mkdir -p "${DATA_DIR}"
    for f in wav.scp utt2spk raw.list; do
      cp -f "${SAMRESNET_DATA}/${f}" "${DATA_DIR}/${f}"
    done
    cp -rf "${SAMRESNET_DATA}/trials" "${DATA_DIR}/"
  else
    echo "  Generating data from scratch ..."
    mkdir -p "${DATA_DIR}/trials"

    find "${VOX1_TEST_WAV}" -name "*.wav" \
      | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF, $0}' \
      | sort > "${DATA_DIR}/wav.scp"

    awk '{print $1}' "${DATA_DIR}/wav.scp" \
      | awk -F"/" '{print $0, $1}' > "${DATA_DIR}/utt2spk"

    for trial_name in veri_test2 list_test_hard2 list_test_all2; do
      url="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/${trial_name}.txt"
      dest="${DATA_DIR}/trials/${trial_name}.txt"
      if [ ! -f "${dest}" ]; then
        wget --no-check-certificate -q "${url}" -O "${dest}"
      fi
    done

    awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
      "${DATA_DIR}/trials/veri_test2.txt" > "${DATA_DIR}/trials/vox1_O_cleaned.kaldi"

    python tools/make_raw_list.py \
      "${DATA_DIR}/wav.scp" \
      "${DATA_DIR}/utt2spk" \
      "${DATA_DIR}/raw.list"
  fi

  echo "  wav.scp entries : $(wc -l < "${DATA_DIR}/wav.scp")"
  echo "  raw.list entries: $(wc -l < "${DATA_DIR}/raw.list")"
  echo "  Data ready."
fi

# ===================== Stage 2: (reserved, data already prepared) =====================

# ===================== Stage 3: Extract embeddings =====================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "[Stage 3] Extracting embeddings (W2V-BERT 2.0 — this is slower than fbank models) ..."
  mkdir -p "${EXP_DIR}/embeddings/vox1" "${EXP_DIR}"

  cp -f "${MODEL_CONFIG}" "${EXP_DIR}/config.yaml"
  # W2V-BERT frontend outputs hidden states, not fbank — skip CMVN
  if ! grep -q "cmvn:" "${EXP_DIR}/config.yaml"; then
    python -c "
import yaml, sys
with open('${EXP_DIR}/config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['dataset_args']['cmvn'] = False
cfg['dataset_args']['num_frms'] = 7000
with open('${EXP_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
    echo "  Injected cmvn: false into config."
  fi

  wavs_num=$(wc -l < "${DATA_DIR}/wav.scp")

  bash tools/extract_embedding.sh \
    --exp_dir "${EXP_DIR}" \
    --model_path "${MODEL_CKPT}" \
    --data_type raw \
    --data_list "${DATA_DIR}/raw.list" \
    --wavs_num "${wavs_num}" \
    --store_dir vox1 \
    --batch_size ${BATCH_SIZE} \
    --num_workers 1 \
    --nj ${NJ} \
    --gpus "${GPUS}"

  echo "  Embeddings: $(wc -l < "${EXP_DIR}/embeddings/vox1/xvector.scp") entries"
fi

# ===================== Stage 4: Score & compute EER/minDCF =====================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "[Stage 4] Cosine scoring and computing EER / minDCF ..."
  mkdir -p "${EXP_DIR}/scores"

  trials="vox1_O_cleaned.kaldi"

  for trial_file in ${trials}; do
    echo "  Scoring ${trial_file} ..."
    python wespeaker/bin/score.py \
      --exp_dir "${EXP_DIR}" \
      --eval_scp_path "${EXP_DIR}/embeddings/vox1/xvector.scp" \
      --cal_mean False \
      --cal_mean_dir "none" \
      "${DATA_DIR}/trials/${trial_file}"
  done

  echo ""
  echo "========== RESULTS (W2V-BERT 2.0) =========="
  for trial_file in ${trials}; do
    python wespeaker/bin/compute_metrics.py \
      --p_target 0.01 \
      --c_fa 1 \
      --c_miss 1 \
      "${EXP_DIR}/scores/${trial_file}.score" \
      2>&1 | tee -a "${EXP_DIR}/scores/vox1_cos_result"
    echo ""
  done
  echo "============================================="
  echo "Full results saved to: ${EXP_DIR}/scores/vox1_cos_result"
fi
