#!/bin/bash
# Evaluate pretrained SimAM_ResNet100 (VoxBlink2 + VoxCeleb2 FT)
# on VoxCeleb1 test set (Vox1-O, Vox1-E, Vox1-H cleaned trials).
#
# Usage:
#   cd /home/taejinp/projects/wespeaker/wespeaker
#   bash examples/voxceleb/v2/eval_samresnet100/run_eval.sh
#
# Prerequisites:
#   conda activate nemo012826
#   pip install kaldiio  (if not installed)

set -euo pipefail

# ──────────────────────── configurable paths ────────────────────────
VOX1_TEST_WAV="/disk_f_nvd/datasets/voxceleb1/test_wav"
MODEL_DIR="/disk_f_nvd/datasets/Yodas/wespeaker/models/voxblink2_samresnet100_ft"
MODEL_CKPT="${MODEL_DIR}/avg_model.pt"
MODEL_CONFIG="${MODEL_DIR}/config.yaml"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"           # this script's dir
WESPEAKER_ROOT="$(cd "${EVAL_DIR}/../../../.." && pwd)" # wespeaker repo root
DATA_DIR="${EVAL_DIR}/data/vox1"
EXP_DIR="${EVAL_DIR}/exp"

GPUS="[0]"
NJ=1            # number of parallel extraction jobs
BATCH_SIZE=1    # must be 1 for variable-length test utterances

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
echo "  SimAM_ResNet100 evaluation on VoxCeleb1"
echo "============================================="
echo "  VOX1_TEST_WAV : ${VOX1_TEST_WAV}"
echo "  MODEL_CKPT    : ${MODEL_CKPT}"
echo "  DATA_DIR      : ${DATA_DIR}"
echo "  EXP_DIR       : ${EXP_DIR}"
echo "============================================="

# ===================== Stage 1: Prepare data =====================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[Stage 1] Preparing wav.scp, utt2spk, and trial lists ..."
  mkdir -p "${DATA_DIR}/trials"

  # wav.scp  (format: spkid/videoid/uttid  /full/path.wav)
  find "${VOX1_TEST_WAV}" -name "*.wav" \
    | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF, $0}' \
    | sort > "${DATA_DIR}/wav.scp"

  # utt2spk
  awk '{print $1}' "${DATA_DIR}/wav.scp" \
    | awk -F"/" '{print $0, $1}' > "${DATA_DIR}/utt2spk"

  echo "  wav.scp entries : $(wc -l < "${DATA_DIR}/wav.scp")"
  echo "  speakers        : $(awk '{print $2}' "${DATA_DIR}/utt2spk" | sort -u | wc -l)"

  # Download cleaned trial lists from VoxCeleb website
  for trial_name in veri_test2 list_test_hard2 list_test_all2; do
    url="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/${trial_name}.txt"
    dest="${DATA_DIR}/trials/${trial_name}.txt"
    if [ ! -f "${dest}" ]; then
      echo "  Downloading ${trial_name}.txt ..."
      wget --no-check-certificate -q "${url}" -O "${dest}"
    fi
  done

  # Convert to Kaldi trial format: enroll_utt  test_utt  target/nontarget
  awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
    "${DATA_DIR}/trials/veri_test2.txt" > "${DATA_DIR}/trials/vox1_O_cleaned.kaldi"
  awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
    "${DATA_DIR}/trials/list_test_hard2.txt" > "${DATA_DIR}/trials/vox1_H_cleaned.kaldi"
  awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
    "${DATA_DIR}/trials/list_test_all2.txt" > "${DATA_DIR}/trials/vox1_E_cleaned.kaldi"

  echo "  Trial lists ready."
fi

# ===================== Stage 2: Make raw list =====================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "[Stage 2] Creating raw.list for dataloader ..."
  python tools/make_raw_list.py \
    "${DATA_DIR}/wav.scp" \
    "${DATA_DIR}/utt2spk" \
    "${DATA_DIR}/raw.list"
  echo "  raw.list entries : $(wc -l < "${DATA_DIR}/raw.list")"
fi

# ===================== Stage 3: Extract embeddings =====================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "[Stage 3] Extracting embeddings ..."
  mkdir -p "${EXP_DIR}/embeddings/vox1" "${EXP_DIR}"

  # Copy config so extract.py can find it at exp_dir/config.yaml
  cp "${MODEL_CONFIG}" "${EXP_DIR}/config.yaml"

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
  echo "========== RESULTS =========="
  for trial_file in ${trials}; do
    python wespeaker/bin/compute_metrics.py \
      --p_target 0.01 \
      --c_fa 1 \
      --c_miss 1 \
      "${EXP_DIR}/scores/${trial_file}.score" \
      2>&1 | tee -a "${EXP_DIR}/scores/vox1_cos_result"
    echo ""
  done
  echo "============================="
  echo "Full results saved to: ${EXP_DIR}/scores/vox1_cos_result"
fi
