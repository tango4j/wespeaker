#!/bin/bash
# Evaluate NVIDIA TitaNet-Large (NeMo / Hugging Face) on VoxCeleb1-O (cleaned trials).
#
# Scoring matches examples/voxceleb/v2/local/score.sh (ResNet293_LM recipe): cohort mean
# is computed from VoxCeleb2 **dev** embeddings, subtracted before cosine similarity.
#
# Model card: https://huggingface.co/nvidia/speakerverification_en_titanet_large
#
# Usage:
#   cd /path/to/wespeaker/wespeaker
#   bash examples/voxceleb/v2/titanet/run_eval.sh
#
# Environment (cohort mean — same as ResNet293_LM / wespeaker v2 score.sh):
#   VOXCELEB2_WAV_ROOT   Root tree of VoxCeleb2 dev wavs (same layout as prepare_data.sh:
#                        find .../voxceleb2_wav -name "*.wav" with spk/video/utt ids).
#                        Default: /disk_f_nvd/datasets/voxceleb2_wav
#   CAL_MEAN             true (default) | false — set false to skip Vox2 extraction & mean.
#   SMOKE_MEAN_FROM_VOX1 Set to 1 only for pipeline smoke tests: duplicate vox1 wav.scp with
#                        prefixed utt ids so mean statistics run without real Vox2 (invalid EER).
#
# Prerequisites:
#   pip install nemo_toolkit['asr'] kaldiio soundfile

set -euo pipefail

# ──────────────────────── configurable paths ────────────────────────
VOX1_TEST_WAV="/disk_f_nvd/datasets/voxceleb1/test_wav"
VOXCELEB2_WAV_ROOT="${VOXCELEB2_WAV_ROOT:-/disk_f_nvd/datasets/voxceleb2_wav}"
CAL_MEAN="${CAL_MEAN:-true}"
SMOKE_MEAN_FROM_VOX1="${SMOKE_MEAN_FROM_VOX1:-0}"

MODEL_NAME="${MODEL_NAME:-nvidia/speakerverification_en_titanet_large}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-}"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
WESPEAKER_ROOT="$(cd "${EVAL_DIR}/../../../.." && pwd)"
DATA_DIR="${EVAL_DIR}/data/vox1"
DATA_VOX2="${EVAL_DIR}/data/vox2_dev"
EXP_DIR="${EVAL_DIR}/exp"
LOCAL_DIR="${EVAL_DIR}/local"

stage=1
stop_stage=4

# ──────────────────────── parse CLI overrides ────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)             stage=$2;              shift 2 ;;
    --stop-stage)        stop_stage=$2;         shift 2 ;;
    --batch-size)        BATCH_SIZE=$2;         shift 2 ;;
    --device)            DEVICE=$2;             shift 2 ;;
    --model-name)        MODEL_NAME=$2;         shift 2 ;;
    --no-cal-mean)       CAL_MEAN=false;        shift ;;
    --smoke-mean-vox1)   SMOKE_MEAN_FROM_VOX1=1; shift ;;
    *)                   echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export PYTHONPATH="${WESPEAKER_ROOT}:${PYTHONPATH:-}"
export PYTHONIOENCODING=UTF-8
cd "${WESPEAKER_ROOT}"

extract_invocation() {
  local out_dir="$1"
  local wav_scp="$2"
  local args=(--wav_scp "${wav_scp}" --output_dir "${out_dir}" --model_name "${MODEL_NAME}" --batch_size "${BATCH_SIZE}")
  if [[ -n "${DEVICE}" ]]; then
    args+=(--device "${DEVICE}")
  fi
  python "${LOCAL_DIR}/extract_titanet_embeddings.py" "${args[@]}"
}

echo "============================================="
echo "  TitaNet (NeMo) evaluation on VoxCeleb1-O"
echo "============================================="
echo "  VOX1_TEST_WAV       : ${VOX1_TEST_WAV}"
echo "  CAL_MEAN (ResNet293): ${CAL_MEAN}"
if [[ "${CAL_MEAN}" == "true" ]]; then
  if [[ "${SMOKE_MEAN_FROM_VOX1}" == "1" ]]; then
    echo "  Vox2 cohort (SMOKE) : duplicate vox1 utts with prefix (not for benchmarks)"
  else
    echo "  VOXCELEB2_WAV_ROOT  : ${VOXCELEB2_WAV_ROOT}"
  fi
fi
echo "  MODEL_NAME          : ${MODEL_NAME}"
echo "  BATCH_SIZE          : ${BATCH_SIZE}"
echo "  DATA_DIR            : ${DATA_DIR}"
echo "  EXP_DIR             : ${EXP_DIR}"
echo "============================================="

# ===================== Stage 1: Vox1 test + trials; Vox2 dev wav.scp (if cal_mean) =====================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[Stage 1a] Vox1 test: wav.scp, utt2spk, trials ..."
  mkdir -p "${DATA_DIR}/trials"

  find "${VOX1_TEST_WAV}" -name "*.wav" \
    | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF, $0}' \
    | sort > "${DATA_DIR}/wav.scp"

  awk '{print $1}' "${DATA_DIR}/wav.scp" \
    | awk -F"/" '{print $0, $1}' > "${DATA_DIR}/utt2spk"

  echo "  wav.scp entries : $(wc -l < "${DATA_DIR}/wav.scp")"
  echo "  speakers        : $(awk '{print $2}' "${DATA_DIR}/utt2spk" | sort -u | wc -l)"

  for trial_name in veri_test2 list_test_hard2 list_test_all2; do
    url="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/${trial_name}.txt"
    dest="${DATA_DIR}/trials/${trial_name}.txt"
    if [ ! -f "${dest}" ]; then
      echo "  Downloading ${trial_name}.txt ..."
      wget --no-check-certificate -q "${url}" -O "${dest}"
    fi
  done

  awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
    "${DATA_DIR}/trials/veri_test2.txt" > "${DATA_DIR}/trials/vox1_O_cleaned.kaldi"
  awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
    "${DATA_DIR}/trials/list_test_hard2.txt" > "${DATA_DIR}/trials/vox1_H_cleaned.kaldi"
  awk '{if($1==0) label="nontarget"; else label="target"; print $2, $3, label}' \
    "${DATA_DIR}/trials/list_test_all2.txt" > "${DATA_DIR}/trials/vox1_E_cleaned.kaldi"

  echo "  Trial lists ready."

  if [[ "${CAL_MEAN}" == "true" ]]; then
    echo "[Stage 1b] Vox2 dev wav.scp (cohort for mean, cf. v2/local/score.sh) ..."
    mkdir -p "${DATA_VOX2}"
    if [[ "${SMOKE_MEAN_FROM_VOX1}" == "1" ]]; then
      awk '{print "vox2mean_" $1, $2}' "${DATA_DIR}/wav.scp" > "${DATA_VOX2}/wav.scp"
      echo "  (smoke) vox2_dev lines: $(wc -l < "${DATA_VOX2}/wav.scp")"
    else
      if [[ ! -d "${VOXCELEB2_WAV_ROOT}" ]]; then
        echo "ERROR: CAL_MEAN=true but VOXCELEB2_WAV_ROOT is not a directory:"
        echo "       ${VOXCELEB2_WAV_ROOT}"
        echo "  Set VOXCELEB2_WAV_ROOT to your VoxCeleb2 dev wav tree, or run with --no-cal-mean,"
        echo "  or SMOKE_MEAN_FROM_VOX1=1 / --smoke-mean-vox1 for a non-benchmark smoke test."
        exit 1
      fi
      find "${VOXCELEB2_WAV_ROOT}" -name "*.wav" \
        | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF, $0}' \
        | sort > "${DATA_VOX2}/wav.scp"
      echo "  vox2_dev wav.scp entries: $(wc -l < "${DATA_VOX2}/wav.scp")"
    fi
  else
    echo "[Stage 1b] Skipped (--no-cal-mean / CAL_MEAN=false)."
  fi
fi

# ===================== Stage 2: reserved (WeSpeaker raw.list N/A) =====================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "[Stage 2] Skipped — NeMo TitaNet reads wav.scp directly."
fi

# ===================== Stage 3: Extract embeddings (Vox1 test; Vox2 dev if cal_mean) =====================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "[Stage 3a] TitaNet embeddings: Vox1 test → Kaldi xvector.scp ..."
  mkdir -p "${EXP_DIR}/embeddings/vox1"
  extract_invocation "${EXP_DIR}/embeddings/vox1" "${DATA_DIR}/wav.scp"
  echo "  Vox1 embeddings: $(wc -l < "${EXP_DIR}/embeddings/vox1/xvector.scp")"

  if [[ "${CAL_MEAN}" == "true" ]]; then
    echo "[Stage 3b] TitaNet embeddings: Vox2 dev (cohort for mean_vec, cf. ResNet293_LM / v2/local/score.sh) ..."
    mkdir -p "${EXP_DIR}/embeddings/vox2_dev"
    extract_invocation "${EXP_DIR}/embeddings/vox2_dev" "${DATA_VOX2}/wav.scp"
    echo "  Vox2 dev embeddings: $(wc -l < "${EXP_DIR}/embeddings/vox2_dev/xvector.scp")"
  fi
fi

# ===================== Stage 4: Cosine score + EER / threshold at EER =====================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "[Stage 4] Cosine scoring (wespeaker/bin/score.py) + metrics ..."
  mkdir -p "${EXP_DIR}/scores"
  trials="vox1_O_cleaned.kaldi"

  if [[ "${CAL_MEAN}" == "true" ]]; then
    CAL_ARGS=(--cal_mean True --cal_mean_dir "${EXP_DIR}/embeddings/vox2_dev")
    echo "  Post-processing: subtract mean of Vox2-dev TitaNet embeddings, then cosine (v2/local/score.sh)."
  else
    CAL_ARGS=(--cal_mean False --cal_mean_dir "none")
    echo "  Post-processing: cosine on raw embeddings (no cohort mean)."
  fi

  for trial_file in ${trials}; do
    echo "  Scoring ${trial_file} ..."
    python wespeaker/bin/score.py \
      --exp_dir "${EXP_DIR}" \
      --eval_scp_path "${EXP_DIR}/embeddings/vox1/xvector.scp" \
      "${CAL_ARGS[@]}" \
      "${DATA_DIR}/trials/${trial_file}"
  done

  {
    echo ""
    echo "========== RESULTS (TitaNet / NeMo) =========="
    echo "  CAL_MEAN=${CAL_MEAN}  SMOKE_MEAN_FROM_VOX1=${SMOKE_MEAN_FROM_VOX1}"
    echo "=============================================="
  } | tee -a "${EXP_DIR}/scores/vox1_cos_result"

  for trial_file in ${trials}; do
    python "${LOCAL_DIR}/report_metrics.py" \
      --p_target 0.01 \
      --c_miss 1 \
      --c_fa 1 \
      "${EXP_DIR}/scores/${trial_file}.score" \
      2>&1 | tee -a "${EXP_DIR}/scores/vox1_cos_result"
    echo "" | tee -a "${EXP_DIR}/scores/vox1_cos_result"
  done
  echo "Full log: ${EXP_DIR}/scores/vox1_cos_result"
fi
