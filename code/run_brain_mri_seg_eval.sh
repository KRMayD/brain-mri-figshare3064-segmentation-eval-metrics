#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

if [ $# -lt 3 ]; then
  echo "usage: bash scripts/run_brain_mri_seg_eval.sh <model_label> <checkpoint_path> <gpu> [csv_path]" >&2
  exit 1
fi

MODEL_LABEL="$1"
CHECKPOINT_PATH="$2"
GPU="$3"
CSV_PATH="${4:-/home/msko021220/finegrained-vlm-training/data/brain_mri_dpo_modality_tumortype_100caption_test30_seed42.csv}"

BASE_DIR="/home/msko021220/finegrained-vlm-training"
PYTHON_BIN="${PYTHON_BIN:-/home/msko021220/.conda/envs/busi2/bin/python}"
MODEL_NAME="${MODEL_NAME:-hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-/home/msko021220/MedCLIP-SAMv2-dpoloss/segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth}"
EVAL_ROOT="${EVAL_ROOT:-${BASE_DIR}/runtime/brain_mri_seg_eval_tumortype_test30}"
MODEL_ROOT="${MODEL_ROOT:-${BASE_DIR}/runtime/brain_mri_seg_eval_models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/outputs/brain_mri_seg_eval}"
PROMPTS_JSON="${PROMPTS_JSON:-${EVAL_ROOT}/prompts.json}"
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

VBETA="${VBETA:-0.1}"
VVAR="${VVAR:-1.0}"
VLAYER="${VLAYER:-7}"
SEED="${SEED:-12}"
POSTPROCESS="${POSTPROCESS:-kmeans}"
NUM_CONTOURS="${NUM_CONTOURS:-2}"
NSD_TOLERANCE_MM="${NSD_TOLERANCE_MM:-2.0}"
EXPECTED_EVAL_COUNT="${EXPECTED_EVAL_COUNT:-}"
STRICT_EVAL="${STRICT_EVAL:-1}"
EVAL_CHECK_ARGS=()
if [ "${STRICT_EVAL}" = "1" ]; then
  EVAL_CHECK_ARGS+=(--strict)
fi
if [ -n "${EXPECTED_EVAL_COUNT}" ]; then
  EVAL_CHECK_ARGS+=(--expected-count "${EXPECTED_EVAL_COUNT}")
fi

MODEL_DIR="${MODEL_ROOT}/${MODEL_LABEL}"
RUN_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${MODEL_DIR}" "${RUN_DIR}" "${LOG_DIR}"

if [ ! -f "${CSV_PATH}" ]; then
  echo "CSV not found: ${CSV_PATH}" >&2
  exit 1
fi
if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [ ! -f "${SAM_CHECKPOINT}" ]; then
  echo "SAM checkpoint not found: ${SAM_CHECKPOINT}" >&2
  exit 1
fi

if [ ! -f "${EVAL_ROOT}/prompts.json" ] || [ ! -d "${EVAL_ROOT}/images" ] || [ ! -d "${EVAL_ROOT}/masks" ]; then
  "${PYTHON_BIN}" "${BASE_DIR}/tools/prepare_brain_mri_seg_eval.py" \
    --csv "${CSV_PATH}" \
    --output-root "${EVAL_ROOT}" \
    "${LIMIT_ARGS[@]}"
fi
if [ ! -f "${PROMPTS_JSON}" ]; then
  echo "prompts JSON not found: ${PROMPTS_JSON}" >&2
  exit 1
fi

cp "${BASE_DIR}/saliency_maps/model/config.json" "${MODEL_DIR}/config.json"
cp "${BASE_DIR}/saliency_maps/model/configuration_biomed_clip.py" "${MODEL_DIR}/configuration_biomed_clip.py"
cp "${BASE_DIR}/saliency_maps/model/modeling_biomed_clip.py" "${MODEL_DIR}/modeling_biomed_clip.py"

if [ ! -f "${MODEL_DIR}/model.safetensors" ] || [ "${FORCE_CONVERT:-0}" = "1" ]; then
  PYTHONPATH="${BASE_DIR}/biomedclip_finetuning/open_clip/src:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${BASE_DIR}/atomic_dpo/convert_atomic_checkpoint.py" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --output-path "${MODEL_DIR}/pytorch_model.bin" \
    --model-name "${MODEL_NAME}"
  "${PYTHON_BIN}" -c "import torch; from safetensors.torch import save_file; sd=torch.load('${MODEL_DIR}/pytorch_model.bin', map_location='cpu'); save_file(sd, '${MODEL_DIR}/model.safetensors')"
  rm -f "${MODEL_DIR}/pytorch_model.bin"
fi

export PYTHONPATH="${BASE_DIR}/saliency_maps:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

{
  echo "model_label=${MODEL_LABEL}"
  echo "checkpoint=${CHECKPOINT_PATH}"
  echo "csv=${CSV_PATH}"
  echo "gpu=${GPU}"
  echo "eval_root=${EVAL_ROOT}"
  echo "prompts_json=${PROMPTS_JSON}"
  echo "run_dir=${RUN_DIR}"
  echo "vbeta=${VBETA} vvar=${VVAR} vlayer=${VLAYER} seed=${SEED}"
  echo "postprocess=${POSTPROCESS} num_contours=${NUM_CONTOURS} nsd_tolerance_mm=${NSD_TOLERANCE_MM}"
  echo "strict_eval=${STRICT_EVAL} expected_eval_count=${EXPECTED_EVAL_COUNT:-unset}"

  "${PYTHON_BIN}" "${BASE_DIR}/saliency_maps/generate_saliency_maps.py" \
    --input-path "${EVAL_ROOT}/images" \
    --output-path "${RUN_DIR}/saliency" \
    --model-name BiomedCLIP \
    --json-path "${PROMPTS_JSON}" \
    --finetuned \
    --local-model-dir "${MODEL_DIR}" \
    --reproduce \
    --vbeta "${VBETA}" \
    --vvar "${VVAR}" \
    --vlayer "${VLAYER}" \
    --seed "${SEED}"

  "${PYTHON_BIN}" "${BASE_DIR}/postprocessing/postprocess_saliency_maps.py" \
    --input-path "${EVAL_ROOT}/images" \
    --sal-path "${RUN_DIR}/saliency" \
    --output-path "${RUN_DIR}/coarse" \
    --postprocess "${POSTPROCESS}" \
    --filter \
    --num-contours "${NUM_CONTOURS}"

  "${PYTHON_BIN}" "${BASE_DIR}/segment-anything/prompt_sam.py" \
    --input "${EVAL_ROOT}/images" \
    --mask-input "${RUN_DIR}/coarse" \
    --output "${RUN_DIR}/sam" \
    --model-type vit_h \
    --checkpoint "${SAM_CHECKPOINT}" \
    --prompts boxes \
    --multicontour

  "${PYTHON_BIN}" "${BASE_DIR}/tools/evaluate_brain_mri_segmentation.py" \
    --gt-dir "${EVAL_ROOT}/masks" \
    --pred-dir "${RUN_DIR}/sam" \
    --output-json "${RUN_DIR}/metrics.json" \
    --output-csv "${RUN_DIR}/per_sample_metrics.csv" \
    --tolerance-mm "${NSD_TOLERANCE_MM}" \
    "${EVAL_CHECK_ARGS[@]}"
} 2>&1 | tee "${LOG_DIR}/eval.log"
