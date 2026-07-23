#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 7 ]; then
  echo "usage: $0 <project_root> <model_dir> <eval_root> <prompts_json> <sam_checkpoint> <output_dir> <gpu>" >&2
  exit 1
fi

project_root="$1"
model_dir="$2"
eval_root="$3"
prompts_json="$4"
sam_checkpoint="$5"
output_dir="$6"
gpu="$7"
python_bin="${PYTHON_BIN:-python}"
mkdir -p "$output_dir"/{saliency,coarse,sam}
export PYTHONPATH="$project_root/saliency_maps:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$gpu"

"$python_bin" "$project_root/saliency_maps/generate_saliency_maps.py" \
  --input-path "$eval_root/images" --output-path "$output_dir/saliency" \
  --model-name BiomedCLIP --json-path "$prompts_json" --finetuned \
  --local-model-dir "$model_dir" --reproduce \
  --vbeta 2.0 --vvar 0.3 --vlayer 9 --seed 12
"$python_bin" "$project_root/postprocessing/postprocess_saliency_maps.py" \
  --input-path "$eval_root/images" --sal-path "$output_dir/saliency" \
  --output-path "$output_dir/coarse" --postprocess kmeans --filter --num-contours 2
"$python_bin" "$project_root/segment-anything/prompt_sam.py" \
  --input "$eval_root/images" --mask-input "$output_dir/coarse" \
  --output "$output_dir/sam" --model-type vit_h --checkpoint "$sam_checkpoint" \
  --prompts boxes --multicontour
"$python_bin" "$project_root/tools/evaluate_brain_mri_segmentation.py" \
  --gt-dir "$eval_root/masks" --pred-dir "$output_dir/sam" \
  --output-json "$output_dir/metrics.json" --output-csv "$output_dir/per_sample_metrics.csv" \
  --tolerance-mm 2.0 --strict --expected-count 3064
