# Brain MRI Figshare3064 Segmentation Evaluation Artifact

This repository preserves every non-image component required to reproduce the documented Figshare3064 metadata-informed segmentation evaluation. Raw MRI images, ground-truth masks, and image-derived saliency/coarse/SAM prediction PNG files are intentionally excluded.

## Included

- Fine-tuned BiomedCLIP inference weight in `artifacts/model/model.safetensors` through Git LFS.
- Its exact local model configuration files.
- Image-specific metadata-informed/oracle prompt mapping for 3,064 Figshare filenames.
- Original pipeline code snapshots, logs, provenance, aggregate metrics, and per-sample metrics.
- A portable evaluation wrapper and an official SAM ViT-H downloader with SHA-256 verification.
- The `busi2` Python package snapshot used at evaluation time.

## Excluded Image Assets

- Input MRI images and ground-truth masks.
- Saliency maps, coarse masks, and SAM prediction images.
- The optimizer-containing raw DDP training checkpoint. It is unnecessary for inference because `model.safetensors` contains the exact converted model weight used by saliency inference.
- SAM ViT-H binary, because GitHub rejects files larger than 2 GiB. Use `scripts/download_sam_vit_h.sh` to retrieve the official, checksum-verified checkpoint.

## Reproduction

1. Clone [KRMayD/finegrained-vlm-training](https://github.com/KRMayD/finegrained-vlm-training) and install its evaluation environment.
2. Clone this repository with Git LFS enabled so `artifacts/model/model.safetensors` is materialized.
3. Download SAM:

```bash
bash scripts/download_sam_vit_h.sh /path/to/sam_vit_h_4b8939.pth
```

4. Prepare an evaluation root containing `images/` and `masks/` using the same filenames as `artifacts/prompts/figshare3064_metadata_oracle_prompts_seed0.json`.
5. Run:

```bash
bash scripts/run_evaluation.sh \
  /path/to/finegrained-vlm-training \
  artifacts/model \
  /path/to/eval_root \
  artifacts/prompts/figshare3064_metadata_oracle_prompts_seed0.json \
  /path/to/sam_vit_h_4b8939.pth \
  /path/to/output \
  0
```

## Recorded Result

- Mean DSC: `0.5500764322`
- Mean NSD: `0.5870636030`
- Test images: `3,064`
- Prompt mode: metadata-informed/oracle, seed 0
- Saliency: `vbeta=2.0`, `vvar=0.3`, `vlayer=9`, seed `12`
- Postprocessing: k-means + filter, top-2 contour box prompts
- NSD tolerance: `2.0 mm`

The evaluated raw checkpoint SHA-256 was `34d72f3248b53b93ccfbe5acfbe26e8bf64d8c360a20cf51a221b1455bb09b6f`.
