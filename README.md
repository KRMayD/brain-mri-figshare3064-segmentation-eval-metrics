# Brain MRI Figshare3064 Segmentation Metrics

This repository contains metrics only for a completed Brain MRI segmentation evaluation. It intentionally excludes all evaluation images, ground-truth masks, saliency maps, coarse masks, SAM predictions, model checkpoints, and model weights.

## Evaluation

- Test set: Figshare Brain MRI, 3,064 images
- Prompt mode: metadata-informed/oracle prompt mapping, seed 0
- Saliency: `vbeta=2.0`, `vvar=0.3`, `vlayer=9`, seed `12`
- Postprocessing: k-means with filtering; top-2 contours converted to box prompts
- Segmenter: SAM ViT-H with box prompts and multicontour mode
- Metric: strict DSC and NSD, NSD tolerance `2.0 mm`

## Result

| Metric | Value |
| --- | ---: |
| Mean DSC | 0.5500764322 |
| Mean NSD | 0.5870636030 |
| Evaluated samples | 3,064 |
| Missing predictions | 0 |

## Files

- `metrics.json`: aggregate metric summary.
- `per_sample_metrics.csv`: per-image DSC/NSD values and filenames.
- `provenance.json`: checkpoint path at evaluation time, test roots, prompt mode, and pipeline settings.

## Checkpoint Identity

The evaluated local checkpoint had SHA-256:

```text
34d72f3248b53b93ccfbe5acfbe26e8bf64d8c360a20cf51a221b1455bb09b6f
```

This checkpoint is not included in this repository.
