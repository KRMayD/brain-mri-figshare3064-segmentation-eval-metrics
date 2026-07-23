#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <output_checkpoint_path>" >&2
  exit 1
fi

output_path="$1"
mkdir -p "$(dirname "$output_path")"
curl -L --fail --retry 3 --continue-at - \
  -o "$output_path" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
expected_sha256='a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e'
actual_sha256="$(sha256sum "$output_path" | awk '{print $1}')"
[ "$actual_sha256" = "$expected_sha256" ] || {
  echo "SAM checksum mismatch: $actual_sha256" >&2
  exit 1
}
