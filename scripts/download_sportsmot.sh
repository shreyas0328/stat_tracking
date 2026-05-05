#!/usr/bin/env bash
# Download SportsMOT into ./data/sportsmot.
#
# SportsMOT is hosted on Google Drive by the original authors at
#   https://github.com/MCG-NJU/SportsMOT
# and there is also a Hugging Face mirror that is more convenient on Colab.
#
# This script tries the HF mirror first (no auth, resumable, fast on Colab),
# and falls back to printing manual instructions if that fails. Adjust the
# HF_REPO variable below if the mirror moves.

set -euo pipefail

OUT_DIR="${1:-data/sportsmot}"
HF_REPO="MCG-NJU/SportsMOT"   # if this 404s, see manual instructions below

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

if command -v huggingface-cli >/dev/null 2>&1; then
    echo "Attempting download from Hugging Face mirror: $HF_REPO ..."
    if huggingface-cli download "$HF_REPO" --repo-type dataset --local-dir . ; then
        echo "Download succeeded."
        exit 0
    fi
    echo "HF download failed, falling back to manual instructions."
fi

cat <<'EOF'
============================================================
Could not download SportsMOT automatically.

Manual steps:
  1. Visit https://github.com/MCG-NJU/SportsMOT and follow the
     "Download" section to get the Google Drive link for the
     dataset archive.
  2. On Colab, you can use gdown to fetch it once you have the
     file ID, e.g.:
        pip install -q gdown
        gdown --id <FILE_ID> -O sportsmot.zip
        unzip -q sportsmot.zip -d data/sportsmot
  3. The expected layout after extraction is:
        data/sportsmot/
          splits_txt/
          dataset/
            train/  val/  test/
============================================================
EOF
exit 1
