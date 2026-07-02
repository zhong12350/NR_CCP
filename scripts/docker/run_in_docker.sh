#!/bin/bash
# Run from macOS host: starts Ubuntu container with NR_CCP mounted at /workspace.
set -euo pipefail

REPO="${1:-$HOME/Desktop/NR_CCP}"

docker run -it --rm \
  -v "$REPO:/workspace" \
  -w /workspace \
  ubuntu:22.04 \
  bash
