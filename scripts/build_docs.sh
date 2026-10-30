#!/usr/bin/env bash
# Build the Doxygen HTML API documentation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v doxygen >/dev/null 2>&1; then
  echo "doxygen is not installed; install it and rerun" >&2
  exit 1
fi
doxygen Doxyfile
echo "HTML: docs/doxygen/html/index.html"
