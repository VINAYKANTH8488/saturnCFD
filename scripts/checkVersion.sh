#!/usr/bin/env bash
# Keep the project version and author consistent across the repository.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

version="$(tr -d '[:space:]' < VERSION)"
fail=0

check() {
  local file="$1" pattern="$2"
  if ! grep -qE "$pattern" "$file"; then
    echo "version/author mismatch: $file (expected: $pattern)" >&2
    fail=1
  fi
}

check CMakeLists.txt "VERSION ${version}"
check src/core/version.hpp "kVersion = \"${version}\""
check Doxyfile "PROJECT_NUMBER.*${version}"
check README.md "Version:\*\* ${version}"
check PLAN.md "Version:\*\* ${version}"
check AUTHORS "Vinaykanth"
check src/core/version.hpp 'kAuthor = "Vinaykanth"'

if [ "${fail}" -ne 0 ]; then
  exit 1
fi
echo "version ${version}, author Vinaykanth: consistent"
