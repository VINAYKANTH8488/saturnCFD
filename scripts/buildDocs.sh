#!/usr/bin/env bash
# Build the Doxygen HTML and PDF API documentation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# User-local toolchains used on the development machine.
export PATH="${HOME}/opt/doxygen-1.12.0/bin:${HOME}/.TinyTeX/bin/x86_64-linux:${PATH}"

if ! command -v doxygen >/dev/null 2>&1; then
  echo "doxygen is not installed; install it and rerun" >&2
  exit 1
fi

version="$(tr -d '[:space:]' < VERSION)"
doxygen Doxyfile

if command -v pdflatex >/dev/null 2>&1 && command -v makeindex >/dev/null 2>&1; then
  # Doxygen 1.12's tabu tables use an array hook removed in newer LaTeX.
  python3 - <<'PYINNER'
from pathlib import Path
p = Path("docs/doxygen/latex/refman.tex")
s = p.read_text()
if "providecommand{\\insert@pcolumn}" not in s:
    s = s.replace(
        "  %%\n  % Packages required by doxygen\n  \\makeatletter",
        "  %%\n  % Packages required by doxygen\n  \\makeatletter\n  \\providecommand{\\insert@pcolumn}{\\insert@column}",
        1,
    )
    s = s.replace(
        r"      pdftitle={saturn\+CFD},%",
        r"      pdftitle={saturn\+CFD},%" + "\n" + r"      pdfauthor={Vinaykanth},%",
        1,
    )
    s = s.replace(
        r"  [1ex]\large 0.\+1.\+0 \\",
        r"  [1ex]\large 0.\+1.\+0 \\" + "\n" + r"  \vspace*{0.5cm}" + "\n" + r"  {\large Vinaykanth}\\",
        1,
    )
    p.write_text(s)
PYINNER
  make -C docs/doxygen/latex >/dev/null
  cp docs/doxygen/latex/refman.pdf "docs/saturnCFD-${version}.pdf"
  echo "PDF:  docs/saturnCFD-${version}.pdf"
else
  echo "pdflatex/makeindex not found; HTML only" >&2
fi

echo "HTML: docs/doxygen/html/index.html"
