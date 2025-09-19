#!/usr/bin/env bash
# Generate a Quarto CI overlay profile that limits render to changed files
# Usage:
#   make_quarto_ci_profile.sh <changed.lst> <output.yml> [--strict]
# Where:
#   <changed.lst> is a NULL-delimited list of paths (what your diff step writes)
#   --strict replaces book.chapters with only the changed files (plus index/glossary/references)
#   (without --strict we only set project.render and keep your normal chapters)

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <changed.lst> <output.yml> [--strict]" >&2
  exit 2
fi

IN="$1"
OUT="$2"
STRICT=0
if [[ "${3:-}" == "--strict" ]]; then
  STRICT=1
fi

if [[ ! -f "$IN" ]]; then
  echo "ERROR: '$IN' not found" >&2
  exit 3
fi

# Read null-delimited files into array
mapfile -d '' -t FILES < "$IN" || true

# Helper to YAML-escape double quotes
yaml_quote() {
  local s="${1//\"/\\\"}"
  printf '"%s"' "$s"
}

# Start clean output
: > "$OUT"

# project.render
{
  echo "project:"
  echo "  render:"
  if [[ ${#FILES[@]} -gt 0 ]]; then
    for f in "${FILES[@]}"; do
      printf '    - %s\n' "$(yaml_quote "$f")"
    done
  else
    # keep valid YAML if empty (runner guards usually skip before we get here)
    echo "    []"
  fi
} >> "$OUT"

# Optional: restrict visible book chapters during CI
if [[ $STRICT -eq 1 ]]; then
  {
    echo ""
    echo "book:"
    echo "  chapters:"
    echo "    - index.qmd"
    echo "    - part: \"**CI**\""
    echo "      chapters:"
    if [[ ${#FILES[@]} -gt 0 ]]; then
      for f in "${FILES[@]}"; do
        printf '        - %s\n' "$(yaml_quote "$f")"
      done
    else
      echo "        []"
    fi
    echo "    - glossary.qmd"
    echo "    - references.qmd"
  } >> "$OUT"
fi

# Keep CI fast: reuse outputs for everything else and avoid multi-format re-exec
{
  echo ""
  echo "execute:"
  echo "  freeze: auto"
  echo ""
  echo "format:"
  echo "  html: {}"
} >> "$OUT"

echo "Wrote CI profile to: $OUT"
