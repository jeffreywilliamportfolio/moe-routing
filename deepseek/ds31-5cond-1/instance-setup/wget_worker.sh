#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <url> <dest_dir> <log_dir>" >&2
  exit 1
fi

url="$1"
dest_dir="$2"
log_dir="$3"

mkdir -p "$dest_dir" "$log_dir"

filename="$(basename "$url")"
tmp_path="$dest_dir/${filename}.part"
final_path="$dest_dir/${filename}"
log_path="$log_dir/${filename}.log"

echo "[worker] downloading $filename"
wget \
  --continue \
  --tries=0 \
  --timeout=30 \
  --waitretry=5 \
  --retry-connrefused \
  --progress=bar:force:noscroll \
  -O "$tmp_path" \
  "$url" 2>&1 | tee "$log_path"

mv "$tmp_path" "$final_path"
echo "[worker] finished $final_path"
