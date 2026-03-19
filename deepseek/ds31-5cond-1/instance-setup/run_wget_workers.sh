#!/usr/bin/env bash
set -euo pipefail

manifest="${1:-download_manifest_deepseek_ud_q2_xl.txt}"
worker_count="${2:-6}"
dest_dir="${3:-/workspace/models/DeepSeek-V3-0324-UD-Q2_K_XL/UD-Q2_K_XL}"
log_dir="${4:-download_logs}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest_path="$script_dir/$manifest"
worker_script="$script_dir/wget_worker.sh"

if [[ ! -f "$manifest_path" ]]; then
  echo "missing manifest: $manifest_path" >&2
  exit 1
fi

if [[ ! -x "$worker_script" ]]; then
  chmod +x "$worker_script"
fi

mkdir -p "$dest_dir" "$log_dir"

mapfile -t urls < <(grep -v '^[[:space:]]*$' "$manifest_path")
if [[ ${#urls[@]} -eq 0 ]]; then
  echo "manifest is empty: $manifest_path" >&2
  exit 1
fi

echo "manifest: $manifest_path"
echo "dest_dir: $dest_dir"
echo "log_dir: $log_dir"
echo "requested_workers: $worker_count"
echo "files: ${#urls[@]}"

running=0
for url in "${urls[@]}"; do
  "$worker_script" "$url" "$dest_dir" "$log_dir" &
  running=$((running + 1))
  if (( running >= worker_count )); then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "all wget workers completed"
