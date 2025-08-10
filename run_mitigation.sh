#!/usr/bin/env bash
set -euo pipefail

sd_versions=(1)
gen_numbers=(4)

export CUDA_VISIBLE_DEVICES=0 

for sd_ver in "${sd_versions[@]}"; do
  if [[ "$sd_ver" == "1" ]]; then
    data_paths=('prompts/sample_mitigation.txt')
    echo "Running for SD Version 1..."
  else
    echo "Unknown sd_ver: $sd_ver" >&2
    continue
  fi

  for data_path in "${data_paths[@]}"; do
    for gen_num in "${gen_numbers[@]}"; do
      echo "--------------------------------------------------------"
      echo "Executing: sd_ver=$sd_ver, data_path=$data_path, gen_num=$gen_num"

      python mitigate_mem.py \
        --sd_ver "$sd_ver" \
        --data_path "$data_path" \
        --gen_num "$gen_num"

      echo "Execution complete."
    done
  done
done

echo "--------------------------------------------------------"
echo "All combinations have been run."
