#!/usr/bin/env bash
set -euo pipefail

sd_versions=(1 2)
gen_numbers=(1 4)

export CUDA_VISIBLE_DEVICES=0 

# Loop through each SD version (1 and 2)
for sd_ver in "${sd_versions[@]}"; do
    # Conditional logic to set the correct data paths based on the SD version
    if [[ "$sd_ver" == "1" ]]; then
        data_paths=('prompts/sd1_mem.txt' 'prompts/sd1_nmem.txt')
        echo "Running for SD Version 1..."
    elif [[ "$sd_ver" == "2" ]]; then
        data_paths=('prompts/sd2_mem.txt' 'prompts/sd2_nmem.txt')
        echo "Running for SD Version 2..."
    else
        echo "Invalid sd_ver: $sd_ver"
        continue # Skip to the next iteration if the version is not 1 or 2
    fi

    # Loop through each data path file
    for data_path in "${data_paths[@]}"; do
        # Loop through each generation number
        for gen_num in "${gen_numbers[@]}"; do
            echo "--------------------------------------------------------"
            echo "Executing: sd_ver=$sd_ver, data_path=$data_path, gen_num=$gen_num"
            
            python detect_mem.py \
                --sd_ver "$sd_ver" \
                --data_path "$data_path" \
                --gen_num "$gen_num"

            echo "Execution complete."
        done
    done
done

echo "--------------------------------------------------------"
echo "All combinations have been run."