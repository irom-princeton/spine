#!/bin/bash

# Bash script for submitting the training jobs for the radiance fields

# list of paths all datasets
all_dset_paths=(
    # "<path to dataset directory containing transforms.json>"
)

# list of all encoder names
all_enc_names=(
    "base"
    "dino"
    "vggt"
)

# option to train a NeRF: Literal[true, false]
train_nerf=true

# output directory
train_output="<output path>"

if [[ "$train_nerf" == "true" ]]; then
    # update the output path
    train_output="${train_output}/nerf"
fi

# job script
job_script="slurm_scripts/train_rad_field_example.sh"

for dset_path in "${all_dset_paths[@]}"; do
    for enc_name in "${all_enc_names[@]}"; do
        echo "Submitting: $dset_path"
        sbatch "$job_script" "$dset_path" "$train_output" "$enc_name" "$train_nerf"
    done
done

