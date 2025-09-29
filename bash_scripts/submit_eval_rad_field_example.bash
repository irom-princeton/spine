#!/bin/bash

# Bash script for submitting the training jobs for the radiance fields

# list of paths all datasets
all_cfg_paths=(
    # "<name of config file for evaluation>"
    "eval_rf_example"
)

# option to evaluate a NeRF: Literal[true, false]
train_nerf=true

# job script
job_script="slurm_scripts/eval_rad_field_example.sh"

for cfg_path in "${all_cfg_paths[@]}"; do
    echo "Submitting: $cfg_path"
    # sbatch "$job_script" "$cfg_path"
    bash "$job_script" "$cfg_path" "$train_nerf"
done

