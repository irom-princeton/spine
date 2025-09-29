#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --ntasks-per-node=4                             ## Processors per node
#SBATCH --mem=60G                                       ## RAM per node
#SBATCH --time=3:00:00                                  ## Walltime
#SBATCH --gres=gpu:1                                    ## Number of GPUs
#SBATCH --job-name=train_rad_field                      ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#
# send mail when the process completes/fails
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=<email address>

# Optional (requires removing extra leading '#' characters)
###SBATCH --exclude=node[101]                           ## Exclude some nodes


# load modules or conda environments here
source ~/.bashrc

# activate virtual environment
conda activate <path to environment>
# or
micromamba activate <path to environment>

# export any necessary environment variables
# export <ENV Variable>

# data path
data_path=$1

# output path for trained model
output_path=$2

# encoder name
encoder_name=$3

# option to train a NeRF
train_nerf=$4

if [[ "$train_nerf" == "true" ]]; then
    cd spine_nerf

    # base model name, one of: (splatfacto, nerfacto)
    base_model_name=nerfacto
else
    cd spine_gsplat

    # base model name, one of: (splatfacto, nerfacto)
    base_model_name=splatfacto
fi

# install the Python Package
pip install -e .

# run
if [[ "$encoder_name" == "base" ]]; then
    # no semantics
    ns-train "${base_model_name}"  --data "${data_path}" \
        --output-dir "${output_path}/${encoder_name}" \
        --viewer.quit-on-train-completion True
else
    ns-train spine  --data "${data_path}" \
        --output-dir "${output_path}/${encoder_name}" \
        --pipeline.datamanager.distill_cam_feats True \
        --pipeline.datamanager.distill_img_feats True \
        --pipeline.datamanager.semantic_extractor "${encoder_name}" \
        --viewer.quit-on-train-completion True
fi