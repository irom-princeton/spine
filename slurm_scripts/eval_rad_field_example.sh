#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --ntasks-per-node=4                             ## Processors per node
#SBATCH --mem=60G                                       ## RAM per node
#SBATCH --time=3:00:00                                  ## Walltime
#SBATCH --gres=gpu:1                                    ## Number of GPUs
#SBATCH --job-name=eval_rad_field                       ## Job Name
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

# option to evaluate a NeRF
train_nerf=$3

if [[ "$train_nerf" == "true" ]]; then
    cd spine_nerf
else
    cd spine_gsplat
fi

# install spine
pip install -e .

cd spine/scripts

# evaluate
python main.py --config_name "${cfg_path}"