#!/bin/bash
#SBATCH --job-name=preprocess-dataset
#SBATCH -N 1 # Same machine
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 32000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p cpu


seg=$1 # token or structure
domain=$2 # dmv va ssa or studentaid
YOUR_DIR=../data # change it to your own local dir

python data_preprocessor.py \
--dataset_config_name multidoc2dial \
--output_dir $YOUR_DIR/mdd_dpr \
--segmentation $seg \
--dpr