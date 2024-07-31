#!/bin/bash

export NEPTUNE_API_TOKEN={YOUR_API_TOKEN}

# download mask file here
# https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

cd /path/to/SwiFUN # move to where 'SwiFUN' is located

TRAINER_ARGS="--accelerator gpu --max_epochs 10 --precision 16 --num_nodes 1 --devices 2 --strategy ddp_find_unused_parameters_false" #--strategy ddp_find_unused_parameters_false" # devices should be 4 for sbatch
MAIN_ARGS='--loggername neptune --dataset_name UKB --image_path {image_path} --task_path {ground_truth_path} --mask_filename Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
DATA_ARGS='--batch_size 2 --eval_batch_size 16 --num_workers 8'
DEFAULT_ARGS='--project_name {project_name}'
OPTIONAL_ARGS='--cope 1 --c_multiplier 2 --clf_head_version v1 --downstream_task tfMRI_3D --use_scheduler --gamma 0.5 --cycle 0.7 --loss_type mse --last_layer_full_MSA True '  
RESUME_ARGS="" 

CUDA_VISIBLE_DEVICES=6,7

python project/main.py $TRAINER_ARGS \
  $MAIN_ARGS \
  $DEFAULT_ARGS \
  $DATA_ARGS \
  $OPTIONAL_ARGS \
  $RESUME_ARGS \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 5e-5 \
  --model swinunetr \
  --time_as_channel \
  --attn_drop_rate 0.3 \
  --depth 2 2 2 2 \
  --embed_dim 24 \
  --sequence_length 20 \
  --window_size 7 \
  --img_size 96 96 96