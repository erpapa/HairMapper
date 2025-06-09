#!/bin/bash

#启用虚拟环境
eval "$(conda shell.bash hook)"
conda activate StyleFlow

cd /data/deep/StyleFlow/HairMapper

# Step 1
python align_images.py \
    --data_dir ./test_data \
    --shape_predictor_model_path ./ckpts/shape_predictor_68_face_landmarks.dat \
    --psp_model_path ./ckpts/psp_ffhq_frontalization.pt \
    --e4e_model_path ./ckpts/e4e_ffhq_encode.pt

# Step 2
python main_mapper.py \
    --data_dir ./test_data \
    --best_model_path ./mapper/checkpoints/final/best_model.pt \
    --face_parsing_model_path ./ckpts/face_parsing.pth \
    --learning_rate 0.01 \
    --num_iterations 100 \
    --loss_weight_feat 0.00003 \
    --loss_weight_id 1.0 \
    --remain_ear \
    --diffuse \
    --dilate_kernel_size 50 \
    --blur_kernel_size 30 \
    --truncation_psi 0.75
