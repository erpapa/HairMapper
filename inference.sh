#!/bin/bash

#启用虚拟环境
eval "$(conda shell.bash hook)"
conda activate StyleFlow

cd /data/deep/StyleFlow/HairMapper

# Step 1
cd ./encoder4editing
python run_e4e_inversion.py \
    --proj_data_dir ../test_data \
    --e4e_model_path ../ckpts/e4e_ffhq_encode.pt \
    --shape_predictor_model_path ../ckpts/shape_predictor_68_face_landmarks.dat

# Step 2
cd ../detect_attributes
python run_dpr_light.py \
    --proj_data_dir ../test_data \
    --dpr_model_path ../ckpts/dpr_model/trained_model_03.t7

# Step 3
cd ../head_pose_estimation
# python head_pose_estimation.py \
#     --proj_data_dir ../test_data \
#     --face_model_path ../ckpts/shape_predictor_68_face_landmarks.dat \
#     --shuff_model_path ../ckpts/head_pose_model/shuff_epoch_120.pkl

python head_pose_detect.py \
    --proj_data_dir ../test_data \
    --face_model_path ../ckpts/shape_predictor_68_face_landmarks.dat \
    --weights_model_path ../ckpts/head_pose_model/resnet50.pt \
    --network resnet50

# Step 4
cd ../styleflow_editing
python run_styleflow_editing.py \
    --proj_data_dir ../test_data \
    --network_pkl ../ckpts/stylegan_model/stylegan2-ffhq-config-f.pkl \
    --flow_model_path ../ckpts/styleflow_model/modellarge10k.pt \
    --exp_direct_path ../ckpts/styleflow_model/expression_direction.pt \
    --exp_recognition_path ../ckpts/exprecog_model/FacialExpRecognition_model.t7 \
    --edit_items delight,norm_attr,multi_yaw

# Step 5
cd ../
python main_encode.py \
    --data_dir ./test_data \
    --e4e_model_path ./ckpts/e4e_ffhq_encode.pt

# Step 6
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
