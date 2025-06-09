import os
import time
import json
import argparse
import dlib

import cv2
import numpy as np

import torch
from torchvision import transforms

from models import get_model
from utils.general import compute_euler_angles_from_rotation_matrices, draw_cube, draw_axis

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_data_dir', type=str, required=False,
                        default='../test_data',
                        help='Directory to save the results. If not specified, '
                             '`../test_data` will be used by default.')
    parser.add_argument('--face_model_path', type=str, required=False,
                        default='../ckpts/shape_predictor_68_face_landmarks.dat',
                        help='Face model path.')
    parser.add_argument('--weights_model_path', type=str, required=False,
                        default='../ckpts/head_pose_model/resnet50.pt',
                        help='Head pose weights path.')
    parser.add_argument('--network', type=str, required=False,
                        default='resnet50',
                        help='Head pose network name.')
    return parser.parse_args()

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def calculate_mouth_open_score(shape_np):
    # 提取关键点
    top_lip = shape_np[62]
    bottom_lip = shape_np[66]
    left_corner = shape_np[48]
    right_corner = shape_np[54]

    # 垂直张开距离
    mouth_open = np.linalg.norm(top_lip - bottom_lip)
    # 水平宽度
    mouth_width = np.linalg.norm(left_corner - right_corner)

    # 防止除以0
    if mouth_width == 0:
        return 0

    # 张开比值（通常范围在 0.0 到 0.6 左右）
    ratio = mouth_open / mouth_width

    # 映射到 0 - 100
    scaled = np.clip(ratio * 300, 0, 100)  # 你可以根据样本调整 300 这个系数
    return scaled

def expand_bbox(x_min, y_min, x_max, y_max, factor=0.2):
    """Expand the bounding box by a given factor."""
    width = x_max - x_min
    height = y_max - y_min

    x_min_new = x_min - int(factor * height)
    y_min_new = y_min - int(factor * width)
    x_max_new = x_max + int(factor * height)
    y_max_new = y_max + int(factor * width)

    return max(0, x_min_new), max(0, y_min_new), x_max_new, y_max_new

def predict_head_pose(proj_data_dir, file_name, device, img_transforms, detector, predictor, head_pose):
    img_path = os.path.join(proj_data_dir, 'inversions', file_name)
    cv_img = cv2.imread(img_path)
    cv_img = cv2.resize(cv_img, [512,512])
    # 转为灰度图识别速度更快
    frame = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = detector(frame, 0)
    if len(faces) == 0:
        return
    shape = predictor(frame, faces[0])
    shape = shape_to_np(shape)
    # 计算微笑分数
    smile_score = calculate_mouth_open_score(shape)
    smile = smile_score * 0.01

    image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    image = img_transforms(image)
    with torch.no_grad():
        image = image.unsqueeze(0)
        image = image.to(device)
        rotation_matrix = head_pose(image)
        euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
        p_pred_deg = euler[:, 0].cpu().item()
        y_pred_deg = euler[:, 1].cpu().item()
        r_pred_deg = euler[:, 2].cpu().item()
        print(f'smile: {smile}, yaw: {-y_pred_deg}, pitch: {p_pred_deg}, roll: {r_pred_deg}')

    parsed_attr = {
        "Age": 23.0,
        "Baldness": 0.05,
        "Beard": 0.0,
        "Expression": smile,
        "Gender": 0,
        "Glasses": 0,
        "Pitch": p_pred_deg,
        "Yaw": -y_pred_deg
    }
    base_name = os.path.splitext(file_name)[0]
    output_attr_dir = os.path.join(proj_data_dir, 'attributes')
    os.makedirs(output_attr_dir, exist_ok=True)
    with open(os.path.join(output_attr_dir, f'{base_name}.json'), 'w') as f:
        json.dump(parsed_attr, f, indent=4, sort_keys=True)

def run_head_pose():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_model_path)

    state_dict = torch.load(args.weights_model_path, map_location='cpu')
    head_pose = get_model(args.network, num_classes=6, pretrained=False)
    head_pose.load_state_dict(state_dict)
    head_pose.to(device)
    head_pose.eval()

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    file_names = sorted(os.listdir(os.path.join(args.proj_data_dir, 'inversions')))
    for file_name in file_names:
        file_ext = os.path.splitext(file_name)[1].lower()
        if (file_ext == '.png' or file_ext == '.jpg'):
            tic = time.time()
            predict_head_pose(args.proj_data_dir, file_name, device, img_transforms, detector, predictor, head_pose)
            toc = time.time()
            print('Editing {} done, took {:.4f} seconds.'.format(file_name, toc - tic))


if __name__ == '__main__':
    run_head_pose()
