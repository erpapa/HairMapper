import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image

from argparse import Namespace
from PIL import ImageFile
sys.path.append(".")
sys.path.append("./ffhq_dataset")
sys.path.append("./encoder4editing")
ImageFile.LOAD_TRUNCATED_IMAGES = True

from encoder4editing.models.psp import pSp
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=False,
                        default='./test_data',
                        help='Directory to save the results. If not specified, '
                             '`data/double_chin_pair/images` will be used by default.')
    parser.add_argument('--model_path', type=str, required=False,
                        default='./ckpts/e4e_ffhq_encode.pt',
                        help='FFHQ Encode model path.')
    parser.add_argument('--landmarks_model_path', type=str, required=False,
                        default='./ckpts/shape_predictor_68_face_landmarks.dat',
                        help='Landmarks model path.')
    return parser.parse_args()

def run_on_batch(inputs, net):
    latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return latents

def image_latent(img_transforms, net, file_path, code_path):
    if os.path.exists(code_path):
        return
    input_image = PIL.Image.open(file_path)
    transformed_image = img_transforms(input_image)
    with torch.no_grad():
        latents = run_on_batch(transformed_image.unsqueeze(0), net)
        latent = latents[0].cpu().numpy()
        latent = np.reshape(latent, (1,18,512))
        np.save(code_path, latent)
        print(f'save to {code_path}')

def run_image_align():
    args = parse_args()
    input_image_dir = os.path.join(args.data_dir, 'input')
    origin_image_dir = os.path.join(args.data_dir, 'origin')
    data_code_dir = os.path.join(args.data_dir, 'code')
    os.makedirs(origin_image_dir, exist_ok=True)
    os.makedirs(data_code_dir, exist_ok=True)

    ckpt = torch.load(args.model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()

    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    landmarks_detector = LandmarksDetector(args.landmarks_model_path)
    for img_name in os.listdir(input_image_dir):
        raw_img_path = os.path.join(input_image_dir, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            img_name = os.path.splitext(img_name)[0]
            face_img_name = f'{img_name}.png' if i == 1 else f'{img_name}_{i}.png'
            face_code_name = f'{img_name}.npy' if i == 1 else f'{img_name}_{i}.npy'
            aligned_face_path = os.path.join(origin_image_dir, face_img_name)
            code_path = os.path.join(data_code_dir, face_code_name) 
            image_align(raw_img_path, aligned_face_path, face_landmarks)
            image_latent(img_transforms, net, aligned_face_path, code_path)
    

if __name__ == "__main__":
    run_image_align()

    
