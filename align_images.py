import os
import sys
import time
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np

from argparse import Namespace
from PIL import Image
from PIL import ImageFile
sys.path.append(".")
sys.path.append("./ffhq_dataset")
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=False,
                        default='./test_data',
                        help='Directory to save the results. If not specified, '
                             '`./test_data` will be used by default.')
    parser.add_argument('--shape_predictor_model_path', type=str, required=False,
                        default='./ckpts/shape_predictor_68_face_landmarks.dat',
                        help='Landmarks model path.')
    parser.add_argument('--psp_model_path', type=str, required=False,
                        default='./ckpts/psp_ffhq_frontalization.pt',
                        help='pixel2style2pixel model path.')
    parser.add_argument('--e4e_model_path', type=str, required=False,
                        default='./ckpts/e4e_ffhq_encode.pt',
                        help='encoder4editing model path.')
    parser.add_argument("--use_e4e",
                        help="if set, use encoder4editing",
                        action="store_true")
    return parser.parse_args()

def run_on_batch(inputs, net, device):
    images, latents, codes = net(inputs.to(device).float(),
                                 resize=False,
                                 randomize_noise=False,
                                 return_latents=True,
                                 return_codes=True)
    return images, latents, codes

def image_latent(img_transforms, net, device, crop_img_path, origin_img_path, code_path):
    if os.path.exists(code_path):
        return
    input_image = Image.open(crop_img_path)
    transformed_image = img_transforms(input_image)
    with torch.no_grad():
        images, latents, codes = run_on_batch(transformed_image.unsqueeze(0), net, device)
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images.cpu().numpy()
        Image.fromarray(images[0], 'RGB').save(origin_img_path)
        code_np = codes[0].cpu().numpy()
        code_np = np.reshape(code_np, (1,18,512))
        np.save(code_path, code_np)
        print(f'save to {code_path}')

def run_image_align():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_image_dir = os.path.join(args.data_dir, 'images')
    crop_image_dir = os.path.join(args.data_dir, 'crop')
    origin_image_dir = os.path.join(args.data_dir, 'origin')
    data_code_dir = os.path.join(args.data_dir, 'code')
    os.makedirs(crop_image_dir, exist_ok=True)
    os.makedirs(origin_image_dir, exist_ok=True)
    os.makedirs(data_code_dir, exist_ok=True)

    if args.use_e4e:
        sys.path.append("./encoder4editing")
        from encoder4editing.models.psp import pSp
        ckpt = torch.load(args.e4e_model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = args.e4e_model_path
        opts= Namespace(**opts)
        net = pSp(opts)
        net.to(device)
        net.eval()
    else:
        sys.path.append("./pixel2style2pixel")
        from pixel2style2pixel.models.psp import pSp
        ckpt = torch.load(args.psp_model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = args.psp_model_path
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
        if 'output_size' not in opts:
            opts['output_size'] = 1024
        opts= Namespace(**opts)
        net = pSp(opts)
        net.to(device)
        net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    landmarks_detector = LandmarksDetector(args.shape_predictor_model_path)
    file_names = sorted(os.listdir(input_image_dir))
    for file_name in file_names:
        base_name = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[1].lower()
        if (file_ext == '.png' or file_ext == '.jpg'):
            tic = time.time()
            raw_img_path = os.path.join(input_image_dir, file_name)
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                face_img_name = f'{base_name}.png' if i == 1 else f'{base_name}_{i}.png'
                face_code_name = f'{base_name}.npy' if i == 1 else f'{base_name}_{i}.npy'
                aligned_face_path = os.path.join(crop_image_dir, face_img_name)
                origin_img_path = os.path.join(origin_image_dir, face_img_name) 
                code_path = os.path.join(data_code_dir, face_code_name) 
                image_align(raw_img_path, aligned_face_path, face_landmarks)
                image_latent(img_transforms, net, device, aligned_face_path, origin_img_path, code_path)
            toc = time.time()
            print('Editing {} done, took {:.4f} seconds.'.format(file_name, toc - tic))
    

if __name__ == "__main__":
    run_image_align()
