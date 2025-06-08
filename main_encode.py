import os
import sys
import time
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
import shutil

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
                             '`./test_data` will be used by default.')
    parser.add_argument('--e4e_model_path', type=str, required=False,
                        default='./ckpts/e4e_ffhq_encode.pt',
                        help='Encoder4editing model path.')
    return parser.parse_args()

def run_on_batch(inputs, net):
    latents = net.encode(inputs.to('cuda').float(),
                         resize=False,
                         randomize_noise=False)
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

def run_image_encode():
    args = parse_args()
    origin_image_dir = os.path.join(args.data_dir, 'origin')
    data_code_dir = os.path.join(args.data_dir, 'code')
    os.makedirs(origin_image_dir, exist_ok=True)
    os.makedirs(data_code_dir, exist_ok=True)

    ckpt = torch.load(args.e4e_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.e4e_model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()

    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    file_names = sorted(os.listdir(os.path.join(args.data_dir, 'inversions')))
    for file_name in file_names:
        base_name = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[1].lower()
        if (file_ext == '.png' or file_ext == '.jpg'):
            tic = time.time()
            edit_img_path = os.path.join(args.data_dir, 'edit', base_name, f'{base_name}_front.png')
            aligned_face_path = os.path.join(origin_image_dir, f'{base_name}.png') 
            shutil.copyfile(edit_img_path, aligned_face_path)
            code_path = os.path.join(data_code_dir, f'{base_name}.npy') 
            image_latent(img_transforms, net, aligned_face_path, code_path)
            toc = time.time()
            print('Editing {} done, took {:.4f} seconds.'.format(f'{base_name}_front.png', toc - tic))
    

if __name__ == "__main__":
    run_image_encode()
