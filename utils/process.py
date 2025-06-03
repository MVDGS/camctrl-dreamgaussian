import os
import glob
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import rembg

class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--model', default='u2net', type=str, help="rembg model, see https://github.com/danielgatis/rembg#models")
    parser.add_argument('--size', default=256, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=bool, default=True, help="recenter, potentially not helpful for multiview zero123")    
    opt = parser.parse_args()

    session = rembg.new_session(model_name=opt.model)

    if os.path.isdir(opt.path):
        print(f'[INFO] processing directory {opt.path}...')
        files = glob.glob(f'{opt.path}/*')
        out_dir = opt.path
    else: # isfile
        files = [opt.path]
        out_dir = os.path.dirname(opt.path)
    
    for file in files:

        out_base = os.path.basename(file).split('.')[0]
        out_rgba = os.path.join(out_dir, out_base + '_rgba.png')
        out_transform_info = os.path.join(out_dir, out_base + '_transform.json')

        # load image
        print(f'[INFO] loading image {file}...')
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        original_height, original_width = image.shape[:2]
        
        # carve background
        print(f'[INFO] background removal...')
        carved_image = rembg.remove(image, session=session) # [H, W, 4]
        mask = carved_image[..., -1] > 0

        # 변환 정보 초기화
        transform_info = {
            'original_width': original_width,
            'original_height': original_height,
            'output_size': opt.size,
            'border_ratio': opt.border_ratio,
            'recenter': opt.recenter
        }

        # recenter
        if opt.recenter:
            print(f'[INFO] recenter...')
            final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
            
            coords = np.nonzero(mask)
            if len(coords[0]) > 0:  # 객체가 있는 경우
                x_min, x_max = coords[0].min(), coords[0].max()
                y_min, y_max = coords[1].min(), coords[1].max()
                h = x_max - x_min
                w = y_max - y_min
                desired_size = int(opt.size * (1 - opt.border_ratio))
                scale = desired_size / max(h, w)
                h2 = int(h * scale)
                w2 = int(w * scale)
                x2_min = (opt.size - h2) // 2
                x2_max = x2_min + h2
                y2_min = (opt.size - w2) // 2
                y2_max = y2_min + w2
                final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
                
                # 변환 정보 저장
                transform_info.update({
                    'bbox': {
                        'x_min': int(x_min), 'x_max': int(x_max),
                        'y_min': int(y_min), 'y_max': int(y_max)
                    },
                    'object_size': {'width': int(w), 'height': int(h)},
                    'scale_factor': float(scale),
                    'final_object_size': {'width': int(w2), 'height': int(h2)},
                    'placement_offset': {'x': int(y2_min), 'y': int(x2_min)}
                })
            else:
                # 객체가 없는 경우 전체 이미지 사용
                final_rgba = cv2.resize(carved_image, (opt.size, opt.size), interpolation=cv2.INTER_AREA)
                transform_info.update({
                    'bbox': None,
                    'object_size': None,
                    'scale_factor': opt.size / max(original_width, original_height),
                    'final_object_size': {'width': opt.size, 'height': opt.size},
                    'placement_offset': {'x': 0, 'y': 0}
                })
        else:
            final_rgba = carved_image
            transform_info.update({
                'bbox': None,
                'object_size': None,
                'scale_factor': 1.0,
                'final_object_size': {'width': carved_image.shape[1], 'height': carved_image.shape[0]},
                'placement_offset': {'x': 0, 'y': 0}
            })
        
        # write image
        cv2.imwrite(out_rgba, final_rgba)
        
        # write transform info
        with open(out_transform_info, 'w') as f:
            json.dump(transform_info, f, indent=2)
        
        print(f'[INFO] saved {out_rgba} and {out_transform_info}')