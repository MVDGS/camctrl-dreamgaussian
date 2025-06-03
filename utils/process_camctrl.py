import os
import glob
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

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

        # load image
        print(f'[INFO] loading image {file}...')
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        
        # carve background
        print(f'[INFO] background removal...')
        carved_image = rembg.remove(image, session=session) # [H, W, 4]
        
        # 객체의 상단과 하단 위치 찾기
        mask = carved_image[..., 3] > 0  # 알파 채널이 0보다 큰 부분이 객체
        rows = np.any(mask, axis=1)  # 각 행에 객체가 있는지 확인
        top = np.argmax(rows)  # 객체가 시작되는 첫 행
        bottom = len(rows) - np.argmax(rows[::-1])  # 객체가 끝나는 마지막 행
        
        # 상단 여백만큼 하단도 제거
        top_margin = top  # 상단 여백
        bottom_margin = len(rows) - bottom  # 하단 여백
        
        # 더 큰 여백만큼 양쪽에서 제거
        margin = max(top_margin, bottom_margin)
        
        # 새로운 이미지 생성 (원본 크기 유지)
        final_image = np.zeros_like(carved_image)
        
        # 객체 부분만 복사
        if margin > 0:
            # 상단과 하단을 동일한 margin만큼 제거
            final_image[margin:-margin] = carved_image[margin:-margin]
        
        # write image
        cv2.imwrite(out_rgba, final_image)