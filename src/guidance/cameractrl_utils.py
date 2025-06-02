from diffusers import DDIMScheduler
import torchvision.transforms.functional as TF

import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    _append_dims,
    StableVideoDiffusionPipelineOutput
)

# tensor2vid 함수가 더 이상 제공되지 않으므로 직접 구현
def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> np.ndarray:
    """Convert a tensor of shape [batch, channels, frames, height, width] to a numpy array of shape [batch, frames, height, width, channels]."""
    mean = torch.tensor(mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1, 1)
    video = video * std + mean
    video = video.clamp(0, 1)
    video = video.cpu().numpy()
    video = np.transpose(video, (0, 2, 3, 4, 1))
    return video

import sys
sys.path.append('./')

from cameractrl.models.pose_adaptor import CameraPoseEncoder
from cameractrl.models.unet import UNetSpatioTemporalConditionModelPoseCond
from cameractrl.pipelines.pipeline_animation import StableVideoDiffusionPipelinePoseCond

from packaging import version as pver

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4
    # V: # of video frames

    B = K.shape[0] # batch size

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )  
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    return plucker

class CameraCtrlGuidance(nn.Module):
    def __init__(self, device, fp16=True, t_range=[0.02, 0.98], pose_adaptor_ckpt="checkpoints/CameraCtrl_svdxt.ckpt", ori_model_path="checkpoints/stable-video-diffusion-img2vid-xt"):
        super().__init__()
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        
        # Stable Video Diffusion XT 기본 모델 경로
        self.ori_model_path = ori_model_path
        
        # 모델 설정값들
        unet_subfolder = "unet"
        down_block_types = ["CrossAttnDownBlockSpatioTemporalPoseCond", "CrossAttnDownBlockSpatioTemporalPoseCond", "CrossAttnDownBlockSpatioTemporalPoseCond", "DownBlockSpatioTemporal"]
        up_block_types = ["UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporalPoseCond", "CrossAttnUpBlockSpatioTemporalPoseCond", "CrossAttnUpBlockSpatioTemporalPoseCond"]
        
        # Pose Encoder 설정
        pose_encoder_kwargs = {
            "downscale_factor": 8,
            "channels": [320, 640, 1280, 1280],
            "nums_rb": 2,  # config 파일에서는 2로 설정
            "cin": 384,
            "ksize": 1,
            "sk": True,
            "use_conv": False,
            "compression_factor": 1,
            "temporal_attention_nhead": 8,
            "attention_block_types": ("Temporal_Self",),
            "temporal_position_encoding": True,
            "temporal_position_encoding_max_len": 25,  # XT는 25프레임
            "rescale_output_factor": 1.0
        }
        
        # Attention Processor 설정
        attention_processor_kwargs = {
            "add_spatial": False,
            "add_temporal": True,
            "attn_processor_name": "attn1",
            "pose_feature_dimensions": [320, 640, 1280, 1280],
            "query_condition": True,
            "key_value_condition": True,
            "scale": 1.0,
            "enable_xformers": True
        }
        
        # 추가 설정
        self.num_inference_steps = 25  # config 파일의 num_inference_steps
        self.min_guidance_scale = 1.0  # config 파일의 min_guidance_scale
        self.max_guidance_scale = 3.0  # config 파일의 max_guidance_scale
        self.video_length = 25  # config 파일의 video_length
        self.first_image_cond = True  # config 파일의 first_image_cond
        self.sample_latent = True  # config 파일의 sample_latent
        
        # 이미지 크기 설정
        self.image_height = 256  # CameraCtrl이 처리할 이미지의 높이
        self.image_width = 256  # CameraCtrl이 처리할 이미지의 너비
        self.original_pose_width = 720  # 입력 이미지의 원본 너비
        self.original_pose_height = 1280  # 입력 이미지의 원본 높이
        self.enable_xformers = True  # xformers 메모리 최적화 사용
        
        # 모델 컴포넌트 로드
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(ori_model_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(ori_model_path, subfolder="feature_extractor")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(ori_model_path, subfolder="image_encoder")
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(ori_model_path, subfolder="vae")
        self.unet = UNetSpatioTemporalConditionModelPoseCond.from_pretrained(
            ori_model_path,
            subfolder=unet_subfolder,
            down_block_types=down_block_types,
            up_block_types=up_block_types
        )
        
        # Pose Encoder 초기화
        self.pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)
        
        # Attention Processor 설정
        print("Setting the attention processors")
        self.unet.set_pose_cond_attn_processor(**attention_processor_kwargs)
        
        # Pose Adaptor 체크포인트 로드
        print(f"Loading weights of camera encoder and attention processor from {pose_adaptor_ckpt}")
        self.ckpt_dict = torch.load(pose_adaptor_ckpt, map_location=self.unet.device)
        
        # Pose Encoder 가중치 로드
        self.pose_encoder_state_dict = self.ckpt_dict['pose_encoder_state_dict']
        self.pose_encoder_m, self.pose_encoder_u = self.pose_encoder.load_state_dict(self.pose_encoder_state_dict)
        assert len(self.pose_encoder_m) == 0 and len(self.pose_encoder_u) == 0
        
        # Attention Processor 가중치 로드
        self.attention_processor_state_dict = self.ckpt_dict['attention_processor_state_dict']
        _, self.attention_processor_u = self.unet.load_state_dict(self.attention_processor_state_dict, strict=False)
        assert len(self.attention_processor_u) == 0  # 가중치가 제대로 로드되었는지 확인
        
        # 모델을 device로 이동
        self.vae.to(device)
        self.image_encoder.to(device)
        self.unet.to(device)
        self.pose_encoder.to(device)
        
        # Pipeline 초기화
        self.pipe = StableVideoDiffusionPipelinePoseCond(
            vae=self.vae,
            image_encoder=self.image_encoder,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            feature_extractor=self.feature_extractor,
            pose_encoder=self.pose_encoder
        ).to(device)

    @torch.no_grad()
    def get_img_embeds(self, x, caption=None, image_transform_info=None):
        """
        이미지 임베딩 생성
        Args:
            x: [batch_size, 3, height, width] 형태의 이미지 텐서 (0-1 범위)
            caption: 이미지에 대한 캡션 (선택사항)
            image_transform_info: 이미지 전처리 변환 정보
        """
        # 이미지 크기 조정
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        
        # CLIP 이미지 인코더를 위한 전처리
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor(
            images=x_pil, 
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt"
        ).pixel_values.to(device=self.device, dtype=self.dtype)
        
        # CLIP 이미지 임베딩 생성
        c = self.pipe.image_encoder(x_clip).image_embeds
        
        # VAE 임베딩 생성
        v = self.pipe.vae.encode(x.to(self.dtype)).latent_dist.mode()
        v = v * self.pipe.vae.config.scaling_factor
        
        self.embeddings = [c, v]
        self.caption = caption
        self.image_transform_info = image_transform_info  # 변환 정보 저장

    def process_camera_params(self, position, rotation, H=256, W=256, image_transform_info=None, scale_factor=None):
        """
        카메라 파라미터를 처리하여 Plucker embedding 생성
        Args:
            position: [x, y, z] 카메라 위치
            rotation: [rx, ry, rz] 카메라 회전
            H, W: 이미지 크기 (정사각형 이미지 사용)
            image_transform_info: 이미지 전처리 변환 정보 (process.py에서 생성)
            scale_factor: 포즈 정규화에 사용된 스케일 팩터
        Returns:
            plucker_embedding: [1, 1, H, W, 6] Plucker embedding
        """
        # 카메라 파라미터를 텐서로 변환
        position = torch.tensor(position, device=self.device).float()
        rotation = torch.tensor(rotation, device=self.device).float()
        
        # 스케일 팩터가 제공된 경우 로그 출력
        if scale_factor is not None:
            print(f'[DEBUG] Using scale factor: {scale_factor:.4f}')
        
        # process.py의 변환 정보를 사용하여 카메라 파라미터 조정
        if image_transform_info is not None:
            original_width = image_transform_info['original_width']
            original_height = image_transform_info['original_height']
            output_size = image_transform_info.get('output_size', 256)
            border_ratio = image_transform_info.get('border_ratio', 0.2)
            
            # process.py의 변환 과정을 역추적하여 카메라 파라미터 조정
            if image_transform_info.get('recenter', True) and image_transform_info.get('bbox') is not None:
                bbox = image_transform_info['bbox']
                img_scale_factor = image_transform_info['scale_factor']
                placement_offset = image_transform_info['placement_offset']
                
                # 원본 이미지에서 객체의 중심점
                obj_center_x = (bbox['y_min'] + bbox['y_max']) / 2  # 주의: bbox의 x,y가 뒤바뀜
                obj_center_y = (bbox['x_min'] + bbox['x_max']) / 2
                
                # 원본 이미지 중심에서 객체 중심까지의 오프셋
                offset_from_center_x = obj_center_x - original_width / 2
                offset_from_center_y = obj_center_y - original_height / 2
                
                # 스케일링 후 256x256 이미지에서의 주점 위치
                # process.py에서는 객체를 중앙에 배치하므로, 최종 이미지의 중심이 주점
                cx = W / 2
                cy = H / 2
                
                # 스케일 팩터 계산: 원본 이미지에서 최종 이미지로의 변환
                # 원본 이미지의 단위 길이가 최종 이미지에서 얼마나 되는지
                original_to_final_scale = img_scale_factor * (output_size / max(original_width, original_height))
                
                # 카메라 내부 파라미터의 스케일 조정
                fx = 1.0 * original_to_final_scale
                fy = 1.0 * original_to_final_scale
                
            else:
                # 리센터링이 없거나 객체가 없는 경우
                img_scale_factor = image_transform_info.get('scale_factor', 1.0)
                
                # 단순 스케일링만 적용
                scale_x = W / original_width
                scale_y = H / original_height
                fx = 1.0 * scale_x
                fy = 1.0 * scale_y
                cx = W / 2
                cy = H / 2
                
        else:
            # 기본값 사용 (원본 코드와 동일)
            scale_x = W / self.original_pose_width
            scale_y = H / self.original_pose_height
            fx = 1.0 * scale_x
            fy = 1.0 * scale_y
            cx = W / 2
            cy = H / 2
        
        # 스케일 팩터가 있는 경우 focal length에도 적용
        if scale_factor is not None:
            fx *= scale_factor
            fy *= scale_factor
        
        K = torch.tensor([[fx, fy, cx, cy]], device=self.device).float()
        
        # 카메라 외부 파라미터 (위치와 회전으로부터 계산)
        # 회전 행렬 계산 (Euler angles to rotation matrix)
        rx, ry, rz = rotation
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(rx), -torch.sin(rx)],
            [0, torch.sin(rx), torch.cos(rx)]
        ], device=self.device)
        
        Ry = torch.tensor([
            [torch.cos(ry), 0, torch.sin(ry)],
            [0, 1, 0],
            [-torch.sin(ry), 0, torch.cos(ry)]
        ], device=self.device)
        
        Rz = torch.tensor([
            [torch.cos(rz), -torch.sin(rz), 0],
            [torch.sin(rz), torch.cos(rz), 0],
            [0, 0, 1]
        ], device=self.device)
        
        R = Rz @ Ry @ Rx  # 최종 회전 행렬
        
        # 카메라-월드 변환 행렬 생성
        c2w = torch.eye(4, device=self.device)
        c2w[:3, :3] = R
        c2w[:3, 3] = position
        c2w = c2w.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        
        # Plucker embedding 생성
        plucker = ray_condition(K.unsqueeze(0), c2w, H, W, self.device)
        return plucker

    @torch.no_grad()
    def refine(self, pred_rgb, position, rotation, guidance_scale=5, steps=50, num_frames=25):
        """
        이미지 개선을 위한 refine 함수
        Args:
            pred_rgb: [batch_size, 3, height, width] 렌더링된 이미지 (정사각형 이미지)
            position: [x, y, z] 카메라 위치
            rotation: [rx, ry, rz] 카메라 회전
            guidance_scale: classifier-free guidance 스케일
            steps: 디노이징 스텝 수
            num_frames: 생성할 프레임 수 (XT는 25프레임)
        """
        batch_size = pred_rgb.shape[0]
        
        # 이미지를 256x256으로 리사이즈 (정사각형 유지)
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        latents = self.pipe.vae.encode(pred_rgb_256.to(self.dtype)).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor

        # Plucker embedding 생성 (256x256 정사각형 이미지에 맞춤, 저장된 변환 정보 사용)
        plucker_embedding = self.process_camera_params(
            position, rotation, H=256, W=256, 
            image_transform_info=getattr(self, 'image_transform_info', None)
        )
        
        # 이미지 임베딩 생성
        image_embeddings = self.pipe.image_encoder(pred_rgb_256).image_embeds

        # 캡션이 있는 경우 텍스트 임베딩 생성
        if self.caption is not None:
            text_embeddings = self.pipe.text_encoder(self.caption)
            image_embeddings = torch.cat([text_embeddings, image_embeddings], dim=0)

        # 디노이징 루프
        self.noise_scheduler.set_timesteps(steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.pipe.unet(
                latents,
                t,
                encoder_hidden_states=image_embeddings,
                pose_features=plucker_embedding
            ).sample
            
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # latent를 이미지로 디코딩
        imgs = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs
    
    def train_step(self, pred_rgb, position, rotation, step_ratio=None, guidance_scale=5, scale_factor=None):
        """
        학습 스텝
        Args:
            pred_rgb: [batch_size, 3, height, width] 렌더링된 이미지
            position: [x, y, z] 카메라 위치
            rotation: [rx, ry, rz] 카메라 회전
            step_ratio: 학습 진행도에 따른 노이즈 스케일 조정
            guidance_scale: classifier-free guidance 스케일
            scale_factor: 포즈 정규화에 사용된 스케일 팩터
        """
        batch_size = pred_rgb.shape[0]
        
        # 이미지를 latent space로 인코딩
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        latents = self.pipe.vae.encode(pred_rgb_256.to(self.dtype)).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor

        # 노이즈 스케일 결정
        if step_ratio is not None:
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        # 노이즈 추가
        noise = torch.randn_like(latents)
        latents_noisy = self.noise_scheduler.add_noise(latents, noise, t)
        
        # CameraCtrl을 통한 예측
        with torch.no_grad():
            # Plucker embedding 생성 (스케일 팩터 포함)
            plucker_embedding = self.process_camera_params(
                position, rotation, H=256, W=256,
                image_transform_info=getattr(self, 'image_transform_info', None),
                scale_factor=scale_factor
            )
            
            # 이미지 임베딩 생성
            image_embeddings = self.pipe.image_encoder(pred_rgb_256).image_embeds
            
            # 노이즈 예측
            noise_pred = self.pipe.unet(
                latents_noisy,
                t,
                encoder_hidden_states=image_embeddings,
                pose_features=plucker_embedding
            ).sample

        # SDS loss 계산
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        
        return loss
    
    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # latents = 1 / self.vae.config.scaling_factor * latents

        # imgs = self.vae.decode(latents).sample
        # imgs = (imgs / 2 + 0.5).clamp(0, 1)

        # return imgs

        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames
    
    def encode_imgs(
        self,
        imgs,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        mode=False
        ):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1 # 
        imgs = imgs.to(self.device)

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample() 
        latents = latents * self.vae.config.scaling_factor

        return latents
    
    
if __name__ == '__main__':
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    import kiui

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='input image path')
    parser.add_argument('--position', type=float, nargs=3, default=[0, 0, 1], help='camera position [x, y, z]')
    parser.add_argument('--rotation', type=float, nargs=3, default=[0, 0, 0], help='camera rotation [rx, ry, rz] in degrees')
    parser.add_argument('--guidance_scale', type=float, default=5.0, help='guidance scale for generation')
    parser.add_argument('--steps', type=int, default=50, help='number of denoising steps')

    opt = parser.parse_args()

    device = torch.device('cuda')

    print(f'[INFO] loading image from {opt.input} ...')
    image = kiui.read_image(opt.input, mode='tensor')
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)

    print(f'[INFO] loading CameraCtrl model ...')
    cameractrl = CameraCtrlGuidance(device)
    
    print(f'[INFO] running model ...')
    cameractrl.get_img_embeds(image)

    # 카메라 위치와 회전을 라디안으로 변환
    position = np.array(opt.position)
    rotation = np.array(opt.rotation) * np.pi / 180.0  # degrees to radians

    print(f'[INFO] generating frame with camera parameters:')
    print(f'  position: {position}')
    print(f'  rotation: {rotation}')

    # 프레임 생성
    with torch.no_grad():
        output = cameractrl.refine(
            image,
            position=position,
            rotation=rotation,
            guidance_scale=opt.guidance_scale,
            steps=opt.steps
        )

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(output[0].permute(1, 2, 0).cpu().numpy())
    plt.title('Generated Frame')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_relative_pose(poses_data, zero_first_frame_scale=True, target_distance=0.7):
    """
    Apply scale factor normalization to camera poses
    Args:
        poses_data: List of pose dictionaries with 'position' and 'rotation' keys
        zero_first_frame_scale: Whether to set first frame at origin
        target_distance: Target distance for normalization (default: 0.7)
    Returns:
        scaled_poses: List of normalized pose dictionaries
    """
    if not poses_data or len(poses_data) == 0:
        return poses_data
    
    # Convert to camera-to-world matrices
    c2ws = []
    for pose in poses_data:
        position = np.array(pose['position'])
        rotation = np.array(pose['rotation'])
        
        # Convert Euler angles to rotation matrix
        rx, ry, rz = rotation
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx  # Rotation matrix
        
        # Create 4x4 camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = position
        c2ws.append(c2w)
    
    # Get the average distance from camera to origin
    distances = [np.linalg.norm(c2w[:3, 3]) for c2w in c2ws]
    avg_distance = np.mean(distances)
    
    # Scale factor to normalize the distances
    # This will make the average distance from camera to origin approximately target_distance
    scale_factor = target_distance / avg_distance
    
    # Scale all camera positions
    scaled_c2ws = []
    for c2w in c2ws:
        scaled_c2w = c2w.copy()
        scaled_c2w[:3, 3] *= scale_factor
        scaled_c2ws.append(scaled_c2w)
    
    source_cam_c2w = scaled_c2ws[0]
    
    if zero_first_frame_scale:
        cam_to_origin = 0
    else:
        cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
    
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    abs2rel = target_cam_c2w @ np.linalg.inv(scaled_c2ws[0])
    ret_poses = [target_cam_c2w] + [abs2rel @ c2w for c2w in scaled_c2ws[1:]]
    
    # Convert back to pose dictionaries
    scaled_poses = []
    for i, c2w in enumerate(ret_poses):
        # Extract position
        position = c2w[:3, 3]
        
        # Extract rotation matrix and convert to Euler angles
        R = c2w[:3, :3]
        
        # Extract Euler angles from rotation matrix
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        scaled_pose = {
            'position': position.tolist(),
            'rotation': [roll, pitch, yaw],
            'scale_factor': scale_factor,
            'target_distance': target_distance,
            'original_avg_distance': avg_distance
        }
        
        # Copy other fields if they exist
        if i < len(poses_data):
            original_pose = poses_data[i]
            for key in original_pose:
                if key not in ['position', 'rotation']:
                    scaled_pose[key] = original_pose[key]
        
        scaled_poses.append(scaled_pose)
    
    return scaled_poses