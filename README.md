# DreamGaussian with Cameractrl guidance

<div align="center">
  <img src="docs/figures/front.gif" width="45%">
  <img src="docs/figures/side.gif" width="45%">
  <p><em>Figure: Front and side view of the generated 3D mesh</em></p>
</div>

This project integrates [CameraCtrl](https://arxiv.org/abs/2404.02101) as guidance into [DreamGaussian](https://arxiv.org/abs/2309.16653) to enhance 3D reconstruction results. By incorporating pose-aware frontal view inputs into the existing pipeline end-to-end, we achieve more refined frontal 3D reconstruction results.

## Key scripts
### `src/guidance/cameractrl_utils.py`
Implementation of `CameraCtrlGuidance class` based on CameraCtrl's inference and pipeline:
- Pipeline configuration
- Image embedding retrieval via `get_img_embeds` method
- Plucker embedding generation from Camera [R|t] via `process_camera_params` method
- SDS Loss calculation in `train_step` method

### `src/main.py`
- Integration of cameractrl guidance into the original dreamgaussian `main.py`
- Detailed training configuration specified in `configs/image_camctrl.yaml`

Training steps:
1. Gaussian initialization based on the first condition image
2. ~50 iterations:
  - Gaussian optimization using only 14 frames from cameractrl
3. 50 ~ 200 iterations:
  - Unseen viewpoint optimization using a mix of Zero123 and CameraCtrl
4. 200+ iterations:
  - Camera pose rotation every 20 iterations followed by optimization from CameraCtrl pose

---

## Environment & Dependencies
- dreamgaussian Docker environment + dreamgaussian conda environment
- Additional dependencies for cameractrl guidance:
```
# for stable-diffusion
huggingface_hub == 0.19.4
diffusers == 0.24.0
accelerate == 0.24.1
transformers == 4.36.2
```
- Checkpoint downloads
Inside `src/checkpoints/`, the following pre-trained models are required:
- stable-video-diffusion-img2vid-xt
- CameraCtrl_svdxt

---

## Pipeline
### Folder Structure
```
camctrl-dreamgaussian/
├── configs/  # Configuration files
│   ├── image_camctrl.yaml  # Dreamgaussian config
│   └── svdxt_320_576_cameractrl.yaml  # CameraCtrl config
├── src/  # Main source code directory
│   ├── checkpoints/  # Model checkpoints
│   ├── guidance/  # Guidance utilities
│   │   └── cameractrl_utils2.py
│   ├── cameractrl/  # CameraCtrl related code
│   ├── diff-gaussian-rasterization/  # Gaussian splatting implementation
│   ├── simple-knn/  # KNN implementation
│   ├── scripts/  # Utility scripts
│   ├── main.py  # Main training script
│   ├── main2.py  # mesh & 2D albedo map training script
│   ├── process.py  # Processing utilities
│   └── process_camctrl.py  # CameraCtrl processing
├── datasets/  # Dataset directory
|   ├── cameractrl_assets/ 
│   │   ├── pose_files  # generated pose files
│   │   └── svd_prompts.json
│   └── condition_image.png  # Input condition image
└── results/  # Output results directory
```
### Inference How-to
1. Prepare one condition image from CameraCtrl and one pose file (.txt) generated from generate_pose_file_from_COLMAP.py. Place the condition image in datasets/ and pose_file.txt in datasets/cameractrl_assets/pose_files/
2. Run process.py:
  ```
    python src/process.py datasets/condition_image.png
  ```
  - Original process.py for zero123

  ```
    python src/process_camctrl.py datasets/condition_image.png
  ```
  - Modified to maintain original image aspect ratio without square resizing
  - If the image ratio is not 720*1280 (typical phone camera), update cameractrl_image_width, cameractrl_image_height, cameractrl_original_pose_width, cameractrl_original_pose_height in src/configs/image_camctrl.yaml
  => Verify creation of condition_image_rgba.png in datasets/

3. Run main.py:
  ```
    python main.py --config src/configs/image_camctrl.yaml input=data/condition_image_rgba.png
  ```
4. Run the original main2.py for mesh refinement:
  ```
    python main2.py --config src/configs/image_camctrl.yaml input=data/condition_image_rgba.png
  ```