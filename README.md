# Cameractrl ver. DreamGaussian How-To

### Environment & Dependencies
- 기존 현아가 제공해준 Docker & Conda 환경 활성화

### Folder Structure
```
dreamgaussian/
├── data/  # put condition image here
│   └── condition_image.png
│   └── cameractrl_assets/
│     └── pose_files/  # put pose file here
│       └── some_pose_file_svdxt.txt
├── configs/
│   ├── image_camctrl.yaml  # config for Dreamgaussian
│   └── svdxt_320_576_cameractrl.yaml  # config for CameraCtrl
├── checkpoints/  # model checkpoints
├── guidance/
│   └── cameractrl_utils2.py
├── main_new.py
└── process_camctrl.py
```

### Pipeline
1. condition image 하나, generate_pose_file_from_COLMAP.py로 뽑은 pose file (.txt) 하나 준비. 위의 위치 참고해서 파일 넣기
2. process_camctrl.py 돌리기:
  ```
    python process_camctrl.py data/condition_image.png
  ```
  - 정사각형으로 resize하는 과정 빼고, 기존 이미지 비율 유지하도록 해둠
  - 이미지 비율이 만약 폰으로 찍은 720*1280이 아니라면 configs/image_camctrl.yaml에서 cameractrl_image_width, cameractrl_image_height, cameractrl_original_pose_width, cameractrl_original_pose_height 수정
  => data/ 안에 condition_image_rgba.png 생성됨 확인

3. main_new.py 돌리기:
  ```
    python main_new.py --config configs/image_camctrl.yaml input=data/condition_image_rgba.png
  ```
4. 기존 코드의 main2.py 돌리기 (mesh refine):
  ```
    python main2.py --config configs/image_camctrl.yaml input=data/condition_image_rgba.png
  ```