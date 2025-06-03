import numpy as np
from pathlib import Path
import os
import shutil

"""
    ** 미리 설정할 것**
    - main 함수 정의 안에 dataset_dir, output_dir 설정
    - num_frames 설정: 영상 몇 프레임으로 등분할 것인지(예: 14 or 25, ...)
    RUN: python generate_pose_file_from_COLMAP.py (argument 없음)
"""

def read_colmap_cameras(cameras_file):
    """COLMAP cameras.txt 파일 읽기"""
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            data = line.strip().split()
            if len(data) == 0:
                continue
            camera_id = int(data[0])
            model = data[1]
            width = int(data[2])
            height = int(data[3])
            params = np.array([float(x) for x in data[4:]])
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_colmap_images(images_file):
    """COLMAP images.txt 파일 읽기"""
    images = []
    with open(images_file, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('#'):
                i += 1
                continue
            if len(lines[i].strip()) == 0:
                i += 1
                continue
                
            # 이미지 데이터 읽기
            data = lines[i].strip().split()
            image_id = int(data[0])
            qw, qx, qy, qz = map(float, data[1:5])  # 쿼터니언
            tx, ty, tz = map(float, data[5:8])      # 위치
            camera_id = int(data[8])
            image_name = data[9]
            
            # 포인트 데이터 읽기 (다음 줄)
            i += 1
            if i < len(lines):
                points_data = lines[i].strip().split()
                points = []
                for j in range(0, len(points_data), 3):
                    if j + 2 < len(points_data):
                        x, y, z = map(float, points_data[j:j+3])
                        points.append([x, y, z])
            
            images.append({
                'image_id': image_id,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'camera_id': camera_id,
                'image_name': image_name,
                'points': points
            })
            i += 1
    return images

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """쿼터니언을 회전 행렬로 변환"""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def create_pose_file(colmap_dir, output_file, fps=30, num_frames=26):
    """COLMAP 결과를 real_estate 형식의 포즈 파일로 변환"""
    # COLMAP 파일 읽기
    cameras = read_colmap_cameras(Path(colmap_dir) / "cameras.txt")
    images = read_colmap_images(Path(colmap_dir) / "images.txt")
    
    # 이미지 이름에서 프레임 번호를 추출하여 정렬
    def extract_frame_number(img_dict):
        """frame_xxxxx.jpg 형태에서 프레임 번호 추출, 없으면 image_id 사용"""
        import re
        image_name = img_dict['image_name']
        match = re.search(r'frame_(\d+)', image_name)
        if match:
            return int(match.group(1))
        else:
            # 프레임 번호를 찾을 수 없으면 원래 image_id 사용
            print(f"Warning: Could not extract frame number from '{image_name}', using image_id {img_dict['image_id']}")
            return img_dict['image_id']
    
    # 이미지를 프레임 번호 순으로 정렬 (프레임 번호가 없으면 image_id 순)
    images.sort(key=extract_frame_number)
    
    print(f"Sorted {len(images)} images by frame number")
    if len(images) > 0:
        first_frame = extract_frame_number(images[0])
        last_frame = extract_frame_number(images[-1])
        print(f"Frame range: {first_frame} to {last_frame}")
    
    # 전체 프레임 수
    total_frames = len(images)
    
    # num_frames개의 균등한 간격의 프레임 인덱스 계산 (첫 프레임 포함)
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    # 선택된 프레임만 사용
    selected_images = [images[i] for i in indices]
    
    # 카메라가 하나만 있는 경우를 위한 처리
    if len(cameras) == 1:
        single_camera_id = list(cameras.keys())[0]
        single_camera = cameras[single_camera_id]
        print(f"Only one camera found (ID: {single_camera_id}), using it for all frames")
    
    # 포즈 데이터 생성
    poses = []
    for idx, img in enumerate(selected_images):  # enumerate로 인덱스 추가
        # 쿼터니언을 회전 행렬로 변환
        R = quaternion_to_rotation_matrix(img['qw'], img['qx'], img['qy'], img['qz'])
        
        # 카메라 파라미터 가져오기
        if len(cameras) == 1:
            # 카메라가 하나만 있는 경우, 모든 프레임에 동일한 카메라 사용
            camera = single_camera
        else:
            # 카메라가 여러 개인 경우, 해당 이미지의 카메라 사용
            camera = cameras[img['camera_id']]
            
        if camera['model'] == 'SIMPLE_PINHOLE':
            fx = camera['params'][0]
            fy = fx
            cx = camera['params'][1]
            cy = camera['params'][2]
        else:  # PINHOLE
            fx = camera['params'][0]
            fy = camera['params'][1]
            cx = camera['params'][2]
            cy = camera['params'][3]
        
        # 타임스탬프를 0부터 순차적으로 증가
        timestamp = idx  # 0부터 시작하는 순차적인 번호
        
        # real_estate 형식으로 포즈 데이터 포맷팅 (k3 제외)
        pose_data = f"{timestamp} {fx} {fy} {cx} {cy} 0.0 0.0 "
        
        # 회전 행렬과 위치 벡터를 3x4 행렬로 변환
        t = np.array([img['tx'], img['ty'], img['tz']])
        Rt = np.column_stack((R, t))
        
        # 3x4 행렬을 한 줄로 변환
        pose_data += " ".join([str(x) for x in Rt.flatten()]) + "\n"
        
        poses.append(pose_data)
    
    # 포즈 파일 작성
    with open(output_file, 'w') as f:
        # 임시 URL 추가 (실제 URL이 없는 경우)
        f.write("https://www.youtube.com/watch?v=placeholder\n")
        f.write("".join(poses))

def process_dataset_folder(dataset_dir, output_dir, num_frames=26):
    """데이터셋 폴더 내의 모든 COLMAP 데이터를 처리"""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # colmap_1 폴더 찾기
    colmap_dirs = list(dataset_path.glob("**/colmap_*"))
    
    for colmap_dir in colmap_dirs:
        # COLMAP 파일이 있는지 확인
        if not (colmap_dir / "cameras.txt").exists() or not (colmap_dir / "images.txt").exists():
            print(f"Skipping {colmap_dir}: Missing COLMAP files")
            continue
        
        # 출력 파일 이름 생성
        output_file = output_path / f"{colmap_dir.name}_svdxt.txt"
        
        print(f"Processing {colmap_dir}...")
        create_pose_file(colmap_dir, output_file, num_frames=num_frames)
        print(f"Created {output_file}")

def main():
    # 데이터셋 폴더와 출력 폴더 설정
    dataset_dir = "dataset"
    output_dir = "CameraCtrl/assets/pose_files"
    num_frames = 25  # 원하는 프레임 수 설정
    
    # 데이터셋 폴더 처리
    process_dataset_folder(dataset_dir, output_dir, num_frames=num_frames)

if __name__ == "__main__":
    main()