import cv2
import os
import glob
import argparse

def extract_frames_from_multiple_videos(video_dir, output_dir, every_n_frame=1):
    os.makedirs(output_dir, exist_ok=True)
    
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    total_saved = 0

    frame_idx = 0
    saved_idx = 0
    
    for video_path in sorted(video_paths):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            continue

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n_frame == 0:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                frame_filename = os.path.join(output_dir, f"frame_{saved_idx:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_idx += 1
                total_saved += 1

            frame_idx += 1

        cap.release()
        print(f"{base_name}: {saved_idx} frames saved.")

    print(f"\n✅ 총 {total_saved}개의 프레임이 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos in a folder")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to the directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for frames")
    parser.add_argument("--every_n_frame", type=int, default=1, help="Extract every N-th frame")

    args = parser.parse_args()

    extract_frames_from_multiple_videos(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        every_n_frame=args.every_n_frame,
    )
