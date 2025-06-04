#!/bin/bash

# image path
IMAGE_DIR="./img"
mkdir -p results

# output path
DB_PATH="results/database.db"
SPARSE_DIR="results/sparse/0"
UNDISTORTED_DIR="results/undistorted"

# 0. Ititialize directories
echo "[Step 0] Cleaning old outputs (if any)..."
rm -rf $DB_PATH $SPARSE_DIR $UNDISTORTED_DIR

# 1. Feature extraction
echo "[Step 1] Running feature extractor..."
colmap feature_extractor \
    --database_path $DB_PATH \
    --image_path $IMAGE_DIR \
    --SiftExtraction.use_gpu 0 \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE

# 2. Feature matching
echo "[Step 2] Running exhaustive matcher..."
colmap exhaustive_matcher \
    --database_path $DB_PATH \
    --SiftMatching.use_gpu 0

# 3. Sparse reconstruction (Mapping)
echo "[Step 3] Running mapper..."
mkdir -p $SPARSE_DIR
colmap mapper \
    --database_path $DB_PATH \
    --image_path $IMAGE_DIR \
    --output_path $SPARSE_DIR

# 4. Undistort images
echo "[Step 4] Undistorting images for Gaussian Splatting..."
mkdir -p $UNDISTORTED_DIR
colmap image_undistorter \
    --image_path $IMAGE_DIR \
    --input_path $SPARSE_DIR/0 \
    --output_path $UNDISTORTED_DIR \
    --output_type COLMAP \
    --max_image_size 2000

echo "[Done] All COLMAP steps completed!"
echo "Use this for Gaussian Splatting: -s $UNDISTORTED_DIR"