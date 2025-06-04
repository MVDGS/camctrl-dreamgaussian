INPUT_DIR=./results/undistorted/sparse
OUTPUT_DIR=./results/undistorted_txt

mkdir -p ./results/undistorted_txt

colmap model_converter \
  --input_path $INPUT_DIR \
  --output_path $OUTPUT_DIR \
  --output_type TXT