# 3D CNN with Global Average Pooling

## What This Project Does
- Takes a local image (input_image.jpg)
- Converts it into a 3D volume by stacking
- Passes it through a 3D CNN model
- Applies MaxPooling to reduce size
- Applies Global Average Pooling (GAP)
- Outputs 128 feature values as a bar chart

## Project Structure
evaluation_project/
├── image/
│   └── input_image.jpg
├── 3d_cnn.ipynb
├── requirements.txt
└── README.md

## How To Run
1. Install libraries:
pip install -r requirements.txt

2. Open notebook:
3d_cnn.ipynb

3. Run all cells top to bottom

## Libraries Used
- TensorFlow
- NumPy
- Matplotlib
- Pillow