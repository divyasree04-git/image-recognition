# Image Recognition with CNN (TensorFlow + Flask)

## Setup
1. Create a Python virtual environment:
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

2. Install dependencies:
   pip install -r requirements.txt

## Training with CIFAR-10 (default)
Run:
   python train.py --mode cifar --output models --epochs 30 --batch 64

This will save models/best_model.h5 and models/final_model.h5.

## Training with custom dataset
Make folder `dataset/` with subfolders for each class:
dataset/dog/*.jpg
dataset/cat/*.jpg
...
Then run:
   python train.py --mode custom --custom_folder dataset --output models --epochs 30 --batch 32 --size 128 128

It will also save models/label_map.json.

## Run Web App
Ensure models/best_model.h5 exists (or change path in app.py).
Start server:
   python app.py

Open http://127.0.0.1:5000 in your browser, upload an image, get prediction.
https://znb9lskw-5000.inc1.devtunnels.ms/

## Notes
- If you used CIFAR-10 (32x32 images), update TARGET_SIZE in app.py to (32,32).
- For better accuracy on real images, prefer training on a custom dataset or use transfer learning (MobileNetV2, ResNet50).
# image-recognition
