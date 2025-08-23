import sys
import numpy as np
from tensorflow.keras.applications import MobileNetV2 # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]
import os

def predict_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        model = MobileNetV2(weights="imagenet")  # Auto-download weights
        preds = model.predict(x)

        decoded = decode_predictions(preds, top=3)[0]
        return decoded
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Image file '{img_path}' not found.")
        sys.exit(1)

    predictions = predict_image(img_path)
    if isinstance(predictions, str):
        print(predictions)
    else:
        print("\nTop Predictions:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            print(f"{i+1}. {label} - {score*100:.2f}%")
