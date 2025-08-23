# utils.py
import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10 # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]
from PIL import Image # pyright: ignore[reportMissingImports]

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return x_train, y_train, x_test, y_test

def create_image_generators(x_train, y_train, x_val, y_val, batch_size=64):
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size)

    val_datagen = ImageDataGenerator()
    val_gen = val_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)

    return train_gen, val_gen

def prepare_custom_dataset_from_folder(folder_path, target_size=(128,128), test_size=0.2):
    # folder structure: folder_path/class_name/*.jpg
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    for idx, cls in enumerate(class_names):
        cls_folder = os.path.join(folder_path, cls)
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                fpath = os.path.join(cls_folder, fname)
                try:
                    im = Image.open(fpath).convert('RGB')
                    im = im.resize(target_size)
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                    images.append(arr)
                    labels.append(idx)
                except Exception as e:
                    print(f"skip {fpath}: {e}")
    images = np.array(images)
    labels = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, stratify=labels, random_state=42)
    return x_train, y_train, x_test, y_test, class_names

def save_label_map(class_names, out_path='label_map.json'):
    with open(out_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Saved label map to {out_path}")

def load_label_map(path='label_map.json'):
    import json
    with open(path, 'r') as f:
        return json.load(f)
