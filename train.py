import argparse
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_cifar10(epochs, batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = build_model((32, 32, 3), 10)
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=epochs, batch_size=batch_size)
    return model, {i: str(i) for i in range(10)}

def train_custom(data_dir, epochs, batch_size):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        data_dir, target_size=(128, 128), batch_size=batch_size,
        class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=(128, 128), batch_size=batch_size,
        class_mode='categorical', subset='validation'
    )

    model = build_model((128, 128, 3), train_gen.num_classes)
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    return model, {v: k for k, v in train_gen.class_indices.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cifar", "custom"], required=True)
    parser.add_argument("--custom_folder", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    if args.mode == "cifar":
        model, label_map = train_cifar10(args.epochs, args.batch)
    else:
        if not args.custom_folder or not os.path.exists(args.custom_folder):
            raise ValueError("Please provide a valid custom dataset folder path.")
        model, label_map = train_custom(args.custom_folder, args.epochs, args.batch)

    # Always save as model.h5 and labels.json in main folder
    model.save("model.h5")
    with open("labels.json", "w") as f:
        json.dump(label_map, f)

    print("\n✅ Training complete. Model saved as model.h5 and labels.json")
