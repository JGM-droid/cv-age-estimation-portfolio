"""Train an age-estimation model with transfer learning on face images."""

from pathlib import Path

import pandas as pd
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
VALIDATION_SPLIT = 0.25
SEED = 12345


def load_train(path: str):
    labels = pd.read_csv(Path(path) / "labels.csv")

    train_datagen = ImageDataGenerator(
        validation_split=VALIDATION_SPLIT,
        rescale=1.0 / 255,
        horizontal_flip=True,
    )

    return train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=str(Path(path) / "final_files"),
        x_col="file_name",
        y_col="real_age",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="raw",
        subset="training",
        seed=SEED,
    )


def load_test(path: str):
    labels = pd.read_csv(Path(path) / "labels.csv")

    test_datagen = ImageDataGenerator(
        validation_split=VALIDATION_SPLIT,
        rescale=1.0 / 255,
    )

    return test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=str(Path(path) / "final_files"),
        x_col="file_name",
        y_col="real_age",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="raw",
        subset="validation",
        seed=SEED,
    )


def create_model(input_shape):
    backbone = ResNet50(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False,
    )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation="relu"))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(model, train_data, test_data, epochs=20):
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=len(train_data),
        validation_steps=len(test_data),
        verbose=2,
    )
    return model


def main(data_path: str = "data/faces"):
    train_data = load_train(data_path)
    test_data = load_test(data_path)
    model = create_model((224, 224, 3))
    train_model(model, train_data, test_data)
    return model


if __name__ == "__main__":
    main()
