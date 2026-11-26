CONV2D_FILTER_1 = 32
CONV2D_KERNAL_SIZE_1 = 3

CONV2D_FILTER_2 = 64
CONV2D_KERNAL_SIZE_2 = 3

CONV2D_FILTER_3 = 128
CONV2D_KERNAL_SIZE_3 = 3

DENSE_HIDDEN = 256

LEARNING_RATE = 1e-4

EPOCHS = 10

import tensorflow as tf

input = tf.keras.Input(shape=(28,28,1))
layer = tf.keras.layers.Conv2D(
    filters=CONV2D_FILTER_1,
    kernel_size=CONV2D_KERNAL_SIZE_1,
    activation="relu"
)(input)
layer = tf.keras.layers.MaxPool2D()(layer)
layer = tf.keras.layers.Conv2D(
    filters=CONV2D_FILTER_2,
    kernel_size=CONV2D_KERNAL_SIZE_2,
    activation="relu"
)(layer)
layer = tf.keras.layers.MaxPool2D()(layer)
layer = tf.keras.layers.Conv2D(
    filters=CONV2D_FILTER_3,
    kernel_size=CONV2D_KERNAL_SIZE_3,
    activation="relu"
)(layer)
layer = tf.keras.layers.MaxPool2D()(layer)
layer = tf.keras.layers.Flatten()(layer)
layer = tf.keras.layers.Dense(units=DENSE_HIDDEN, activation="relu")(layer)
outputs = tf.keras.layers.Dense(units=62, activation="softmax")(layer)

model = tf.keras.Model(inputs=input, outputs=outputs, name="Letter_Number_Classifier")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=[
        'accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc")
    ]
)


from huggingface_hub import snapshot_download
import glob, gzip
import numpy as np
from datetime import datetime

start = datetime.now().timestamp()

path = snapshot_download(repo_id="Heliosoph/EMNIST", repo_type="dataset")

def find_file(name: str):
    return glob.glob(f"{path}/**/{name}", recursive=True)[0]

def load_images(file_name):
    with gzip.open(find_file(file_name), "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28).transpose(0, 2, 1) / 255.0

def load_labels(file_name):
    with gzip.open(find_file(file_name), "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

x_train = load_images("emnist-byclass-train-images-idx3-ubyte.gz")
y_train = load_labels("emnist-byclass-train-labels-idx1-ubyte.gz")
x_test  = load_images("emnist-byclass-test-images-idx3-ubyte.gz")
y_test  = load_labels("emnist-byclass-test-labels-idx1-ubyte.gz")

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(5_000)
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
)

validation_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .shuffle(5_000)
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="./model_outputs/num_lett/best_model.keras",
    monitor="val_loss",
    save_best_only=True,
)

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

def format_duration(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"
print("Duration:", format_duration(datetime.now().timestamp()-start))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.show()