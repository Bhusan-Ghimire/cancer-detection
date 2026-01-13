import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.regularizers import l2
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# =========================
# 1. CONFIGURATION
# =========================
IMAGE_DIR = "./ISIC-images"
CSV_PATH = "./metadata.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5
RANDOM_STATE = 42

TOTAL_IMAGES = 800       # final dataset size
SKIP_IMAGES = 10000       # skip earlier ones reproducibly

# =========================
# 2. LOAD & ALIGN METADATA
# =========================
df = pd.read_csv(CSV_PATH)

df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df = df.iloc[SKIP_IMAGES:SKIP_IMAGES + TOTAL_IMAGES]

image_paths = [
    os.path.join(IMAGE_DIR, isic_id + ".jpg")
    for isic_id in df["isic_id"]
]

labels = df["melanocytic"].astype(int).values

print("Malignant ratio:", labels.mean())

# =========================
# 3. TF.DATA PIPELINE
# =========================
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
    img = tf.cast(img, tf.float32) / 255.0 # tensor to numpy array
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE) 
dataset = dataset.shuffle(512, seed=RANDOM_STATE)

# Split: 70% train / 15% val / 15% test
train_size = int(0.7 * TOTAL_IMAGES)
val_size = int(0.15 * TOTAL_IMAGES)

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size).take(val_size)
test_ds = dataset.skip(train_size + val_size)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)

# =========================
# 4. MODEL DEFINITION
# =========================

data_augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom((-0.3, 0.3))
]) #aters each training images in-place in each epoch, without changing the original train_ds dataset.

#model built from scratch
model_from_scratch = models.Sequential([
    tf.keras.Input(shape=(224,224,3)),

    layers.Conv2D(32, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, kernel_regularizer=l2(1e-3)),
    layers.LeakyReLU(negative_slope=0.1),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

#model built on top of transfre learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False # Freeze the pre-trained weights

transfer_model = models.Sequential([
    tf.keras.Input(shape=(224,224,3)),
    data_augmentation_layer,
    base_model,
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    # since the output of base model already has very less output shape, we need not do pooling.
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model = transfer_model

model.summary()

# =========================
# 5. COMPILE MODEL
# =========================
model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

# Handle class imbalance
weights = compute_class_weight(
    class_weight= "balanced",
    classes= np.array([0,1]),
    y= labels
)
priority_factor = 1.2 #never exceed penalizing factor more than 2
class_weight = {
    0: weights[0], 
    1: weights[1] * priority_factor
}

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# =========================
# 6. TRAIN
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stopping]
)

# =========================
# 7. EVALUATION
# =========================
y_true = []
y_pred = []

for x, y in test_ds:
    preds = model.predict(x)
    y_true.extend(y.numpy())
    y_pred.extend((preds >= 0.50).astype(int).flatten())

ConfusionMatrixDisplay.from_predictions(
    y_true=y_true,
    y_pred=y_pred,
    display_labels=["Normal", "Cancerous"]
)

plt.title("Confusion Matrix - Cancer detection")
plt.show()

# =========================
# 8. SINGLE IMAGE INFERENCE
# =========================
def predict_single_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]
    label = "Cancerous" if prob >= 0.5 else "Normal"

    print(f"Prediction: {label}")
    print(f"Confidence: {prob:.4f}")

# Example inference
sample_image = image_paths[0]
predict_single_image(sample_image)
