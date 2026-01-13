import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# 1. PARAMETERS
IMAGE_DIR = './ISIC-images'
initial_gap = 6000
N_IMAGES = initial_gap+300  # Number of images to extract for training
IMG_SIZE = (224, 224)
RANDOM_STATE = 42

df = pd.read_csv("./metadata.csv")

# 2. PREPROCESSING FUNCTION (Preserves Aspect Ratio)
def preprocess_image(image_path):
    # Load raw image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Resize with padding to 224x224 (prevents "squishing")
    img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
    
    # Normalize to [0, 1]
    img = img / 255.0
    return img

# 3. EXTRACT AND PROCESS N IMAGES
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][:N_IMAGES]
X = []
for filename in image_files:
    path = os.path.join(IMAGE_DIR, filename)
    X.append(preprocess_image(path))
X = np.array(X)
X = X[initial_gap:]

output_labels = df["melanocytic"]
y = output_labels.astype(int)[:N_IMAGES]
y = y[initial_gap:]
#note: 68% of examples in dataset are malignant
#note: X[a][b] -> a_th image, b_th row from top --> X[a][b] contains array of pixels for b_th row of image 

X_train, X_, y_train, y_ = train_test_split(X,y,random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_,y_, test_size=0.5, random_state=RANDOM_STATE, stratify=y_)
del X_, y_

model = models.Sequential([
    tf.keras.Input(shape=(224,224,3)),
    layers.Conv2D(32, kernel_size=(3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(32, kernel_size=(3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2,2), strides=2),

    layers.Conv2D(64, kernel_size=(3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(64, kernel_size=(3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2,2), strides=2),

    # layers.Conv2D(128, kernel_size=(3,3), padding='same'),
    # layers.BatchNormalization(),
    # layers.Activation("relu"),
    # layers.Conv2D(128, kernel_size=(3,3), padding='same'),
    # layers.BatchNormalization(),
    # layers.Activation("relu"),
    # layers.MaxPooling2D(pool_size=(2,2), strides=2),

    # layers.Conv2D(256, kernel_size=(3,3), padding='same'),
    # layers.BatchNormalization(),
    # layers.Activation("relu"),
    # layers.MaxPooling2D(pool_size=(2,2), strides=2),

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.008)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

print(model.summary())


print("Malignant class ratio: ",y.sum()/len(y))

model.compile(
    loss = BinaryCrossentropy(),
    optimizer =Adam(learning_rate=5e-4)
)
earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 10,
    verbose=True,
    mode="auto",
    restore_best_weights = True
)
# 5. TRAIN THE MODEL
# print("\nStarting Training...")
# model.fit(X_train, y_train, epochs=5, batch_size=8, callbacks=[earlystopping], validation_data=(X_val, y_val))

# y_pred = model.predict(X_test) 
# # y_pred = (y_pred>=0.5).astype(int)

# print(y_pred)

# ConfusionMatrixDisplay.from_predictions(
#     y_true=y_val,
#     y_pred=y_pred,
#     labels=["Malignant", "Benign"]
# )

# plt.show()

# # 6. INFERENCE ON A SAMPLE IMAGE
# # Let's take the first image again as a test
# sample_path = os.path.join(IMAGE_DIR, image_files[0])
# sample_img = preprocess_image(sample_path)
# sample_img = np.expand_dims(sample_img, axis=0)  # Add batch dimension: (1, 224, 224, 3)

# prediction = model.predict(sample_img)
# label = "Malignant" if prediction[0][0] > 0.5 else "Benign"

# print(f"\nInference for {image_files[0]}:")
# print(f"Raw Output (Sigmoid): {prediction[0][0]:.4f}")
# print(f"Predicted Class: {label}")



'''
make the image and labels selection process random
try different ralues of reg. parameter
'''