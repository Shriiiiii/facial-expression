import os
import zipfile
import requests
import io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === STEP 1: Automatically download and extract dataset ===
DOWNLOAD_URL = 'https://github.com/ShridharG7/datasets/releases/download/v1.0/fer2013.zip'  # Replace with your actual ZIP URL

if not os.path.exists("fer2013/train") or not os.path.exists("fer2013/test"):
    print("Downloading FER2013 dataset...")
    response = requests.get(DOWNLOAD_URL)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("fer2013")
    print("Dataset downloaded and extracted.")
else:
    print("FER2013 dataset already exists.")

# === STEP 2: Data generators ===
train_path = 'fer2013/train'
test_path = 'fer2013/test'

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_gen = datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64,
    subset='validation'
)

test_gen = datagen.flow_from_directory(
    test_path,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64,
    shuffle=False
)

# === STEP 3: Build and train model ===
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[early_stop, checkpoint]
)

# === STEP 4: Evaluation and Visualization ===
loss, acc = model.evaluate(test_gen)
print(f'Test Accuracy: {acc * 100:.2f}%')

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# === STEP 5: Predict Emotion Function ===
def predict_emotion(image_path):
    img = load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img = img_to_array(img).reshape(1, 48, 48, 1) / 255.0
    pred = model.predict(img)
    classes = list(train_gen.class_indices.keys())
    print("Predicted Emotion:", classes[np.argmax(pred)])

# === STEP 6: Take input image and predict ===
img_path = input("Enter the path to any image to predict emotion: ")
predict_emotion(img_path)
