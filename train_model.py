import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
import os
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Dense, MaxPooling2D, Conv2D, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# تحديد مسار البيانات
DATASET_FOLDER = './input/CK+48'

if not os.path.exists(DATASET_FOLDER):
    raise FileNotFoundError(f"Dataset folder '{DATASET_FOLDER}' not found.")

sub_folders = sorted(os.listdir(DATASET_FOLDER))  # التأكد من ترتيب المجلدات لتجنب التغييرات العشوائية
images = []
labels = []

# قراءة الصور والتسميات
for sub_folder in sub_folders:
    path = os.path.join(DATASET_FOLDER, sub_folder)
    if not os.path.isdir(path):
        continue

    sub_folder_images = os.listdir(path)
    label = sub_folders.index(sub_folder)

    # تصنيف المشاعر
    if label in [4, 6]:
        new_label = 0  # إيجابي
    elif label in [0, 5]:
        new_label = 1  # سلبي
    else:
        new_label = 2  # محايد

    for image in sub_folder_images:
        image_path = os.path.join(path, image)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        img = cv2.resize(img, (48, 48))
        images.append(img)
        labels.append(new_label)

# تحويل البيانات إلى مصفوفات NumPy
images_x = np.array(images, dtype=np.float32) / 255.0
labels_y = np.array(labels)

# تحويل التصنيفات إلى ترميز فئة واحدة
num_of_classes = 3
labels_y_encoded = tf.keras.utils.to_categorical(labels_y, num_classes=num_of_classes)

# تقسيم البيانات
X_train, X_test, Y_train, Y_test = train_test_split(images_x, labels_y_encoded, test_size=0.25, random_state=10)

# تحويل البيانات إلى الشكل المناسب لشبكة CNN
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# بناء النموذج
input_layer = Input(shape=(48, 48, 1))

conv1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.001), activation='relu')(input_layer)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001), activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001), activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001), activation='relu')(pool3)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

flatten = Flatten()(pool4)
dense_1 = Dense(128, activation='relu')(flatten)
drop_1 = Dropout(0.2)(dense_1)
output_layer = Dense(num_of_classes, activation='softmax')(drop_1)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# تحديد نقطة الحفظ
MODEL_PATH = './output/emotion_model.h5'
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

checkpointer = ModelCheckpoint(MODEL_PATH, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False)

# تدريب النموذج
history = model.fit(X_train, Y_train, batch_size=32, validation_data=(X_test, Y_test), epochs=50,
                    callbacks=[checkpointer])

# رسم النتائج
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].plot(history.history['loss'], label='Train Loss', color='blue', marker='o')
ax[0].plot(history.history['val_loss'], label='Test Loss', color='red', marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(history.history['accuracy'], label='Train Accuracy', color='blue', marker='o')
ax[1].plot(history.history['val_accuracy'], label='Test Accuracy', color='red', marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.show()