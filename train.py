from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.python.ops.gen_array_ops import fill

from tensorflow.python.ops.gen_math_ops import Imag


# initialize learning rate, number of epochs, and batch size (for training)
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# directory of dataset
DIR = r'dataset'
CATEGORIES = ['with_mask', 'without_mask']

# create a list of images using the dataset
print("[INFO] Loading images...")

# array of tuples, tuple has image array and with/without mask label
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIR, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# perform one-hot encoding on labels array
# https://en.wikipedia.org/wiki/One-hot
# a one-hot is a group of bits among which the legal combinations
# of values are only those with a single high (1) bit and all the others low (0).

# one-hot encoding ensures that machine learning does not assume that
# higher numbers are more important.

# with mask would be converted to [1. 0.]
# without mask would be converted to [0. 1.]
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# convert both data & labels arrays to numpy arrays
data = np.array(data, dtype='float32')
labels = np.array(labels)

# split arrays into random train and test subsets
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=25)

# generate more images using image data augmentation via ImageDataGenerator
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# create base model using pretrained model for images
# include top false because we'll add the FC layer later
baseModel = MobileNetV2(weights='imagenet', include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

# place head FC model on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# freeze layers in base model so they won't be updated
for layer in baseModel.layers:
    layer.trainable = False

# compile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the head
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

# show classification report
print(classification_report(testY.argmax(axis=1),
      predIdxs, target_names=lb.classes_))

# save model
print("[INFO] saving model...")
model.save("mask_detection.model", save_format='h5')

# plot training & loss accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_acc')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc='lower left')
plt.savefig('plot.png')
