import matplotlib.pyplot
matplotlib.use('Agg')

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os


dataset = r"F:\Data\Sports Dataset"
moddel = r"F:\Data\Video Classification\activity.model"
label = r"F:\Data\Video Classification\lb.pickle"
plot = r"F:\Data\Video Classification/"

LABELS = set(["Boxing","Hockey","Chess","Cricket","Fencing"])

imagePaths = list(paths.list_images(dataset))
data = []
labels = []
print(imagePaths)
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    if label not in LABELS:
        continue

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    data.append(image)
    labels.append(label)



data = np.array(data)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)


trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range =0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean

valAug.mean = mean



headmodel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headmodel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

model = headmodel.output
model = AveragePooling2D(pool_size=(2, 2))(model)
model = Flatten(name="flatten")(model)
model = Dense(512, activation="relu")(model)
model = Dropout(0.5)(model)
model = Dense(len(lb.classes_), activation="softmax")(model)

moodel = Model(inputs=headmodel.input, outputs=model)

for layer in headmodel.layers:
    layer.trainable = False

opt = SGD(lr=1e-4, momentum=0.85, decay=1e-4 / 5)
moodel.compile(loss="categorical_crossentropy", optimizer=opt,
               metrics=["accuracy"])

H = moodel.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=32),
    steps_per_epoch=len(trainX) // 32,
    validation_data=valAug.flow(testX, testY),
    validation_steps=len(testX) // 32,
    epochs=5)


predictions = moodel.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))


N = 5
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="test_loss")
#plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="test_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot")


moodel.save(moddel)

f = open("label", "wb")
f.write(pickle.dumps(lb))
f.close()
