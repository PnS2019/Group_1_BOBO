"""
PokemonRecognizer V1.0
Dejan Bozin
Florian Bolli

Executable on Rasperry Pi 3
run with:
>> python3 pokemonRacognizer.py
Hold camera against a fluffy pokemon
>> (No command, just hit enter)
Wait 1 second

And the Raspberry Pi will tell you something about the pokemon on the picture.


This script can not only recognize pokemon, but also train the model with train data.
If you want to train it again, set TRAIN = True and chose a MODEL_NAME.
For a training, the classes have to be in a folder relative to this dircetion:
/Data/TrainSmall/{NAME_OF_CLASS}/
And the test data should be:
/Data/TestSmall/{NAME_OF_CLASS}/
In our case we have 11 Classes:
Bulbasaur, Charmander, Dratini, Eevee, Jigglypuff, Jolteon, Meowth, Pichu, Pikatchi, Squirtle, Vaporeon
If Train is set to False, the Network will load the weights saved in the file declared in MODEL_NAME
"""

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from pnslib import utils
from PIL import Image
from skimage import transform

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import webbrowser
import pygame


from skimage import transform
import time
#Should the network be trained again?
TRAIN = False

#Should the camera output be shown on desktop version?
DESKTOP = True

#Name of the Weight file to read or write (Depending on TRAIN)
MODEL_NAME = "training_10Epochs.h5"

#11 Real classes, Flaeron was Just an experiment how the network deals with very little train data... Didnt work, as expected
num_classes = 12



# dimensions of our images.
img_width, img_height = 75, 100

train_data_dir = 'Data/TrainSmall'
validation_data_dir = 'Data/TestSmall'
nb_train_samples = 15000
nb_validation_samples = 800
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (750, 1000)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(750, 1000))


# define a model
num_train_samples = 15000
num_test_samples = 50
input_shape = (75,100,3)

kernel_sizes = [(7, 7), (5, 5)]
num_kernels = [20, 25]

pool_sizes = [(2, 2), (2, 2)]
pool_strides = [(2, 2), (2, 2)]

num_hidden_units = 200

x = Input(shape=input_shape)
y = Conv2D(num_kernels[0], kernel_sizes[0], activation='relu')(x)
y = MaxPooling2D(pool_sizes[0], pool_strides[0])(y)
y = Conv2D(num_kernels[1], kernel_sizes[1], activation='relu')(y)
y = MaxPooling2D(pool_sizes[1], pool_strides[1])(y)
y = Flatten()(y)
y = Dense(num_hidden_units, activation='relu')(y)
y = Dense(num_classes, activation='softmax')(y)
model = Model(x, y)

print("[MESSAGE] Model is defined.")

# print model summary
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer="sgd",
              metrics=['accuracy'])

# Image generator for generating more train Data
# And rescale the colors to a float between 0 and 1 instead of in between o to 255
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Image generator for test data
# Testdata has to be treated the same way as the train data => Also rescale
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,)



#Train the network if TRAIN = true (Takes very much time)
if(TRAIN):

    model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

    model.save_weights(MODEL_NAME)

#if TRAIN = false, just load the training data from file
else:
    model.load_weights(MODEL_NAME)


#Dealing with test data
data = []
labels = []
i = 0
for d, l in validation_generator:
    data.append(d)
    labels.append(l)
    i += 1
    if i == 1:
        break


test_data = data[0]
test_label = labels[0]

#Predict test data for getting a feeling about the accuracy
pred=model.predict(test_data)
#Get the hights neuron in the outputlayer with argmax
predicted_class_indices=np.argmax(pred,axis=1)
predicted_class_ground_truth=np.argmax(test_label,axis=1)


#Dealing with the write labels for the classes
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print("LABELS")
print(labels)

predictions = [labels[k] for k in predicted_class_indices]
truths = [labels[k] for k in predicted_class_ground_truth]

print("PREDICTIONS: ")
print(predictions)
print("TRUTHS: ")
print(truths)

print("...")
print("...")
print("loaded, ro recognice a pokemon, press Enter, to exit, type 'exit'")
#Programm now ready for single recognition


#makes a picture and saves it as snapshot.jpg
def snap():
    print('You made a snap')
    camera.capture('snapshot.jpg')


#Plays an mp3 file in the Voice/ folder with the name "prediction"
def tellPrediction(prediction):
    print("PREDICTION: ")
    print(prediction)

    pygame.mixer.init()
    pygame.mixer.music.set_volume(0.2)
    pygame.mixer.music.load("Voice/"+prediction +".mp3")
    pygame.mixer.music.play()

#Shows the image called snapshot.png
def showImage():
    img = Image.open('snapshot.png')
    img.show()


def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (75, 100, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

if(DESKTOP):
    camera.start_preview()
    time.sleep(2)
    camera.preview.window=(200,200,256,192)
    time.sleep(2)
    camera.preview.window=(0,0,512,384)


#MAIN_LOOP
#Wait for inputs and recognize pokemon taken by the camera
while True:

    inputStr = input()

    #Empty command => Try to recognize
    if(inputStr == ""):
        snap()
        testimage = load('snapshot.jpg')
        pred = model.predict(testimage)
        print(pred)
        predicted_class_index=np.argmax(pred,axis=1)
        print("ARGMAx")
        print(predicted_class_index)
        index = predicted_class_index[0]
        prediction = labels[index]


        print("PREDICTION: ")
        print(prediction)
        #Sound output
        tellPrediction(prediction)

    #Some settings...
    if(inputStr == "desktop"):
        camera.start_preview()
    if(inputStr == "ssh"):
        camera.stop_preview()

    if(inputStr == "exit"):
        break
