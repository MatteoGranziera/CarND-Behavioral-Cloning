import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

data_path = '../sim_data/run1/'

logs = pd.read_csv(data_path + 'driving_log.csv')

logs = logs.values

images = []
measurements = []

images_path = data_path + 'IMG/'

for i in tqdm(range(len(logs))):
    line = logs[i]
    center_path = line[0]
    center_path = images_path + center_path.split('/')[-1]
    center_img = cv2.imread(center_path)
    images.append(center_img)
    
    measure = float(line[3])
    measurements.append(measure)
    
X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()

# Normalization
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))

model.add(Convolution2D(6,(5,5), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,(5,5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')