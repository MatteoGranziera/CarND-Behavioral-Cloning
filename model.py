import random
import pandas as pd
import numpy as np
import cv2
import sklearn
import tensorflow as tf
from PIL import Image

from tqdm import tqdm

from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D
from keras.callbacks import EarlyStopping
# from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

# Hyperparameters
EPHOCS=10
BATCH_SIZE = 128
STEERING_FACTOR = 1
STEERING_CORRECTION = -0.2
ACTIVATION = 'relu'

# Preprocess
MODEL_NAME = 'model.h5'
GENERATE_AUGMENTED = True
CROP_TOP = 70
CROP_BOTTOM = 25
CROP_LEFT = 5
CROP_RIGHT = 5
STEERING_MIN = 0.1
STEERING_FILTER_PERC = 0.2

# Extra
CONTINUE_MODEL = False

# Config
data_paths = [
    '../data/track1_lap1/',
    '../data/track1_lap2/',
    '../data/track1_lap3_r/',
    '../data/recovery/',
    '../data/corrections/',
    '../data/corrections/',
#     '../data/track1_lap_slow/',
    '../data/smooth/', 
#     '../data/straight/',
#     '../data/straight/',
#   '../data/track2_lap1/',
#   '../data/track2_lap2/',
#   '../data/recovery_track2/',
]


# Enable memory grow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                
                # Load image
                center_image = Image.open(batch_sample[0])
                center_image = center_image.convert('YCbCr')
                center_image = np.asarray(center_image)
                
                center_angle = float(batch_sample[3])
                        
                if GENERATE_AUGMENTED == True:
                    # Get augmentation type from last column
                    augmentation_type = batch_sample[7]
                    
                    # Flipped image
                    if augmentation_type == 1:
                        center_image = np.fliplr(center_image) * STEERING_FACTOR + STEERING_CORRECTION
                        center_angle = float(-center_angle)
                
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
         
# Load logs
df_logs = pd.DataFrame()

for i in range(len(data_paths)):

    images_path = data_paths[i] + 'IMG/'
    
    df = pd.read_csv(data_paths[i] + 'driving_log.csv', 
#                      nrows=64,
                     header=None,
                     names=['center_image', 'left_image', 'center_image', 'steering', 'throttle', 'break', 'speed'], 
                     dtype={'center_image':str, 
                            'left_image':str,
                            'center_image':str, 
                            'steering':float, 
                            'throttle':float,
                            'break':float,
                            'speed':float })


    df = df[(abs(df['steering']) > STEERING_MIN) | (random.random() > STEERING_FILTER_PERC) ]
    
    # Replace path with the correct
    df.iloc[:, 0] = df.iloc[:,0].apply(lambda p: images_path + p.split('/')[-1])
    df.iloc[:, 1] = df.iloc[:,1].apply(lambda p: images_path + p.split('/')[-1])
    df.iloc[:, 2] = df.iloc[:,2].apply(lambda p: images_path + p.split('/')[-1])
    
    df_logs = df_logs.append(df)

# Add augmented data
if GENERATE_AUGMENTED == True:
    print("Add augmented rows...")
    
    # Create a copy for each augmentation
    df_flip_logs = df_logs.copy()
    
    # Add column augmentation 0 for original images
    df_logs['augmentation'] = 0
    df_flip_logs['augmentation'] = 1

    # Append all rows
    df_logs = df_logs.append(df_flip_logs)

# Get numpy array
logs = df_logs.values

print()
print()
print("####### Configuration ######")
print()
print("Shape: ", logs.shape)
print("Continue training: ", str(CONTINUE_MODEL))
print("Generate augmented: ", str(GENERATE_AUGMENTED))
print("Model name: ", str(MODEL_NAME))
print("Batch size: ", str(BATCH_SIZE))
print()
print("####### Data ######")
print()
print("First row: ")
print(logs[0])
print()

input("Press Enter to start training...")

# Split in train and validation sets
train_samples, validation_samples = train_test_split(logs, test_size=0.2)

# Create generator for train and validation sets
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

def build_model():
    # BUILD MODEL #
    model = Sequential()
        
    # Crop image
    model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (CROP_LEFT,CROP_RIGHT)), input_shape=(160,320,3)))
    
    # Normalization range -0.5 <-> 0.5
    model.add(Lambda(lambda x: x / 255. - 0.5))

    model.add(Convolution2D(24,(5,5), strides=(2, 2), activation=ACTIVATION))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(36,(5,5), strides=(2, 2), activation=ACTIVATION))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(48,(5,5), strides=(2, 2), activation=ACTIVATION))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(64,(3,3), activation=ACTIVATION))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(64,(3,3), activation=ACTIVATION))
    
    model.add(Flatten())
    
    model.add(Dropout(0.2))
    model.add(Dense(1100, activation=ACTIVATION))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation=ACTIVATION))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation=ACTIVATION))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=ACTIVATION))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    return model

if(CONTINUE_MODEL == True):
    model = load_model(MODEL_NAME)
else:
    model = build_model()
    
    # Compile
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=1.e-4,
                           patience=2,
                           mode='min')
# Run training
model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_samples)/BATCH_SIZE), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(len(validation_samples)/BATCH_SIZE), 
            epochs=EPHOCS, 
            verbose=1,
            callbacks=[early_stop])

model.save(MODEL_NAME)