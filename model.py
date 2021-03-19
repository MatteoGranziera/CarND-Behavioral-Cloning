import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

X_train = np.zeros((1, 160, 320, 3), dtype=np.int32)
y_train = np.zeros((1, 1), dtype=np.int32)

model = Sequential()

model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train)

model.save('model.h5')