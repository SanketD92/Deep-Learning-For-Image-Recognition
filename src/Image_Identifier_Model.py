import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0     # Normalize image data for quicker processing

model = Sequential()

# Layer 1
# Convolution is used to find cross-correlation between the filter and the image window
model.add(Conv2D(128, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
# After convolution, output size decreases but features identified increases number of output images, which requires sampling or pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Additional Layers based on conv_layer loop
# Deeper convolution filter identifies more complex features of the input image

# Layer 2
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# The first Dense layer should be preceded by Flatten
model.add(Flatten())

# Dense Layers based on dense_layer loop
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',   # binary since data is dogs or cats
                optimizer='adam',
                metrics=['accuracy'],
                )

model.fit(X, y,
            batch_size=50,                    # batch size depends on data size
            epochs=10,
            validation_split=0.3)

model.summary()

print(mode.summary())

# Test
model.predict()