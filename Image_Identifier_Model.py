from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0     # Normalize image data for quicker processing

dense_layers = [0, 1, 2, 3]
layer_sizes = [32, 64, 128, 256, 512]       # These don't need to be in 2's power ranges, just used as an example
conv_layers = [1, 2, 3, 4, 5]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()
            
            # Layer 1
            # Convolution is used to find cross-correlation between the filter and the image window
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            # After convolution, output size decreases but features identified increases number of output images, which requires sampling or pooling
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # Additional Layers based on conv_layer loop
            # Deeper convolution filter identifies more complex features of the input image
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # The first Dense layer should be preceded by Flatten
            model.add(Flatten())

            # Dense Layers based on dense_layer loop
            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',   # binary since data is dogs or cats
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, y,
                      batch_size=32,                    # batch size depends on data size
                      epochs=20,
                      validation_split=0.3,
                      callbacks=[tensorboard])
