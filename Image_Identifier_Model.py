import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import normalize
import pickle

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33) # can reduce gpu fraction less than 100% when running multiple models in parallel
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle","rb"))
Y = pickle.load(open("Y.pickle","rb"))

X = tf.keras.utils.normalize(X) # Normalize image data for quicker processing

model = Sequential()
# Layer 1
# Convolution is used to find cross-correlation between the filter and the image window
model.add(Conv2D(32,(3,3), input_shape = X.shape[1:])) # (Number of output filters, Size of convolution window, H x W of input irrelevant of colour)
model.add(Activation("relu"))
# After convolution, output size decreases but features identified increases number of output images, which requires sampling or pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
model.add(Conv2D(32,(3,3))) # Deeper convolution filter identifies more complex features of the input image
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 3
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 4
model.add(Flatten()) # The first Dense layer should be preceded by Flatten
model.add(Dense(64))
model.add(Activation("relu"))

# Layer 5
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",  # binary since data is dogs or cats
optimizer="adam",metrics=["accuracy"])

model.fit(X,Y,epochs=10, batch_size = 500) # batch size depends on data size