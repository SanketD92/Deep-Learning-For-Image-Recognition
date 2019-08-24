# https://www.youtube.com/watch?v=wQ8BIBpya2k

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 28*28 images of hand-written digits 0-9
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test,y_test) = mnist.load_data()

# Normalize image bits 0-255 to 0-1 for quicker input processing
x_train = tf.keras.utils.normalize(x_train,axis=1)  
x_test = tf.keras.utils.normalize(x_test,axis=1)

# Model Architecture - Adding neuron layers and output layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) # relu = rectified linear
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) #classification layer for output - softmax for probability

# Training the model
model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy', 
metrics =['accuracy']) # There are different types of optimizers/loss/metrics that can be used
model.fit(x_train,y_train,epochs=3) # Epoch = 1 forward pass and one backward pass of all the training examples. batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc) # Lower the loss, better the model is learning

# Saving and loading a model
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

# Test model
predictions = new_model.predict([x_test])
plt.imshow(x_test[0])
plt.show()
np.argmax(predictions[0])