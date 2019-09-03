# Deep-Learning Models with TensorFlow & Keras
<p align="center">
<img src="assets/robo_dude.png" width="625"/>
</p>

## Basics of a Neural Network
Usually we feed a computer instructions to compute data in a predetermined way, and the output is therefore a function of that algorithm programmed as such. A neural network is the exact opposite. If we imagine a black box for representing the neural network, that box would contain the following:

<p align="center">
<img src="assets/neural_network.jpeg" width="625"/>
</p>

The input layer is where the pre-identified training data comes in and "teach" the neural network. The output layer is where the neural network gathers instructions to predict new data and classify it into previously identified as accurately as possible. This accuracy is dependent upon the size of the training data (the more you teach a model with new data, the better it gets at predicting) and the configuration of the hidden layers. The hidden layers are responsible for extracting features from training data which are then collated and compiled into a model.

> One hidden layer means you just have a neural network. Two or more hidden layers? Boom, you've got a deep neural network! Multiple hidden layers allows the network to create non-linear relationships between the input and the output.

## Create Training Data
I'll be using one of the most prevalent datasets for image recognition - Cats vs Dogs. Since the size of the dataset (or the pickled version of it) is quite large, I'll include the link for where to get it:
https://www.microsoft.com/en-us/download/details.aspx?id=54765&WT.mc_id=rss_alldownloads_devresources

'''python
DATADIR = "Where\You\Placed\Your\DataSet" # "C:\\SomeFolder\\PetImages"
CATEGORIES = ["Dog","Cat"]
IMG_SIZE = 100

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) # grayscale because colour isn't a differentiating factor
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
'''

> All training images have certain features that are helpful in differentiating between the given categories, and in order to only use those differentiating features in the hidden layers, we need to get rid of the the non-feature data from these images (for example - color and image size are components of the data but do not determine whether the image is of a cat or a dog).

## Customize model
```python
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
                      batch_size=50,                    # batch size depends on data size
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])
```


## Performance tracking using TensorBoard

## Testing the Selected Model


