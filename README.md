# Deep-Learning Models with TensorFlow & Keras
## Create Training Data

## Customize model

## Performance tracking using TensorBoard

## Testing the Selected Model



Input Sinogram             |  Output should be close to
:-------------------------:|:-------------------------:
![](/Assets/Phantom_Sinogram.jpg)  |  ![](Assets/Phantom.png)

<p align="center">
<img src="Assets/Moment_zero.jpg" width="425"/>
</p>


>The striking result of it being a flat line is because no matter which angle the projection is being taken from, the sum of the attenuation intensity will be constant since the object features are static.

Filtered Sinogram             |  Filtered vs Original
:-------------------------:|:-------------------------:
![](/Assets/Filtered_Sinogram.jpg)  |  ![](Assets/Filtered_vs_Original_Sinogram_45deg.png)

<p align="center">
<img src="Assets/Filtered_Backprojected.jpg" width="425"/>
</p>

> As compared to our simple back-projected image, we see that the Ram-Lak filter has been able to remove low frequency noise (haze), improve contrast, thereby improving the total signal-to-noise ratio. The resolution also seems to have increased but mainly due to the increased sharpness and improved contrast.


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

            tensorboard = TensorBoard(log_dir="logs2/{}".format(NAME))

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

