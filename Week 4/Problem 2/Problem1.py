import os
import datetime
import numpy

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorboard import notebook

import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

# Double checking versions
print(sklearn.__version__)
print(tf.__version__)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Evaluates Tensorflow DCNN models
# ---------------------------------------------------------------------------------------------------------------------------------------
def eval_model(model, x_test, y_test):
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred = model.predict(x_test);
    indexes = tf.argmax(pred, axis=1)
    print(indexes)
    i = tf.cast([], tf.int32)
    indexes = tf.gather_nd(indexes, i)
    print(indexes)
    
    cm = confusion_matrix(y_test, indexes)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)
    print(tf.executing_eagerly())



# ---------------------------------------------------------------------------------------------------------------------------------------
# Dense Model
# ---------------------------------------------------------------------------------------------------------------------------------------
def fully_connected_model():
    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Loading fashion mnist dataset
    # ---------------------------------------------------------------------------------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    fig = plt.figure(figsize=[10, 10])
    for i in range(100):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.imshow(x_train[i,:])
        
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255


    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Training the model
    # ---------------------------------------------------------------------------------------------------------------------------------------
    # create an input, we need to specify the shape of the input, in this case it's a vectorised images with a 784 in length
    inputs = keras.Input(shape=(784,), name='img')
    # first layer, a dense layer with 64 units, and a relu activation. This layer recieves the 'inputs' layer as it's input
    x = layers.Dense(256, activation='relu')(inputs)
    # second layer, another dense layer, this layer recieves the output of the previous layer, 'x', as it's input
    x = layers.Dense(64, activation='relu')(x)
    # output layer, length 10 units. This layer recieves the output of the previous layer, 'x', as it's input
    outputs = layers.Dense(10, activation='softmax')(x)

    # create the model, the model is a collection of inputs and outputs, in our case there is one of each
    model = keras.Model(inputs=inputs, outputs=outputs, name='fashion_mnist_model')
    # print a summary of the model
    model.summary()
    # plot the shape of the model - saved as model.png
    # keras.utils.plot_model(model, show_shapes=True)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=keras.optimizers.RMSprop(),
                metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=20,
                        validation_split=0.2)

    eval_model(model, x_test, y_test) 
    

def CNN_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255
    y_train = y_train.reshape(y_train.shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255
    y_test = y_test.reshape(y_test.shape[0], 1)

    # our input now has a different shape, 28x28x1, as we have 28x28 single channel images
    inputs = keras.Input(shape=(28, 28, 1, ), name='img')
    # rather than use a fully connected layer, we'll use 2D convolutional layers, 8 filters, 3x3 size kernels
    x = layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu')(inputs)
    # 2x2 max pooling, this will downsample the image by a factor of two
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    # more convolution, 16 filters, followed by max poool
    x = layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    # final convolution, 32 filters
    x = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
    # a flatten layer. Matlab does a flatten automatically, here we need to explicitly do this. Basically we're telling
    # keras to make the current network state into a 1D shape so we can pass it into a fully connected layer
    x = layers.Flatten()(x)
    # a single fully connected layer, 64 inputs
    x = layers.Dense(64, activation='relu')(x)
    # and now our output, same as last time
    outputs = layers.Dense(10, activation='softmax')(x)

    # build the model, and print the summary
    model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='fashion_mnist_cnn_model')
    model_cnn.summary()

    keras.utils.plot_model(model_cnn, show_shapes=True)

    model_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    optimizer=keras.optimizers.RMSprop(),
                    metrics=['accuracy'])
    history = model_cnn.fit(x_train, y_train,
                            batch_size=64,
                            epochs=20,
                            validation_split=0.2)

    eval_model(model_cnn, x_test, y_test)

def advanced_CNN_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255
    y_train = y_train.reshape(y_train.shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255
    y_test = y_test.reshape(y_test.shape[0], 1)

    

    # our model, input again, still in an image shape
    inputs = keras.Input(shape=(28, 28, 1, ), name='img')
    # run pairs of conv layers, all 3s3 kernels
    x = layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(x)
    # batch normalisation, before the non-linearity
    x = layers.BatchNormalization()(x)
    # spatial dropout, this will drop whole kernels, i.e. 20% of our 3x3 filters will be dropped out rather
    # than dropping out 20% of the invidual pixels
    x = layers.SpatialDropout2D(0.2)(x)
    # max pooling, 2x2, which will downsample the image
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    # rinse and repeat with 2D convs, batch norm, dropout and max pool
    x = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    # final conv2d, batch norm and spatial dropout
    x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.2)(x)
    # flatten layer
    x = layers.Flatten()(x)
    # we'll use a couple of dense layers here, mainly so that we can show what another dropout layer looks like 
    # in the middle
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    # the output
    outputs = layers.Dense(10, activation=None)(x)

    # build the model, and print a summary
    model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='fashion_mnist_cnn_model')
    model_cnn.summary()

    keras.utils.plot_model(model_cnn, show_shapes=True)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=keras.optimizers.RMSprop(),
                metrics=['accuracy'])
    history = model_cnn.fit(x_train, y_train,
                            batch_size=64,
                            epochs=20,
                            validation_split=0.2,
                            callbacks=[tensorboard_callback])

    eval_model(model_cnn, x_test, y_test)


# dense_model()
# CNN_model()
advanced_CNN_model()
plt.show()