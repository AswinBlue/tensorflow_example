import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAveragePooling2D


class mnistModule:

    def __init__(self):
        pass

    def get_sequential_model(self):
        return tensorflow.keras.Sequential(
            [
                # you can choose whatever shape of model you want.
                # but this model is verified optimized for classification
                Input(shape=(28, 28, 1)), # image shape is 28 by 28 size, and 1 channel (gray scale only)
                # 2d convolution layer acquire 4-dimension data as input
                Conv2D(32, (3,3), activation='relu'),  # 32 filter, each filter is 3 by 3, activation function is relu (rectivied linear unit)
                Conv2D(62, (3,3), activation='relu'),
                MaxPool2D(),
                BatchNormalization(),   # normalizing input save functional costs, so we normalize output of each layers

                # by normalizing, gradient moves faster towards the global minimum of the cost function
                
                Conv2D(128, (3,3), activation='relu'),
                MaxPool2D(),
                BatchNormalization(),

                GlobalAveragePooling2D(), # get average according to axes
                Dense(64, activation='relu'), # vector containing values. no filtered layer
                Dense(10, activation='softmax') # final output. 10 different classes having probability
                # end of neural network architecture

                # now we have images as input, 10 classes as output
            ]
        )

    def run(self):
        (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

        print("x_train.shape = ", x_train.shape)
        print("y_train.shape = ", y_train.shape)
        print("x_test.shape = ", x_test.shape)
        print("y_test.shape = ", y_test.shape)

        # self.display_examples(x_train, y_train)
        x_train = x_train.astype('float32') / 255 # 0 ~ 255 (color)
        x_test = x_test.astype('float32') / 255

        # change 2 dimentions(28,28) to 3 dimentions(28,28,1)
        x_train = np.expand_dims(x_train, axis=-1) # add 1 dimention at the end of the 'x_train'
        x_test = np.expand_dims(x_test, axis=-1)

        model = self.get_sequential_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        # optimizer : algorithm to optimize cost function (finding global minimum)
        # loss : loss function, penalize neural network weights when they predict the wrong thing
        # metrics : how to guide training. indicate our model becoming better or worse

        # MODEL TRAINING
        model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
        # (1) images, (2) labels corresponding to those images
        # batch_size : how many images our model going to see
        # epochs : the number of times the model is going to see training data
        # validation_split : save some input data to validate our model. that data will not be used in training

        # EVALUATION
        model.evaluate(x_test, y_test, batch_size=64)

    def display_examples(self, examples, labels):
        plt.figure(figsize=(10,10))

        for i in range(25):
            idx = np.random.randint(0, examples.shape[0]-1) # choose random number between 0 ~ MAX_INDEX_OF_examples
            img = examples[idx]
            label = labels[idx]

            plt.subplot(5, 5, i + 1) # 5 by 5 plot, current index is 'i+1' (must be 1 ~ 25)
            plt.title(str(label)) # set title of current subplot
            plt.tight_layout() # set layout margin more big
            plt.imshow(img, cmap='gray') # set image of current subplot as 'gray' version

        plt.show() # show plt we made
