'''
Convolutional neural network for mnist fashion dataset with a custom callback.
'''
import tensorflow as tf
from myCallback import myCallback

# Import the mnist fashion dataset.
# Data are 28x28 pixel vales of 0 -> 255, labels are 0-9 (indicating type of clothing)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize input data to a scale of 0 -> 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Example of layers added outside of initial model creation
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1))) # Feed single 28x28 image into 32 convolutional kernels of 3x3 size. Output is 32 26*26 images.
model.add(tf.keras.layers.MaxPooling2D(2, 2)) # Perform max pooling over 2x2 subareas in each of the 32 26x26 images. Output is 32 13x13 images.
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu')) # Feed each of the 32 13x13 images into 32 convolutional kernels of 3x3 size. Output is 32 11x11 images.
model.add(tf.keras.layers.MaxPooling2D(2,2)) # Perform max pooling over 2x2 subareas in each of the 32 11x11 images. Output is 32 5x5 images.
model.add(tf.keras.layers.Flatten()) # Input shape is not necessary...will provide 5*5*32 = 800 nodes.
model.add(tf.keras.layers.Dense(128, activation='relu')) # 128-node hidden layer with a rectified linear activation. if x > 0: return x else return 0
model.add(tf.keras.layers.Dense(10, activation='softmax')) # Output layer for values 0-9, logits turned into probabilities. Reminder: softmax is exp(x) / reduce_sum(exp(x))

# Call model.summary() to get a nice summary of layers and associated outputs.
model.summary()


# Compile model and specify optimizer to be 'adam', a stochastic gradient descent method that is based on
# adaptive estimation of first-order and second-order moments.
# Loss function will be given by SCCE
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy' , metrics=['accuracy']) # Note: can pass in string of optimizer instance (e.g. tf.optimizers.Adam())

# Instantiate a custom callback that stops training for a certain accuracy and prints a message when training ends.
cb = myCallback()
# Start training the model with custom callback instance
model.fit(x_train, y_train, epochs = 5, callbacks = [cb])

# Evaluate model on validation data.
# Verbose 1 gives progress bar. Verbose 2 recommended for non-interactive session.
model.evaluate(x_test, y_test, verbose=1)
