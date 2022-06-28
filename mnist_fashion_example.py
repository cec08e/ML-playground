'''
Simple mnist fashion dataset example with a custom callback.
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
model.add(tf.keras.layers.Flatten()) # Input shape is not necessary
model.add(tf.keras.layers.Dense(128, activation='relu')) # 128-node hidden layer with a rectified linear activation. if x > 0: return x else return 0
model.add(tf.keras.layers.Dense(10, activation='softmax')) # Output layer for values 0-9, logits turned into probabilities. Reminder: softmax is exp(x) / reduce_sum(exp(x))

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
