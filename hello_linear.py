'''
A simple hello world example for solving a linear equation y = m*x + b
using an embarrasingly simple NN with TensorFlow.
'''
import tensorflow as tf
import numpy as np

# Set the characteristic values of the linear equation y = m*x + b
m = 2
b = 1

# Generate a training set and a testing set with 1000 values each.
x_total = np.linspace(-10, 10, 2000, dtype=float)
np.random.shuffle(x_total)
[x_train, x_test] = np.array_split(x_total, 2)

y_train = m*x_train + b
y_test = m*x_test + b

# Compile a NN with a single input and a single output. We are looking to train the
# single weight and bias to recover m and b.
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train model
model.fit(x_train, y_train, epochs=10)

# Test model
print(model.predict([20.0])) # Should be 41!
model.evaluate(x_test,  y_test, verbose=2)
