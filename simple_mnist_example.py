'''
The standard mnist digits dataset example.
'''
import tensorflow as tf

# Load up the mnist dataset from keras' datasets.
mnist = tf.keras.datasets.mnist

# load_data routine returns two two-element tuples of the
# training and test sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalization of grayscale data

# Single hidden layer with a dropout routine.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten 28x28 input to 784 input nodes
  tf.keras.layers.Dense(128, activation='relu'), # 128-node hidden layer with a rectified linear activation
  tf.keras.layers.Dropout(0.2), # Prevents overfitting and memorization of training data. Drops links to 20% of hidden nodes per minibatch.
  tf.keras.layers.Dense(10) # Output layer for values 0-9
])

# Uncomment to see the raw logit values on untrained model.
#predictions = model(x_train[:1]).numpy()
#print(predictions)
#print(tf.nn.softmax(predictions).numpy())   # Softmax function converts logits to probabilities: softmax = exp(logit) / reduce_sum(exp(logit))
# Softmax values will convert logits to approx ~1/10 on untrained model.


# Check out https://fmorenovr.medium.com/sparse-categorical-cross-entropy-vs-categorical-cross-entropy-ea01d0392d28 for more loss discussion.
# Space CCE loss will evaluate the Softmax value of the logit in the target (true) index. Useful in cases where size of output layer (classes) is
# large or information about "near-misses" not necessary. CE loss is given by \sum_i -t_i*log(y_i) for target index i where t_i is truth value and y_i is softmax of logit.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#print(loss_fn(y_train[:1], predictions).numpy()) # Take in target and prediction logits. In untrained case, softmax of all logits is ~.1. Therefore, SCCE will be approx -log(.1)
#print(y_train[:1])

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 5)
model.evaluate(x_test,  y_test, verbose=2)
