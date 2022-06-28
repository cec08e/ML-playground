'''
A simple callback class example to be used in the fashion mnist NN
'''

import tensorflow as tf

# The Callback class is an abstract base class with several
# redefineable methods: on_epoch_begin, on_epoch_end, ...
# Look at Callback class source to see all the methods.
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs = None):
        # print(logs.keys()) to see available keys
        # If accuracy above 86%, stop training.
        if logs['accuracy'] > .86:
            self.model.stop_training = True


    def on_train_end(self, logs = None):
        print("Training is done!")
