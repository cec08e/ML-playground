'''
 Binary classifier for computer-generated images of horses and humans.
 This script assumes the existence of a local directory containing
 Laurence Moroney's horses and humans dataset: http://www.laurencemoroney.com/horses-or-humans-dataset/
'''
import tensorflow as tf
import os
import numpy as np

# Relative location of the training and validation subdirectories for the horse-or-human dataset

parent_dir = os.getcwd().rsplit('/',1)[0]

train_dir = parent_dir + "/datasets/horse-or-human/train"
valid_dir = parent_dir + "/datasets/horse-or-human/validation"
pred_dir = parent_dir + "/datasets/horse-or-human/pred"

# Generate batches of tensor image data with real-time data augmentation.
# The rescale parameter defaults to none or zero. We will multiply all data by 1/255 to rescale.
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)
# Images are "re"sized to 300x300 (modifiable to other values), images loaded in batches of size 128 (32 for validation), binary classifier
train_generator = test_datagen.flow_from_directory(train_dir, target_size = (300,300), batch_size = 128, class_mode ='binary')
valid_generator = test_datagen.flow_from_directory(valid_dir, target_size = (300,300), batch_size = 32, class_mode ='binary')

# ConvNet with 3 convolutional layers.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)), # 16 convolutions of 3x3 filters. Input is a single 300x300 image with 3 values (RGB) per pixel. Output is 16 298x298 images
    tf.keras.layers.MaxPooling2D((2,2)), # 2x2 Max pooling layer. Output is 16 149x149 images.
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), # 32 convolutions of 3x3 filters. Output is 32 147x147 images.
    tf.keras.layers.MaxPooling2D((2,2)), # 2x2 Max pooling layer. Output is 32 73x73 images.
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # 64 convolutions of 3x3 filters. Output is 64 71x71 images.
    tf.keras.layers.MaxPooling2D((2,2)), # 2x2 Max pooling layer. Output is 64 35x35 images.
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # 64 convolutions of 3x3 filters. Output is 64 33x33 images.
    tf.keras.layers.MaxPooling2D((2,2)), # 2x2 Max pooling layer. Output is 64 16x16 images.
    tf.keras.layers.Flatten(), # Flatten to 64*35*35
    tf.keras.layers.Dense(512, activation='relu'), # Dense layer of 512 nodes.
    tf.keras.layers.Dense(1, activation='sigmoid') # A single binary classifier node using the sigmoid activation function.
])

model.summary()

# Set the loss to the Binary Cross Entropy, and use an optimizer with a variable learning rate.
model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.RMSprop(learning_rate=.001), metrics = ['accuracy'])

# Need to consider that we are not passing in a static training list, but a training generator of batch size 128.
# Since there are 1027 images in training directory, we need 8 (steps per epoch) batches of 128 at a time.
history = model.fit(train_generator, steps_per_epoch = 8, epochs = 13, validation_data = valid_generator, validation_steps = 8, verbose = 2)

# Try to test with real images
pred_list = ['horse1.jpg', 'horse2.jpg', 'horse3.jpg', 'human1.jpg', 'human2.jpg', 'human3.jpg' ]

for fname in pred_list:
    img = tf.keras.preprocessing.image.load_img(pred_dir + "/" + fname, target_size=(300, 300))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x/255.0
    x = np.expand_dims(x, axis=0) # Add a new axis a index 0, equivalent to [a,b] --> [[a,b]]
    #print(fname, x)
    #images = np.vstack([x])
    #print(images)
    classes = model.predict(x, batch_size=1)
    print("Prediction: ", classes[0])
    if classes[0]>0.5:
        print(fname + " is a human")
    else:
        print(fname + " is a horse")
