'''
A CNN classifier for cat and dog images based off of the filtered cats and dogs dataset
found here: https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
Larger cats and dogs dataset found here: https://www.kaggle.com/c/dogs-vs-cats
'''

import tensorflow as tf
import os
import numpy as np

# Relative location of the training and validation subdirectories for the cats and dogs dataset
parent_dir = os.getcwd().rsplit('/',1)[0]
train_dir = parent_dir + "/datasets/cats_and_dogs_filtered/train"
valid_dir = parent_dir + "/datasets/cats_and_dogs_filtered/validation"
pred_dir = parent_dir + "/datasets/cats_and_dogs_filtered/pred"

# Print the total number of training and validation images for each class
print('Total training cat images :', len(os.listdir(train_dir+"/cats")))
print('Total training dog images :', len(os.listdir(train_dir+"/dogs")))
print('Total validation cat images :', len(os.listdir(valid_dir+"/cats")))
print('Total validation dog images :', len(os.listdir(valid_dir+"/cats")))



# Generate batches of tensor image data with real-time data augmentation.
# The rescale parameter defaults to none or zero. We will multiply all data by 1/255 to rescale.
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)
# Images are resized to 150x150 (modifiable to other values), images loaded in batches of size 20, binary classifier
train_generator = test_datagen.flow_from_directory(train_dir, target_size = (150,150), batch_size = 20, class_mode ='binary')
valid_generator = test_datagen.flow_from_directory(valid_dir, target_size = (150,150), batch_size = 20, class_mode ='binary')


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)), # Output is 16 148x148 images.
    tf.keras.layers.MaxPooling2D(2,2), # Output is 16 74x74 images
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), # Output is 32 72x72 images
    tf.keras.layers.MaxPooling2D(2,2), # Output is 32 36x36 images
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # Output is 64 34x34 images
    tf.keras.layers.MaxPooling2D(2,2), # Output is 64 17x17 images
    tf.keras.layers.Flatten(),     # Flatten the results into 64x17x17 to feed into a DNN
    tf.keras.layers.Dense(512, activation='relu'),     # 512 neuron hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')     # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
])

model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),loss='binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_generator, steps_per_epoch=100, epochs=12, validation_data=valid_generator, validation_steps=50, verbose=2)

pred_list = ['cat1.jpg', 'cat2.jpg', 'cat3.jpg','cat4.jpg', 'dog1.jpg', 'dog2.jpg', 'dog3.jpg', 'dog4.jpg' ]

for fname in pred_list:
    img = tf.keras.preprocessing.image.load_img(pred_dir + "/" + fname, target_size=(150, 150))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x/255.0
    x = np.expand_dims(x, axis=0) # Add a new axis a index 0, equivalent to [a,b] --> [[a,b]]
    classes = model.predict(x, batch_size=1)
    print("Prediction: ", classes[0])
    if classes[0]>0.5:
        print(fname + " is a dog")
    else:
        print(fname + " is a cat")
