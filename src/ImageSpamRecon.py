# Neural Networks

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 73
epochs = 10
dataset_dir = 'dataset_path'

train_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary')

# steps_per_epoch = train_generator.samples // train_generator.batch_size

training_images, training_labels = next(train_generator)
testing_images, testing_labels = next(test_generator)

training_labels = training_labels.astype(int)
testing_labels = testing_labels.astype(int)

class_names = ['ham', 'spam']

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i]])

plt.show()

# training_images = training_images[:744] # 80% dataset
# training_labels = training_labels[:744]
# testing_images = testing_images[:186] #20% dataset
# testing_labels = testing_labels[:186]

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid')) #1 class

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(train_generator, epochs=epochs ,validation_data=test_generator)

# loss, accuracy = model.evaluate(test_generator)
# print(f"Loss : {loss}")
# print(f"Accuracy : {accuracy}")

# model.save('ImageSpamRecon.keras')

#------------------------------

model = models.load_model('ImageSpamRecon.keras')

img = cv.imread('img_path')
img = cv.resize(img, (128, 128))
# img = img / 255
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
print(prediction)
index = prediction[0]
if (index > 0.5):
    print(f'Prediction is {class_names[1]} \n')
else:
    print(f'Prediction is {class_names[0]} \n')
