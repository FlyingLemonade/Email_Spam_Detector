import cv2 as cv
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def spamRecon(img_path=None, train_model=False):
    batch_size = 73
    epochs = 10
    dataset_dir = '../dataset/ImageDatasets'
    class_names = ['ham', 'spam']
    model_path = './CNN_Model/ImageSpamRecon.keras'

    if train_model:
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

        # Uncomment to see the first 16 training images
        # training_images, training_labels = next(train_generator)
        # for i in range(16):
        #     plt.subplot(4, 4, i + 1)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(training_images[i], cmap=plt.cm.binary)
        #     plt.xlabel(class_names[int(training_labels[i])])
        # plt.show()

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, epochs=epochs, validation_data=test_generator)
        model.save(model_path)
        print(f"Model trained and saved to {model_path}")
        return

    # Load the pre-trained model
    model = models.load_model(model_path)

    if img_path:
        # Load and preprocess the image
        img = cv.imread(img_path)
        img = cv.resize(img, (128, 128))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.array([img]) / 255.0  # Normalize the image

        # Perform prediction
        prediction = model.predict(img)
        print(prediction)

        # Interpret prediction
        if prediction[0] > 0.5:
            return "Spam"
        else:
            return "Ham"

    return "No image path provided"

# Example usage:
# To train the model
# spamRecon(train_model=True)

# To make a prediction on an image
# result = spamRecon(img_path='C:\\Users\\Lenovo\\Pictures\\ktp.jpg')
# print(result)
# spamRecon(img_path="..\dataset\ImageDatasets\ham\zzz_0013_4950651bbd_m.jpg",train_model=False)

