import numpy as np
from PIL import Image
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import glob

def load_data(name):
    df = pd.read_csv(name + '.csv')
    df_sorted = df.sort_values(by='image_id')

    labelDict = dict()
    labels = df_sorted.values
    for entry in labels:
        try:
            labelDict[entry[0]] = entry[1]
        except:
            labelDict[entry[0]] = 0

    images = []
    labels = []
    filenames = []
    for filename in os.listdir(name):
        if filename.endswith('.png'):
            file_path = os.path.join(name, filename)
            image = Image.open(file_path)
            image_array = np.array(image)
            if filename[:-4] in labelDict.keys():
                filenames.append(filename[:-4])
                images.append(image_array)
                labels.append(labelDict[filename[:-4]])

        for i in range(len(images)):
            if images[i].shape != (80, 80, 3):
                images[i] = np.stack((images[i], images[i], images[i]), axis=-1)

    images = np.array(images)
    labels = np.array(labels)

    return filenames, images, labels

training_filenames, training_images, training_labels = load_data("train")
print(f"Training: {np.shape(training_images)} - {np.shape(training_labels)}")

validation_filenames, validation_images, validation_labels = load_data("validation")
print(f"Validation: {np.shape(validation_images)} - {np.shape(validation_labels)}")

test_filenames, test_images, test_labels = load_data("test")
print(f"Test: {np.shape(test_images)} - {np.shape(test_labels)}")

# Standardize images

standard_training_images = (training_images - np.mean(training_images, axis=(0, 1, 2))) / np.std(training_images, axis=(0, 1, 2))
standard_validation_images = (validation_images - np.mean(validation_images, axis=(0, 1, 2))) / np.std(validation_images, axis=(0, 1, 2))
standard_test_images = (test_images - np.mean(test_images, axis=(0, 1, 2))) / np.std(test_images, axis=(0, 1, 2))

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical

model_train_images = standard_training_images
model_train_labels = to_categorical(training_labels, num_classes=3)
model_validation_images = standard_validation_images
model_validation_labels = to_categorical(validation_labels, num_classes=3)

train_ds = tf.data.Dataset.from_tensor_slices((model_train_images, model_train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((model_validation_images, model_validation_labels))
test_ds = tf.data.Dataset.from_tensor_slices(standard_test_images)

def augment(image, label):
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[72, 72, 3])
    return image, label

def crop_center(image, label):
    image = tf.image.crop_to_bounding_box(image, 4, 4, 72, 72)
    return image, label

train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(crop_center, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

if not os.path.exists("tmp"):
    os.makedirs("tmp")

files = glob.glob('tmp/*')
for f in files:
    os.remove(f)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"tmp/save.keras",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=5, 
    min_lr=0.00001
)


my_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(72, 72, 3)),
    BatchNormalization(),
    
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu'),    
    BatchNormalization(),
    Dropout(0.2),
    
    Flatten(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(3, activation='softmax')
])
my_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
my_model.summary()

history = my_model.fit(
    train_ds,
    epochs=3,
    validation_data=val_ds,
    callbacks=[model_checkpoint_callback, reduce_lr],
)

loss, accuracy = my_model.evaluate(val_ds)
print(f'Validation accuracy: {accuracy}')

# import matplotlib.pyplot as plt

# plt.plot(history.history['accuracy'], label='train_accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.plot(history.history['loss'], label='train_loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.savefig("/kaggle/working/plots/cnn-2-0")
# plt.show()


# from sklearn.metrics import confusion_matrix

# def print_confusion_matrix(model, images, labels):
#     y_prediction = model.predict(images)
#     y_prediction = np.argmax(y_prediction, axis=1)
    
#     # Calculate the confusion matrix
#     result = confusion_matrix(labels, y_prediction, normalize='pred')
#     print(result)

# print_confusion_matrix(my_model, model_validation_images[:,4:76,4:76,:], validation_labels)

def predict_with_multiple_crops(model, images):
    prediction = model.predict(images[:,4:76,4:76,:])
    
    flipped_images = images[:,::-1,:,:]
    
    crops = [[0, 72], [4, 76], [8, 80]]
    
    for i1, i2 in crops:
        for j1, j2 in crops:
            cropped = images[:,i1:i2,j1:j2,:]
            print(np.shape(cropped))
            prediction += model.predict(cropped)
            
            cropped = flipped_images[:,i1:i2,j1:j2,:]
            print(np.shape(cropped))
            prediction += model.predict(cropped)
            
    prediction = np.argmax (prediction, axis = 1)
    return prediction

def evaluate(prediction, labels):
    return np.sum(prediction==labels) / len(labels)

def printPrediction(prediction, name):
    files = []
    folder_path = 'test'
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            files.append(filename[:-4])
    d = dict()
    for i in range(len(files)):
        d[files[i]] = prediction[i]


    df = pd.read_csv('sample_submission.csv')
    for i in range(len(files)):
        df.at[i, 'label'] = d[df.at[i, 'image_id']]
    df.to_csv(name, index=False)
    


model_path = 'tmp/save.keras'
loaded_model = load_model(model_path)
loaded_model.summary()
val_loss, val_accuracy = loaded_model.evaluate(val_ds)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')


val_accuracy_cropped = evaluate(predict_with_multiple_crops(loaded_model, standard_validation_images), validation_labels)
print(f'Multi Crop Validation accuracy: {val_accuracy_cropped}')


prediction = predict_with_multiple_crops(loaded_model, standard_test_images)
printPrediction(prediction, "output.csv")