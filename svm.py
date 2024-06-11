import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import skimage.measure


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

standard_training_images = (training_images - np.mean(training_images, axis=(0, 1, 2))) / np.std(training_images, axis=(0, 1, 2))
standard_validation_images = (validation_images - np.mean(validation_images, axis=(0, 1, 2))) / np.std(validation_images, axis=(0, 1, 2))
standard_test_images = (test_images - np.mean(test_images, axis=(0, 1, 2))) / np.std(test_images, axis=(0, 1, 2))

# ### MONO PIXEL COLOR
# def keep_strongest_color(rgb) :
#     c = np.argmax(rgb)
#     for i in range(3):
#         if i != c:
#             rgb[i] = 0
#     return rgb

# def process_images(images):
#     for i in range(len(images)):
#         if i % 500 == 0:
#             print(i)
#         for j in range(len(images[i])):
#             for k in range(len(images[i][j])):
#                 images[i][j][k] = keep_strongest_color(images[i][j][k])
#     return images
# mono_training_images = process_images(training_images[:1000])
# mino_validation_images = process_images(validation_images[:1000])
# mono_test_images = process_images(test_images[:1000])

# def to_grayscale(images):
#     return np.mean(images, axis=(3))

# def print_images(images, labels, count, cmap=None, title="title"):
#     for i in range(count):
#         plt.imshow(images[i], cmap=cmap)
#         plt.title(f'Label: {labels[i]}')
#         plt.show()

#         img = images[i]
#         # mins = np.min(img, axis=(0, 1))
#         # maxs = np.max(img, axis=(0, 1))
#         # img = (img - mins) / (maxs - mins) * 255.0
#         Image.fromarray(img.astype(np.uint8)).save(f"documentation/plots/{title}-{labels[i]}-{str(i)}.png")


reduced_training_images = (skimage.measure.block_reduce(training_images, (1,4,4,1), np.max)).reshape((10500, 10*40*3))
print(np.shape(reduced_training_images))
reduced_validation_images = (skimage.measure.block_reduce(validation_images, (1,4,4, 1), np.max)).reshape((3000, 10*40*3))

clf = svm.SVC(decision_function_shape='ovr', verbose=True, max_iter=-1, kernel="rbf")
clf.fit(reduced_training_images, training_labels)
score = clf.score(reduced_validation_images, validation_labels)

print("score on test: " + str(score))