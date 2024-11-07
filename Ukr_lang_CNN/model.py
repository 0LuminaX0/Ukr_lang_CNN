import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import datetime
import ast
import math

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    y_data = df['label'].values
    x_data = df['image_data'].apply(lambda x: ast.literal_eval(x)).values
    x_data = np.array(x_data.tolist())
    x_data = x_data.reshape(-1, 28, 28)
    
    return x_data, y_data

# Load the dataset from CSV
#csv_file = 'ukr_mnist_data.csv'
csv_file = './data/augmented_glyphs_2.csv'
x_train, y_train = load_data_from_csv(csv_file)

# shuffle data !!!
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)

x_train = x_train[indices]
y_train = y_train[indices]
#--------------

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('y_train unique:', np.unique(y_train))
print('y_train unique length:', len(np.unique(y_train)))

(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape
IMAGE_CHANNELS = 1

print('IMAGE_WIDTH:', IMAGE_WIDTH)
print('IMAGE_HEIGHT:', IMAGE_HEIGHT)
print('IMAGE_CHANNELS:', IMAGE_CHANNELS)

# Display the first image from the dataset with its label
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.title(f"Label: {y_train[0]}")
#plt.show()

# Display multiple images in a grid
numbers_to_display = 144
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10, 10))

for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(str(y_train[i]))
plt.show()

x_train_with_chanels = x_train.reshape(
    x_train.shape[0],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
)
print('x_train_with_chanels:', x_train_with_chanels.shape)

x_train_normalized = x_train_with_chanels / 255
print("x_train_normalized example:", x_train_normalized[0][18])

#==============================================================
#                          Model start
#==============================================================

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    layers.MaxPooling2D((2, 2)),
    # Stop 3rd layer (now optional)
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    layers.Flatten(),
    #layers.Dropout(0.4),  # dropout after flattening
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    #layers.Dropout(0.5),  # Dropout before output layer
    layers.Dense(33, activation='softmax')  # 33 for 33 letters
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,  # Stop if val_loss does not improve for N consecutive epochs
    restore_best_weights=True
)

training_history = model.fit(
    x_train_normalized,
    y_train_encoded,
    epochs=15,  # Allow for more training if early stopping doesn't trigger
    validation_split=0.2,  # 20% of data as validation
    callbacks=[tensorboard_callback, early_stopping]
)

plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(training_history.history['loss'], label='training set')
plt.plot(training_history.history['val_loss'], label='test set')
plt.legend()
plt.show()

plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['accuracy'], label='training set')
plt.plot(training_history.history['val_accuracy'], label='test set')
plt.legend()
plt.show()

model_name = 'ukrlang_recognition_mlp.keras'
model.save(model_name)