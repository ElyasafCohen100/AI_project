import json
import datetime as dt
import random
import keras.models
import numpy as np
import skimage as ski
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf


# STEP 0: CREATE CNN LAYERS
def create_model():
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(4, 3, 1, "same", activation="sigmoid", input_shape=[48, 48, 3]),
            keras.layers.Conv2D(8, 3, 1, "same", activation="sigmoid"),
            keras.layers.Flatten(),
            keras.layers.Dense(7, "sigmoid")
        ]
    )
    return model


# STEP 1: LOAD DATASET FILES AND PREPARE DATA
print("Opening combined JSON file")
f = open('combined-Facial-expression-data-label-48-48-RGB-7-labels.json', 'r')
data = json.load(f)
features = np.array(data[0])
labels = np.array(data[1])
f.close()

# Opening 2nd JSON file
print("Opening 2nd JSON file")
f = open('new-combined-train-validation-datasets-48-48-RGB.json', 'r')
data = json.load(f)
features = np.concatenate((features, np.array(data[0])))
labels = np.concatenate((labels, np.array(data[1])))
f.close()

print(len(features))

# Generate random seed for split function
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

print("splitting the data")
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.25, shuffle=True)

# STEP 2: CREATE MODEL
model = create_model()
print("Model successfully created")

# STEP 3: Create an Instance of Early Stopping Callback
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# STEP 4: Compile the model and specify loss function, optimizer and metrics values to the model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

# STEP 5: Start training the model.
print("Start training the model.")
conv_model_training_history = model.fit(x=features_train, y=labels_train, epochs=50, batch_size=4,
                                        shuffle=True, validation_split=0.2,
                                        callbacks=[early_stopping_callback])

# STEP 6: SAVING THE MODEL
model_evaluation_history = model.evaluate(features_test, labels_test)
# Get the loss and accuracy from model_evaluation_history.
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

# Define the string date format.
# Get the current Date and Time in a DateTime Object.
# Convert the DateTime object to string according to the style mentioned in date_time_format string.
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.
model_file_name = f'conv_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.keras'

print("Saving the model")
# Save your Model.
model.save(model_file_name)
print("Saved")

# test the model for one prediction
image_data = ski.io.imread("sf2.jpg")
new_image = image_data.tolist()
print(model.predict([new_image]))
print("Done.")
