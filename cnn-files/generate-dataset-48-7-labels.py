# STEP 1: EXTRUCT ALL DATASET DIRECTORIES
import json
import os

import cv2
import skimage as ski
from gray2color import gray2color

DATASET_DIR_NAME = r"dataset\validation"
dir_list = os.listdir(DATASET_DIR_NAME)

# STEP 2: FOR EACH DIRECTORY EXTRUCT ALL IMAGES NAMES
all_images_data = []
all_images_features = []
for class_index, class_dir_name in enumerate(dir_list):
    images_list = os.listdir(os.path.join(DATASET_DIR_NAME, class_dir_name))

    images_number = len(images_list)
    counter = 0
    print(str(class_index + 1) + ". " + class_dir_name)

    image_data_list = []
    image_features_list = []
    # STEP 3: READ ALL IMAGES INTO ONE LIST
    for image_name in images_list:
        image_data = ski.io.imread(os.path.join(DATASET_DIR_NAME, class_dir_name, image_name))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)  # Expend each pixel to RGB
        all_images_data.append(image_data.tolist())

        # STEP 4: ENTER THE CLASSIFICATION FOR EACH IMAGE
        label = [0, 0, 0, 0, 0, 0, 0]
        label[class_index] = 1
        all_images_features.append(label)

        # show progress percentages
        counter += 1
        ratio = counter / images_number * 100
        if int(ratio) % 25 == 0:
            print(str(int(ratio)) + "%")

# STEP 5: COMBINE DATA AND FEATURES INTO ONE LIST
combined_data_features_list = [all_images_data, all_images_features]

# STEP 6: SERIALIZE LIST TO JSON
print("Dumps to JSON format")
data_in_json = json.dumps(combined_data_features_list, indent=4)

# STEP 7: SAVE DATA TO JSON FILE
print("Saving into JSON file")
json_file_name = "new-dataset-48-48-RGB-validation.json"
with open(json_file_name, 'w') as outfile:
    print("Saving to json file: " + json_file_name)
    outfile.write(data_in_json)
    print("Done.")
