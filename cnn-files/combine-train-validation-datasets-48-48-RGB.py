# Opening JSON file
import json
import numpy as np


# print("open Facial expression file")
# features = np.load(r"archive(7)\Facial expression.npy")
#
# print("open Facial expression label file")
# labels = np.load(r"archive(7)\Facial expression label.npy")

print("open train JSON file")
f = open('new-dataset-48-48-RGB-train.json', 'r')
train_data = json.load(f)
train_features = np.array(train_data[0]).tolist()
train_labels = np.array(train_data[1]).tolist()

# Closing file
f.close()

print("open validation JSON file")
f = open('new-dataset-48-48-RGB-validation.json', 'r')
validation_data = json.load(f)
validation_features = np.array(validation_data[0]).tolist()
validation_labels = np.array(validation_data[1]).tolist()

# Closing file
f.close()

features_to_copy = len(validation_features)
for index, valid_feature in enumerate(validation_features):
    train_features.append(valid_feature)
    train_labels.append(validation_labels[index])
    pres = (index / features_to_copy) * 100
    if pres % 5 == 0:
        print("{:.1f}".format(pres) + "% have been copied.")


combined_data_features_list = [train_features, train_labels]

print(len(combined_data_features_list))
print(len(combined_data_features_list[0]))
print(len(combined_data_features_list[0][0]))
print(len(combined_data_features_list[0][0][0]))
print(len(combined_data_features_list[0][0][0][0]))
print(len(combined_data_features_list[1]))

print("start saving process")
data_in_json = json.dumps(combined_data_features_list, indent=4)
json_file_name = "new-combined-train-validation-datasets-48-48-RGB.json"
with open(json_file_name, 'w') as outfile:
    print("Saving to json file: " + json_file_name)
    outfile.write(data_in_json)
    print("Done.")

