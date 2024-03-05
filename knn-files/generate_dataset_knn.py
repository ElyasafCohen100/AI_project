import os
import main
from PIL import Image


def generate_faceVector_dataset(dir_name):
    """
    Generate a dataset of face vectors from images in the specified directory.

    Parameters:
    - dir_name (str): The name of the directory containing the image dataset.

    Returns:
    - list: A list of lists, where each inner list represents a face vector
            extracted from an image. Each inner list contains numerical
            values representing the face features along with the class
            label of the image.
    """
    # Construct the full path to the dataset directory
    DATASET_DIR_NAME = r"dataset\\" + dir_name

    # Get a list of subdirectories (class labels) in the dataset directory
    dir_list = os.listdir(DATASET_DIR_NAME)

    # Initialize an empty list to store all face vectors
    all_images_data = []

    detect_faces_counter = 0
    # Iterate over each subdirectory (class label) in the dataset
    for class_index, class_dir_name in enumerate(dir_list):
        # Get a list of image filenames in the current subdirectory
        images_list = os.listdir(os.path.join(DATASET_DIR_NAME, class_dir_name))

        # Iterate over each image in the current subdirectory
        for image_name in images_list:
            # Open the image using PIL (Python Imaging Library)
            image = Image.open(os.path.join(DATASET_DIR_NAME, class_dir_name, image_name))

            # Detect facial landmarks in the image and generate a face vector
            landmarks = main.detect_landmarks(image)
            if (len(landmarks) == 0):
                continue
            detect_faces_counter += 1
            print(str(detect_faces_counter) + " faces detected")
            face_vector = main.generateVector(landmarks)

            # Append the class label to the face vector
            face_vector += [class_dir_name]

            # Convert the face vector to a list and append it to the dataset
            all_images_data.append(face_vector)
    print("Done.")

    # Return the list of all face vectors extracted from the dataset
    return all_images_data
