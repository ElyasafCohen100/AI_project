# Import necessary modules
import heap  # Import heap module for heap data structure operations
import math  # Import math module for mathematical operations
import generate_dataset_knn  # Import generate_dataset_knn module for dataset generation

# Set the value of k for k-NN algorithm
k = 5

# Define the train function to train the k-NN classifier
def train():
    """
    Trains the k-NN classifier using the training dataset.

    Returns:
        list: A list of classified images after training.
    """
    # Generate face vector dataset from training images
    images = generate_dataset_knn.generate_faceVector_dataset("train")
    print(images)
    # Pass the dataset for training and return the classified images
    return pass_on_dataset(images, images, "train")

# Define the pass_on_dataset function to classify images based on the k-NN algorithm
def pass_on_dataset(sourceImages, compImages, type):
    """
    Classifies images in the dataset based on the k-NN algorithm.

    Parameters:
        sourceImages (list): A list of source images for classification.
        compImages (list): A list of comparison images for classification.
        type (str): Type of dataset (train/test/classified).

    Returns:
        list: A list of classified images.
    """
    # Initialize an empty list to store classified images
    classifiedImages = []
    # Iterate over each source image in the dataset
    for image in sourceImages:
        # Create a heap data structure
        f = heap.create()
        # Iterate over each comparison image in the dataset
        for compImage in compImages:
            # Exclude self-comparisons during training
            if compImage != image or type != "train":
                # Insert distance and class label into the heap
                heap.insert(f, [distance(image, compImage), compImage[len(compImage) - 1]])
        # Initialize an empty list to store class labels for k-nearest neighbors
        classifying = []
        # Retrieve k-nearest neighbors from the heap
        for _ in range(k):
            s = heap.remove(f)
            classifying.append(s[1])
        # Count occurrences of each class label among k-nearest neighbors
        counter = 0
        imClass = classifying[0]
        for i in classifying:
            curr_frequency = classifying.count(i)
            if curr_frequency > counter:
                counter = curr_frequency
                imClass = i
        # Assign the most frequent class label to the source image
        classifiedImages.append(image)
        classifiedImages[len(classifiedImages) - 1][len(image) - 1] = imClass
    # Return the list of classified images
    return classifiedImages

# Define the test function to evaluate the accuracy of the classifier
def test(dataset):
    """
    Tests the accuracy of the k-NN classifier using a validation dataset.

    Parameters:
        dataset (list): The dataset for testing.

    Returns:
        float: The accuracy of the classifier.
    """
    # Generate face vector dataset from validation images
    images = generate_dataset_knn.generate_faceVector_dataset("validation")
    # Classify images in the validation dataset and store the result
    classifiedImages = pass_on_dataset(images, dataset, "test")
    # Initialize counter to count correct classifications
    counter = 0
    # Get the length of the dataset
    length = len(classifiedImages)
    # Iterate over each classified image
    for i in range(length):
        # Compare the class label of each classified image with the ground truth
        if classifiedImages[i][len(classifiedImages[i])-1] == images[i][len(images[i])-1]:
            counter += 1
    # Calculate the accuracy of the classifier
    accuracy = counter / length
    # Print the accuracy of the classifier
    print("The accuracy for k =", k, "is:", accuracy)

# Define the knn function to classify a single image vector
def knn(imageVec, dataset):
    """
    Classifies an image vector using the k-NN algorithm.

    Parameters:
        imageVec (list): The image vector to classify.
        dataset (list): The dataset for classification.

    Returns:
        str: The class label of the classified image.
    """
    # Classify the image vector using the k-NN algorithm
    classifiedImages = pass_on_dataset([imageVec], dataset, "classified")
    # Return the class label of the classified image
    return classifiedImages[0][len(imageVec) - 1]

# Define the distance function to calculate the Euclidean distance between two vectors
def distance(sourceVec, compVector):
    """
    Calculates the Euclidean distance between two vectors.

    Parameters:
        sourceVec (list): The source vector.
        compVector (list): The comparison vector.

    Returns:
        float: The Euclidean distance between the vectors.
    """
    # Initialize the sum of squared differences
    sum = 0
    # Iterate over each element in the vectors (except the class label)
    for i in range(len(sourceVec) - 1):
        # Add the squared difference of corresponding elements to the sum
        sum += math.pow((sourceVec[i] - compVector[i]), 2)
    # Return the square root of the sum (Euclidean distance)
    return math.sqrt(sum)
