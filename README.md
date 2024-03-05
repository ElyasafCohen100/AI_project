# AI_project

The project contain two models. One is the CNN model, and the second is KNN model.


## Project structure

### CNN
The CNN model is already trained and redy to use.

The CNN model contain 2 convolutional layers, and one dens layer.
The model trained on dataset of ~35,000 48*48 pixels grayscale extend to RGB images. 
(The dataset source is kaggle.com. Can't pinpoint to the direct dataset.)

The evaluation score of the model is: 58.25%, and the loss is: 1.028.

The code to train the model is in the folder [*cnn-files*](/cnn-files).
- *training-48-7-labels.py* - Contain the training code. The code is based on 2 JSON files (NOT INCLUDED IN THIS REPOSITORY, ~13 GB of data).
- *generate-dataset-48-7-labels.py* - Takes all the imaged dataset and convert them into JSON file.
- *test.py* - Testing the model by loading the saved model (the long file name in format keras) and predict on ar3.jpg file.

To run the model you can simply run the [*test.py*](/cnn-files/test.py) file.

### KNN
The model is pre-trained, and the code contain the training functions.

The code contain a Multiclass KNN Model original implementation.
The KNN model train on the dataset folder (Included in [*knn-files*](/knn-files) folder).
For prediction the model using K=5 neighbors.

The model extruct all the face key points and the main distances between them for classifying between each 7 labels.

The model code is in [*knn-files*](/knn-files) folder.
- *main.py* - The main function which activate the train and test function and eventually its open the computer camera and capture user photo by pressing on the space bar.
- *knn.py* - Contain all the train and test functions.
- *heap.py* - Despite his name, the file contain implementation of priority queue using a minimum heap.
- *generate-dataset.py* - Load all the images dataset and convert it to face landmarks list.

To run the model all you need is to run the [*main.py*](/knn-files/main.py) file. 
(remember, this is pre-trained model, so it will be trained each time you run it.)

## Dependencies
#### For the CNN model you will need:
- json
- datetime
- random
- keras
- numpy
- skimage
- tensorflow
- os
- cv2

#### For the KNN model you will need:
- cv2
- dlib
- math
- numpy
- openai
- PIL
- os

### Pay attention!
The final code should generate an image inspired by the captured face emotion. 
For doing so, the code is using "openai" API which require an API_KEY. 
We did not enter our API_KEY so line 206 in [*main.py*](/knn-files/main.py) file is commented. 
For testing the generate_image function please
* [ ] Uncomment 206 line in [*main.py*](/knn-files/main.py) file.
* [ ] Insert an API_KEY in line 25 in the same file.

We hope you will find this repo usefully for whatever you need.


    ElyasafCohen100, danyots, yohanan400. 