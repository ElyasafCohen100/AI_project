import cv2
import dlib
import math
import knn
# Load the pre-trained facial landmark detector
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


def detect_landmarks(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    detector = dlib.get_frontal_face_detector()
    face = detector(gray_image)
    landmarks = predictor(gray_image, face)
    landmarksList = []
    # Draw landmarks on the image
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        coordinate_dict = {'X': x, 'Y': y}
        landmarksList += [coordinate_dict]
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    return landmarksList

def generateVector(analyzedFaces):
    facesVectors=[]
    LeyebowL =distancePoint(analyzedFaces[37],analyzedFaces[18],'Y')
    LeyebowM =distancePoint(analyzedFaces[37],analyzedFaces[19],'Y')
    LeyebowR = distancePoint(analyzedFaces[38],analyzedFaces[20],'Y')
    ReyebowL = distancePoint(analyzedFaces[25],analyzedFaces[44],'Y')
    ReyebowM = distancePoint(analyzedFaces[24],analyzedFaces[44],'Y')
    ReyebowR = distancePoint(analyzedFaces[23],analyzedFaces[43],'Y')
    LeyeL= distancePoint(analyzedFaces[37],analyzedFaces[41],'Y')
    LeyeR = distancePoint(analyzedFaces[38],analyzedFaces[40],'Y')
    ReyeL =distancePoint(analyzedFaces[43],analyzedFaces[47],'Y')
    ReyeR =distancePoint(analyzedFaces[44],analyzedFaces[46],'Y')
    lengthMouseL = distancePoint(analyzedFaces[61],analyzedFaces[67],'Y')
    lengthMouseM = distancePoint(analyzedFaces[62], analyzedFaces[66],'Y')
    lengthMouseR = distancePoint(analyzedFaces[63], analyzedFaces[65],'Y')
    widthMouse = distancePoint(analyzedFaces[54],analyzedFaces[48],'X')
    lengthUnit = distancePoint(analyzedFaces[0],analyzedFaces[1],'Y')
    widthUnit = distancePoint(analyzedFaces[0], analyzedFaces[16], 'X')
    facesVectors += [[LeyebowL/lengthUnit,LeyebowM/lengthUnit,LeyebowR/lengthUnit,ReyebowL/lengthUnit,ReyebowM/lengthUnit,ReyebowR/lengthUnit,LeyeL/lengthUnit,LeyeR/lengthUnit,ReyeL/lengthUnit,ReyeR/lengthUnit,lengthMouseL/lengthUnit,lengthMouseM/lengthUnit,lengthMouseR/lengthUnit,widthMouse/widthUnit]]
    return facesVectors



def distancePoint(dict1,dict2,str):
    return math.fabs(dict2[str]-dict1[str])

def vectorDistance(vectorImage1,vectorImage2):
    sum=0
    for i in range(14):
        sum += math.pow(vectorImage1[i]-vectorImage2[i],2)
    return math.sqrt(sum)



def main():
    #train the data
    dataset=knn.train()
    #test the data
    knn.test(dataset)
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Press Space to Capture', frame)

        # Check for spacebar press
        key = cv2.waitKey(1)
        if key == ord(' '):
            # Capture the current frame as a photo
            captured_photo = frame.copy()

            # Display the captured photo
            cv2.imshow('Captured Photo', captured_photo)

            # Detect facial landmarks on the captured photo
            landmarksList=detect_landmarks(captured_photo)
            #generate distance vector
            distanceVec = generateVector(landmarksList)
            #classify by knn
            imClass = knn.knn(distanceVec,dataset)
            #print the result
            print ("You are "+imClass)
            #display facial landmarks on the captured photo
            cv2.imshow('Facial Landmarks', captured_photo)

            # Wait for any key press to continue
            cv2.waitKey(0)

            # Close the windows
            cv2.destroyAllWindows()
            break

    # Release the webcam
    cap.release()
if __name__ == "__main__":
    # Call the main function when the script is run
    main()