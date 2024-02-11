import cv2
import dlib
import math

# Load the pre-trained facial landmark detector
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


def detect_landmarks(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_image)
    analyzedFaces = []
    # Iterate over detected faces
    for face in faces:
        # Predict facial landmarks for each face
        landmarks = predictor(gray_image, face)
        landmarksList = []
        # Draw landmarks on the image
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            coordinate_dict = {'X': x, 'Y': y}
            landmarksList += [coordinate_dict]
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        analyzedFaces += [landmarksList]
    return analyzedFaces

def generateVector(analyzedFaces):
    facesVectors=[]
    for face in analyzedFaces:
        LeyebowL =distancePoint(face[37],face[18],'Y')
        LeyebowM =distancePoint(face[37],face[19],'Y')
        LeyebowR = distancePoint(face[38],face[20],'Y')
        ReyebowL = distancePoint(face[25],face[44],'Y')
        ReyebowM = distancePoint(face[24],face[44],'Y')
        ReyebowR = distancePoint(face[23],face[43],'Y')
        LeyeL= distancePoint(face[37],face[41],'Y')
        LeyeR = distancePoint(face[38],face[40],'Y')
        ReyeL =distancePoint(face[43],face[47],'Y')
        ReyeR =distancePoint(face[44],face[46],'Y')
        lengthMouseL = distancePoint(face[61],face[67],'Y')
        lengthMouseM = distancePoint(face[62], face[66],'Y')
        lengthMouseR = distancePoint(face[63], face[65],'Y')
        widthMouse = distancePoint(face[54],face[48],'X')
        lengthUnit = distancePoint(face[0],face[1],'Y')
        widthUnit = distancePoint(face[0], face[16], 'X')
        facesVectors += [[LeyebowL/lengthUnit,LeyebowM/lengthUnit,LeyebowR/lengthUnit,ReyebowL/lengthUnit,ReyebowM/lengthUnit,ReyebowR/lengthUnit,LeyeL/lengthUnit,LeyeR/lengthUnit,ReyeL/lengthUnit,ReyeR/lengthUnit,lengthMouseL/lengthUnit,lengthMouseM/lengthUnit,lengthMouseR/lengthUnit,widthMouse/widthUnit]]
    return facesVectors



def distancePoint(dict1,dict2,str):
    return math.fabs(dict2[str]-dict1[str])

def vectorDistance(vectorImage1,vectorImage2):
    sum=0
    for i in range(14):
        sum += math.pow(vectorImage1[i]-vectorImage2[i],2)
    return math.sqrt(sum)

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

        # Detect and display facial landmarks on the captured photo
        detect_landmarks(captured_photo)
        cv2.imshow('Facial Landmarks', captured_photo)

        # Wait for any key press to continue
        cv2.waitKey(0)

        # Close the windows
        cv2.destroyAllWindows()
        break

# Release the webcam
cap.release()