import cv2
import dlib

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
