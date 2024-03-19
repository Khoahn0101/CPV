import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model for face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"E:\CPV\Face\trainingData.yml")

# Load the trained model for sign language gesture recognition
model_dict = pickle.load(open(r"E:\CPV\Dataset\model.p", 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect hands and predict sign language
def detect_and_predict(frame):
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Detect faces
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)

        # If two faces are detected
        if len(faces) == 2:
            closest_face_0 = None
            closest_face_1 = None
            min_distance_0 = float('inf')
            min_distance_1 = float('inf')

            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                for (x, y, w, h) in faces:
                    distance = np.sqrt((x1 - x)**2 + (y1 - y)**2)
                    label, _ = face_recognizer.predict(gray_img[y:y+h, x:x+w])
                    if label == 0 and distance < min_distance_0:
                        closest_face_0 = (x1, y1, x2, y2)
                        min_distance_0 = distance
                    elif label == 1 and distance < min_distance_1:
                        closest_face_1 = (x1, y1, x2, y2)
                        min_distance_1 = distance

            if closest_face_0 is not None and closest_face_1 is not None:
                x1_0, y1_0, x2_0, y2_0 = closest_face_0
                x1_1, y1_1, x2_1, y2_1 = closest_face_1
                if min_distance_0 < min_distance_1:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                    cv2.rectangle(frame, (x1_0, y1_0), (x2_0, y2_0), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1_0, y1_0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # If only one face is detected
        elif len(faces) == 1:
            closest_face_0 = None
            min_distance_0 = float('inf')

            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                for (x, y, w, h) in faces:
                    distance = np.sqrt((x1 - x)**2 + (y1 - y)**2)
                    label, _ = face_recognizer.predict(gray_img[y:y+h, x:x+w])
                    if label == 0 and distance < min_distance_0:
                        closest_face_0 = (x1, y1, x2, y2)
                        min_distance_0 = distance

            if closest_face_0 is not None:
                x1_0, y1_0, x2_0, y2_0 = closest_face_0
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                cv2.rectangle(frame, (x1_0, y1_0), (x2_0, y2_0), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1_0, y1_0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    return frame

# Main function to capture video from webcam and detect hands
def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = detect_and_predict(frame)

        cv2.imshow('frame', frame)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
