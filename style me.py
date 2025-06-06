import cv2
import mediapipe as mp
import pyttsx3


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()


engine = pyttsx3.init()


gesture_dict = {
    (0, 1, 1, 0, 0): "Hello",   
    (0, 1, 0, 0, 0): "Yes",     
    (1, 1, 1, 1, 1): "Stop",    
    (0, 0, 0, 0, 0): "No"       
}


def speak(text):
    engine.say(text)
    engine.runAndWait()


def fingers_up(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_states = []

    
    
    finger_states.append(1 if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x else 0)

    for tip_id in finger_tips[1:]:
        finger_states.append(1 if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y else 0)

    return tuple(finger_states)


cap = cv2.VideoCapture(0)
prev_gesture = None

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = fingers_up(hand_landmarks)

            if gesture in gesture_dict:
                text = gesture_dict[gesture]
                cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                if gesture != prev_gesture:
                    speak(text)
                    prev_gesture = gesture
            else:
                prev_gesture = None

    cv2.imshow("Hand Gesture Interpreter", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()