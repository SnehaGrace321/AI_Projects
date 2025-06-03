import cv2
from deepface import DeepFace
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize webcam
cap = cv2.VideoCapture(0)

last_spoken_time = 0
speak_interval = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = results[0]['dominant_emotion']

    # Display emotion on frame
    cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Speak emotion at intervals
    current_time = time.time()
    if current_time - last_spoken_time > speak_interval:
        engine.say(f"You seem {emotion}")
        engine.runAndWait()
        last_spoken_time = current_time

    # Show video frame
    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
