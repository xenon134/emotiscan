import os; os.system("")  # enable ansi esc. codes
print('Fetching dependancies ... ', end='', flush=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'   # hide annoying tensorflow warnings
import cv2
import numpy as np
import sys

from tensorflow.keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from keras.layers import MaxPooling2D

print('\rBuilding the neural network ... ', end='', flush=True)
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))


emotion_model.load_weights('model.h5')

# Using openCV haarcascade xml detect the bounding boxes of face in the webcam and predict the emotions

cv2.ocl.setUseOpenCL(False)

emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

running = True
def printFps():
    while running:
        fps = '%02d' % round(1/lag)
        print(end=f'\rCamera running at {fps} Hz FPS.', file=sys.stderr)
    print(file=sys.stderr)
import time, threading
t1 = time.time()
lag = float('inf')

try:
    print('\rRequesting camera access ... ', end='', flush=True)
    cap = cv2.VideoCapture(1)
    print(end='\rCamera active.')
    threading.Thread(target=printFps).start()

    sys.stdout = open('nul', 'w', encoding='utf-8')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotions[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', frame)

        lag = time.time() - t1
        t1 = time.time()

        cv2.waitKey(1)  # otherwise it freezes up
        if cv2.getWindowProperty('Video',cv2.WND_PROP_VISIBLE) < 1:  # X button pressed
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    running = False
