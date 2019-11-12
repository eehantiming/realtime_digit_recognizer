from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import cv2
import numpy as np
from collections import deque
import time

num_classes = 10
input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

## LOAD MODEL HERE
model.load_weights('modelaug_100.h5')

## deque to stabilize output and reduce false positive
window = deque(['nil']*20,maxlen=20)
img_counter = 0 #for saving frames 
start_time = time.time()
frame_id = 1 #to calc FPS

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret , img = cap.read()
    width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    size = int(min(width,height)/2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting to grayscale

    # blur = cv2.GaussianBlur(gray, (5,5), 0) #blurring
    blur = gray

    x, y, w, h = int((width-size)/2), int((height-size)/2), size, size
    thresh  = blur[y:y + h, x:x + w]

    Captured_Image = cv2.resize(thresh, (28,28)) #resizing in 28 * 28 shape
    Captured_Image = cv2.bitwise_not(Captured_Image)
    ret, final = cv2.threshold(Captured_Image, 150, 255, cv2.THRESH_TOZERO)
    cv2.imshow("resized", final)
    
    k=cv2.waitKey(30) & 0xFF # press spacebar to save frame
    if k == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, final)
        print("{} written!".format(img_name))
        img_counter += 1

    final = np.array(final)/255  # creating numpy array and dividing by 255 to normalise
    final = final.reshape(1, 28, 28, 1) #reshape vector to input of model

    y_pred = model.predict(final)
    pred = np.argmax(y_pred)
    if y_pred[0][pred] > 0.9:
    	out = y_pred[0][pred]
    else: 
    	out = 'nil'
    	pred = 'nil'
    print('\r',out,end='',flush = True)
    window.append(pred)
    mode = max(set(window), key=window.count)

    elapsed_time = time.time() - start_time

    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0))
    cv2.putText(img, "Predicted Digit is " + str(mode), (30, 320),cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
    fps = frame_id / elapsed_time
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    cv2.imshow("Window",img)
    frame_id += 1
    
    k=cv2.waitKey(10) & 0xFF # Esc to quit
    if k==27:
        print('\n')
        break

cap.release()
cv2.destroyAllWindows()
