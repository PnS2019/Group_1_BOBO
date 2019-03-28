
import numpy as np
import cv2
import pickle as pkl

# First try to collect data...
# f.e. one can show 4 fingers to the camera, then press and hold '4' with the other hand.
# slightly move the hand which shows 4 fingers
# about 5 pics per second are collected and saved into /data/4

#TODO : Save the data set in a format which is readable for keras


# open the camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow('frame', frame)
i = 0
train_x = []
train_y = []
with open("train.pkl", "w") as f:
    pkl.dump([train_x, train_y], f)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame

    if cv2.waitKey(1) & 0xFF == ord('1'):
        cv2.imwrite('data/1/01_' + str(i)+ '.png',frame)
        cv2.imshow('frame', frame)
        train_x.append(frame)
        trany_y.append(1)
        i=i+1
    if cv2.waitKey(1) & 0xFF == ord('2'):
        cv2.imwrite('data/2/02_' + str(i)+ '.png',frame)
        cv2.imshow('frame', frame)
        i=i+1
    if cv2.waitKey(1) & 0xFF == ord('3'):
        cv2.imwrite('data/3/03_' + str(i)+ '.png',frame)
        cv2.imshow('frame', frame)
        i=i+1
    if cv2.waitKey(1) & 0xFF == ord('4'):
        cv2.imwrite('data/4/04_' + str(i)+ '.png',frame)
        cv2.imshow('frame', frame)
        i=i+1
    if cv2.waitKey(1) & 0xFF == ord('5'):
        cv2.imwrite('data/5/05_' + str(i)+ '.png',frame)
        cv2.imshow('frame', frame)
        i=i+1
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

with open("train.pkl", "w") as f:
    pkl.dump([train_x, train_y], f)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


print("hello")
