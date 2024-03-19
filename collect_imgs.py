import cv2
import sys

count = 0

vidStream = cv2.VideoCapture(0)

while True:
    ret, frame = vidStream.read()
    cv2.imshow('Frame', frame)

    cv2.imwrite(r'E:\CPV\Face\Data\0\image%04i.jpg' %count, frame)
    count += 1

    if cv2.waitKey(10) == 27:
        break
