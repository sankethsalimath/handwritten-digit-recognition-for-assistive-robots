####
# Code Developed by Sanketh Salimath, Omkar Tulankar
####


import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
import time
import pyfirmata

board = pyfirmata.Arduino('COM3')

model = load_model("C:/Users/Sanket/AppData/Local/Programs/Python/Python36/projects/set2/model3.h5")
def prediction():
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, gray = cv2.threshold(gray_blur, 120, 255, cv2.THRESH_BINARY)
        #gray = cv2.Canny(gray, 150, 150)
        gray_not = cv2.bitwise_not(gray)
        gray_resize = cv2.resize(gray_not, (28, 28))
        gray2 = gray_resize.astype("float")/255.0
        image = img_to_array(gray2)
        image2 = np.expand_dims(image, axis=0)
        #predict digit
        prediction = model.predict(image2).argmax()
        print(prediction)
        #print(max(prediction))
        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray_not)
        cv2.imshow('resized', gray_resize)
        bit = [1 if x=='1' else 0 for x in "{:08b}".format(prediction)]
        print(bit)
        for count, elem in enumerate(bit):
            board.digital[count+2].write(elem)
            #print("pin number: ", count + 2)
            #print("pin Mode: ", elem)
        #
        #time.sleep(5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ######
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


prediction()
