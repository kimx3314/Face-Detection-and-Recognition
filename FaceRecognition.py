'''
Facial Landmarks/Recognition Final Project
by Sean Sungil Kim

Assumptions:
First image can only have 1 face
Second image can have more than 1 face
'''


from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import face_recognition
import matplotlib.pyplot as plt


# building the argparser
parser = argparse.ArgumentParser()
parser.add_argument("-sp", "--shape_predictor", required = True, help = "path to facial landmark predictor")
parser.add_argument("-i1", "--image_1", required = True, help = "path to input image 1")
parser.add_argument("-i2", "--image_2", required = True, help = "path to input image 2")
parser.add_argument("-nu", "--num_upsmpl", default = 1, required = False, help = "up-sample value")
args = vars(parser.parse_args())


# initializing dlib's face detector (HOG-based) and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


# reading, resizing, and converting the 2 inputs to grayscale
# it is given that image1 only has 1 face (face of interest)
image1 = cv2.imread(args["image_1"])
image1 = imutils.resize(image1, width = 500)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

image2 = cv2.imread(args["image_2"])
image2 = imutils.resize(image2, width = 500)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# detecting faces in the image, changing num_upsmpl value is possible below
num_upsmpl = int(args["num_upsmpl"])

rects1 = detector(gray1, num_upsmpl)
rects2 = detector(gray2, num_upsmpl)

face_locs1 = face_recognition.face_locations(image1, num_upsmpl)
face_locs2 = face_recognition.face_locations(image2, num_upsmpl)


# comparing known facial encodings to the unknown encodings
known_encoding = face_recognition.face_encodings(image1, face_locs1)[0]


# looping over the face detections
# first image
for (i, rect) in enumerate(rects1):
	# determining the facial landmarks for the face rectangles
	# converting the facial landmark x, y coordinates to a NumPy array
	shape1 = predictor(gray1, rect)
	shape1 = face_utils.shape_to_np(shape1)

	# converting dlib's rectangle to a OpenCV-style bounding box
	(x1, y1, w1, h1) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image1, (x1 - 10, y1 - 10), (x1 + w1 + 10, y1 + h1 + 10), (0, 255, 0), 2)

	# showing the face
	cv2.putText(image1, "TARGET", (x1 - 10, y1 - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

	# plotting the facial landmarks
	for x, y in shape1:
		cv2.circle(image1, (x, y), 1, (0, 0, 255), -1)


# looping over the face detections
# second image, can have more than 1 face in the image
for (i, rect) in enumerate(rects2):
        # face location recognition/comparison
        # comparing face encodings
        unknown_encoding = face_recognition.face_encodings(image2, face_locs2)[i]
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)

        # deciding if the faces are a match or not
        if results[0]:
                bCol = (0, 255, 0)
                bTxt = "MATCH"
        else:
                bCol = (0, 0, 255)
                bTxt = "NO MATCH"
        
	# determining the facial landmarks for the face rectangles
	# converting the facial landmark (x, y)-coordinates to a NumPy array
        shape2 = predictor(gray2, rect)
        shape2 = face_utils.shape_to_np(shape2)

        # converting dlib's rectangle to a OpenCV-style bounding box
        (x2, y2, w2, h2) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image2, (x2 - 10, y2 - 10), (x2 + w2 + 10, y2 + h2 + 10), bCol, 2)

        # showing the face
        cv2.putText(image2, bTxt, (x2 - 10, y2 - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bCol, 2)

	# plotting the facial landmarks
        for x, y in shape2:
                cv2.circle(image2, (x, y), 1, (0, 0, 255), -1)


# showing the output image with the face detections + facial landmarks
plt.subplot(121), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.subplot(122), plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

plt.show()
