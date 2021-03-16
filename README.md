## Image Processing
### Face Detection and Recognition
using Python

by Sean Sungil Kim



This program detects all faces and facial landmarks within two input images, utilizing the dlib's pre-trained shape predictor (which you can find and download here: http://dlib.net/files/). This program is able to recognize/compare facial landmarks as well.

It is important to notice that it is required to only have 1 face of interest in the first image. This is because the program selects the first detected facial landmarks in the first image to compare to all detected facial landmarks in the second image.

If the first image has multiple faces, the program will STILL RUN and detect all recognized faces in the first and second image. However, it will still only select the first array of facial landmarks for comparison purposes.

First you will need to download the FaceRecognition.py program and the shape predictor in the link provided above, in addition to the two images of interest. They should be in the same directory for convenience. (If they are in separate directories, you need to input the path to the file in the terminal/command prompt).

If all files are downloaded, you need to open terminal/command prompt and set the current directory to the directory the files are in, as shown below:

![alt text](https://github.com/kimx3314/Face-Detection-and-Recognition-without-Complex-Model-Building/blob/master/RESULTS/README_Support_Image1.png)

If you require help on what each of the arguments mean, you can refer to "help" as shown below:

![alt text](https://github.com/kimx3314/Face-Detection-and-Recognition-without-Complex-Model-Building/blob/master/RESULTS/README_Support_Image2.png)

Example code:

#### python FaceRecognition.py --shape_predictor shape_predictor_68_face_landmarks.dat --image_1 image1.jpg --image_2 image7.jpg --num_upsmpl 1

Explanation of the above example code:

The --shape_predictor should be followed by the name of the shape predictor dat file, --image_1 should be followed by the name of the first image, --image_2 should be followed by the name of the second image, --num_upsmpl should be followed by the up-sample value.

A successful example run would look like the following (all files used are attached in this repository):

![alt text](https://github.com/kimx3314/Face-Recognition-without-complex-model-building-/blob/master/README_Support_Image3.png)
