### Image Processing
### Face Recognition
with Python

by Sean Sungil Kim



This program detects all faces and facial landmarks within two input images, utilizing the dlib's pre-trained shape predictor (which you can find and download here: http://dlib.net/files/). This program is able to recognize/compare facial landmarks as well.

It is important to notice that it is required to only have 1 face of interest in the first image. This is because the program selects the first detected facial landmarks in the first image to compare to all detected facial landmarks in the second image.

If the first image has multiple faces, the program will STILL RUN and detect all recognized faces. However, it will still only select the first array of facial landmarks for comparison purposes.
