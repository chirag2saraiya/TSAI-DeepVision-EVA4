Static Website Link: http://www.ai-genie.com.s3-website.ap-south-1.amazonaws.com/

# Problem Statement

1- Create a Face Alignment App on AWS Lambda , where if someone uploads a face (check that by using dlib face detector), you return aligned face. Image with more than 1 face is not processed for alignment. 
2- Implement Face Swap Application as extension of above concept

# DLib

![DLib](http://dlib.net/) is an open-source Machine Learning library written in C++ with python binding. It is extremely portable and can be used on multiple platforms (including mobiles).

# DLib's LandMark Detection Model

Dlib has 2 Type of Land mark detection models 
- 68-Point Model
- 5-Point Model

# Face Alignment

Face Alignment is helpful in Face recognition systems.We will be using a 5-point landmark detector released by dlib and also see how to perform Face Alignment using the model.
![aligned]()

# DLib's 5-Point Model

In the new 5-point model, the landmark points consist of 2 points at the corners of the eye; for each eye and one point on the nose-tip. It is shown in the image given below.
![landmark]()

# Face Swap Application

### Input Image
![input]()

### Output Image
![output]()
