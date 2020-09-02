# Problem Statement

- Collect 10 facial images of 10 people you know (stars, politicians, etc). Add it to [LFW](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz) dataset. 

- Train with the help of [repo](https://github.com/davidsandberg/facenet) and upload the Face Recognition model to AWS Lambda

# Approach:

**Face Detection**

We divided the problem into 2 stages
1. 1st stage consists of Face detection i.e detecting a face and it's coordinates in an image. We did this by using MTCNN architecture.
![](https://miro.medium.com/max/2506/1*ICM3jnRB1unY6G5ZRGorfg.png)

MTCNN consists of 3 networks P-Network, R-Network and O-Network. P-Network provides basic bounding boxes by doing a 12x12 sliding window which then are fine-tuned by R-Network which gives tuned bounding boxes and confidence scores for each box. O-Network takes the outputs of R-Network and gives an output of bounding boxes, confidence scores and facial landmarks. 
The landmarks obtained from the network are helpful in correcting posture of face which is needed for face recognition

![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/03/Pipeline-for-the-Multi-Task-Cascaded-Convolutional-Neural-Network-862x1024.png)

**Face Recognition**
For face recognition we used a Inception-Resnet based architecture to recognize the faces sent by detector. Detector crops the image across the bounding box which is sent to recognizer.

![](https://1.bp.blogspot.com/-O7AznVGY9js/V8cV_wKKsMI/AAAAAAAABKQ/maO7n2w3dT4Pkcmk7wgGqiSX5FUW2sfZgCLcB/s1600/image00.png)

# Training

We have followed [this](https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79) blog for training LFW + 10 Indian politician dataset.

- Amit_Shah
- Arvind_Kejriwal
- Jayalalitha
-	Mamata_Banerjee
-	Mayawati
-	Narendra_Modi
-	Rahul_Gandhi
-	Shashi_Tharoor
-	Shushma_Swaraj
-	Smriti_Irani
-	Sonia_Gandhi
-	Yogi_Adityanath

# Uploading to AWS Lamda

- Uploaded model trained via above method to S3 bucket.
- Used same [dlib](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/tree/master/03-Face-Recognition-Part-1) lambda for detecting and aligning face
- Querying image to train model uploaded to S3 bucket

# Reference

- https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79

- https://github.com/davidsandberg/facenet


