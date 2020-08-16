try:
    import unzip_requirements
except ImportError:
    pass

import dlib
import numpy as np
import faceBlendCommon as fbc
import cv2
import matplotlib
from io import BytesIO

matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Landmark model location
PREDICTOR_PATH =  "./shape_predictor_5_face_landmarks.dat"

from PIL import Image

import boto3
import os
import tarfile
import io
import base64
import json

from requests_toolbelt.multipart import decoder

print('Import End....')


# Dimensions of output image
h = 600
w = 600

#Get the face detector
faceDetector = dlib.get_frontal_face_detector()
#The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)


def align_face_image(image_bytes):
    try:
        print('align-face: start')
        im_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        im = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        #Detect landmarks
        print('Detect Landmarks=====')
        points = fbc.getLandmarks(faceDetector, landmarkDetector, im)
        print('Detected  =====')
        points = np.array(points)
        print("Number of detected points: {}".format(len(points)))
        if len(points) != 5:
            return False,""

        # Convert image to floating point in the range 0 to 1
        im = np.float32(im)/255.0
        # Normalize image to output coordinates.
        print('Normalize image to output coordinates.')
        imNorm, points = fbc.normalizeImagesAndLandmarks((h, w), im, points)
        imNorm = np.uint8(imNorm*255)
        print('save Image......123')
        retval, buffer = cv2.imencode('.jpeg', imNorm)
        return True,buffer
    except Exception as e:
        print('align-face: start',repr(e))
        raise(e)

def detect_face(event, context):
    try:
        print('classify_image: start')
        content_type_header = event['headers']['content-type']
        print('classify_image: content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('classify_image: Body loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        result,image = align_face_image(image_bytes=picture.content)
        print('Aligned face return') 
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
       
        if result==True:
            face = "True"
            encode = base64.b64encode(image)
            encode = encode.decode("utf-8") 
        else:
            face = "False"
            encode = ""

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"img": encode, "result": face})
          }
    except Exception as e:
        print('classify_image',repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
