try:
    import unzip_requirements
except ImportError:
    pass

import dlib
import numpy as np
import faceBlendCommon as fbc
import cv2
import matplotlib
import sys,time

matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Landmark model location
PREDICTOR_PATH =  "./shape_predictor_68_face_landmarks.dat"

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


def swap_face_image(image_bytes1, image_bytes2):
    try:
        print('swap-face: start')
        im_arr1 = np.frombuffer(image_bytes1, dtype=np.uint8)
        img1 = cv2.imdecode(im_arr1, flags=cv2.IMREAD_COLOR)
        print('Collect Second Image')
        im_arr2 = np.frombuffer(image_bytes2, dtype=np.uint8)
        img2 = cv2.imdecode(im_arr2, flags=cv2.IMREAD_COLOR)

        print('Collected TWO Image')
        img1Warped = np.copy(img2)
        # Initialize the dlib facial landmakr detector
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # Read array of corresponding points

        print('Getting Landmark')
        points1 = fbc.getLandmarks(detector, predictor, img1)
        points2 = fbc.getLandmarks(detector, predictor, img2)
        # Find convex hull

        print('Getting Hull Index')
        hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
        # Create convex hull lists
        hull1 = []
        hull2 = []
        for i in range(0, len(hullIndex)):
            hull1.append(points1[hullIndex[i][0]])
            hull2.append(points2[hullIndex[i][0]])

        # Calculate Mask for Seamless cloning
        hull8U = []
        for i in range(0, len(hull2)):
            hull8U.append((hull2[i][0], hull2[i][1]))

        mask = np.zeros(img2.shape, dtype=img2.dtype)

        print('Fill Convex Poly')
        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

        # Find Centroid
        m = cv2.moments(mask[:,:,1])
        center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
        
        print('Fill Delaunay')
        # Find Delaunay traingulation for convex hull points
        sizeImg2 = img2.shape
        rect = (0, 0, sizeImg2[1], sizeImg2[0])

        dt = fbc.calculateDelaunayTriangles(rect, hull2)

        # If no Delaunay Triangles were found, quit
        if len(dt) == 0:
            quit()
        
        tris1 = []
        tris2 = []
        for i in range(0, len(dt)):
            tri1 = []
            tri2 = []
            for j in range(0, 3):
                tri1.append(hull1[dt[i][j]])
                tri2.append(hull2[dt[i][j]])

            tris1.append(tri1)
            tris2.append(tri2)

        print('Alpha Blending......')
        # Simple Alpha Blending
        # Apply affine transformation to Delaunay triangles
        for i in range(0, len(tris1)):
            fbc.warpTriangle(img1, img1Warped, tris1[i], tris2[i])
        
        print('Seamless Clone.......')
        # Clone seamlessly.
        output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
        retval, buffer = cv2.imencode('.jpeg', output)

        print('Sending image.....')
        return buffer
    except Exception as e:
        print('align-face: start',repr(e))
        raise(e)

def face_swap(event, context):
    try:
        print('classify_image: start')
        content_type_header = event['headers']['content-type']
        print('classify_image: content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('classify_image: Body loaded')

        picture1 = decoder.MultipartDecoder(body, content_type_header).parts[0]
        picture2 = decoder.MultipartDecoder(body, content_type_header).parts[1]

        image = swap_face_image(image_bytes1=picture1.content, image_bytes2=picture2.content)
        print('Aligned face return') 
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
       
        encode = base64.b64encode(image)
        encode = encode.decode("utf-8") 

        print('Returning Image.......')
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"img": encode})
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
