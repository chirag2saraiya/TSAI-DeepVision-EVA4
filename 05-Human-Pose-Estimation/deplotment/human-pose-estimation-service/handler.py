try:
    import unzip_requirements
except ImportError:
    pass
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image,ImageDraw

import boto3
import os
import tarfile
import io
import base64
import json

import numpy as np
import copy

from config import config
from config import update_config
import pose_resnet

from requests_toolbelt.multipart import decoder

print('import ends....')

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'gauravp-human-pose-estimation-models'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'pose_resnet_50_256x256.pth.tar'

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Main: Creating Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())

        print("Main: Creating Model")
        update_config('256x256_d256x3_adam_lr1e-3.yaml')
        model = pose_resnet.get_pose_net(config, is_train=False)
        print('Main: Model created')
    
        print("Main: Loading Model wts...")
        model.load_state_dict(torch.load(bytestream,map_location=torch.device('cpu')))
        print("Main: Model Loaded...")
except Exception as e:
    print('Main:', repr(e))
    raise(e)

def get_predictions(model,image_bytes):
    # Read input image
    print('Getting Predictions')
    print('Loading Image')
    image = Image.open(io.BytesIO(image_bytes))
    # define transformations for input image
    # model specifies input of size 256x256 so we must resize image
    image_transform = transforms.Compose([
                              transforms.Resize((256,256)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    # indicate model evaluation mode
    model.eval()

    print('Calling model')
    # pass the input through model
    predictions = model(image_transform(image).unsqueeze(0))
    # reduce extra dimension
    predictions = predictions.squeeze(0)
    # get a numpy copy of the prediction tensor
    predictions = copy.deepcopy(predictions.cpu().detach().numpy())
    return predictions

def plot_pose(image_bytes,predictions):
    num_joints,pred_height,pred_width = predictions.shape

    joint_dict = {0:"r_ankle",1:"r_knee", 2: "r_hip",
                3: "l_hip", 4:"l_knee", 5: "l ankle",
                6: "pelvis",7: "thorax",8: "upper_neck",
                9: "head_top",10: "r_wrist", 11:"r_elbow", 
                12: "r_shoulder", 13:"l_shoulder", 
                14:"l_elbow", 15: "l_wrist"}

    joint_pairs = [
    # TORSO
                [9, 8],   # head_top <-> upper_neck
                [8, 7],   # upper_neck <-> thorax
                [7, 6],   # thorax <-> pelvis
    # RIGHT ARM  
                [7, 12],  # thorax <-> r_shoulder
                [12, 11], # r_shoulder <-> r_elbow
                [11, 10], # r_elbow <-> r_wrist
    # LEFT ARM
                [7, 13],  # thorax <-> l_shoulder
                [13, 14], # l_shoulder <-> l_elbow
                [14, 15], # l_elbow <-> l_wrist             
    # RIGHT LEG
                [6, 2], # pelvis <-> r_hip
                [2, 1], # r_hip <-> r_knee
                [1, 0], # r_knee <-> r_ankle
    # LEFT LEG
                [6, 3], # pelvis <-> l_hip
                [3, 4], # l_hip <-> l_knee
                [4, 5] # l_knee <-> l_ankle
    ]

    joint_coords = [np.unravel_index(np.argmax(p, axis=None), p.shape) for p in predictions]


    # Read the image
    print('plot_pose: Reading Image')
    pose_img = Image.open(io.BytesIO(image_bytes))
    (img_width,img_height) = pose_img.size
    print('plot_pose: Input image size:',img_width,'x',img_height)

    
    # scale coordinates to match input image dimensions
    print('Scaling joint coordinates to input image size')
    new_joint_coords = [(j[1]*img_width//pred_width,j[0]*img_height//pred_height) for j in joint_coords]

    pose_img_draw = ImageDraw.Draw(pose_img)
    # plot circles
    radius = 20
    for joint in new_joint_coords:
        bounding_lt = (joint[0]-radius//2,joint[1]-radius//2)
        bounding_rb = (joint[0]+radius//2,joint[1]+radius//2)
        pose_img_draw.ellipse([bounding_lt,bounding_rb],width=3,fill='blue')

    # Draw lines connecting the joint pairs
    for pair in joint_pairs:
        pose_img_draw.line([new_joint_coords[pair[0]],new_joint_coords[pair[1]]], fill='red', width=2)

    return pose_img

def estimate_pose(event, context):
    try:
        print('estimate_pose: start')
        content_type_header = event['headers']['content-type']
        print('estimate_pose: content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('estimate_pose: Body loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        
        predictions = get_predictions(model,picture.content)
        pose_img = plot_pose(picture.content,predictions)

        print('estimate_pose: Return pose image') 
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        print('estimate_pose: Encoding PIL image') 
        buf = io.BytesIO()
        pose_img.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        base64_pil_img = base64.b64encode(byte_im)
        base64_pil_img = base64_pil_img.decode("utf-8") 
        
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"img": base64_pil_img})
          }
    except Exception as e:
        print('encoded_pose_img: Error',repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }    

