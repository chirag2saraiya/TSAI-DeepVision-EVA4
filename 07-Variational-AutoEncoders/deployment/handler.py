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


from requests_toolbelt.multipart import decoder

print('import ends....')

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'vae-car-s3'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'vae-car-model.pt'

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print("Model Loaded...")

except Exception as e:
    print(repr(e))
    raise(e)   

def get_decoded_car(model,image_bytes):
    # Read input image
    print('Getting Decoded image')
    print('Loading Image')
    image = Image.open(io.BytesIO(image_bytes))
    # define transformations for input image
    # model specifies input of size 96x96 so we must resize image
    image_transform = transforms.Compose([
                              transforms.Resize((96,96)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
    # indicate model evaluation mode
    model.eval()

    print('Calling model')
    # pass the input through model
    op_image = model(image_transform(image).unsqueeze(0))

    return op_image

def get_generated_image(event, context):
    try:
        print('estimate_pose: start')
        content_type_header = event['headers']['content-type']
        print('estimate_pose: content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('estimate_pose: Body loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        
        op_image = get_decoded_car(model,picture.content)
        print("convert image to numpy==========")
        op_np = op_image[0].detach().numpy()
        print(op_np.shape)
        op_np = np.squeeze(op_np,0)
        print("Transpose Image=================")
        print(op_np.shape)
        final_img = np.transpose(op_np, (1, 2, 0))
        print("Convert Numpy to PIL image====================")
        final_img = Image.fromarray(np.uint8(final_img*255))
        print('Base 64 Encode=====================') 
        buf = io.BytesIO()
        final_img.save(buf, format='JPEG')
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


