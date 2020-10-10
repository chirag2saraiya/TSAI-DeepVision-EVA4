try:
    import unzip_requirements
except ImportError:
    pass
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import boto3
import os
import tarfile
import io
import base64
import json

from requests_toolbelt.multipart import decoder
from style_transfer_utils import *

device = "cpu"

print('Import End....')


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'gauravp-neural-style-transfer-models'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'vgg19_model.pth'

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Main: Creating Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Main: Loading Model...")
        model = torch.load(bytestream)
        print("Main: Model Loaded...")
except Exception as e:
    print('Main:', repr(e))
    raise(e)


def transform_image(image_bytes):
    try:
        print('transform_image: start')
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        print('transform_image: Image opened')
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print('transform_image: start',repr(e))
        raise(e)


def get_prediction(image_bytes):
    print('get_prediction: start')
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()


def classify_image(event, context):
    try:
        print('classify_image: start')
        content_type_header = event['headers']['content-type']
        #print(event['body'])
        print('classify_image: content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('classify_image: Body loaded')

        style_picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        content_picture = decoder.MultipartDecoder(body, content_type_header).parts[1]
        print('classify_image: pictures loaded')

        style_img = image_loader(style_picture.content)
        content_img = image_loader(content_picture.content)
        
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        input_img = content_img.clone()

        output = run_style_transfer(model, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img,num_steps=100)

        pil_img = transforms.ToPILImage()(output.squeeze(0))

        # Generate jpeg Byte array stream
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
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
