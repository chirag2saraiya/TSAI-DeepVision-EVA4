try:
    import unzip_requirements
except ImportError:
    pass

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import boto3
import os
import tarfile
import io
import base64
import json
import numpy as np

from generator import Generator

from requests_toolbelt.multipart import decoder

print('Import End....')

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'dcgan-car-model-s3'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'dcgan-car.pt'

s3 = boto3.client('s3')
n_noise = 100
IMAGE_DIM = (32, 32, 3)

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model...")
        #model = torch.load(bytestream)
        G_test = Generator(out_channel=IMAGE_DIM[-1]).to("cpu")
        G_test.eval()
        G_test.load_state_dict(torch.load('G_c.pkl',map_location=torch.device('cpu')))
        print("Model Loaded...")
except Exception as e:
    print(repr(e))
    raise(e)

def get_sample_image(G, n_noise):
    """
        save sample 100 images
    """
    z = torch.randn(10, n_noise).to('cpu')
    y_hat = G(z).view(10, 3, 32, 32).permute(0, 2, 3, 1)
    result = (y_hat.detach().cpu().numpy()+1)/2.
    return result


def generate_image(event, context):
    try:
        #content_type_header = event['headers']['content-type']
        image = get_sample_image(G_test, n_noise)[0] 
        print("Got image....")
        print(type(image))
        #image = np.ascontiguousarray(image)
        #encode = base64.b64encode(image)
        print("Encoded.............")
        buf = io.BytesIO()
        image= Image.fromarray(image, 'RGB')
        image.save(buf, format='JPEG')
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
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
