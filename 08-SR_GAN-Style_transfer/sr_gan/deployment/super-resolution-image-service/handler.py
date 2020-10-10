try:
    import unzip_requirements
except ImportError:
    pass
import torch
from PIL import Image

import boto3
import os
import io
import base64
import json


import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


from requests_toolbelt.multipart import decoder
print('import ends....')


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3BUCKET' in os.environ else 'gauravp-super-resolution-models'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'srgan_generator_JIT_model.pt'

print('Downloading model...')
s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print("Model Loaded...")
        model.eval()

except Exception as e:
    print(repr(e))
    raise(e)        


def get_sr_image(event, context):

    try:
        content_type_header = event['headers']['content-type']
        #print(event['body'])
        body = base64.b64decode(event["body"])
        print("Body Loaded")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
               
	    # load image from bytes using PIL
        image = Image.open(io.BytesIO(picture.content))	    
        image = (ToTensor()(image)).unsqueeze(0)
        print('Input image shape: ',image.shape)
        out = model(image)
        print('Output image shape: ',out.shape)
        pil_img = ToPILImage()(out[0].data.cpu())
	    
        # Generate jpeg Byte array stream
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        base64_pil_img = base64.b64encode(byte_im)
        base64_pil_img = base64_pil_img.decode("utf-8") 
        
        print('generate_image: Returning Image.......')
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
        print('generate_image',repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
