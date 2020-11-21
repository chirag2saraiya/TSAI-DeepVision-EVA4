try:
    import unzip_requirements
except ImportError:
    pass

import boto3
import os
import io
import base64
import json
from PIL import Image
import torch

from requests_toolbelt.multipart import decoder
from imageCaptionUtil import caption_image_beam_search

print('import ends....')

print('loading wordmap')
with open('WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json', 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}  # idx2word
print('Done')

print('loading checkpoint')
checkpoint = torch.load('BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar', map_location='cpu')
decoderModel = checkpoint['decoder']
decoderModel = decoderModel.to('cpu')
decoderModel.eval()
encoderModel = checkpoint['encoder']
encoderModel = encoderModel.to('cpu')
encoderModel.eval()
print('done')

#print('Loading Decoder model')
#decoder = torch.load('decoderModel.pth',map_location='cpu')
#print('done')

#print('Loading Encoder model')
#encoder = torch.load('encoderModel.pth',map_location='cpu')
#print('done')



def getCaption(event, context):
    try:
        print('getCaption: start')
        content_type_header = event['headers']['content-type']
        print('getCaption: content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('getCaption: Body loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        
        img = Image.open(io.BytesIO(picture.content))
        print('getCaption: Image opened')

        print('getCaption: Sending image for Image Captioning')

        seq, _ = caption_image_beam_search(encoderModel, decoderModel, img, word_map, 5)

        img_caption = " ".join([rev_word_map[ind] for ind in seq][1:-1])
        print('Caption: ',img_caption)

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"Caption": img_caption})
          }
    except Exception as e:
        print('getCaption: Error',repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }    
