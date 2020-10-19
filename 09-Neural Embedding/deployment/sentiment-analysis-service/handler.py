try:
    import unzip_requirements
except ImportError:
    pass

import json
import boto3
import os
import tarfile
import io
import base64
import json
from requests_toolbelt.multipart import decoder

from torchtext import data
import torch
import spacy
import dill
from modelDef import RNN

device = "cpu"

print('Import End....')


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'gauravp-eva-sentiment-analysis-models'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'sentiment_analysis_model.pth'
TORCH_TEXT_FIELD =  os.environ['TORCH_TEXT_FIELD'] if 'TORCH_TEXT_FIELD' in os.environ else 'TEXT_fields.pkl'

s3 = boto3.client('s3')


try:
    print('loading spacy')
    nlp = spacy.load('/tmp/pkgs-from-layer/en_core_web_sm/en_core_web_sm-2.2.5')
    
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Main: Creating model Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Main: Loading Model...")
        model = torch.load(bytestream,map_location=torch.device('cpu'))
        print("Main: Model Loaded...")

    if os.path.isfile(TORCH_TEXT_FIELD) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=TORCH_TEXT_FIELD)
        print("Main: Creating torchtext Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print('Main: loading torchtext Fields')
        TEXT = torch.load(bytestream, pickle_module=dill)
        print("Main: torchtext Fields Model Loaded...")


except Exception as e:
    print('Main:', repr(e))
    raise(e)


def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()




def predictSentiment(event, context):
    
    
    print(event['body'])
    bodyTemp = event["body"]
    print("Body Loaded")
    
    body = json.loads(bodyTemp)
    print(body,type(body))
    inReview = body["inReview"]
    print(inReview)
    print(type(inReview))

    predValue = predict_sentiment(model,inReview)
    print('Predicted value',predValue)
    review = "Positive" if predValue > 0.5 else "Negative"
    
    response = {
        "statusCode": 200,
    #    "body":json.dumps(body)
        "body": json.dumps({"input": inReview , "predictionValue":predValue, "sentiment": review})
    }

    return response

