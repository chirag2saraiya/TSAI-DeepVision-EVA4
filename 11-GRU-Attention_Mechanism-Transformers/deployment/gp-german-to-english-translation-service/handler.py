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

import torch 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dill
from model_utils import *

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"    
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

device = 'cpu'
DEVICE = 'cpu'
print('Import End....')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)



def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)
  



def predict_sentiment(model, sentence,spacy_model,src_fields,DEVICE='cpu'):
    model.eval()
    tokenized = [tok.text for tok in spacy_model.tokenizer(sentence)]
    print(tokenized)
    indexed = [src_fields.vocab.stoi[t] for t in tokenized]
    print(indexed)
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(0)
    print(tensor)
    print(tensor.size())
    pad_index = 0
    src_mask = (tensor != pad_index).unsqueeze(0).to(DEVICE)
    print(src_mask)
    print(src_mask.size())
    src_lengths = torch.LongTensor([tensor.size()[1]]).to(DEVICE)
    print(src_lengths)
    print(src_lengths.size())
    result, _ = greedy_decode(
          model, tensor, src_mask, src_lengths,
          max_len=25, sos_index=TRG.vocab.stoi[SOS_TOKEN], eos_index=TRG.vocab.stoi[EOS_TOKEN])
        
    return result        


def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]
            
    return [str(t) for t in x]


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'gauravp-s11-german-english-translation-models'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'translation_model.pt'
SRC_TEXT_FIELD= os.environ['SRC_TEXT_FIELD'] if 'SRC_TEXT_FIELD' in os.environ else 'SRC_fields.pkl'
TRG_TEXT_FIELD= os.environ['TRG_TEXT_FIELD'] if 'TRG_TEXT_FIELD' in os.environ else 'TRG_fields.pkl'

s3 = boto3.client('s3')


try:
    print('loading spacy')
    os.system('cp -r de_core_news_sm* /tmp/pkgs-from-layer/')
    spacy_de = spacy.load('/tmp/pkgs-from-layer/de_core_news_sm/de_core_news_sm-2.2.5')
    
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Main: Creating model Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Main: Loading Model...")
        model = torch.load(bytestream,map_location=torch.device('cpu'))
        print("Main: Model Loaded...")

    if os.path.isfile(SRC_TEXT_FIELD) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=SRC_TEXT_FIELD)
        print("Main: Creating torchtext Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print('Main: loading SRC torchtext Fields')
        SRC = torch.load(bytestream, pickle_module=dill)
        print("Main: SRC torchtext Fields Model Loaded...")

    if os.path.isfile(TRG_TEXT_FIELD) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=TRG_TEXT_FIELD)
        print("Main: Creating torchtext Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print('Main: loading TRG torchtext Fields')
        TRG = torch.load(bytestream, pickle_module=dill)
        print("Main: TRG torchtext Fields Model Loaded...")

except Exception as e:
    print('Main:', repr(e))
    raise(e)





def hello(event, context):
    
    
    print(event['body'])
    bodyTemp = event["body"]
    print("Body Loaded")
    
    body = json.loads(bodyTemp)
    print(body,type(body))
    inText = body["inText"]
    print(inText)
    print(type(inText))

    #text = "als ich 11 jahre alt war , wurde ich eines morgens von den <unk> heller freude geweckt"
    #print(text)
    result = predict_sentiment(model,inText,spacy_de,SRC,DEVICE)
    output = lookup_words(result,vocab= TRG.vocab)
    outText = ' '.join(output)
    

    response = {
        "statusCode": 200,
        "body": json.dumps({"input": inText , "output":outText})
    }
    

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
