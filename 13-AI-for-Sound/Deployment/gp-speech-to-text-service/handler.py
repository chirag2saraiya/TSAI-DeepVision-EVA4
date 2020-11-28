try:
    import unzip_requirements
except ImportError:
    pass

import json
import torch
import torchaudio
import torch.nn.functional as F

import base64
from requests_toolbelt.multipart import decoder


CLASSES = [
    'six', 'nine', 'on', 'left', 'three', 'five', 'go', 'bird', 'seven', 'off', 'wow', 'two', 'stop', 'zero', 'up', 'house', 'happy', 'cat', 'sheila', 'down', 'right', 'four', 'one', 'tree', 'eight', 'bed', 'marvin', 'dog', 'yes', 'no'
]


class SpeechRNN(torch.nn.Module):
  
  def __init__(self):
    super(SpeechRNN, self).__init__()
    
    self.lstm = torch.nn.GRU(input_size = 13, 
                              hidden_size= 256, 
                              num_layers = 2, 
                              batch_first=True)
    
    self.out_layer = torch.nn.Linear(256, 30)
    
    self.softmax = torch.nn.LogSoftmax(dim=1)
    
  def forward(self, x):
    
    out, _ = self.lstm(x)
    
    x = self.out_layer(out[:,-1,:])
    
    return self.softmax(x)


print('Loading Model')    
model = torch.load('speechToTextModel.pth.tar',map_location=torch.device('cpu'))
#model = SpeechRNN()
#model.load_state_dict(torch.load('speechToTextModelWeights.pth',map_location=torch.device('cpu')))
model.eval()

print('Model loaded')


def get_prediction(filename,model):

    mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=13, log_mels=True)

    waveform, _ = torchaudio.load(filename, normalization=True)
            
    # if the waveform is too short (less than 1 second) we pad it with zeroes
    if waveform.shape[1] < 16000:
        waveform = F.pad(input=waveform, pad=(0, 16000 - waveform.shape[1]), mode='constant', value=0)

    # then, we apply the transform
    mfcc = mfcc_transform(waveform).squeeze(0).transpose(0,1).unsqueeze(0)
    print(mfcc.shape)

    output = model(mfcc).max(1)[1].item()
    
    return (CLASSES[output])



def speechToText(event, context):

    try:
        print('speechToText: start')
        content_type_header = event['headers']['content-type']
        print('speechToText: content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('speechToText: Body loaded')

        audio = decoder.MultipartDecoder(body, content_type_header).parts[0]
        
        print('speechToText:Audio loaded')

        # write bytes data into audio file
        audio_filename = '/tmp/test.wav'
        with open(audio_filename, 'wb') as f:
            f.write(audio)
        print('speechToText: File written')

        print('speechToText: Sending file for prediction')
        audioText = get_prediction(audio_filename,model)    

        print('Text: ',audioText)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"audioText": audioText})
          }
    except Exception as e:
        print('speechToText: Error',repr(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }    