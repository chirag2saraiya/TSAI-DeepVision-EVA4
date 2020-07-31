import torch
import torch.nn as nn

from torch import optim, cuda
from torch.optim import lr_scheduler


"""
Function to Load in Pretrained Model
We can refactor all that code into a single function that returns a pretrained model.
This only accepts the vgg16 ,resnet50 and mobilenet_v2 at the moment but can be extended to use other models.
"""
def get_pretrained_model(model_name):
  """Retrieve a pre-trained model from torchvision

  Params
  -------
      model_name (str): name of the model (currently only accepts vgg16 and resnet50)

  Return
  --------
      model (PyTorch model): cnn

  """
  train_on_gpu = cuda.is_available()

  if model_name == 'vgg16':
      model = models.vgg16(pretrained=True)

      # Freeze early layers
      for param in model.parameters():
          param.requires_grad = False
      n_inputs = model.classifier[6].in_features

      # Add on classifier
      model.classifier[6] = nn.Sequential(
          nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
          nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

  elif model_name == 'resnet50':
      model = models.resnet50(pretrained=True)

      for param in model.parameters():
          param.requires_grad = False

      n_inputs = model.fc.in_features
      model.fc = nn.Sequential(
          nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
          nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

  elif model_name == 'mobilenet_v2':
      model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
      # Freeze Early layers
      for param in model.parameters():
        param.requires_grad = False
      # Add custom classifier
      classifierInFeatures = model.classifier[1].in_features
      model.classifier[1] = nn.Sequential(
      nn.Linear(classifierInFeatures, 256), nn.ReLU(), nn.Dropout(0.3),
      nn.Linear(256, 4))

  # Move to gpu and parallelize
  if train_on_gpu:
      model = model.to('cuda')

  #if multi_gpu:
      #model = nn.DataParallel(model)

  return model

def get_optimiser():
  criterion = nn.CrossEntropyLoss()   # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
  # Observe that all parameters are being optimized
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  return criterion,optimizer

def get_scheduler():
  # Decay LR by a factor of 0.1 every 7 epochs
  scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
  return scheduler
