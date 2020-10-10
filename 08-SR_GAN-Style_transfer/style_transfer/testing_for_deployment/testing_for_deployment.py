from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

import copy
import time

from utils import *

style_img = image_loader("./picasso.jpg")
content_img = image_loader("./dancing.jpg")

assert style_img.size() == content_img.size(),"we need to import style and content images of the same size"

#cnn = torch.load('vgg19_model.pth')
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


input_img = content_img.clone()

start = time.perf_counter()
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img,num_steps=50)

stop = time.perf_counter()

print('time taken',(stop-start))
print(type(output))
print(output.shape)
print(output.size())

image = output.cpu().clone()  # we clone the tensor to not do changes on it
image = image.squeeze(0)      # remove the fake batch dimension
image = transforms.ToPILImage()(image)

print(type(image))
print(image.size)
