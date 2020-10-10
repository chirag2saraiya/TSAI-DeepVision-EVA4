import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator


UPSCALE_FACTOR = 4
TEST_MODE = False 
IMAGE_NAME = 'test.jpeg'
MODEL_NAME = 'srgan_generator_JIT.pt'

#model = Generator(UPSCALE_FACTOR).eval()
model = torch.jit.load(MODEL_NAME)
model.eval()


image = Image.open(IMAGE_NAME)
image = (ToTensor()(image)).unsqueeze(0)
print(image.shape)


start = time.time()
out = model(image)
print(out.shape)
elapsed = (time.time() - start)
out_img = ToPILImage()(out[0].data.cpu())
print(out_img.shape)
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME.split('/')[-1])
stop = time.time()
print('time: ',(stop-start),' Secs')