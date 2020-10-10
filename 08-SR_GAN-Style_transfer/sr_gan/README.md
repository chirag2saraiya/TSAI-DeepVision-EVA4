# Image Super-Resolution

### Program Statement:
Train SRGAN on the drone images dataset created. Target is to train an SRGAN network (pre-trained models are fine) to be awesome at drone/flying object Super Resolution.

### Hierarchically-structured taxonomy


### Problem Definition 
Image super-resolution aims at recovering the corresponding HR images from the LR images.

### Image Quality Assessment
The image quality assessment includes:

1. subjective methods based on humans' perception
2. objective computational methods:
    - PSNR
    - SSIM

Objective computational methods are most used (time-saving), but often are unable to capture the human visual perception, and that may lead to large differences in IQA results.

### Peak Signal-to-Noise Ratio - PSNR

For Image SR, PSNR is defined via the maximum pixel value (denoted by L) and the mean squared error (MSE) between images.Since the PSNR is only related to the pixel-level MSE, only caring about the differences between corresponding pixels instead of visual perception, it often leads to poor performance in representing the reconstruction quality in real scenes, where we're usually more concerned with human perceptions.

### Structural Similarity - SSIM

SSIM (proposed to be closer to human perception compared to PSNR) measures the structural similarity between images in terms of luminance, contrast, and structures. For an Image I with N pixels, the luminance LaTeX: \mu_Iμ I and contrast LaTeX: \sigma_Iσ I are estimated as the mean and standard deviation of the intensity of the image, 

### Upsampling Methods
- Nearest-neighbor Interpolation
- Bilinear Interpolation
- Bicubic Interpolation
- Convolutional Methods

### Loss Functions
- Pixel Loss :        measures the pixel-wise difference between two images and mainly include L1 or L2 loss. 
- Charbonnier Loss :  This loss function is used in LapSRN instead of the generic L2 loss. The results show that Charbonnier loss deals better with outliers and                           produces sharper images compared to those generated with L2 loss, which are generally smoother.
- Content Loss or Perceptual Loss : Extract the feature map of HR images and fake HR image from VGG19 and compute the MSE between these two features. 
- Texture Loss :      On account that the reconstructed image should have the same style (color, textures, contrast, etc) with the target image, texture loss is                           introduced in EnhanceNet. This loss function tries to optimize the Gram matrix of feature outputs inspired by the Style Transfer loss                               function.
