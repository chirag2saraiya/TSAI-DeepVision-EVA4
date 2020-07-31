import pandas as pd
import torch
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os


class DataClass():
    data_dir = ""
    train_dir = ""
    valid_dir = ""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_data_path(self):
        self.train_dir = self.data_dir + 'train/'
        self.valid_dir = self.data_dir + 'val/'
        return self.train_dir,self.valid_dir


"""
Below we take a look at the number of images in each category and the size of the images.
"""
    def show_number_of_data_in_class(self):
        classes = []
        n_train = []
        n_valid = []
        n_total = []
        # Collect data for train Directory
        # Iterate through each category
        for d in os.listdir(self.train_dir):
            classes.append(d)
            n_train.append(len(os.listdir(self.train_dir + d)))
            n_valid.append(len(os.listdir(self.valid_dir + d)))

        n_total = [sum(x) for x in zip(n_train, n_valid) ]
        # Dataframe of Classes
        classDf = pd.DataFrame({'Classes': classes,
                       'n_train': n_train,
                       'n_valid': n_valid,
                       'n_total': n_total} )
        print(classDf)


"""
Image Preprocessing
To prepare the images for our network, we have to resize them to 224 x 224 and normalize each color channel by subtracting a mean value and dividing by a standard deviation.
We will also augment our training data in this stage.
These operations are done using image transforms, which prepare our data for a neural network.

Data Augmentation
Because there are a limited number of images in some categories, we can use image augmentation to artificially increase the number of images "seen" by the network.
This means for training, we randomly resize and crop the images and also flip them horizontally.
A different random transformation is applied each epoch (while training), so the network effectively sees many different versions of the same image.
All of the data is also converted to Torch Tensors before normalization. The validation and testing data is not augmented but is only resized and normalized.
The normalization values are standardized for Imagenet.

"""
    def get_image_transform(self):
        # Image transformations
        image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
                            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                            transforms.RandomRotation(degrees=15),
                            transforms.ColorJitter(),
                            transforms.RandomHorizontalFlip(),
                            transforms.CenterCrop(size=224),  # Image net standards
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])  # Imagenet standards
                            ]),
                        # Validation does not use augmentation
        'val':
        transforms.Compose([
                            transforms.Resize(size=256),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        }
        return image_transforms

"""
To show how augmentation works, we need to write a function that will plot a tensor as an image.
"""
    def imshow_tensor(self, image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        # Set the color channel as the third dimension
        image = image.numpy().transpose((1, 2, 0))
        # Reverse the preprocessing steps
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        # Clip the image pixel values
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        return ax, image
"""
Data Iterators
To avoid loading all of the data into memory at once, we use training DataLoaders.
First, we create a dataset object from the image folders, and then we pass these to a DataLoader.
At training time, the DataLoader will load the images from disk, apply the transformations, and yield a batch.
To train and validation, we'll iterate through all the batches in the respective DataLoader.
One crucial aspect is to shuffle the data before passing it to the network.
This means that the ordering of the image categories changes on each pass through the data (one pass through the data is one training epoch).
"""
    def get_dataloader(self, batch_size):
        image_transforms = self.get_image_transform()
        dataset_train = torchvision.datasets.ImageFolder(root=self.train_dir, transform=image_transforms['train'])
        dataset_valid = torchvision.datasets.ImageFolder(root=self.valid_dir, transform=image_transforms['val'])
        data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True, num_workers=4)
        return data_loader_train,data_loader_valid

    def image_show(self, class_names,  num_image, num_row, num_col, data_loader_name):
        inputs, classes = next(iter(data_loader_name))
        plt.figure(figsize=(15, 15))
        for i in range(num_image):
            ax = plt.subplot( num_row, num_col, i + 1)
            _ = self.imshow_tensor(inputs[i], ax=ax,title=class_names[classes[i]])
            plt.tight_layout()
