"""Data Exploration
We can do a brief exploration of the available images. There are a total of 100 categories, which I've already split into training (50%), validation (25%), and testing (25%) folders. The data is clean and each class (I use the terms "class" and "category" interchangeably in this notebook) is stored in a separate folder. The architecture of the folders is thus:
/datadir
    /train
            /class_1
            /class_2
            .
            .
    /valid
            /class_1
            /class_2
            .
            .
This is a standard organization of the data for cnns and makes it simple to associate the correct labels with the images.
"""

import subprocess
import sys

try:
    import splitfolders
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'split-folders'])
finally:
    import splitfolders

def split_data_folder(input_folder, output_folder, train_ratio=0.8):
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    test_ratio = 1.0-train_ratio
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(train_ratio, test_ratio))
