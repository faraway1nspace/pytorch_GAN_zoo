# one-off script to resize images to k*512, in order to reduce their size before uploading to cloud service

# ARGUMENTS
path_to_db = "../../../data/" # path to old database
path_to_new_db = "/tmp/data/" # path to new database
thres_pixels = 700 # thresshold larger than which, the images will be downscale to

import os
import shutil
import sys

# path to the home-directory, working directory, and repository code (to import)
home_path = os.path.expanduser('~') # get home path
path_to_wd = "Documents/ScriptsPrograms/ml_art/bin/gan_zoo_pytorch/pytorch_GAN_zoo/"
path_to_repo_code = "Documents/ScriptsPrograms/ml_art/bin/gan_zoo_pytorch/pytorch_GAN_zoo/"

# change home directory; add repositry code directory to sys.path
os.chdir(os.path.join(home_path, path_to_wd))
sys.path.append(os.path.join(home_path, path_to_repo_code))

import importlib
#import argparse
import json
import re

from models.utils.utils import getVal, getLastCheckPoint, loadmodule
from models.utils.config import getConfigOverrideFromParser, updateParserWithConfig
import numpy as np

import albumentations as A
import cv2
import magic
from models.utils.image_transform import pil_loader, ToTensorV2

def get_img_dim(path_to_img):
    """uses regex and the output from magic module to extract the text description of image size; return number dimensions"""
    img_info_str = magic.from_file(path_to_img)
    # find', 1234x1234, '
    #m =re.search('\,\s\d+x\d+\,\s',img_info_str)
    m = re.search('\,\s\d+\s*x\s*\d+\,\s',img_info_str)
    if not bool(m):
        raise ValueError("""no resolution found in %s""" % img_info_str)
    # split by x
    res_xy_str  =re.sub('[\s\,]','',m.group()).split('x')
    # make into integers
    res = [int(x) for x in re.sub('[\s\,]','',m.group()).split('x')]
    return res

def check_image_requires_resizing(image_dimensions, threshold):
    """check whether any dimenion is lower than the mandated threshold"""
    return all(list(map(lambda l: l>threshold, image_dimensions)))

def new_dimension( image_dimension, threshold):
    """what is the new dimension of the resized image"""
    min_dim = min(image_dimension)
    idx_min_dim = image_dimension.index(min_dim)
    # get the scale necessary to rescale 
    scale_ = threshold/image_dimension[idx_min_dim]
    # downscaled dimension
    dim_new = [int(d*scale_) for d in image_dimension]
    return dim_new

def downscale(path_to_image, new_dimension, path_to_new_image = None, plot=None):
    """ use albumentations to downscale an image"""
    if plot is None:
        plot = False
    
    img = cv2.imread(path_to_image)
    # ensure proper color
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_image = cv2.resize(img, tuple(new_dimension), interpolation=cv2.INTER_AREA)
    # whether to save an image
    if not (path_to_new_image is None):
        cv2.imwrite(path_to_new_image,new_image)
    
    # whether to plot an image
    if plot:
        plt.imshow(img); plt.show()
    
    return new_image

# make the new path
if not os.path.isdir(path_to_new_db):
    os.mkdir(path_to_new_db)

# make the subdirectories (if they don't exist)
subdirs = [d for d in os.listdir(path_to_db)]
for subdir_ in subdirs:
    if not os.path.isdir(os.path.join(path_to_new_db,subdir_)):
        os.mkdir(os.path.join(path_to_new_db,subdir_))

# get all the files, and their dimensions
paths_to_files = []
for subdir_ in subdirs:
    for f in os.listdir(os.path.join(path_to_db,subdir_)): #
        # paths, old and new
        old_path_ = os.path.join(path_to_db,subdir_,f)
        new_path_ = os.path.join(path_to_new_db,subdir_,f)
        # get dimensions
        dimensions_ = get_img_dim(old_path_)
        paths_to_files.append((old_path_, new_path_, dimensions_))

# loop through files: copy the ones that are 
for path_old, path_new, old_dimension in paths_to_files:
    # check whether an image needs rescaling
    if check_image_requires_resizing(image_dimensions=old_dimension, threshold=thres_pixels):
        dim_new = new_dimension( old_dimension, thres_pixels) # new dimension
        # new image
        new_image = downscale(path_old, dim_new, path_to_new_image = path_new, plot=False)
    else:
        shutil.copyfile(path_old, path_new) 

# play with opening and saving images in ONE color channel format
# i = 0
# i+=1
# path_old, path_new, old_dimension = paths_to_files[i]   #
# path_new_1 = re.sub("/tmp/data/impression_pureabstract/", "/tmp/foo/cv_", path_new)
# path_new_2 = re.sub("../../../data/impression_pureabstract/", "/tmp/foo/orig_", path_old)
# path_new_3 = re.sub("../../../data/impression_pureabstract/", "/tmp/foo/cov_", path_old)

# # copy over the original
# shutil.copyfile(path_old, path_new_2)

# # read in for saving
# img = cv2.imread(path_old)
# cv2.imwrite(path_new_1,img)

# # convert color and save
# img2 = copy.copy(img)
# #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# img2 = cv2.resize(img2, tuple([int(i*0.8) for i in rev(img2.shape[0:2])]), interpolation=cv2.INTER_AREA)
# cv2.imwrite(path_new_3,img2)
